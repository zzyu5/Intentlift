"""Contract-preserving CUDA lowering-state/schedule search over immutable M0.

Candidate legality is decided only by registered C++ whole-graph probes.
Legacy Source-C priors influence enumeration order and nothing else.  A target
Layer-C plugin is a separate object: it resolves one exact regenerated and
re-admitted candidate by opaque key plus implementation digest.  Runtime
measurements select among candidates that already passed structural admission
and compiled to cubins; measurements are never used as a correctness oracle.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.pipeline import (
    LoweredProgram,
    lower_to_cuda,
    lower_to_cuda_fused,
)
from intentir.backend.cuda.schedules import (
    ELEMENTWISE_FAST_PATH_ID,
    PADDED_ROW_FAST_PATH_ID,
    ROW_FAST_PATH_ID,
    ElementwiseSchedule,
    PaddedRowSchedule,
    RowSchedule,
)
from intentir.backend.contract_capabilities import (
    B_CAP_RANK0_HOST_CAPTURE,
    B_CAP_READONLY_INPUTS,
    derive_search_capabilities,
)
from intentir.backend.materialize import (
    BackendSearchState,
    CPrior,
    MaterializedJ,
    materialize_target_lowering,
)
from intentir.backend.ablation import ExperimentArm, SearchAxes
from intentir.backend.search_and_close import (
    BackendClosureError,
    BackendClosureResult,
    CandidateQDResult,
    ConcreteLoweringState,
    ObservationClass,
    SearchAndCloseController,
)
from intentir.frontend.canonical_j import (
    LOGICAL_FULL_AXIS_STEP_MATERIALIZATION,
    STRUCTURED_SCAN_LOOP_MATERIALIZATION,
)
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.layer_c import (
    ImplementationChoice,
    LayerCBinding,
    LayerCExtensionPoint,
    LayerCInterface,
    TargetIdentity,
    TargetLayerCPlugin,
    TargetLowering,
    TargetPipeline,
    TargetRealizationContract,
    TargetSplit,
    binding_interface_digest,
    canonical_layer_c_json,
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
from intentir.verify.gate1.model import RefinementKind, SemanticArchive

from .program import analyze_j_for_cuda
from .selection import (
    DEFAULT_PASSES,
    RAGGED_FAST_PATH_ID,
    LoweringTarget,
    read_intent_module,
    select_variant,
)


SEARCH_PROTOCOL = "intentir.backend.cuda.measured-search.v5"
CONTRACT_VIEW_VERSION = "intentir.backend.cuda.contract-view.v2"

B_TRANSFORM_EPILOGUE_LOAD_MOTION = (
    "cuda.b.epilogue_readonly_load_motion.v1"
)
B_TRANSFORM_RANK0_HOST_CAPTURE = "cuda.b.rank0_host_value_capture.v1"
B_TRANSFORM_UNIFORM_HOST_CAPTURE = "cuda.b.uniform_host_value_capture.v1"
LINALG_FUSED_FAST_PATH_ID = "cuda.linalg.elementwise-fused.v1"
SPLITK_VARARGS_FAST_PATH_ID = "cuda.splitk.reduce-varargs.v1"
DIRECT_CONTRACTION_FAST_PATH_ID = "cuda.contraction.tiled.v1"
MMA_CONTRACTION_FAST_PATH_ID = "cuda.contraction.mma-sync.v1"
TUNED_LINEAR_FAST_PATH_ID = "cuda.linear.f16_f32.bias_saved.mma_sync.v1"
SHIFTED_EXP_CANCELLATION_FAST_PATH_ID = (
    "cuda.algebra.exp_shift_product_cancellation.v1"
)
GATHER_FAST_PATH_ID = "cuda.gather-scatter.block.v1"
GATHER_COMPOSED_FAST_PATH_ID = "cuda.gather-chain.composed-index.v1"
ROWMASK_ZERO_FILL_FAST_PATH_ID = "cuda.elementwise.rowmask.zero-fill.v1"
SEARCH_PARTITION_UNPARTITIONED = "unpartitioned"
SEARCH_PARTITION_SCHEDULE_SPINE = "schedule_spine"
SEARCH_PARTITION_EXPANDED = "expanded"
CUDA_LAYER_C_BACKEND_KIND = "cuda"


Benchmark = Callable[[LoweredProgram, "CudaSearchCandidate"], float]


class CudaContractError(ValueError):
    """Raised before candidate compilation when complete Layer B is unsupported."""


def _lowering_path_kind_for_fast_path(fast_path_id: str) -> str:
    """Map registered pass identities to stable, non-semantic path labels."""

    mapping = {
        ELEMENTWISE_FAST_PATH_ID: "elementwise",
        PADDED_ROW_FAST_PATH_ID: "padded_row_reduction",
        ROW_FAST_PATH_ID: "row_reduction",
        SHIFTED_EXP_CANCELLATION_FAST_PATH_ID: "elementwise_shifted_exp",
        "cuda.row_scan.sparsemax.sorted_prefix.v1": "row_scan",
        "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1": "row_scan",
        SPLITK_VARARGS_FAST_PATH_ID: "splitk_weighted_merge",
        GATHER_FAST_PATH_ID: "gather",
        GATHER_COMPOSED_FAST_PATH_ID: "gather_composed",
        ROWMASK_ZERO_FILL_FAST_PATH_ID: "elementwise_rowmask_zero_fill",
    }
    # Unknown registered paths keep their exact compiler identity.  This avoids
    # inventing a semantic "family" or a kernel-name recipe merely to group
    # structurally different C++ lowering paths.
    return mapping.get(fast_path_id, fast_path_id)


def derive_default_fast_path_id(
    state: BackendSearchState,
    *,
    intent_module: str | None = None,
) -> str | None:
    """Resolve E1's first C++-admitted pass identity from J and target only."""

    derive_cuda_contract_view(state)
    selection = select_variant(
        intent_module or state.base_ir,
        target=LoweringTarget(
            backend="cuda",
            chip=state.target.architecture,
            features=frozenset(state.target.features),
        ),
    )
    return selection.fast_path_id


def derive_default_lowering_path_kind(
    state: BackendSearchState,
    *,
    intent_module: str | None = None,
) -> str:
    """Resolve E1's admitted path label without compiling a target binary."""

    fast_path_id = derive_default_fast_path_id(
        state,
        intent_module=intent_module,
    )
    return (
        "generic"
        if fast_path_id is None
        else _lowering_path_kind_for_fast_path(fast_path_id)
    )


def schedule_only_cuda_path_identity(
    state: BackendSearchState,
    *,
    intent_module: str | None = None,
) -> tuple[str, str | None]:
    """Choose the production canonical structure held fixed by CF/E2.

    The fixed structure is the first complete, compiler-admitted production
    lowering of J.  Complete N-D/index/broadcast consumption and the standard
    read-once row graph belong to DirectLower rather than Full.  Optional host
    capture, numeric evaluation, materialization, reduction topology and stage
    changes remain disabled by the Schedule-Only axes and are opened by Full.

    For row reductions the canonical collective remains ``gpu.all_reduce``;
    Full may replace it with compact shuffle/shared, multi-accumulator,
    multi-row or multi-stage states.  For paired rotation the canonical
    elementwise pass is the direct consumer of the typed J graph; merely
    avoiding the generic materialized fallback is not a Full contribution.
    """

    fast_path_id = derive_default_fast_path_id(
        state,
        intent_module=intent_module,
    )
    if fast_path_id is None:
        return "generic", None
    # Tensor-core substitution changes the contraction implementation.  The
    # direct contraction is the registered SIMT tiled path; Schedule-Only may
    # tune that structure, while Full additionally opens MMA/Split-K states.
    if fast_path_id.startswith("cuda.linear.") or ".mma" in fast_path_id:
        selection = select_variant(
            intent_module or state.base_ir,
            target=LoweringTarget(
                backend="cuda",
                chip=state.target.architecture,
                features=frozenset(state.target.features),
            ),
            only_fast_path=DIRECT_CONTRACTION_FAST_PATH_ID,
        )
        if selection.fast_path_id is not None:
            return (
                _lowering_path_kind_for_fast_path(selection.fast_path_id),
                selection.fast_path_id,
            )
        return "generic", None
    return _lowering_path_kind_for_fast_path(fast_path_id), fast_path_id


@dataclass(frozen=True)
class CudaContractView:
    """Ephemeral B- and target-conditioned admission view of immutable J.

    The facts are derived by the existing CUDA Layer-B consumer.  They are not
    a second semantic object and never own or rewrite any part of J.
    """

    target: Mapping[str, Any]
    consumed: tuple[str, ...]
    unsupported: tuple[str, ...]
    capabilities: tuple[str, ...]
    boundary_projection: Mapping[str, Any] | None
    version: str = CONTRACT_VIEW_VERSION

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": dict(self.target),
            "supported": self.supported,
            "consumed": list(self.consumed),
            "unsupported": list(self.unsupported),
            "capabilities": list(self.capabilities),
            "boundary_projection": (
                dict(self.boundary_projection)
                if self.boundary_projection is not None
                else None
            ),
        }


def derive_cuda_contract_view(state: BackendSearchState) -> CudaContractView:
    """Consume complete Layer B before enumeration or target compilation."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA ContractView requires BackendSearchState")
    if state.target.family != "cuda":
        raise CudaContractError("CUDA search requires a cuda BackendTarget")
    analysis = analyze_j_for_cuda(
        state.base.j,
        materialization=state.base,
    )
    view = CudaContractView(
        target=state.target.to_dict(),
        consumed=tuple(analysis["consumed"]),
        unsupported=tuple(analysis["unsupported"]),
        capabilities=derive_search_capabilities(state.base),
        boundary_projection=analysis["boundary_projection"],
    )
    if not view.supported:
        raise CudaContractError(
            "CUDA search cannot consume complete Layer B: "
            + ", ".join(view.unsupported)
        )
    return view


@dataclass(frozen=True)
class CudaSearchCandidate:
    candidate_id: str
    lowering_path_kind: str
    fast_path_id: str | None
    pass_argument: str | None
    schedule: Mapping[str, int | bool | str]
    c_order_score: float
    b_transformations: tuple[str, ...] = ()
    required_b_capabilities: tuple[str, ...] = ()
    requested_trace: tuple[str, ...] = ()
    requested_observation_class: ObservationClass = (
        ObservationClass.HARD_PRESERVING
    )
    compiler_pipeline: str = "direct"
    search_partition: str = SEARCH_PARTITION_UNPARTITIONED

    @property
    def generic(self) -> bool:
        return self.compiler_pipeline in {
            "generic_materialized",
            RAGGED_FAST_PATH_ID,
        }

    @property
    def upstream_fused(self) -> bool:
        return self.compiler_pipeline == "linalg_fused"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "lowering_path_kind": self.lowering_path_kind,
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "schedule": dict(self.schedule),
            "c_order_score": self.c_order_score,
            "b_transformations": list(self.b_transformations),
            "required_b_capabilities": list(
                self.required_b_capabilities
            ),
            "requested_trace": list(self.requested_trace),
            "requested_observation_class": (
                self.requested_observation_class.value
            ),
            "compiler_pipeline": self.compiler_pipeline,
            "search_partition": self.search_partition,
        }

    def concrete_state(self, program: LoweredProgram) -> ConcreteLoweringState:
        record = program.transformation_record
        trace = (
            record.transformation_trace
            if record is not None
            else self.requested_trace
        )
        if not trace:
            trace = (f"cuda.{self.lowering_path_kind}.v1",)
        observation_class = (
            ObservationClass.OBSERVATION_CHANGING
            if record is not None and record.observation_changing
            else self.requested_observation_class
        )
        actual_schedule = dict(self.schedule)
        if record is not None:
            actual_schedule.update(
                {
                    "rows_per_block": record.rows_per_block,
                    "warps_per_row": record.warps_per_row,
                    "d_partitions": record.d_partitions,
                    "accumulator_chains": record.accumulator_chains,
                }
            )
        if record is not None and record.launch_plans:
            # The candidate records the requested pass options.  The concrete
            # state records what C++ actually emitted: in particular row
            # vector-width=0 is an auto request that may become native vec4.
            # Keeping these two layers distinct prevents an auto-vectorized
            # winner from being reported as a scalar implementation.
            primary = record.launch_plans[0]
            actual_schedule.update(
                {
                    "block_threads": primary.block[0],
                    "actual_grid_blocks": primary.grid[0],
                    "actual_grid_stride_unroll": record.grid_stride_unroll,
                    "actual_logical_tile_elements": record.logical_tile_elements,
                    "actual_smem_bytes": primary.smem,
                    "actual_per_thread_tile": primary.per_thread_tile,
                    "vector_width": primary.vector_width,
                    "actual_pipeline_depth": primary.pipeline_depth,
                    "actual_split_factor": primary.split_factor,
                    "actual_launch_plan_count": len(record.launch_plans),
                    "warps_per_row": record.warps_per_row,
                }
            )
            if record.prefix_packet_width:
                actual_schedule["prefix_packet_width"] = (
                    record.prefix_packet_width
                )
        return ConcreteLoweringState(
            state_id=self.candidate_id,
            lowering_path_kind=(
                record.lowering_path_kind
                if record is not None
                else self.lowering_path_kind
            ),
            transformation_trace=tuple(trace),
            schedule=actual_schedule,
            observation_class=observation_class,
            compiler_record=program.to_record(),
        )


def _generic_cuda_candidate(*, ragged: bool = False) -> CudaSearchCandidate:
    if ragged:
        return CudaSearchCandidate(
            _candidate_id("generic", None, RAGGED_FAST_PATH_ID),
            "generic",
            None,
            None,
            {},
            0.0,
            requested_trace=(RAGGED_FAST_PATH_ID,),
            compiler_pipeline=RAGGED_FAST_PATH_ID,
        )
    return CudaSearchCandidate(
        _candidate_id("generic", None, None),
        "generic",
        None,
        None,
        {},
        0.0,
        requested_trace=("cuda.generic.intent_to_linalg.v1",),
        compiler_pipeline="generic_materialized",
    )


@dataclass(frozen=True)
class SearchAttempt:
    candidate: CudaSearchCandidate
    status: str
    reason: str
    compile_ms: float | None = None
    latency_us: float | None = None
    lowering_record: Mapping[str, Any] | None = None
    concrete_state: ConcreteLoweringState | None = None
    qd_search: CandidateQDResult | None = None
    finalist_samples_us: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "compile_ms": self.compile_ms,
            "latency_us": self.latency_us,
            "lowering": (
                dict(self.lowering_record)
                if self.lowering_record is not None
                else None
            ),
            "concrete_state": (
                self.concrete_state.to_dict()
                if self.concrete_state is not None
                else None
            ),
            "qd_search": (
                self.qd_search.to_dict()
                if self.qd_search is not None
                else None
            ),
            "finalist_samples_us": list(self.finalist_samples_us),
        }


@dataclass(frozen=True)
class CudaSearchResult:
    state: BackendSearchState
    contract_view: CudaContractView
    attempts: tuple[SearchAttempt, ...]
    winner: CudaSearchCandidate | None
    winner_program: LoweredProgram | None
    winner_latency_us: float | None
    benchmark_key: str | None
    elapsed_ms: float
    time_to_best_ms: float | None
    finalist_count: int
    finalist_remeasurements: int
    b_transformations_enabled: bool
    winner_state: ConcreteLoweringState | None = None
    backend_closure: BackendClosureResult | None = None
    backend_closure_error: str | None = None
    ablation_axes: SearchAxes | None = None
    fixed_lowering_path_kind: str | None = None
    fixed_fast_path_id: str | None = None
    hard_admission: tuple[Mapping[str, Any], ...] = ()
    candidate_universe_count: int = 0
    candidate_budget: int = 0
    budget_consumed: int = 0
    c_fixed_candidate_id: str | None = None
    finalist_actual_count: int = 0
    schedule_spine_candidate_count: int = 0
    expanded_candidate_count: int = 0

    def to_record(self) -> dict[str, Any]:
        best_latency: float | None = None
        best_candidate: str | None = None
        curve: list[dict[str, Any]] = []
        consumed_budget = 0
        for attempt_index, attempt in enumerate(self.attempts, start=1):
            consumed_budget += 1
            improved = (
                attempt.latency_us is not None
                and (
                    best_latency is None
                    or attempt.latency_us < best_latency
                )
            )
            if improved:
                best_latency = attempt.latency_us
                best_candidate = attempt.candidate.candidate_id
            curve.append(
                {
                    "candidate_attempt": attempt_index,
                    "candidate_budget": consumed_budget,
                    "attempt_budget": consumed_budget,
                    "candidate_id": attempt.candidate.candidate_id,
                    "status": attempt.status,
                    "latency_us": attempt.latency_us,
                    "improved": improved,
                    "best_latency_us": best_latency,
                    "best_candidate_id": best_candidate,
                }
            )

        def first_attempt(predicate: Callable[[SearchAttempt], bool]) -> int | None:
            return next(
                (
                    index
                    for index, attempt in enumerate(self.attempts, start=1)
                    if predicate(attempt)
                ),
                None,
            )

        admitted = tuple(
            item
            for item in self.attempts
            if item.concrete_state is not None
            and item.status in {"compiled", "measured", "benchmark_failed"}
        )
        actual_records = tuple(
            record
            for item in admitted
            for record in (
                (
                    item.concrete_state.compiler_record.get(
                        "transformation_record"
                    )
                    if item.concrete_state is not None
                    else None
                ),
            )
            if isinstance(record, Mapping)
        )
        reduction_topologies = sorted(
            {
                str(record["reduction_topology"])
                for record in actual_records
                if record.get("reduction_topology") not in {None, "none"}
            }
        )
        rows_per_block_values = sorted(
            {
                int(record.get("rows_per_block", 1))
                for record in actual_records
            }
        )
        d_partition_values = sorted(
            {
                int(record.get("d_partitions", 1))
                for record in actual_records
            }
        )
        accumulator_chain_values = sorted(
            {
                int(record.get("accumulator_chains", 1))
                for record in actual_records
            }
        )
        actual_fma_counts = [
            int(record.get("fma_count", 0)) for record in actual_records
        ]
        approximation_modes = sorted(
            {
                str(record.get("approximation_mode"))
                for record in actual_records
                if record.get("approximation_mode") not in {None, "none"}
            }
        )
        status_counts = {
            status: sum(item.status == status for item in self.attempts)
            for status in (
                "compiled",
                "measured",
                "rejected",
                "compile_failed",
                "qd_rejected",
                "missing_qd",
                "benchmark_failed",
            )
        }
        admitted_pipeline_ids = sorted(
            {
                str(item.concrete_state.compiler_record.get("pipeline_id"))
                for item in admitted
                if item.concrete_state is not None
                and item.concrete_state.compiler_record.get("pipeline_id")
            }
        )
        materialized_kernel_counts = [
            int(item.concrete_state.compiler_record.get("kernel_count", 0))
            for item in admitted
            if item.concrete_state is not None
            and item.concrete_state.compiler_record.get("pipeline_id")
            == "cuda.pipeline.linalg-materialized.v1"
        ]
        whole_graph_kernel_counts = [
            int(item.concrete_state.compiler_record.get("kernel_count", 0))
            for item in admitted
            if item.concrete_state is not None
            and item.concrete_state.compiler_record.get("pipeline_id")
            != "cuda.pipeline.linalg-materialized.v1"
        ]
        fusion_pair_observed = (
            any(count > 1 for count in materialized_kernel_counts)
            and any(count == 1 for count in whole_graph_kernel_counts)
        )

        # Formal marginal gains are computed from candidates remeasured inside
        # this one search visit.  Never compare cached/initial ranking samples
        # against fresh finalist medians, and never use a separately launched E2
        # process as the primary lowering attribution for E3.
        fresh_attempts = tuple(
            item
            for item in self.attempts
            if item.latency_us is not None and item.finalist_samples_us
        )

        def best_attempt(
            pool: tuple[SearchAttempt, ...],
        ) -> SearchAttempt | None:
            return (
                min(pool, key=lambda item: item.latency_us)
                if pool
                else None
            )

        fresh_global_best = best_attempt(fresh_attempts)
        fresh_spine = tuple(
            item
            for item in fresh_attempts
            if item.candidate.search_partition
            == SEARCH_PARTITION_SCHEDULE_SPINE
        )
        fresh_expanded = tuple(
            item
            for item in fresh_attempts
            if item.candidate.search_partition == SEARCH_PARTITION_EXPANDED
        )
        spine_best = best_attempt(fresh_spine)
        expanded_only_best = best_attempt(fresh_expanded)
        schedule_spine_ablation = None
        if (
            self.ablation_axes is not None
            and self.ablation_axes.search_structural_states
            and spine_best is not None
            and fresh_global_best is not None
        ):
            assert spine_best.latency_us is not None
            assert fresh_global_best.latency_us is not None
            schedule_spine_ablation = {
                "same_run_measurement": True,
                "same_total_search_visit": True,
                "schedule_spine_candidate_count": (
                    self.schedule_spine_candidate_count
                ),
                "schedule_spine_measured_count": len(fresh_spine),
                "expanded_candidate_count": self.expanded_candidate_count,
                "expanded_state_measured_count": len(fresh_expanded),
                "schedule_spine_best_candidate_id": (
                    spine_best.candidate.candidate_id
                ),
                "schedule_spine_best_us": spine_best.latency_us,
                "schedule_spine_best_batch_us": list(
                    spine_best.finalist_samples_us
                ),
                # "expanded best" is the nested E3 global best, so it may be
                # the spine winner and the gain is guaranteed not to fall below
                # one merely because a new state loses.
                "expanded_best_candidate_id": (
                    fresh_global_best.candidate.candidate_id
                ),
                "expanded_best_us": fresh_global_best.latency_us,
                "expanded_best_batch_us": list(
                    fresh_global_best.finalist_samples_us
                ),
                "expanded_over_schedule_spine_speedup": (
                    spine_best.latency_us / fresh_global_best.latency_us
                ),
                "expanded_state_best_candidate_id": (
                    expanded_only_best.candidate.candidate_id
                    if expanded_only_best is not None
                    else None
                ),
                "expanded_state_best_us": (
                    expanded_only_best.latency_us
                    if expanded_only_best is not None
                    else None
                ),
            }

        c_fixed_schedule_ablation = None
        if (
            self.ablation_axes is not None
            and self.ablation_axes.baseline_kind is None
            and not self.ablation_axes.search_structural_states
            and self.c_fixed_candidate_id is not None
            and fresh_global_best is not None
        ):
            c_fixed_attempt = next(
                (
                    item
                    for item in fresh_attempts
                    if item.candidate.candidate_id
                    == self.c_fixed_candidate_id
                ),
                None,
            )
            if c_fixed_attempt is not None:
                assert c_fixed_attempt.latency_us is not None
                assert fresh_global_best.latency_us is not None
                c_fixed_schedule_ablation = {
                    "same_run_measurement": True,
                    "same_total_search_visit": True,
                    "c_fixed_candidate_id": self.c_fixed_candidate_id,
                    "c_fixed_us": c_fixed_attempt.latency_us,
                    "c_fixed_batch_us": list(
                        c_fixed_attempt.finalist_samples_us
                    ),
                    "schedule_only_best_candidate_id": (
                        fresh_global_best.candidate.candidate_id
                    ),
                    "schedule_only_best_us": fresh_global_best.latency_us,
                    "schedule_only_best_batch_us": list(
                        fresh_global_best.finalist_samples_us
                    ),
                    "schedule_over_c_fixed_speedup": (
                        c_fixed_attempt.latency_us
                        / fresh_global_best.latency_us
                    ),
                }

        return {
            "protocol": SEARCH_PROTOCOL,
            "materialization": self.state.base.to_record(),
            "attach_c": self.state.to_record(),
            "contract_view": self.contract_view.to_dict(),
            "benchmark_key": self.benchmark_key,
            "attempts": [item.to_dict() for item in self.attempts],
            "hard_admission": [dict(item) for item in self.hard_admission],
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "winner_latency_us": self.winner_latency_us,
            "winner_state": (
                self.winner_state.to_dict()
                if self.winner_state is not None
                else None
            ),
            "backend_closure": (
                self.backend_closure.to_dict()
                if self.backend_closure is not None
                else None
            ),
            "backend_closure_error": self.backend_closure_error,
            "ablation_axes": (
                self.ablation_axes.to_dict()
                if self.ablation_axes is not None
                else None
            ),
            "fixed_lowering_path_kind": self.fixed_lowering_path_kind,
            "fixed_fast_path_id": self.fixed_fast_path_id,
            "schedule_spine_ablation": schedule_spine_ablation,
            "c_fixed_schedule_ablation": c_fixed_schedule_ablation,
            "registered_state_axes": {
                "reduction_topology": True,
                "fma_contraction": True,
                "precision_mode": False,
                "approximation_mode": True,
                "per_thread_reduction_tree": True,
                "precision_reason": (
                    "no typed contraction in the frozen 18-kernel catalog"
                ),
            },
            "elapsed_ms": self.elapsed_ms,
            "time_to_best_ms": self.time_to_best_ms,
            "finalist_tournament": {
                "count": self.finalist_actual_count,
                "requested_count": self.finalist_count,
                "actual_count": self.finalist_actual_count,
                "remeasurements_per_candidate": self.finalist_remeasurements,
                "aggregation": (
                    "median of fresh round-robin measurements; initial ranking "
                    "sample excluded"
                ),
            },
            "b_search": {
                "transformations_enabled": self.b_transformations_enabled,
                "available_capabilities": list(
                    self.contract_view.capabilities
                ),
                "transformation_candidate_count": sum(
                    bool(item.candidate.b_transformations)
                    for item in self.attempts
                ),
            },
            "actual_state_evidence": {
                "admitted_fast_path_ids": sorted(
                    {
                        item.candidate.fast_path_id
                        for item in admitted
                        if item.candidate.fast_path_id is not None
                    }
                ),
                "admitted_lowering_path_kinds": sorted(
                    {
                        item.concrete_state.lowering_path_kind
                        for item in admitted
                        if item.concrete_state is not None
                    }
                ),
                "admitted_pipeline_ids": admitted_pipeline_ids,
                "materialized_kernel_counts": materialized_kernel_counts,
                "whole_graph_kernel_counts": whole_graph_kernel_counts,
                "concrete_post_pass_mlir_states": sum(
                    item.concrete_state is not None for item in admitted
                ),
                "concrete_topology_states": len(
                    {
                        (
                            item.concrete_state.lowering_path_kind,
                            tuple(item.concrete_state.transformation_trace),
                        )
                        for item in admitted
                        if item.concrete_state is not None
                    }
                ),
                "reduction_topologies": reduction_topologies,
                "rows_per_block_values": rows_per_block_values,
                "d_partition_values": d_partition_values,
                "accumulator_chain_values": accumulator_chain_values,
                "actual_fma_counts": actual_fma_counts,
                "approximation_modes": approximation_modes,
                "structural_path_expansion_observed": (
                    len(admitted_pipeline_ids) > 1
                    or len(
                        {
                            item.candidate.fast_path_id
                            for item in admitted
                            if item.candidate.fast_path_id is not None
                        }
                    )
                    > 1
                ),
                "reduction_topology_expansion_observed": (
                    len(reduction_topologies) > 1
                ),
                "fma_contraction_observed": any(
                    count > 0 for count in actual_fma_counts
                ),
                "fusion_materialization_pair_observed": (
                    fusion_pair_observed
                ),
                "precision_or_approximation_observed": (
                    len(approximation_modes) > 1
                    or "native_tanh" in approximation_modes
                ),
                "schedule_spine_concrete_states": sum(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_SCHEDULE_SPINE
                    for item in admitted
                ),
                "expanded_concrete_states": sum(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                    for item in admitted
                ),
                "expanded_state_search_observed": any(
                    item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                    for item in admitted
                ),
                "evidence_rule": (
                    "count only compiler-admitted, compiled post-pass MLIR; "
                    "requested toggles and duplicate MLIR do not count"
                ),
            },
            "budget_curve": curve,
            "counts": {
                "candidate_universe": self.candidate_universe_count,
                "candidate_budget": self.candidate_budget,
                "budget_consumed": self.budget_consumed,
                "hard_paths_probed": len(self.hard_admission),
                "hard_paths_admitted": sum(
                    item.get("admitted") is True for item in self.hard_admission
                ),
                "hard_paths_rejected": sum(
                    item.get("admitted") is False for item in self.hard_admission
                ),
                "planned": len(self.attempts),
                "admitted": sum(
                    item.status in {"compiled", "measured"}
                    for item in self.attempts
                ),
                "measured": sum(
                    item.latency_us is not None for item in self.attempts
                ),
                "rejected": sum(
                    item.status == "rejected" for item in self.attempts
                ),
                "compile_failed": sum(
                    item.status == "compile_failed" for item in self.attempts
                ),
                "qd_rejected": status_counts["qd_rejected"],
                "missing_qd": status_counts["missing_qd"],
                "benchmark_failed": status_counts["benchmark_failed"],
                "status_counts": status_counts,
                "first_admitted_attempt": first_attempt(
                    lambda item: item.status in {"compiled", "measured"}
                ),
                "first_measured_attempt": first_attempt(
                    lambda item: item.latency_us is not None
                ),
                "winner_attempt": (
                    first_attempt(
                        lambda item: self.winner is not None
                        and item.candidate.candidate_id
                        == self.winner.candidate_id
                    )
                    if self.winner is not None
                    else None
                ),
            },
        }


def _candidate_id(
    lowering_path_kind: str,
    fast_path_id: str | None,
    pass_argument: str | None,
) -> str:
    compiler_path = fast_path_id or "generic"
    options = pass_argument or "default"
    return f"{lowering_path_kind}:{compiler_path}:{options}"


def _cuda_candidate_registry_coordinates(
    candidate: CudaSearchCandidate,
) -> dict[str, Any]:
    """Return closed registry coordinates without compiler command text."""

    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA candidate coordinates require CudaSearchCandidate")
    return {
        "backend_kind": CUDA_LAYER_C_BACKEND_KIND,
        "lowering_path_kind": candidate.lowering_path_kind,
        "fast_path_id": candidate.fast_path_id,
        "schedule": dict(candidate.schedule),
        "b_transformations": list(candidate.b_transformations),
        "required_b_capabilities": list(candidate.required_b_capabilities),
        "requested_trace": list(candidate.requested_trace),
        "requested_observation_class": (
            candidate.requested_observation_class.value
        ),
        "compiler_pipeline": candidate.compiler_pipeline,
    }


def cuda_candidate_key(candidate: CudaSearchCandidate) -> str:
    """Return an opaque key over safe, regenerated registry coordinates.

    Candidate ids and pass arguments may contain compiler command text, so
    neither is transported in target Layer C.  Search scores and partition
    labels are likewise excluded because they do not identify an executable
    implementation.
    """

    payload = canonical_layer_c_json(
        _cuda_candidate_registry_coordinates(candidate)
    )
    return "cuda.candidate." + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cuda_candidate_digest(candidate: CudaSearchCandidate) -> str:
    """Digest complete trusted compile content for stale-plugin detection.

    The internal pass argument is covered by this one-way digest, but is never
    placed in the plugin or accepted back as caller-authored compiler input.
    """

    payload = _cuda_candidate_registry_coordinates(candidate)
    payload["internal_pass_argument"] = candidate.pass_argument
    return layer_c_content_digest(payload)


def _cuda_loop_attachment(
    j: SemanticArchive,
) -> tuple[str, int | str | None] | None:
    if not isinstance(j, SemanticArchive):
        raise TypeError("CUDA Layer-C interface requires SemanticArchive")
    records = {
        record["path"]: record
        for record in j_elaboration(j).operation_records
    }
    typed_rows = j_elaboration(j).typed_operation_rows
    if typed_rows is None:
        raise CudaContractError(
            "CUDA structured loop requires canonical typed operation records"
        )
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
            raise CudaContractError(
                "CUDA structured loop lacks a stable J node attachment"
            )
        if authored_by_path.get(path, node_id) != node_id:
            raise CudaContractError(
                "CUDA structured loop node id disagrees with canonical J"
            )
        if operation.node_id is not None and operation.node_id != node_id:
            raise CudaContractError(
                "CUDA structured loop typed id disagrees with canonical J"
            )
        loops.append((node_id, operation))
    if not loops:
        return None
    if len(loops) != 1:
        raise CudaContractError(
            "CUDA Layer-C currently requires at most one top-level loop"
        )
    node_id, loop = loops[0]
    if loop.trip is None:
        raise CudaContractError("CUDA structured loop lacks a trip axis")
    # A dynamic value trip is controlled by a runtime tensor, not by a
    # semantic J dimension.  It is therefore a subject-only attachment: no
    # target tile may be invented for it and the optional tile extension point
    # must remain absent.
    return node_id, loop.trip.axis


def cuda_layer_c_interface(j: SemanticArchive) -> LayerCInterface:
    """Declare the complete pre-materialization CUDA extension surface."""

    attachment = _cuda_loop_attachment(j)
    if attachment is None:
        return LayerCInterface()
    subject_ref, axis_ref = attachment
    if axis_ref is None:
        return LayerCInterface()
    interface = LayerCInterface(
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
    validate_layer_c_attachments(j, interface)
    return interface


def _cuda_plugin_bindings(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
    *,
    loop_schedule_tile: int | None = None,
) -> tuple[LayerCBinding, ...]:
    del candidate
    if loop_schedule_tile is None:
        return ()
    interface = cuda_layer_c_interface(state.base.j)
    if not interface.points:
        raise CudaContractError(
            "CUDA dynamic value loop has no semantic axis and cannot bind "
            "loop_schedule_tile"
        )
    return (LayerCBinding("loop_schedule_tile", loop_schedule_tile),)


def _cuda_target_realization_contract(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
    choice: ImplementationChoice,
    *,
    loop_schedule_tile: int | None,
) -> TargetRealizationContract:
    """Project the admitted CUDA candidate into the shared typed grammar."""

    attachment = _cuda_loop_attachment(state.base.j)
    subject_ref, axis_ref = attachment if attachment is not None else (None, None)
    requested_depth = next(
        (
            candidate.schedule[name]
            for name in ("pipeline_depth", "pipeline_stages", "stage_count")
            if name in candidate.schedule
        ),
        1,
    )
    depth = (
        requested_depth
        if isinstance(requested_depth, int) and not isinstance(requested_depth, bool)
        and requested_depth > 0
        else 1
    )
    pipeline = TargetPipeline(
        stage=candidate.compiler_pipeline,
        depth=depth,
        # ``loop_schedule_tile`` is a split/tile factor.  It must never be
        # reused as a pipeline-depth binding: the two values have different
        # meanings and can legitimately differ for one candidate.
        slot=None,
        subject_ref=subject_ref,
        axis_ref=axis_ref,
    )
    splits: tuple[TargetSplit, ...] = ()
    if loop_schedule_tile is not None:
        if subject_ref is None or axis_ref is None:
            raise CudaContractError(
                "CUDA target tile requires a J loop attachment"
            )
        splits = (
            TargetSplit(
                slot="loop_schedule_tile",
                subject_ref=subject_ref,
                axis_ref=axis_ref,
                factor=loop_schedule_tile,
            ),
        )
    return TargetRealizationContract(
        pipeline=(pipeline,),
        split=splits,
        lowering=TargetLowering(
            backend_kind=choice.backend_kind,
            candidate_key=choice.candidate_key,
            candidate_digest=choice.candidate_digest,
        ),
    )


def _cuda_target_binding_context(
    state: BackendSearchState,
    contract_view: CudaContractView,
    *,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
) -> TargetBindingContext:
    target_interface = cuda_layer_c_interface(state.base.j)
    return TargetBindingContext(
        j=state.base.j,
        target=state.target,
        interface=target_interface,
        capabilities=contract_view.capabilities,
        allowed_authorizations=cuda_allowed_authorizations(state),
        # C_target is not Source-C.  Shared E_C is populated only when the
        # backend caller supplies it explicitly; omission means this context
        # binds a target-only plugin.
        e_c=e_c,
    )


def cuda_allowed_authorizations(
    state: BackendSearchState,
) -> tuple[str, ...]:
    """Return only J authorizations that a CUDA candidate can address.

    This is intentionally narrower than ``J.boundary``.  Source-side index
    widening, unrelated casts, and permissions for a backend operation that
    CUDA does not currently register are not generic CUDA capabilities.  The
    candidate-specific resolver below repeats the kind/subject/dtype checks
    after it has resolved the opaque candidate key.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA authorization derivation requires BackendSearchState")
    nodes = _cuda_node_operations(state.base.j)
    value_types = j_elaboration(state.base.j).value_type_by_name

    def result_dtype(operation: Any) -> str | None:
        result = getattr(operation, "result", None)
        value_type = value_types.get(result) if isinstance(result, str) else None
        return value_type.dtype if value_type is not None else None
    reduction_subjects = {
        item.node_ref
        for item in state.base.j.boundary.reductions
        if item.node_ref in nodes and nodes[item.node_ref].op in {"reduce", "scan"}
    }
    contraction_subjects = {
        node_ref
        for node_ref, operation in nodes.items()
        if operation.op == "contraction"
    }
    supported = {
        item.auth_id
        for item in state.base.j.boundary.realization_authorizations
        if (
            item.kind is RefinementKind.REASSOCIATION
            and item.subject_ref in reduction_subjects
        )
        or (
            item.kind is RefinementKind.ACCUMULATION_DTYPE
            and item.subject_ref in contraction_subjects
            and item.target_dtype is not None
            and item.target_dtype.casefold() == "f32"
        )
    }
    return tuple(sorted(supported))


def _cuda_node_operations(
    j: SemanticArchive,
) -> dict[str, Any]:
    """Resolve stable J node ids to their canonical Intent operations."""

    result = j_elaboration(j).construction_records.typed_operation_by_ref(j.node_ids)
    if result is None:
        raise ValueError("CUDA search requires canonical operation records")
    return {
        node_ref: result[node_ref]
        for node_ref in j.node_ids
        if node_ref in result
    }


def _cuda_is_floating_dtype(dtype: Any) -> bool:
    if not isinstance(dtype, str):
        return False
    normalized = dtype.removeprefix("torch.").casefold()
    return normalized.startswith(("f", "bf", "float", "half"))


def _cuda_reduction_candidate(candidate: CudaSearchCandidate) -> bool:
    fast_path_id = candidate.fast_path_id or ""
    return (
        candidate.lowering_path_kind in {
            "row_reduction",
            "padded_row_reduction",
            "row_scan",
            "splitk_weighted_merge",
        }
        or fast_path_id.startswith("cuda.row_reduction.")
        or fast_path_id.startswith("cuda.row_scan.")
        or fast_path_id.startswith("cuda.splitk.")
        or fast_path_id.startswith("cuda.loop_contraction.")
    )


def _cuda_contraction_candidate(candidate: CudaSearchCandidate) -> bool:
    fast_path_id = candidate.fast_path_id or ""
    return fast_path_id in {
        DIRECT_CONTRACTION_FAST_PATH_ID,
        MMA_CONTRACTION_FAST_PATH_ID,
        TUNED_LINEAR_FAST_PATH_ID,
        "cuda.contraction.mma-warp-specialized.v1",
        "cuda.conv.implicit-gemm-mma.v1",
    }


def _cuda_matching_authorizations(
    state: BackendSearchState,
    *,
    kind: RefinementKind,
    subject_ref: str,
    source_dtype: str | None = None,
    target_dtype: str | None = None,
) -> tuple[Any, ...]:
    """Find exact J/B authorization matches for one target subject."""

    matches = tuple(
        item
        for item in state.base.j.boundary.realization_authorizations
        if item.kind is kind
        and item.subject_ref == subject_ref
        and (source_dtype is None or item.source_dtype == source_dtype)
        and (target_dtype is None or item.target_dtype == target_dtype)
        and (
            kind is not RefinementKind.REASSOCIATION
            or (item.source_dtype is None and item.target_dtype is None)
        )
    )
    return matches


def _cuda_required_authorizations(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
) -> tuple[str, ...]:
    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA authorization derivation requires BackendSearchState")
    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA authorization derivation requires CudaSearchCandidate")

    nodes = _cuda_node_operations(state.base.j)
    refs: list[str] = []

    # A target reduction topology may change floating accumulation order only
    # when J already describes either a reassociable reduction or an explicit
    # reassociation authorization for that exact reduction subject.  The
    # ordinary ``assoc=float_nondeterministic`` contract is sufficient by
    # itself; an explicit matching authorization is carried as an attachment,
    # while an unrelated u* is never reused as a permission token.
    if _cuda_reduction_candidate(candidate):
        reductions = tuple(
            item
            for item in state.base.j.boundary.reductions
            if item.node_ref in nodes and nodes[item.node_ref].op in {"reduce", "scan"}
        )
        if not reductions:
            raise CudaContractError(
                "CUDA reduction candidate has no J reduction/scan subject"
            )
        for reduction in reductions:
            operation = nodes[reduction.node_ref]
            dtype = result_dtype(operation)
            assoc = operation.attrs.get("assoc")
            if assoc in {"associative", "float_nondeterministic"}:
                matches = _cuda_matching_authorizations(
                    state,
                    kind=RefinementKind.REASSOCIATION,
                    subject_ref=reduction.node_ref,
                )
                if len(matches) > 1:
                    raise CudaContractError(
                        "CUDA reduction subject has duplicate reassociation "
                        f"authorizations: {reduction.node_ref}"
                    )
                if matches:
                    refs.append(matches[0].auth_id)
                continue
            matches = _cuda_matching_authorizations(
                state,
                kind=RefinementKind.REASSOCIATION,
                subject_ref=reduction.node_ref,
            )
            if not matches:
                raise CudaContractError(
                    "CUDA reduction candidate lacks an exact reassociation "
                    f"authorization for {reduction.node_ref}"
                )
            if len(matches) != 1:
                raise CudaContractError(
                    "CUDA reduction subject has duplicate reassociation "
                    f"authorizations: {reduction.node_ref}"
                )
            if _cuda_is_floating_dtype(dtype) or assoc is None:
                refs.append(matches[0].auth_id)

    # Current CUDA contraction passes accumulate in f32.  That is already the
    # canonical J result type for the admitted f16/bf16 contraction graph, so
    # no authorization is needed in the normal case.  If a future admitted J
    # exposes a different result dtype, only an exact typed accumulation
    # authorization for the concrete contraction node may bridge it.
    if _cuda_contraction_candidate(candidate):
        contractions = tuple(
            (node_ref, operation)
            for node_ref, operation in nodes.items()
            if operation.op == "contraction"
        )
        if not contractions:
            raise CudaContractError(
                "CUDA contraction candidate has no J contraction subject"
            )
        for node_ref, operation in contractions:
            dtype = result_dtype(operation)
            if dtype == "f32":
                continue
            matches = _cuda_matching_authorizations(
                state,
                kind=RefinementKind.ACCUMULATION_DTYPE,
                subject_ref=node_ref,
                source_dtype=dtype if isinstance(dtype, str) else None,
                target_dtype="f32",
            )
            if len(matches) != 1:
                raise CudaContractError(
                    "CUDA contraction candidate requires exactly one typed "
                    f"f32 accumulation authorization for {node_ref}"
                )
            refs.append(matches[0].auth_id)

    # Observation-changing numeric states such as native PTX exp/tanh, FMA,
    # reciprocal approximation, row-mask load skipping, and split-K topology
    # are qualified by SearchAndClose Q/D.  They are not silently converted
    # into arbitrary J permissions because the current closed authorization
    # grammar has no matching subject/kind for those target-only changes.
    return tuple(sorted(set(refs)))


def cuda_plugin_loop_schedule_tile(
    j: SemanticArchive,
    target: Any,
    plugin: TargetLayerCPlugin,
) -> int | None:
    """Validate pre-M0 plugin facts and resolve its optional scan tile only."""

    if not isinstance(plugin, TargetLayerCPlugin):
        raise TypeError("CUDA materialization requires TargetLayerCPlugin")
    if plugin.target != TargetIdentity.from_target(target):
        raise CudaContractError("CUDA plugin target does not match lowering target")
    if plugin.j_digest != layer_c_content_digest(j.to_json()):
        raise CudaContractError("CUDA plugin J digest is stale or mismatched")
    expected_interface = cuda_layer_c_interface(j)
    if plugin.interface != expected_interface:
        raise CudaContractError(
            "CUDA plugin interface differs from J extension points"
        )
    if plugin.interface_digest != binding_interface_digest(expected_interface):
        raise CudaContractError("CUDA plugin interface digest is stale")
    if plugin.choice.backend_kind != CUDA_LAYER_C_BACKEND_KIND:
        raise CudaContractError("CUDA plugin has the wrong backend kind")
    validate_layer_c_attachments(j, plugin.interface)
    values = {item.slot: item.value for item in plugin.bindings}
    tile = values.get("loop_schedule_tile")
    if tile is None:
        return None
    if isinstance(tile, bool) or not isinstance(tile, int) or tile <= 0:
        raise CudaContractError(
            "CUDA loop_schedule_tile binding must be a positive integer"
        )
    return tile


def _admitted_cuda_universe(
    state: BackendSearchState,
    *,
    intent_module: str | None = None,
) -> tuple[CudaSearchCandidate, ...]:
    candidates = enumerate_cuda_candidates(
        state,
        enable_b_transformations=True,
        intent_module=intent_module,
    )
    admitted, _records = admit_cuda_candidate_paths(
        state,
        candidates,
        intent_module=intent_module,
    )
    return admitted


def resolve_cuda_candidate_key(
    state: BackendSearchState,
    candidate_key: str,
    candidate_digest: str,
    *,
    intent_module: str | None = None,
) -> CudaSearchCandidate:
    """Resolve one exact key+digest in the regenerated admitted universe."""

    if not isinstance(candidate_key, str) or not candidate_key:
        raise CudaContractError("CUDA Layer-C candidate key must be non-empty")
    if not isinstance(candidate_digest, str) or not candidate_digest:
        raise CudaContractError("CUDA Layer-C candidate digest must be non-empty")
    matches = [
        item
        for item in _admitted_cuda_universe(
            state,
            intent_module=intent_module,
        )
        if cuda_candidate_key(item) == candidate_key
        and cuda_candidate_digest(item) == candidate_digest
    ]
    if len(matches) != 1:
        raise CudaContractError(
            "CUDA Layer-C candidate key+digest is absent or ambiguous for this J/target"
        )
    return matches[0]


def _target_materialization_for_tile(
    base: MaterializedJ,
    loop_schedule_tile: int | None,
) -> MaterializedJ | None:
    """Materialize a validated target tile without changing the search base."""

    if loop_schedule_tile is None:
        return None
    try:
        return materialize_target_lowering(
            base,
            loop_schedule_tile=loop_schedule_tile,
        )
    except (TypeError, ValueError) as exc:
        raise CudaContractError(
            "CUDA target Layer-C tile cannot be materialized from neutral M0: "
            f"{exc}"
        ) from exc


def build_cuda_target_plugin(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
    *,
    loop_schedule_tile: int | None = None,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
) -> TargetLayerCPlugin:
    """Freeze one currently admitted CUDA candidate as target Layer C."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA plugin construction requires BackendSearchState")
    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA plugin construction requires CudaSearchCandidate")
    contract_view = derive_cuda_contract_view(state)
    target_materialization = _target_materialization_for_tile(
        state.base,
        loop_schedule_tile,
    )
    intent_module = (
        target_materialization.intent_mlir
        if target_materialization is not None
        else state.base_ir
    )
    matches = [
        item
        for item in _admitted_cuda_universe(
            state,
            intent_module=intent_module,
        )
        if cuda_candidate_key(item) == cuda_candidate_key(candidate)
        and cuda_candidate_digest(item) == cuda_candidate_digest(candidate)
    ]
    if len(matches) != 1:
        raise CudaContractError(
            "CUDA candidate is absent or ambiguous in the rebuilt admitted universe"
        )
    candidate = matches[0]
    required_capabilities = tuple(sorted(candidate.required_b_capabilities))
    missing = sorted(set(required_capabilities) - set(contract_view.capabilities))
    if missing:
        raise CudaContractError(
            "CUDA candidate lacks internally derived capabilities: "
            + ", ".join(missing)
        )
    authorizations = _cuda_required_authorizations(state, candidate)
    choice = ImplementationChoice(
        backend_kind=CUDA_LAYER_C_BACKEND_KIND,
        candidate_key=cuda_candidate_key(candidate),
        candidate_digest=cuda_candidate_digest(candidate),
        authorization_refs=authorizations,
    )
    target_contract = _cuda_target_realization_contract(
        state,
        candidate,
        choice,
        loop_schedule_tile=loop_schedule_tile,
    )
    return bind_target(
        _cuda_target_binding_context(state, contract_view, e_c=e_c),
        bindings=_cuda_plugin_bindings(
            state,
            candidate,
            loop_schedule_tile=loop_schedule_tile,
        ),
        choice=choice,
        required_capabilities=required_capabilities,
        contract=target_contract,
    )


def build_cuda_target_artifact(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
    *,
    loop_schedule_tile: int | None = None,
    e_c: LayerCInterface | Mapping[str, Any] | None = None,
    e_c_mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None = None,
) -> TargetPluginArtifact:
    """Build the complete explicitly typed CUDA target handoff."""

    target_interface = cuda_layer_c_interface(state.base.j)
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
    plugin = build_cuda_target_plugin(
        state,
        candidate,
        loop_schedule_tile=loop_schedule_tile,
        e_c=shared_e_c,
    )
    if e_c_mapping is None:
        if shared_e_c == plugin.interface:
            e_c_mapping = tuple(
                TargetExtensionMapping(item.slot, item.slot)
                for item in shared_e_c.points
            )
        elif not shared_e_c.points:
            e_c_mapping = ()
        else:
            raise CudaContractError(
                "CUDA shared E_C and target interface differ; an explicit "
                "e_c_mapping is required"
            )
    return TargetPluginArtifact(
        e_c=shared_e_c,
        plugin=plugin,
        e_c_mapping=tuple(e_c_mapping),
    )


def resolve_cuda_target_plugin(
    state: BackendSearchState,
    plugin: TargetLayerCPlugin,
) -> CudaSearchCandidate:
    """Rebuild and re-admit J+target candidates, then resolve one exact plugin."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA plugin resolution requires BackendSearchState")
    if not isinstance(plugin, TargetLayerCPlugin):
        raise TypeError("CUDA lowering requires TargetLayerCPlugin")
    contract_view = derive_cuda_contract_view(state)
    context = _cuda_target_binding_context(state, contract_view)
    plugin = deserialize_target_plugin(
        context,
        serialize_target_plugin(plugin),
    )
    if plugin.choice.backend_kind != CUDA_LAYER_C_BACKEND_KIND:
        raise CudaContractError("CUDA plugin has the wrong backend kind")
    loop_schedule_tile = cuda_plugin_loop_schedule_tile(
        state.base.j,
        state.target,
        plugin,
    )
    target_materialization = _target_materialization_for_tile(
        state.base,
        loop_schedule_tile,
    )
    intent_module = (
        target_materialization.intent_mlir
        if target_materialization is not None
        else state.base_ir
    )
    candidate = resolve_cuda_candidate_key(
        state,
        plugin.choice.candidate_key,
        plugin.choice.candidate_digest,
        intent_module=intent_module,
    )
    if plugin.bindings != _cuda_plugin_bindings(
        state,
        candidate,
        loop_schedule_tile=loop_schedule_tile,
    ):
        raise CudaContractError(
            "CUDA plugin bindings differ from the rebuilt candidate requirements"
        )
    required_capabilities = tuple(sorted(candidate.required_b_capabilities))
    if plugin.required_capabilities != required_capabilities:
        raise CudaContractError(
            "CUDA plugin capabilities differ from the rebuilt candidate requirements"
        )
    expected_authorizations = _cuda_required_authorizations(state, candidate)
    if plugin.choice.authorization_refs != expected_authorizations:
        raise CudaContractError(
            "CUDA plugin authorizations differ from the rebuilt candidate requirements"
        )
    expected_contract = _cuda_target_realization_contract(
        state,
        candidate,
        plugin.choice,
        loop_schedule_tile=loop_schedule_tile,
    )
    try:
        validate_target_plugin(
            context,
            plugin,
            expected_choice=ImplementationChoice(
                backend_kind=CUDA_LAYER_C_BACKEND_KIND,
                candidate_key=cuda_candidate_key(candidate),
                candidate_digest=cuda_candidate_digest(candidate),
                authorization_refs=expected_authorizations,
            ),
            expected_bindings=_cuda_plugin_bindings(
                state,
                candidate,
                loop_schedule_tile=loop_schedule_tile,
            ),
            expected_contract=expected_contract,
        )
    except (TypeError, ValueError) as exc:
        raise CudaContractError(str(exc)) from exc
    return candidate


def canonical_cuda_candidate(
    state: BackendSearchState,
    *,
    intent_module: str | None = None,
) -> CudaSearchCandidate:
    """Return E1's deterministic J/target-selected compiler state.

    This performs the same registered C++ structural probes as production
    lowering.  It does not enumerate schedules and does not read Source,
    Replay, Q/D, benchmark identity, or measurements.
    """

    derive_cuda_contract_view(state)
    path_kind, fast_path_id = schedule_only_cuda_path_identity(
        state,
        intent_module=intent_module,
    )
    if fast_path_id is None:
        return _generic_cuda_candidate(
            ragged=backend_compile._has_ragged_ops(
                read_intent_module(intent_module or state.base_ir)
            )
        )
    descriptor = next(
        (
            item
            for item in DEFAULT_PASSES
            if item.fast_path_id == fast_path_id
            and state.target.architecture in item.supported_chips
        ),
        None,
    )
    if descriptor is None:
        raise CudaContractError(
            f"canonical CUDA path {fast_path_id!r} is not registered for target"
        )
    return CudaSearchCandidate(
        _candidate_id(
            path_kind,
            fast_path_id,
            descriptor.pass_argument,
        ),
        path_kind,
        fast_path_id,
        descriptor.pass_argument,
        dict(descriptor.schedule),
        0.0,
        requested_trace=(fast_path_id,),
        requested_observation_class=ObservationClass.OBSERVATION_CHANGING,
    )


_CUDA_ORDERING_PRIOR_ROLES = frozenset(
    {
        "tile_extent",
        "reduction_tile_extent",
        "execution_warps",
        "constexpr",
        "tile_output_axis0",
        "tile_output_axis1",
        "tile_contraction_axis",
        "query_tile",
        "key_tile",
        "feature_tile",
        "launch_block_x",
    }
)

# CUDA row reductions balance parallelism against per-thread register/dependency
# pressure around a small strip of logical columns.  C describes the logical
# reduction tile, so Retile(C,target) maps it to threads through this strip size
# rather than treating a Source BLOCK_SIZE as either CUDA threads or one exact
# compiler tile.  All neighboring thread counts remain in the same universe and
# are still measured; this constant affects bounded-search order only.
_CUDA_ROW_PREFERRED_ELEMENTS_PER_THREAD = 8


def _numeric_priors(state: BackendSearchState) -> tuple[CPrior, ...]:
    """Return only positive scalar C hints that can rank existing schedules.

    The complete authored record is retained so scoring can respect the role of
    each value.  C remains an ordering hint: this function neither constructs a
    candidate nor participates in hard admission.
    """

    return tuple(
        item
        for item in state.priors
        if item.role in _CUDA_ORDERING_PRIOR_ROLES
        and isinstance(item.value, (int, float))
        and not isinstance(item.value, bool)
        and item.value > 0
    )


def project_cuda_c_fixed_candidate(
    state: BackendSearchState,
    candidates: tuple[CudaSearchCandidate, ...],
) -> tuple[CudaSearchCandidate, dict[str, Any]]:
    """Mechanically project authored C onto one legal canonical-path schedule.

    This is deliberately weaker than target-aware Retile ordering.  It may use
    only the role/value carried by C and fixed compiler defaults.  In
    particular it cannot inspect workload extents, occupancy, A-level operator
    mix, a cost model, or measurements.  The returned candidate therefore is a
    causal fixed-schedule baseline; opening the remaining candidates is the
    Schedule-Only treatment.

    ``candidates`` must be the admitted canonical-path schedule space.  The
    nearest legal coordinate is selected when C names a value that the target
    schedule cannot represent exactly.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA C projection requires BackendSearchState")
    if not candidates:
        raise CudaContractError("CUDA C projection has no admitted schedule")

    candidate_paths = {
        (item.lowering_path_kind, item.fast_path_id) for item in candidates
    }
    if len(candidate_paths) != 1:
        raise CudaContractError(
            "CUDA C projection requires exactly one fixed structural path"
        )
    projection_kind, projection_fast_path = next(iter(candidate_paths))
    eligible = tuple(
        item
        for item in candidates
        if item.lowering_path_kind == projection_kind
        and item.fast_path_id == projection_fast_path
    )
    if not eligible:
        raise CudaContractError(
            "CUDA C projection has no admitted candidate on the canonical path"
        )

    # Defaults are compiler policy, not workload policy.  They resolve schedule
    # coordinates that C does not author and make ties deterministic.  Local-B,
    # structural and numeric choices stay disabled in this baseline.
    if projection_kind == "row_reduction":
        fixed_default: Mapping[str, int | bool | str] = RowSchedule(
            64,
            32,
            promote_rank0_scalars=False,
            defer_epilogue_col_loads=False,
            compact_reduction_topology=False,
        ).to_dict()
    elif projection_kind == "elementwise":
        fixed_default = ElementwiseSchedule(64, False, 1).to_dict()
    elif projection_kind == "padded_row_reduction":
        fixed_default = PaddedRowSchedule(64, False, False).to_dict()
    elif projection_kind == "splitk_weighted_merge":
        fixed_default = {
            "rows_per_block": 4,
            "warps_per_row": 1,
            "d_partitions": 1,
            "block_threads": 128,
            "vector_width": 1,
        }
    elif projection_kind == "linalg_fused":
        fixed_default = {"tile_width": 128}
    elif projection_kind in {"gather", "gather_composed"}:
        fixed_default = {"block_threads": 64}
    elif projection_kind == DIRECT_CONTRACTION_FAST_PATH_ID:
        fixed_default = {
            "tile_rows": 64,
            "tile_cols": 128,
            "tile_k": 32,
            "num_warps": 4,
            "pipeline_stages": 3,
        }
    elif projection_kind == "cuda.flash_forward.structured_scan.direct.v1":
        fixed_default = {
            "tile_rows": 64,
            "tile_cols": 128,
            "pipeline_stages": 1,
            "num_warps": 4,
        }
    else:
        fixed_default = {}

    available_keys = {
        key for candidate in eligible for key in candidate.schedule
    }
    consumed: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    targets: list[tuple[str, float, CPrior]] = []
    for prior in state.priors:
        value = prior.value
        if (
            prior.role not in _CUDA_ORDERING_PRIOR_ROLES
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            ignored.append(
                {**prior.to_record(), "reason": "not_a_positive_schedule_prior"}
            )
            continue

        coordinate: str | None = None
        projected_value = float(value)
        if prior.role == "execution_warps":
            if "num_warps" in available_keys:
                coordinate = "num_warps"
            elif "block_threads" in available_keys:
                coordinate = "block_threads"
                projected_value *= 32.0
        elif prior.role == "launch_block_x" and "block_threads" in available_keys:
            coordinate = "block_threads"
        elif prior.role == "tile_output_axis0" and "tile_rows" in available_keys:
            coordinate = "tile_rows"
        elif prior.role == "tile_output_axis1" and "tile_cols" in available_keys:
            coordinate = "tile_cols"
        elif prior.role == "query_tile" and "tile_rows" in available_keys:
            coordinate = "tile_rows"
        elif prior.role == "key_tile" and "tile_cols" in available_keys:
            coordinate = "tile_cols"
        elif (
            prior.role == "tile_contraction_axis" and "tile_k" in available_keys
        ):
            coordinate = "tile_k"
        elif prior.role in {
            "tile_extent",
            "reduction_tile_extent",
            "constexpr",
        }:
            if projection_kind == "row_reduction" and "tile_cols" in available_keys:
                coordinate = "tile_cols"
            elif (
                projection_kind == "elementwise"
                and "logical_tile_elements" in available_keys
            ):
                coordinate = "logical_tile_elements"
            elif (
                projection_kind == "padded_row_reduction"
                and "block_threads" in available_keys
            ):
                coordinate = "block_threads"
            elif projection_kind == "linalg_fused" and "tile_width" in available_keys:
                coordinate = "tile_width"

        if coordinate is None:
            ignored.append(
                {**prior.to_record(), "reason": "no_corresponding_schedule_coordinate"}
            )
            continue
        targets.append((coordinate, projected_value, prior))
        consumed.append(
            {
                **prior.to_record(),
                "target_schedule_coordinate": coordinate,
                "requested_target_value": projected_value,
            }
        )

    def scalar_distance(actual: Any, expected: Any) -> float:
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and actual > 0
            and expected > 0
        ):
            return abs(math.log2(float(actual) / float(expected)))
        return 0.0 if actual == expected else 1.0

    def projection_key(
        candidate: CudaSearchCandidate,
    ) -> tuple[float, float, str]:
        prior_distances = [
            scalar_distance(candidate.schedule.get(coordinate), value)
            for coordinate, value, _prior in targets
            if coordinate in candidate.schedule
        ]
        prior_distance = (
            statistics.fmean(prior_distances) if prior_distances else 0.0
        )
        default_distances = [
            scalar_distance(candidate.schedule.get(key), value)
            for key, value in fixed_default.items()
            if key in candidate.schedule
        ]
        default_distance = (
            statistics.fmean(default_distances) if default_distances else 0.0
        )
        return prior_distance, default_distance, candidate.candidate_id

    selected = min(eligible, key=projection_key)
    prior_distance, default_distance, _ = projection_key(selected)
    record = {
        "protocol": "intentir.backend.cuda.c-projection.v2",
        "selection_rule": (
            "nearest legal authored-C schedule coordinate, then fixed compiler "
            "defaults; no workload, occupancy, cost-model or performance input"
        ),
        "source_priors": [item.to_record() for item in state.priors],
        "consumed_priors": consumed,
        "ignored_priors": ignored,
        "canonical_lowering_path": {
            "lowering_path_kind": projection_kind,
            "fast_path_id": projection_fast_path,
        },
        "fixed_unspecified_schedule_defaults": dict(fixed_default),
        "selected_candidate_id": selected.candidate_id,
        "selected_target_schedule": dict(selected.schedule),
        "authored_c_distance": prior_distance,
        "fixed_default_distance": default_distance,
        "workload_or_cost_model_used": False,
        "performance_measurement_used": False,
    }
    return selected, record


def _c_score(
    schedule: Mapping[str, int | bool | str],
    priors: tuple[CPrior, ...],
    *,
    row_elements_per_thread: int = _CUDA_ROW_PREFERRED_ELEMENTS_PER_THREAD,
    elementwise_thread_hint: int | None = None,
) -> float:
    if not priors:
        return 0.0

    def positive_values(names: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(
            float(schedule[name])
            for name in names
            if isinstance(schedule.get(name), (int, float))
            and not isinstance(schedule.get(name), bool)
            and float(schedule[name]) > 0
        )

    distances: list[float] = []
    for prior in priors:
        row_threads = positive_values(("block_threads",))
        is_row_schedule = bool(row_threads and "tile_cols" in schedule)
        is_elementwise_schedule = bool(
            row_threads and "flatten_contiguous" in schedule
        )
        if prior.role == "execution_warps":
            values = positive_values(("num_warps",))
            if not values:
                threads = positive_values(("block_threads",))
                values = tuple(value / 32.0 for value in threads)
        elif prior.role == "tile_output_axis0":
            values = positive_values(("tile_rows",))
        elif prior.role == "tile_output_axis1":
            values = positive_values(("tile_cols",))
        elif prior.role == "tile_contraction_axis":
            values = positive_values(("tile_k",))
        elif prior.role in {
            "reduction_tile_extent",
            "tile_extent",
            "constexpr",
        } and is_row_schedule:
            # A Source reduction tile describes logical row coverage, not a CUDA
            # thread count.  Retile it through the target's preferred logical
            # strip per thread.  For example a 4096-column row maps first to 512
            # threads, while 1024 and 128 columns map to 128 and 32 threads.
            values = tuple(
                thread_count * row_elements_per_thread
                for thread_count in row_threads
            )
        elif prior.role in {
            "reduction_tile_extent",
            "tile_extent",
            "constexpr",
        } and is_elementwise_schedule:
            values = positive_values(("logical_tile_elements",))
            if not values:
                values = row_threads
        elif prior.role == "reduction_tile_extent":
            values = positive_values(
                ("tile_cols", "tile_rows", "tile_width", "block_threads")
            )
        else:
            # tile_extent and constexpr are both concrete launch/tile hints in
            # accepted C.  They rank only dimensions already present in the
            # candidate; they never create a new dimension or exact state.
            values = positive_values(
                ("tile_cols", "tile_rows", "tile_width", "block_threads")
            )
        if not values:
            continue
        value = float(
            elementwise_thread_hint
            if is_elementwise_schedule
            and not positive_values(("logical_tile_elements",))
            and elementwise_thread_hint is not None
            and prior.role in {"tile_extent", "constexpr"}
            else prior.value
        )
        distances.append(min(abs(math.log2(candidate / value)) for candidate in values))
    if not distances:
        return float("inf")
    # Every authored C relation contributes.  The former Cartesian minimum
    # discarded all but the easiest-to-match value and produced large zero-score
    # ties, especially when tile and reduction hints coexisted.
    return statistics.fmean(distances)


def _row_elements_per_thread(state: BackendSearchState) -> int:
    """Map J compute density to a CUDA row-retile ordering prior.

    Dense per-element transcendental maps benefit from exposing more lanes to
    hide special-function latency.  Per-row normalization functions such as one
    rsqrt do not trigger this rule.  This is source/name independent and never
    changes hard admission or the candidate universe.
    """

    elaboration = j_elaboration(state.base.j)
    typed_rows = elaboration.typed_operation_rows
    if typed_rows is None:
        return 8
    value_types = elaboration.value_type_by_name
    for _path, operation in typed_rows:
        result_type = (
            value_types.get(operation.result)
            if isinstance(operation.result, str)
            else None
        )
        if (
            operation.op == "map_unary"
            and operation.attrs.get("fn") in {"exp", "log", "sigmoid", "tanh"}
            and result_type is not None
            and result_type.is_tensor
            and len(result_type.shape) >= 2
        ):
            return 4
    return 8


def _materialized_row_extents(
    state: BackendSearchState,
) -> tuple[int | None, int | None]:
    """Return concrete (row-count, feature-count) bounds for schedule pruning.

    This is not a kernel recipe or a performance heuristic.  It only removes
    row-group requests that the C++ matcher can prove impossible (a 32-lane
    row group needs one complete static feature strip per lane).  The actual
    legality remains in the target pass and every surviving state is compiled.
    """

    extents = state.base.extents

    def resolve_shape(shape: Sequence[int | str]) -> tuple[int | str, ...]:
        return tuple(
            extents.get(dim, dim) if isinstance(dim, str) else dim
            for dim in shape
        )

    # Logical outputs have one authoritative order and type surface: the
    # canonical output construction records.  Reading ``intent.outputs`` here
    # would make schedule pruning independently reconstruct an output shape
    # view after Gate1 already accepted the record bundle.
    output_bindings = j_elaboration(state.base.j).output_bindings
    shapes = [
        *[resolve_shape(item.shape) for item in state.base.j.intent.inputs],
        *[
            resolve_shape(binding.output_type.shape)
            for binding in output_bindings
        ],
    ]
    rank2 = [shape for shape in shapes if len(shape) == 2]
    concrete = [
        shape
        for shape in rank2
        if all(type(dim) is int and dim > 0 for dim in shape)
    ]
    if not concrete:
        return None, None
    rows = max(shape[0] for shape in concrete)
    cols = max(shape[1] for shape in concrete)
    return rows, cols


def _row_tile_candidates(
    state: BackendSearchState, threads: int
) -> tuple[int, ...]:
    """Keep one requested tile for each distinct C++ register/tiled schedule.

    Several large ``tile-cols`` requests collapse to the same register path once
    the complete per-thread strip fits.  They are not different target states
    and should not multiply every numeric/structural combination.
    """

    _rows, cols = _materialized_row_extents(state)
    tiles = (16, 32, 64, 128, 256)
    if cols is None:
        return tiles
    per_thread = (cols + threads - 1) // threads
    representatives: list[int] = []
    seen: set[tuple[int, int]] = set()
    for tile in tiles:
        per_thread_tile = per_thread
        split = 1
        if per_thread > tile:
            for candidate_split in range(2, per_thread + 1):
                if (
                    per_thread % candidate_split == 0
                    and per_thread // candidate_split <= tile
                ):
                    split = candidate_split
                    per_thread_tile = per_thread // candidate_split
                    break
        key = (per_thread_tile, split)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(tile)
    return tuple(representatives)


def _row_numeric_opportunities(
    state: BackendSearchState,
) -> tuple[bool, bool, int]:
    """Return A-visible numeric opportunities and last-axis reduction count."""

    typed_rows = j_elaboration(state.base.j).typed_operation_rows
    if typed_rows is None:
        raise CudaContractError(
            "CUDA numeric opportunity analysis requires canonical operations"
        )
    nodes = [operation._structural_payload() for _path, operation in typed_rows]
    producers = {
        str(node["result"]): node
        for node in nodes
        if isinstance(node.get("result"), str)
    }

    def produced_by_mul(name: Any) -> bool:
        producer = producers.get(str(name))
        return bool(
            producer
            and producer.get("op") == "map_binary"
            and producer.get("attrs", {}).get("fn") == "mul"
        )

    has_fma = any(
        (
            node.get("op") == "reduce"
            and node.get("attrs", {}).get("fn") == "sum"
            and any(produced_by_mul(item) for item in node.get("operands", []))
        )
        or (
            node.get("op") == "map_binary"
            and node.get("attrs", {}).get("fn") == "add"
            and any(produced_by_mul(item) for item in node.get("operands", []))
        )
        for node in nodes
    )
    has_division = any(
        (
            node.get("op") == "map_binary"
            and node.get("attrs", {}).get("fn") == "div"
        )
        or (
            node.get("op") == "reduce"
            and node.get("attrs", {}).get("fn") == "mean"
        )
        for node in nodes
    )
    last_axis_reductions = sum(
        node.get("op") == "reduce"
        and node.get("attrs", {}).get("axes") == [1]
        for node in nodes
    )
    return has_fma, has_division, last_axis_reductions


def _has_cross_row_float_sum(state: BackendSearchState) -> bool:
    """Return whether A exposes an axis-0 sum feeding a rank-0 output."""

    if not any(
        len(binding.output_type.shape) == 0
        for binding in j_elaboration(state.base.j).output_bindings
    ):
        return False
    typed_rows = j_elaboration(state.base.j).typed_operation_rows
    if typed_rows is None:
        return False
    return any(
        operation.op == "reduce"
        and operation.attrs.get("fn") == "sum"
        and operation.attrs.get("axes") == [0]
        for _path, operation in typed_rows
    )


def _raw_moment_variance_opportunity(state: BackendSearchState) -> bool:
    """Match A's mean followed by a centered-square row reduction."""

    typed_rows = j_elaboration(state.base.j).typed_operation_rows
    if typed_rows is None:
        return False
    body = tuple(operation for _path, operation in typed_rows)
    producers = {
        item.result: item
        for item in body
        if isinstance(item.result, str) and item.result
    }

    def depends(value: str, target: str, seen: set[str] | None = None) -> bool:
        if value == target:
            return True
        visited = seen if seen is not None else set()
        if value in visited:
            return False
        visited.add(value)
        producer = producers.get(value)
        return producer is not None and any(
            depends(operand, target, visited) for operand in producer.operands
        )

    def is_zero(value: str) -> bool:
        producer = producers.get(value)
        return (
            producer is not None
            and producer.op == "const"
            and producer.attrs.get("value") in {0, 0.0, False}
        )

    def unwrap_full_select(value: str) -> str:
        producer = producers.get(value)
        if producer is None or producer.op != "select" or len(producer.operands) != 3:
            return value
        _condition, positive, negative = producer.operands
        if is_zero(negative):
            return positive
        if is_zero(positive):
            return negative
        return value

    reductions = [
        item
        for item in body
        if item.op == "reduce"
        and item.attrs.get("axes") == [1]
        and item.attrs.get("fn") in {"sum", "mean"}
        and item.attrs.get("assoc") == "float_nondeterministic"
    ]
    if len(reductions) != 2:
        return False
    mean_reduce, variance_reduce = reductions
    if not mean_reduce.operands or not variance_reduce.operands:
        return False
    squared_id = unwrap_full_select(variance_reduce.operands[0])
    square = producers.get(squared_id)
    if (
        square is None
        or square.op != "map_binary"
        or square.attrs.get("fn") != "mul"
        or len(square.operands) != 2
        or square.operands[0] != square.operands[1]
    ):
        return False
    centered_id = unwrap_full_select(square.operands[0])
    centered = producers.get(centered_id)
    return bool(
        centered is not None
        and centered.op == "map_binary"
        and centered.attrs.get("fn") == "sub"
        and len(centered.operands) == 2
        and centered.operands[0] == mean_reduce.operands[0]
        and depends(centered.operands[1], mean_reduce.result)
    )


def _row_materialization_opportunity(state: BackendSearchState) -> bool:
    """Prove that A exposes the exact shared-row spill/elision choice.

    The C++ lowering admits this state only when one authored rank-2 output is
    also an ancestor of the sole row reduction and of another authored output.
    Recover the same necessary graph fact from A before enumeration so kernels
    without that intermediate do not spend finite search budget on requests
    that the compiler must reject.  The C++ matcher remains the authoritative
    sufficient legality check for vector width, aliasing, and expression form.
    """

    typed_rows = j_elaboration(state.base.j).typed_operation_rows
    if typed_rows is None:
        return False
    body = tuple(operation for _path, operation in typed_rows)
    output_bindings = j_elaboration(state.base.j).output_bindings
    output_names = {item.output for item in output_bindings}
    output_type_by_name = {
        item.output: item.output_type for item in output_bindings
    }
    nodes_by_result = {
        result: item
        for item in body
        for result in (
            (item.result,)
            if isinstance(item.result, str)
            else tuple(item.result)
            if isinstance(item.result, list)
            else ()
        )
        if isinstance(result, str)
    }
    reductions = [
        item
        for item in body
        if item.op == "reduce" and item.attrs.get("axes") == [1]
    ]
    if len(reductions) != 1:
        return False
    reduction_operands = reductions[0].operands
    if len(reduction_operands) != 1:
        return False
    reduction_input = reduction_operands[0]
    if not isinstance(reduction_input, str):
        return False

    def depends_on(value: str, ancestor: str, visiting: set[str]) -> bool:
        if value == ancestor:
            return True
        if value in visiting:
            return False
        node = nodes_by_result.get(value)
        if node is None:
            return False
        operands = node.operands
        visiting.add(value)
        try:
            return any(
                isinstance(operand, str)
                and depends_on(operand, ancestor, visiting)
                for operand in operands
            )
        finally:
            visiting.remove(value)

    def contains_select(value: str, visiting: set[str]) -> bool:
        if value in visiting:
            return False
        node = nodes_by_result.get(value)
        if node is None:
            return False
        if node.op == "select":
            return True
        operands = node.operands
        visiting.add(value)
        try:
            return any(
                isinstance(operand, str) and contains_select(operand, visiting)
                for operand in operands
            )
        finally:
            visiting.remove(value)

    for output in output_names:
        producer = nodes_by_result.get(output)
        if producer is None or producer.op != "map_binary":
            continue
        output_type = output_type_by_name.get(output)
        if output_type is None or len(output_type.shape) != 2:
            continue
        if not depends_on(reduction_input, output, set()):
            continue
        if not any(
            other != output and depends_on(other, output, set())
            for other in output_names
        ):
            continue
        if contains_select(output, set()):
            continue
        return True
    return False


def _referenced_intent_inputs(state: BackendSearchState) -> frozenset[str]:
    operation_records = j_elaboration(state.base.j).operation_records
    if not operation_records:
        return frozenset()
    referenced: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            operands = value.get("operands")
            if isinstance(operands, list):
                referenced.update(
                    str(item) for item in operands if isinstance(item, str)
                )
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    # Walk the authored operation payloads already owned by the canonical
    # construction table.  This keeps search heuristics from rebuilding a
    # second body/path interpretation while retaining the existing conservative
    # reference collection for operation-specific fields.
    for record in operation_records:
        wire = record.get("wire")
        if isinstance(wire, Mapping):
            collect(wire)
    return frozenset(referenced)


def _elementwise_thread_hint(state: BackendSearchState) -> int | None:
    """Retile a logical row packet to enough CUDA threads for target occupancy."""

    extents = state.base.extents
    for output in j_elaboration(state.base.j).output_bindings:
        if len(output.output_type.shape) != 2:
            continue
        rows, columns = (
            extents[item] if isinstance(item, str) else item
            for item in output.output_type.shape
        )
        if not isinstance(rows, int) or not isinstance(columns, int):
            return None
        # A tiny outer row domain cannot occupy the GPU even when one block uses
        # many threads.  The registered C++ lowering already switches from a
        # per-row register tile to a flat grid-stride state when N/T > 32.  Rank
        # the largest power-of-two T that still crosses that transition first:
        # for rows=2,N=4096 this is T=64, exposing many independent blocks
        # instead of only two 1024-thread blocks.  This is a target-aware Retile
        # prior over an unchanged candidate universe, not an admission rule.
        if rows < 64 and columns > 32 * 32:
            transition_threads = 1 << math.floor(
                math.log2((columns - 1) // 32)
            )
            return max(32, min(1024, transition_threads))

        # Otherwise keep roughly 64K logical threads in flight across
        # independent rows, while never assigning more lanes than the row can
        # use.  This maps 128x1024 to 512 threads and 2048x4096 to 64 threads on
        # the same target.
        desired = math.ceil(65536 / max(1, rows))
        desired = max(64, min(1024, desired))
        desired = 1 << max(6, round(math.log2(desired)))
        row_cap = 1 << max(5, math.ceil(math.log2(max(1, columns))))
        return min(desired, row_cap, 1024)
    return None


def enumerate_cuda_candidates(
    state: BackendSearchState,
    *,
    enable_b_transformations: bool = True,
    axes: SearchAxes | None = None,
    fixed_lowering_path_kind: str | None = None,
    fixed_fast_path_id: str | None = None,
    intent_module: str | None = None,
) -> tuple[CudaSearchCandidate, ...]:
    """Enumerate concrete pass states only after complete B admission.

    ``state.base`` is always neutral M0.  ``intent_module`` is an optional,
    separately materialized target view used only for registered whole-graph
    probes (for example a validated structured-scan tile); it never replaces
    the J-derived facts used to construct the candidate universe.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA candidate enumeration requires BackendSearchState")
    if axes is not None:
        enable_b_transformations = axes.enable_local_b_transformations
    if (
        axes is not None
        and not axes.search_structural_states
        and fixed_lowering_path_kind is None
        and fixed_fast_path_id is None
    ):
        raise ValueError(
            "fixed-structure schedule search requires a concrete path identity"
        )
    # A fixed concrete path is the complete Schedule-Only search space.  The
    # return logic below excludes the generic materialized pipeline, which
    # would silently turn it back into a multi-path experiment.
    contract_view = derive_cuda_contract_view(state)
    source = read_intent_module(intent_module or state.base_ir)
    if backend_compile._has_ragged_ops(source):
        if fixed_fast_path_id is not None:
            raise CudaContractError(
                "CUDA Ragged lowering is represented by the generic candidate; "
                "the descriptor-aware pass is not a tunable fast-path choice"
            )
        if fixed_lowering_path_kind not in {None, "generic"}:
            raise CudaContractError(
                "CUDA Ragged lowering has no specialized lowering path kind; "
                "use the generic descriptor-aware candidate"
            )
        return (_generic_cuda_candidate(ragged=True),)
    capabilities = set(contract_view.capabilities)
    runtime_names = set(state.base.runtime_input_names)
    referenced_inputs = _referenced_intent_inputs(state)
    has_runtime_rank0 = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and not item.shape
        for item in state.base.j.intent.inputs
    )
    has_column_input = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and len(item.shape) == 1
        for item in state.base.j.intent.inputs
    )
    has_uniform_input = any(
        item.name in runtime_names
        and item.name in referenced_inputs
        and (not item.shape or item.shape == [1])
        for item in state.base.j.intent.inputs
    )
    priors = (
        _numeric_priors(state)
        if axes is None or axes.use_c_prior
        else ()
    )
    row_elements_per_thread = _row_elements_per_thread(state)
    has_fma_opportunity, has_division_opportunity, last_axis_reduction_count = (
        _row_numeric_opportunities(state)
    )
    has_cross_row_float_sum = _has_cross_row_float_sum(state)
    has_materialization_opportunity = _row_materialization_opportunity(state)
    has_raw_moment_opportunity = _raw_moment_variance_opportunity(state)
    elementwise_thread_hint = _elementwise_thread_hint(state)
    specialized: list[CudaSearchCandidate] = []

    elementwise_schedule_bases = [
        *(ElementwiseSchedule(threads, False, width)
          for threads in (32, 64, 128, 256, 512, 1024)
          for width in (1, 2, 4)),
        *(ElementwiseSchedule(
              threads,
              False,
              1,
              grid_stride_unroll=unroll,
          )
          for threads in (32, 64, 128, 256, 512, 1024)
          for unroll in (2, 4, 8)),
        *(ElementwiseSchedule(
              threads,
              False,
              1,
              logical_tile_elements=tile,
          )
          for threads in (32, 64, 128, 256, 512, 1024)
          for tile in (128, 256, 512, 1024, 2048, 4096, 8192)
          if tile >= threads and tile % threads == 0 and tile // threads <= 32),
        *(ElementwiseSchedule(threads, True, width)
          for threads in (64, 128, 256)
          for width in (2, 4)),
    ]
    has_paired_rotation = "intent.paired_rotation_version" in state.base_ir
    has_three_plane_rotation = "intent.paired_rotation_sections" in state.base_ir
    has_shared_position_rotation = (
        'intent.paired_rotation = "shared_position"' in state.base_ir
    )
    if has_paired_rotation:
        # The paired C++ consumer owns a deliberately smaller schedule domain
        # than the generic elementwise path.  Filter it before budgeting so
        # unsupported flatten/unroll/tile requests do not consume the complete
        # Schedule-Only prefix and starve Full structural states.
        elementwise_schedule_bases = [
            ElementwiseSchedule(threads, False, width)
            for threads in (32, 64, 128, 256, 512, 1024)
            for width in ((1, 2) if has_shared_position_rotation else (1,))
        ]
    approximation_modes = (
        ("stable-ex2", "native-ptx")
        if axes is None or axes.search_approximation
        else ("stable-ex2",)
    )
    paired_head_group_modes = (
        (0, 2, 4, 8, 16)
        if has_paired_rotation
        and (axes is None or axes.search_structural_states)
        else (0,)
    )
    paired_direct_plane_modes = (
        (False, True)
        if has_three_plane_rotation
        and (axes is None or axes.search_structural_states)
        else (False,)
    )
    elementwise_schedules = [
        replace(
            base,
            promote_uniform_scalars=promote,
            tanh_approximation=approximation,
            paired_head_reuse=paired_head_group > 0,
            paired_head_group_size=(
                paired_head_group if paired_head_group > 0 else 4
            ),
            paired_direct_plane=paired_direct_plane,
        )
        for base in elementwise_schedule_bases
        for promote in ((False, True) if has_uniform_input else (False,))
        for approximation in approximation_modes
        for paired_head_group in paired_head_group_modes
        for paired_direct_plane in paired_direct_plane_modes
        if not (paired_head_group > 0 or paired_direct_plane)
        or (
            not base.flatten_contiguous
            and base.vector_width in {1, 2}
            and (base.vector_width == 1 or not has_three_plane_rotation)
            and (base.vector_width == 1 or not paired_direct_plane)
            and base.grid_stride_unroll == 1
            and base.logical_tile_elements == 0
            and approximation == "stable-ex2"
        )
    ]
    for schedule in elementwise_schedules:
        values = schedule.to_dict()
        transformations: list[str] = []
        requirements: list[str] = []
        if schedule.promote_uniform_scalars and has_uniform_input:
            transformations.append(B_TRANSFORM_UNIFORM_HOST_CAPTURE)
            requirements.append(B_CAP_READONLY_INPUTS)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "elementwise",
                    ELEMENTWISE_FAST_PATH_ID,
                    schedule.pass_argument,
                ),
                "elementwise",
                ELEMENTWISE_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(
                    values,
                    priors,
                    elementwise_thread_hint=elementwise_thread_hint,
                ),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.elementwise.row.vectorized.v1",
                    *(
                        ("cuda.numeric.tanh_native_ptx.f32.request.v1",)
                        if schedule.tanh_approximation == "native-ptx"
                        else ()
                    ),
                    *(
                        ("cuda.elementwise.paired_head_angle_reuse.request.v1",)
                        if schedule.paired_head_reuse
                        else ()
                    ),
                    *(
                        ("cuda.elementwise.paired_direct_plane.request.v1",)
                        if schedule.paired_direct_plane
                        else ()
                    ),
                    *transformations,
                ),
                (
                    ObservationClass.OBSERVATION_CHANGING
                    if schedule.tanh_approximation == "native-ptx"
                    else ObservationClass.HARD_PRESERVING
                ),
            )
        )

    # Full-only execution-domain row-mask topology.  The C++ matcher requires
    # the exact cast(i1[M])->f32 and pointwise X[M,N]*mask[M] graph.  Skipping X
    # loads on a false row changes NaN/Inf*0 behavior, so Q/D remains mandatory.
    if axes is None or axes.search_structural_states:
        for threads in (32, 64, 128, 256):
            for flat_width in (0, 2, 4, 8):
                flatten = flat_width > 0
                logical_tiles = (0, 4096, 8192) if flatten else (0,)
                for logical_tile in logical_tiles:
                    pass_argument = (
                        "-convert-intent-elementwise-to-gpu="
                        f"block-threads={threads} row-vector-width=1 "
                        "row-mask-zero-fill=true tanh-approximation=stable-ex2"
                        + (
                            " flatten-contiguous=true "
                            f"flat-vector-width={flat_width} "
                            f"logical-tile-elements={logical_tile}"
                            if flatten
                            else ""
                        )
                    )
                    schedule = {
                        "block_threads": threads,
                        "vector_width": flat_width if flatten else 1,
                        "row_mask_zero_fill": True,
                        "flatten_contiguous": flatten,
                        "logical_tile_elements": logical_tile,
                    }
                    specialized.append(
                        CudaSearchCandidate(
                            _candidate_id(
                                "elementwise_rowmask_zero_fill",
                                ROWMASK_ZERO_FILL_FAST_PATH_ID,
                                pass_argument,
                            ),
                            "elementwise_rowmask_zero_fill",
                            ROWMASK_ZERO_FILL_FAST_PATH_ID,
                            pass_argument,
                            schedule,
                            _c_score(
                                schedule,
                                priors,
                                elementwise_thread_hint=elementwise_thread_hint,
                            ),
                            requested_trace=(
                                "cuda.elementwise.rowmask.zero-fill.v1",
                                *(("cuda.elementwise.rowmask.flat-grid.v1",) if flatten else ()),
                            ),
                            requested_observation_class=(
                                ObservationClass.OBSERVATION_CHANGING
                            ),
                        )
                    )

    # Full-only algebraic implementation state.  The C++ pass proves the exact
    # exp(x-shift)*exp(shift) graph and requires cancellation to remove every
    # live reduction before it delegates to the ordinary elementwise MLIR
    # lowering.  The candidate remains observation-changing because floating
    # overflow/rounding differs from the stabilized expression; Q/D, not this
    # enumerator, decides whether the execution-domain result is admissible.
    for threads in (128, 256, 512, 1024):
        pass_argument = (
            "-convert-intent-elementwise-to-gpu="
            f"block-threads={threads} cancel-shifted-exp=true "
            "tanh-approximation=stable-ex2"
        )
        values = {
            "block_threads": threads,
            "vector_width": 0,
            "algebraic_exp_shift_cancellation": True,
        }
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "elementwise_shifted_exp",
                    SHIFTED_EXP_CANCELLATION_FAST_PATH_ID,
                    pass_argument,
                ),
                "elementwise_shifted_exp",
                SHIFTED_EXP_CANCELLATION_FAST_PATH_ID,
                pass_argument,
                values,
                _c_score(
                    values,
                    priors,
                    elementwise_thread_hint=elementwise_thread_hint,
                ),
                requested_trace=(SHIFTED_EXP_CANCELLATION_FAST_PATH_ID,),
                requested_observation_class=(
                    ObservationClass.OBSERVATION_CHANGING
                ),
            )
        )

    # Row vector-width=0 is the C++ auto state: it emits vec4 exactly when the
    # structural gate proves a contiguous full register tile, otherwise scalar.
    # An explicit width=4 request therefore either duplicates that exact MLIR or
    # fails admission; it is not a distinct concrete state and must not consume
    # finite search budget.  Width=1 remains the deliberate scalar ablation.
    row_schedule_bases = [
        *(RowSchedule(threads, tile, vector_width=0, fuse_fma=False)
          for threads in (32, 64, 128, 256, 512, 1024)
          for tile in _row_tile_candidates(state, threads)),
        *(RowSchedule(threads, 32, vector_width=1, fuse_fma=False)
          for threads in (64, 128, 256)),
    ]
    if axes is None or axes.search_reduction_topology:
        concrete_rows, concrete_cols = _materialized_row_extents(state)
        # The subgroup-rows implementation is a register-tile path with
        # exactly 32 lanes per logical row.  Do not transport tile requests
        # that are statically too small for the feature strip (or row groups
        # that cannot divide the concrete row domain); they can never reach
        # the C++ state and otherwise consume the whole E3 search budget as
        # predictable probe failures.
        min_subgroup_tile = (
            (concrete_cols + 31) // 32 if concrete_cols is not None else 0
        )
        row_schedule_bases.extend(
            RowSchedule(32, tile, rows_per_block=rows)
            for rows in (2, 4, 8, 16, 32)
            for tile in _row_tile_candidates(state, 32)
            if tile >= min_subgroup_tile
            and (concrete_rows is None or concrete_rows % rows == 0)
        )
        # Per-thread multi-accumulator trees are a bounded Full-only topology.
        # Give each tree several independent workgroup schedules while avoiding
        # a Cartesian copy of the already-large local/numeric candidate space.
        # The C++ matcher remains the hard gate for the static feature strip.
        for threads in (64, 128, 256, 512, 1024):
            tiles = _row_tile_candidates(state, threads)
            if not tiles:
                continue
            register_tile = max(tiles)
            for chains in (2, 4):
                if (
                    concrete_cols is not None
                    and (concrete_cols + threads - 1) // threads < chains
                ):
                    continue
                row_schedule_bases.append(
                    RowSchedule(
                        threads,
                        register_tile,
                        accumulator_chains=chains,
                    )
                )
        if last_axis_reduction_count == 1:
            row_schedule_bases.extend(
                RowSchedule(
                    threads,
                    32,
                    vector_width=1,
                    promote_rank0_scalars=False,
                    defer_epilogue_col_loads=False,
                    reduction_parts=parts,
                )
                for threads in (128, 256)
                for parts in (2, 4, 8, 16, 32)
            )
    if axes is not None and not axes.search_reduction_topology:
        # Schedule-Only keeps the concrete collective topology on the canonical
        # multi-warp hierarchy while still tuning its schedule parameters.
        row_schedule_bases = [
            item for item in row_schedule_bases if item.block_threads > 32
        ]
    fma_modes = (
        (False, True)
        if (axes is None or axes.search_fma) and has_fma_opportunity
        else (False,)
    )
    approximate_div_modes = (
        (False, True)
        if (axes is None or axes.search_precision) and has_division_opportunity
        else (False,)
    )
    leader_finalize_modes = (
        (False, True)
        if (axes is None or axes.search_reduction_topology)
        and last_axis_reduction_count > 1
        else (False,)
    )
    materialize_shared_row_output_modes = (
        (True, False)
        if (axes is None or axes.search_structural_states)
        and has_materialization_opportunity
        else (True,)
    )
    cross_row_finalize_modes = (
        (False, True)
        if (axes is None or axes.search_reduction_topology)
        and has_cross_row_float_sum
        else (False,)
    )
    raw_moment_modes = (
        (False, True)
        if (axes is None or axes.search_reduction_topology)
        and has_raw_moment_opportunity
        else (False,)
    )
    compact_reduction_modes = (
        (False, True)
        if axes is None or axes.search_reduction_topology
        else (False,)
    )
    promote_rank0_modes = (True, False) if has_runtime_rank0 else (False,)
    defer_column_modes = (True, False) if has_column_input else (False,)
    row_schedules = [
        replace(
            schedule,
            fuse_fma=fuse_fma,
            approximate_div=approximate_div,
            leader_finalize=leader_finalize,
            promote_rank0_scalars=promote,
            defer_epilogue_col_loads=defer,
            materialize_shared_row_output=materialize_row_output,
            cross_row_finalize=cross_row_finalize,
            raw_moment_variance=raw_moment,
            compact_reduction_topology=compact_reduction,
        )
        for schedule in row_schedule_bases
        for fuse_fma in fma_modes
        for approximate_div in approximate_div_modes
        for leader_finalize in leader_finalize_modes
        for promote in promote_rank0_modes
        for defer in defer_column_modes
        for materialize_row_output in materialize_shared_row_output_modes
        for cross_row_finalize in cross_row_finalize_modes
        for raw_moment in raw_moment_modes
        for compact_reduction in (
            (True,)
            if schedule.rows_per_block > 1
            else compact_reduction_modes
        )
        if schedule.reduction_parts == 1
        and (compact_reduction or not leader_finalize)
        and (compact_reduction or not cross_row_finalize)
        and (compact_reduction or not raw_moment)
        and (
            not cross_row_finalize
            or (schedule.rows_per_block == 1 and schedule.reduction_parts == 1)
        )
        and (
            not leader_finalize
            or (schedule.block_threads > 32 and schedule.rows_per_block == 1)
        )
        and (
            not raw_moment
            or (
                schedule.block_threads >= 64
                and schedule.rows_per_block == 1
                and schedule.reduction_parts == 1
            )
        )
    ]
    row_schedules.extend(
        schedule for schedule in row_schedule_bases if schedule.reduction_parts > 1
    )
    for schedule in row_schedules:
        values = schedule.to_dict()
        transformations: list[str] = []
        requirements: list[str] = []
        if schedule.promote_rank0_scalars and has_runtime_rank0:
            transformations.append(B_TRANSFORM_RANK0_HOST_CAPTURE)
            requirements.append(B_CAP_RANK0_HOST_CAPTURE)
        if schedule.defer_epilogue_col_loads and has_column_input:
            transformations.append(B_TRANSFORM_EPILOGUE_LOAD_MOTION)
            requirements.append(B_CAP_READONLY_INPUTS)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        changes_observable_evaluation = (
            schedule.compact_reduction_topology
            or schedule.fuse_fma
            or schedule.approximate_div
            or schedule.leader_finalize
            or schedule.rows_per_block != 1
            or schedule.reduction_parts != 1
            or not schedule.materialize_shared_row_output
            or schedule.cross_row_finalize
            or schedule.raw_moment_variance
            or schedule.accumulator_chains != 1
        )
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "row_reduction", ROW_FAST_PATH_ID, schedule.pass_argument
                ),
                "row_reduction",
                ROW_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(
                    values,
                    priors,
                    row_elements_per_thread=row_elements_per_thread,
                ),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.row_reduction.read_once.v1",
                    (
                        "cuda.reduction.gpu_all_reduce.v1"
                        if not schedule.compact_reduction_topology
                        else "cuda.reduction.two_stage_cross_row_finalize.v1"
                        if schedule.cross_row_finalize
                        else "cuda.reduction.raw_moment_hierarchical.v1"
                        if schedule.raw_moment_variance
                        else "cuda.reduction.multi_stage_partial_finalize.v1"
                        if schedule.reduction_parts > 1
                        else "cuda.reduction.per_thread_multi_accumulator_tree.v1"
                        if schedule.accumulator_chains > 1
                        else "cuda.reduction.subgroup_rows.v1"
                        if schedule.rows_per_block > 1
                        else "cuda.reduction.workgroup_leader_broadcast.v1"
                        if schedule.leader_finalize
                        else "cuda.reduction.subgroup_shuffle.v1"
                        if schedule.block_threads == 32
                        else "cuda.reduction.workgroup_hierarchical.v1"
                    ),
                    *(
                        ("cuda.numeric.fma.f32.request.v1",)
                        if schedule.fuse_fma
                        else ()
                    ),
                    *(
                        ("cuda.numeric.div_reciprocal_approx.f32.request.v1",)
                        if schedule.approximate_div
                        else ()
                    ),
                    *(
                        ("cuda.elide.intermediate_row_materialization.v1",)
                        if not schedule.materialize_shared_row_output
                        else ()
                    ),
                    *transformations,
                ),
                (
                    ObservationClass.OBSERVATION_CHANGING
                    if changes_observable_evaluation
                    else ObservationClass.HARD_PRESERVING
                ),
            )
        )

    padded_schedule_bases = [
        PaddedRowSchedule(32, False),
        PaddedRowSchedule(64, False),
        PaddedRowSchedule(128, False),
        PaddedRowSchedule(32, True),
        PaddedRowSchedule(64, True),
        PaddedRowSchedule(128, True),
    ]
    # The subgroup-feature mapping is a real C++ lowering state: one subgroup
    # owns a padded output row and lanes load contiguous feature elements.  It
    # is only admitted when IntentIR searches topology; Schedule-Only remains
    # on the fixed serial-row state.  The C++ probe supports a bounded strided
    # feature strip
    # (including natural head dimensions 64/128/256) and remains the hard gate.
    if axes is None or axes.search_reduction_topology:
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                1,
                lanes_per_row,
            )
            for threads in (64, 128, 256, 512, 1024)
            for vectorize in (False, True)
            for lanes_per_row in (32,)
        )
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                1,
                lanes_per_row,
            )
            for threads in (128, 256)
            for vectorize in (False, True)
            for lanes_per_row in (4, 8, 16)
        )
        # The lowering already supports assigning several independent output
        # rows to each logical subgroup.  This is a genuine launch/scheduling
        # coordinate of the subgroup-feature state: it reduces the block grid
        # and exposes row-level ILP without changing the per-row reduction tree.
        # Keep the one-row states first so existing C-prior ordering remains
        # stable, then search the complete bounded 2/4/8 domain implemented by
        # the C++ pass.
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                rows_per_subgroup,
                lanes_per_row,
            )
            for rows_per_subgroup in (2, 4, 8)
            for threads in (64, 128, 256, 512, 1024)
            for vectorize in (False, True)
            for lanes_per_row in (32,)
        )
        padded_schedule_bases.extend(
            PaddedRowSchedule(
                threads,
                vectorize,
                True,
                "subgroup-feature",
                rows_per_subgroup,
                lanes_per_row,
            )
            for rows_per_subgroup in (2, 4, 8)
            for threads in (128, 256)
            for vectorize in (False, True)
            for lanes_per_row in (4, 8, 16)
        )
    for schedule in (
        replace(base, promote_rank0_scalars=promote)
        for base in padded_schedule_bases
        for promote in (True, False)
    ):
        values = schedule.to_dict()
        transformations = []
        requirements = []
        if schedule.promote_rank0_scalars and has_runtime_rank0:
            transformations.append(B_TRANSFORM_RANK0_HOST_CAPTURE)
            requirements.append(B_CAP_RANK0_HOST_CAPTURE)
        if not set(requirements) <= capabilities:
            continue
        if transformations and not enable_b_transformations:
            continue
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "padded_row_reduction",
                    PADDED_ROW_FAST_PATH_ID,
                    schedule.pass_argument,
                ),
                "padded_row_reduction",
                PADDED_ROW_FAST_PATH_ID,
                schedule.pass_argument,
                values,
                _c_score(values, priors),
                tuple(transformations),
                tuple(requirements),
                (
                    "cuda.padded_row_reduction.masked_short_dot.v1",
                    (
                        "cuda.reduction.padded_subgroup_feature.v1"
                        if schedule.reduction_topology == "subgroup-feature"
                        else "cuda.reduction.padded_serial_row.v1"
                    ),
                    *(
                        ("cuda.reduction.padded_subgroup_multirow.v1",)
                        if schedule.rows_per_subgroup > 1
                        else ()
                    ),
                    *transformations,
                ),
                ObservationClass.OBSERVATION_CHANGING,
            )
        )

    # The typed unrolled SplitK graph has one concrete whole-graph compiler path
    # and genuine launch-schedule axes: output rows per warp block and contiguous
    # D-packet width.  Keep the pass default (rows=4, warps-per-row=1, width=1)
    # byte-identical to the canonical compiler state so the Schedule-Only prefix
    # is nested.  Multi-warp-per-row is a distinct structural state and is only
    # opened with the Full structural axes.
    # The compiler probe still decides
    # whether the current J has the required weighted-merge SSA topology.
    broadcast_modes = (
        (False, True)
        if axes is None or axes.search_structural_states
        else (False,)
    )
    warps_per_row_modes = (
        (1, 2, 4, 8)
        if axes is None or axes.search_structural_states
        else (1,)
    )
    d_partition_modes = (
        (1, 2, 4, 8)
        if axes is None or axes.search_structural_states
        else (1,)
    )
    for rows_per_block in (4, 1, 2, 8):
        for warps_per_row in warps_per_row_modes:
            if warps_per_row > 1 and rows_per_block != 1:
                continue
            if rows_per_block * warps_per_row > 32:
                continue
            for d_partitions in d_partition_modes:
                if d_partitions > 1 and (
                    rows_per_block != 1 or warps_per_row != 1
                ):
                    continue
                for vector_width in (1, 2, 4):
                    for broadcast_scalars in broadcast_modes:
                        options = []
                        if rows_per_block != 4:
                            options.append(f"rows-per-block={rows_per_block}")
                        if warps_per_row != 1:
                            options.append(f"warps-per-row={warps_per_row}")
                        if d_partitions != 1:
                            options.append(f"d-partitions={d_partitions}")
                        if vector_width != 1:
                            options.append(f"vector-width={vector_width}")
                        if broadcast_scalars:
                            options.append("warp-broadcast-scalars=true")
                        pass_argument = (
                            "-convert-intent-splitk-reduce-varargs-to-gpu"
                            if not options
                            else "-convert-intent-splitk-reduce-varargs-to-gpu="
                            + " ".join(options)
                        )
                        values = {
                            "rows_per_block": rows_per_block,
                            "warps_per_row": warps_per_row,
                            "d_partitions": d_partitions,
                            "block_threads": rows_per_block * warps_per_row * 32,
                            "vector_width": vector_width,
                            "broadcast_row_scalars": broadcast_scalars,
                        }
                        specialized.append(
                            CudaSearchCandidate(
                                _candidate_id(
                                    "splitk_weighted_merge",
                                    SPLITK_VARARGS_FAST_PATH_ID,
                                    pass_argument,
                                ),
                                "splitk_weighted_merge",
                                SPLITK_VARARGS_FAST_PATH_ID,
                                pass_argument,
                                values,
                                _c_score(values, priors),
                                requested_trace=(
                                    "cuda.splitk.unrolled_weighted_merge.v2",
                                    "cuda.materialization.whole_graph_elimination.v1",
                                    *(
                                        ("cuda.splitk.warp_broadcast_row_scalars.v1",)
                                        if broadcast_scalars and warps_per_row == 1
                                        else ()
                                    ),
                                    *(
                                        ("cuda.splitk.workgroup_broadcast_row_scalars.v1",)
                                        if broadcast_scalars and warps_per_row > 1
                                        else ()
                                    ),
                                    *(
                                        ("cuda.splitk.multi_warp_d_partition.v1",)
                                        if warps_per_row > 1
                                        else ()
                                    ),
                                    *(
                                        ("cuda.splitk.cross_cta_d_partition.v1",)
                                        if d_partitions > 1
                                        else ()
                                    ),
                                ),
                                requested_observation_class=(
                                    ObservationClass.OBSERVATION_CHANGING
                                ),
                            )
                        )

    # A sorted-prefix graph has additional physical mappings unavailable to the
    # canonical full-row scan: multiple warp-owned rows may share a CTA, or one
    # warp may own only the active-prefix scan while a wider CTA writes the full
    # row epilogue.  The latter decouples prefix work from output parallelism.
    # Enumerate it from registered path identities and let the C++ graph matcher
    # fail closed for every non-Sparsemax graph; no kernel-name admission occurs.
    sorted_prefix_ids = {
        "cuda.row_scan.sparsemax.sorted_prefix.v1",
        "cuda.row_scan.sparsemax.sorted_prefix.execution_domain.v1",
    }
    if axes is None or axes.search_reduction_topology:
        base_prefix_geometries = (
            # output threads, rows/CTA, tile cols, prefix threads, cross-multiply
            (256, 1, 32, 0, False),
            (128, 1, 32, 0, False),
            (64, 1, 64, 0, False),
            (32, 1, 128, 0, False),
            (32, 2, 128, 0, False),
            (32, 4, 128, 0, False),
            (32, 8, 128, 0, False),
            (32, 16, 128, 0, False),
            (64, 1, 64, 32, False),
            (128, 1, 32, 32, False),
            (256, 1, 32, 32, False),
            (512, 1, 16, 32, False),
            (1024, 1, 8, 32, False),
            (256, 2, 32, 32, False),
            (256, 4, 32, 32, False),
            (512, 2, 16, 32, False),
            (512, 1, 16, 32, True),
            (1024, 1, 8, 32, True),
            (512, 2, 16, 32, True),
        )
        prefix_geometries = tuple(
            (*geometry, prefetch_epilogue_input, frontier_ballot)
            for prefetch_epilogue_input in (False, True)
            for geometry in base_prefix_geometries
            for frontier_ballot in (
                (False, True) if geometry[1] == 1 else (False,)
            )
        )
        for descriptor in DEFAULT_PASSES:
            if descriptor.fast_path_id not in sorted_prefix_ids:
                continue
            if state.target.architecture not in descriptor.supported_chips:
                continue
            execution_domain = descriptor.fast_path_id.endswith(
                ".execution_domain.v1"
            )
            for (
                threads_per_row,
                rows_per_block,
                tile_cols,
                prefix_threads,
                cross_multiply,
                prefetch_epilogue_input,
                frontier_ballot,
            ) in prefix_geometries:
                prefix_packet_widths = (
                    (4, 2, 1)
                    if prefetch_epilogue_input
                    else (4,)
                )
                for prefix_packet_width in prefix_packet_widths:
                    options = [
                        f"block-threads={threads_per_row}",
                        f"tile-cols={tile_cols}",
                        "vector-width=4",
                        f"rows-per-block={rows_per_block}",
                        "require-sorted-prefix=true",
                    ]
                    if prefix_packet_width != 4:
                        options.insert(
                            -1, f"prefix-packet-width={prefix_packet_width}"
                        )
                    if prefix_threads:
                        options.insert(-1, f"prefix-threads={prefix_threads}")
                    if cross_multiply:
                        options.insert(
                            -1, "cross-multiply-support-predicate=true"
                        )
                    if prefetch_epilogue_input:
                        options.insert(
                            -1, "prefetch-epilogue-input-before-summary=true"
                        )
                    if frontier_ballot:
                        options.insert(-1, "frontier-ballot-support=true")
                    if execution_domain:
                        options.insert(
                            -1, "allow-execution-domain-sorted-prefix=true"
                        )
                    pass_argument = (
                        "-convert-intent-row-reduction-to-gpu="
                        + " ".join(options)
                    )
                    values = {
                        "threads_per_row": threads_per_row,
                        "rows_per_block": rows_per_block,
                        "block_threads": threads_per_row * rows_per_block,
                        "tile_cols": tile_cols,
                        "vector_width": 4,
                        "prefix_packet_width": prefix_packet_width,
                    }
                    if prefix_threads:
                        values["prefix_threads"] = prefix_threads
                    if cross_multiply:
                        values["cross_multiply_support_predicate"] = True
                    if prefetch_epilogue_input:
                        values["prefetch_epilogue_input_before_summary"] = True
                    if frontier_ballot:
                        values["frontier_ballot_support"] = True
                    topology_trace = (
                        "cuda.reduction.sorted_prefix_multirow_warp_scan_wide_epilogue_cross_multiply.v1"
                        if rows_per_block > 1 and cross_multiply
                        else "cuda.reduction.sorted_prefix_multirow_warp_scan_wide_epilogue.v1"
                        if rows_per_block > 1 and prefix_threads
                        else "cuda.reduction.sorted_prefix_multirow_warp_early_stop.v1"
                        if rows_per_block > 1
                        else "cuda.reduction.sorted_prefix_warp_scan_wide_epilogue_cross_multiply.v1"
                        if cross_multiply
                        else "cuda.reduction.sorted_prefix_warp_scan_wide_epilogue.v1"
                        if prefix_threads
                        else "cuda.reduction.sorted_prefix_early_stop.v1"
                    )
                    if frontier_ballot:
                        topology_trace = (
                            topology_trace.removesuffix(".v1")
                            + "_frontier_ballot.v1"
                        )
                    specialized.append(
                        CudaSearchCandidate(
                            _candidate_id(
                                "row_scan",
                                descriptor.fast_path_id,
                                pass_argument,
                            ),
                            "row_scan",
                            descriptor.fast_path_id,
                            pass_argument,
                            values,
                            _c_score(values, priors),
                            requested_trace=(
                                topology_trace,
                                *(
                                    (
                                        "cuda.memory.prefetch_epilogue_input_before_summary.v1",
                                    )
                                    if prefetch_epilogue_input
                                    else ()
                                ),
                            ),
                            requested_observation_class=(
                                ObservationClass.OBSERVATION_CHANGING
                            ),
                        )
                    )
        # Barrier-free sorted-prefix state: every output warp recomputes the
        # short active prefix and keeps tau in warp-local SSA.  This is a real
        # topology change, not another block/tile schedule, so it is deliberately
        # absent from the Schedule-Only spine.  The C++ matcher remains the hard
        # graph/precondition gate for both registered path identities.
        base_replicated_prefix_geometries = (
            # output threads, tile cols
            (64, 64),
            (128, 32),
            (256, 32),
            (512, 16),
            (1024, 8),
        )
        replicated_prefix_geometries = tuple(
            (*geometry, prefetch_epilogue_input, frontier_ballot)
            for prefetch_epilogue_input in (False, True)
            for frontier_ballot in (False, True)
            for geometry in base_replicated_prefix_geometries
        )
        for descriptor in DEFAULT_PASSES:
            if descriptor.fast_path_id not in sorted_prefix_ids:
                continue
            if state.target.architecture not in descriptor.supported_chips:
                continue
            execution_domain = descriptor.fast_path_id.endswith(
                ".execution_domain.v1"
            )
            for (
                threads_per_row,
                tile_cols,
                prefetch_epilogue_input,
                frontier_ballot,
            ) in replicated_prefix_geometries:
                for cross_multiply in (False, True):
                    options = [
                        f"block-threads={threads_per_row}",
                        f"tile-cols={tile_cols}",
                        "vector-width=4",
                        "rows-per-block=1",
                        "prefix-threads=32",
                        "replicate-prefix-per-warp=true",
                        "require-sorted-prefix=true",
                    ]
                    if cross_multiply:
                        options.insert(
                            -1, "cross-multiply-support-predicate=true"
                        )
                    if prefetch_epilogue_input:
                        options.insert(
                            -1, "prefetch-epilogue-input-before-summary=true"
                        )
                    if frontier_ballot:
                        options.insert(-1, "frontier-ballot-support=true")
                    if execution_domain:
                        options.insert(
                            -1, "allow-execution-domain-sorted-prefix=true"
                        )
                    pass_argument = (
                        "-convert-intent-row-reduction-to-gpu="
                        + " ".join(options)
                    )
                    values = {
                        "threads_per_row": threads_per_row,
                        "rows_per_block": 1,
                        "block_threads": threads_per_row,
                        "tile_cols": tile_cols,
                        "vector_width": 4,
                        "prefix_threads": 32,
                        "replicate_prefix_per_warp": True,
                    }
                    if cross_multiply:
                        values["cross_multiply_support_predicate"] = True
                    if prefetch_epilogue_input:
                        values[
                            "prefetch_epilogue_input_before_summary"
                        ] = True
                    if frontier_ballot:
                        values["frontier_ballot_support"] = True
                    topology_trace = (
                        "cuda.reduction.sorted_prefix_replicated_warp_cross_multiply.v1"
                        if cross_multiply
                        else "cuda.reduction.sorted_prefix_replicated_warp.v1"
                    )
                    if frontier_ballot:
                        topology_trace = (
                            topology_trace.removesuffix(".v1")
                            + "_frontier_ballot.v1"
                        )
                    specialized.append(
                        CudaSearchCandidate(
                            _candidate_id(
                                "row_scan",
                                descriptor.fast_path_id,
                                pass_argument,
                            ),
                            "row_scan",
                            descriptor.fast_path_id,
                            pass_argument,
                            values,
                            _c_score(values, priors),
                            requested_trace=(
                                topology_trace,
                                *(
                                    (
                                        "cuda.memory.prefetch_epilogue_input_before_summary.v1",
                                    )
                                    if prefetch_epilogue_input
                                    else ()
                                ),
                            ),
                            requested_observation_class=(
                                ObservationClass.OBSERVATION_CHANGING
                            ),
                        )
                    )

    # Canonical contraction schedule spine.  Every state invokes the same WMMA
    # C++ pass and changes only BM/BN/warps/pipeline depth; this is the exact
    # C-Fixed -> Schedule-Only retile space.  Full separately admits the
    # register-level mma.sync state below.
    tensor_core_chips = {"sm_90a", "sm_120a"}
    if state.target.architecture in tensor_core_chips:
        for block_m in (16, 32, 64, 128):
            for block_n in (32, 64, 128):
                for pipeline_stages in (1, 2, 3, 4, 5):
                    pass_argument = (
                        "-convert-intent-contraction-to-gpu="
                        f"num-stages={pipeline_stages} block-m={block_m} "
                        f"block-n={block_n}"
                    )
                    schedule = {
                        "tile_rows": block_m,
                        "tile_cols": block_n,
                        "tile_k": 32,
                        "num_warps": block_m // 16,
                        "pipeline_stages": pipeline_stages,
                    }
                    specialized.append(
                        CudaSearchCandidate(
                            _candidate_id(
                                DIRECT_CONTRACTION_FAST_PATH_ID,
                                DIRECT_CONTRACTION_FAST_PATH_ID,
                                pass_argument,
                            ),
                            DIRECT_CONTRACTION_FAST_PATH_ID,
                            DIRECT_CONTRACTION_FAST_PATH_ID,
                            pass_argument,
                            schedule,
                            _c_score(schedule, priors),
                            requested_trace=(DIRECT_CONTRACTION_FAST_PATH_ID,),
                            requested_observation_class=(
                                ObservationClass.OBSERVATION_CHANGING
                            ),
                        )
                    )

        # Full implementation state: known-layout register mma.sync plus its own
        # independent schedule space.  The exact typed Flash Linear epilogue is
        # kept in registers and writes both observable f16 outputs; the canonical
        # WMMA spine above materializes the f32 block tile instead.
        def append_mma_candidate(
            block_m: int,
            block_n: int,
            block_k: int,
            pipeline_stages: int,
            smem_pad_k: int,
            warps_n: int = 1,
            group_m: int = 8,
        ) -> None:
            pass_argument = (
                "-convert-intent-contraction-to-gpu-mma="
                f"num-stages={pipeline_stages} block-m={block_m} "
                f"block-n={block_n} block-k={block_k} "
                f"smem-pad-k={smem_pad_k}"
            )
            if warps_n != 1:
                pass_argument += f" warps-n={warps_n}"
            if group_m != 8:
                pass_argument += f" group-m={group_m}"
            schedule = {
                "tile_rows": block_m,
                "tile_cols": block_n,
                "tile_k": block_k,
                "num_warps": (block_m // 16) * warps_n,
                "warps_n": warps_n,
                "group_m": group_m,
                "pipeline_stages": pipeline_stages,
                "smem_pad_k": smem_pad_k,
            }
            requested_trace = [MMA_CONTRACTION_FAST_PATH_ID]
            if warps_n != 1:
                requested_trace.append(
                    "cuda.contraction.mma-sync.warp-n-partition.v1"
                )
            specialized.append(
                CudaSearchCandidate(
                    _candidate_id(
                        MMA_CONTRACTION_FAST_PATH_ID,
                        MMA_CONTRACTION_FAST_PATH_ID,
                        pass_argument,
                    ),
                    MMA_CONTRACTION_FAST_PATH_ID,
                    MMA_CONTRACTION_FAST_PATH_ID,
                    pass_argument,
                    schedule,
                    _c_score(schedule, priors),
                    requested_trace=tuple(requested_trace),
                    requested_observation_class=(
                        ObservationClass.OBSERVATION_CHANGING
                    ),
                )
            )

        # Give the register-pressure-reducing 2-D warp map an early, bounded
        # lane.  It changes the tensor-core execution structure, while each
        # point still computes the same typed contraction and epilogue.
        group_m_mma_schedules = (
            (64, 128, 64, 2, 8, 4, 1),
            (64, 128, 64, 2, 8, 4, 2),
            (64, 128, 64, 2, 8, 4, 4),
            (64, 128, 64, 2, 8, 4, 16),
            (64, 128, 64, 2, 8, 4, 32),
            (64, 128, 64, 3, 16, 2, 16),
            (64, 128, 64, 3, 16, 1, 16),
        )
        for config in group_m_mma_schedules:
            append_mma_candidate(*config)

        warp_n_mma_schedules = (
            (64, 128, 96, 2, 0, 2),
            (64, 128, 96, 2, 8, 2),
            (64, 128, 96, 2, 16, 2),
            (64, 128, 128, 2, 0, 2),
            (64, 128, 128, 2, 0, 4),
            (64, 128, 192, 1, 0, 2),
            (64, 128, 192, 1, 8, 2),
            (64, 128, 192, 1, 0, 4),
            (64, 128, 192, 1, 8, 4),
            (64, 128, 256, 1, 0, 2),
            (64, 128, 256, 1, 0, 4),
            (128, 128, 32, 2, 8, 2),
            (128, 128, 32, 3, 8, 2),
            (128, 128, 64, 1, 8, 2),
            (128, 128, 64, 2, 8, 2),
            (128, 128, 64, 2, 16, 2),
            (128, 128, 64, 1, 8, 4),
            (128, 128, 64, 2, 8, 4),
            (128, 64, 64, 1, 8, 2),
            (128, 64, 64, 2, 8, 2),
            (128, 64, 64, 3, 8, 2),
            (32, 128, 128, 1, 8, 8),
            (32, 128, 128, 1, 16, 8),
            (64, 128, 128, 1, 8, 8),
            (64, 128, 128, 1, 16, 8),
            (32, 128, 64, 2, 8, 2),
            (32, 128, 64, 3, 8, 2),
            (32, 128, 64, 2, 8, 4),
            (64, 64, 64, 2, 8, 2),
            (64, 64, 64, 3, 8, 2),
            (64, 128, 32, 2, 8, 2),
            (64, 128, 32, 4, 8, 2),
            (64, 128, 64, 1, 8, 2),
            (64, 128, 64, 2, 8, 2),
            (64, 128, 64, 3, 8, 2),
            (64, 128, 64, 3, 16, 2),
            (64, 128, 64, 1, 8, 4),
            (64, 128, 64, 2, 8, 4),
            (64, 128, 64, 3, 8, 4),
            (64, 128, 64, 3, 16, 4),
        )
        for config in warp_n_mma_schedules:
            append_mma_candidate(*config)

        mma_schedules = (
            (16, 32, 32, 3, 8),
            (16, 32, 32, 5, 8),
            (32, 32, 32, 3, 8),
            (32, 64, 32, 3, 8),
            (32, 64, 64, 3, 8),
            (32, 128, 32, 2, 8),
            (32, 128, 64, 2, 8),
            (32, 128, 64, 3, 8),
            (64, 64, 32, 2, 8),
            (64, 64, 32, 4, 8),
            (64, 128, 32, 1, 8),
            (64, 128, 32, 2, 8),
            (64, 128, 32, 3, 8),
            (64, 128, 32, 4, 8),
            (64, 128, 64, 1, 0),
            (64, 128, 64, 1, 8),
            (64, 128, 64, 1, 16),
            (64, 128, 64, 2, 0),
            (64, 128, 64, 2, 8),
            (64, 128, 64, 2, 16),
            (64, 128, 64, 3, 0),
            (64, 128, 64, 3, 8),
            (64, 128, 64, 3, 16),
            (64, 128, 128, 1, 0),
            (64, 128, 128, 1, 8),
            (64, 128, 128, 1, 16),
            (128, 64, 32, 2, 8),
            (128, 64, 64, 1, 8),
            (128, 64, 64, 2, 8),
            (128, 64, 128, 1, 8),
            (128, 128, 32, 2, 8),
            (128, 128, 64, 2, 8),
        )
        for block_m, block_n, block_k, pipeline_stages, smem_pad_k in mma_schedules:
            append_mma_candidate(
                block_m,
                block_n,
                block_k,
                pipeline_stages,
                smem_pad_k,
            )

        splitk_mma_schedules = (
            (64, 128, 64, 2, 8, 4, 2),
            (64, 128, 64, 3, 8, 4, 2),
            (64, 128, 64, 2, 16, 4, 2),
            (64, 128, 64, 2, 8, 4, 4),
        )
        for (
            block_m,
            block_n,
            block_k,
            pipeline_stages,
            smem_pad_k,
            warps_n,
            split_k_parts,
        ) in splitk_mma_schedules:
            pass_argument = (
                "-convert-intent-contraction-to-gpu-mma="
                f"num-stages={pipeline_stages} block-m={block_m} "
                f"block-n={block_n} block-k={block_k} "
                f"smem-pad-k={smem_pad_k} warps-n={warps_n} "
                f"split-k-parts={split_k_parts}"
            )
            schedule = {
                "tile_rows": block_m,
                "tile_cols": block_n,
                "tile_k": block_k,
                "num_warps": (block_m // 16) * warps_n,
                "warps_n": warps_n,
                "pipeline_stages": pipeline_stages,
                "smem_pad_k": smem_pad_k,
                "split_k_parts": split_k_parts,
                "stage_count": 2,
            }
            specialized.append(
                CudaSearchCandidate(
                    _candidate_id(
                        MMA_CONTRACTION_FAST_PATH_ID,
                        MMA_CONTRACTION_FAST_PATH_ID,
                        pass_argument,
                    ),
                    MMA_CONTRACTION_FAST_PATH_ID,
                    MMA_CONTRACTION_FAST_PATH_ID,
                    pass_argument,
                    schedule,
                    _c_score(schedule, priors),
                    requested_trace=(
                        MMA_CONTRACTION_FAST_PATH_ID,
                        "cuda.contraction.mma-sync.warp-n-partition.v1",
                        "cuda.contraction.split_k_partial_finalize.v1",
                    ),
                    requested_observation_class=(
                        ObservationClass.OBSERVATION_CHANGING
                    ),
                )
            )

    output_rank = max(
        (
            len(binding.output_type.shape)
            for binding in j_elaboration(state.base.j).output_bindings
        ),
        default=1,
    )
    for width in (64, 128, 256):
        tile = (width,) if output_rank <= 1 else (1, width)
        tile_text = "x".join(str(item) for item in tile)
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    "linalg_fused",
                    LINALG_FUSED_FAST_PATH_ID,
                    f"linalg-fuse-elementwise-ops;tile={tile_text}",
                ),
                "linalg_fused",
                LINALG_FUSED_FAST_PATH_ID,
                f"linalg-fuse-elementwise-ops;tile={tile_text}",
                {
                    "tile_shape": tile_text,
                    "tile_width": width,
                },
                _c_score({"tile_cols": width}, priors),
                requested_trace=(
                    "cuda.linalg.elementwise_fusion.v1",
                    "cuda.linalg.parallel_loop_tiling.v1",
                ),
                compiler_pipeline="linalg_fused",
            )
        )

    # Runtime-index schedules.  The canonical descriptor evaluates a typed
    # gather graph in its authored order; the composed descriptor is a distinct
    # Full-only index/materialization topology admitted only when the C++ matcher
    # consumes the complete two-level chain.
    gather_descriptors = tuple(
        item
        for item in DEFAULT_PASSES
        if item.fast_path_id in {GATHER_FAST_PATH_ID, GATHER_COMPOSED_FAST_PATH_ID}
        and state.target.architecture in item.supported_chips
    )
    for descriptor in gather_descriptors:
        composed = descriptor.fast_path_id == GATHER_COMPOSED_FAST_PATH_ID
        if composed and axes is not None and not axes.search_structural_states:
            continue
        for threads in (32, 64, 128, 256):
            pass_argument = (
                "-convert-intent-gather-to-gpu="
                f"block-threads={threads}"
                + (" compose-gather-chain=true" if composed else "")
            )
            schedule = {
                "block_threads": threads,
                "compose_gather_chain": composed,
            }
            specialized.append(
                CudaSearchCandidate(
                    _candidate_id(
                        _lowering_path_kind_for_fast_path(descriptor.fast_path_id),
                        descriptor.fast_path_id,
                        pass_argument,
                    ),
                    _lowering_path_kind_for_fast_path(descriptor.fast_path_id),
                    descriptor.fast_path_id,
                    pass_argument,
                    schedule,
                    _c_score(schedule, priors),
                    requested_trace=(
                        "cuda.gather-chain.composed-index.v1"
                        if composed
                        else "cuda.gather-chain.materialized-row.v1",
                    ),
                    requested_observation_class=ObservationClass.HARD_PRESERVING,
                )
            )

    # Every target-compatible C++ whole-graph pass is a real structural search
    # path.  The three high-volume paths above own schedule grids; all other
    # registered passes enter with their compiler default exactly once.  The
    # structural probe still decides admission for the current J, so this is
    # neither a kernel allowlist nor a Python-emitted implementation.
    scheduled_ids = {
        ELEMENTWISE_FAST_PATH_ID,
        PADDED_ROW_FAST_PATH_ID,
        ROW_FAST_PATH_ID,
        SPLITK_VARARGS_FAST_PATH_ID,
        DIRECT_CONTRACTION_FAST_PATH_ID,
        MMA_CONTRACTION_FAST_PATH_ID,
        TUNED_LINEAR_FAST_PATH_ID,
        SHIFTED_EXP_CANCELLATION_FAST_PATH_ID,
        GATHER_FAST_PATH_ID,
        GATHER_COMPOSED_FAST_PATH_ID,
        ROWMASK_ZERO_FILL_FAST_PATH_ID,
        *sorted_prefix_ids,
    }
    for descriptor in DEFAULT_PASSES:
        if descriptor.fast_path_id in scheduled_ids:
            continue
        if state.target.architecture not in descriptor.supported_chips:
            continue
        path_kind = _lowering_path_kind_for_fast_path(
            descriptor.fast_path_id
        )
        specialized.append(
            CudaSearchCandidate(
                _candidate_id(
                    path_kind,
                    descriptor.fast_path_id,
                    descriptor.pass_argument,
                ),
                path_kind,
                descriptor.fast_path_id,
                descriptor.pass_argument,
                dict(descriptor.schedule),
                0.0,
                requested_trace=(descriptor.fast_path_id,),
                requested_observation_class=(
                    ObservationClass.OBSERVATION_CHANGING
                ),
            )
        )

    if fixed_fast_path_id is not None:
        specialized = [
            item
            for item in specialized
            if item.fast_path_id == fixed_fast_path_id
        ]
    elif fixed_lowering_path_kind is not None:
        specialized = [
            item
            for item in specialized
            if item.lowering_path_kind == fixed_lowering_path_kind
        ]
    if fixed_fast_path_id is not None and not specialized:
        raise CudaContractError(
            f"fixed CUDA path {fixed_fast_path_id!r} is not registered for the target"
        )
    if (
        fixed_lowering_path_kind not in {None, "generic"}
        and not specialized
    ):
        raise CudaContractError(
            "fixed CUDA lowering path kind "
            f"{fixed_lowering_path_kind!r} has no registered concrete state"
        )

    # A C tile/thread prior has no authority to rank one lowering path above
    # another. Use C only to order schedules inside each registered path kind;
    # expanded arms later prepend the fixed-structure schedule spine before
    # round-robin path discovery so IntentIR cannot forget Schedule-Only under
    # a finite budget.
    def search_lane(item: CudaSearchCandidate) -> str:
        local = "local" if item.b_transformations else "base"
        if item.fast_path_id == SHIFTED_EXP_CANCELLATION_FAST_PATH_ID:
            return "elementwise.shifted-exp-cancellation.base"
        if int(item.schedule.get("split_k_parts", 1)) > 1:
            return "contraction.split-k.base"
        if item.lowering_path_kind == "row_reduction":
            topology = (
                "all_reduce"
                if item.schedule.get("compact_reduction_topology") is False
                else "cross_row_finalize"
                if item.schedule.get("cross_row_finalize") is True
                else "raw_moment"
                if item.schedule.get("raw_moment_variance") is True
                else "multi_stage"
                if int(item.schedule.get("reduction_parts", 1)) > 1
                else (
                    "multi_accumulator"
                    f"{int(item.schedule.get('accumulator_chains', 1))}"
                )
                if int(item.schedule.get("accumulator_chains", 1)) > 1
                else "subgroup_rows"
                if int(item.schedule.get("rows_per_block", 1)) > 1
                else "workgroup_leader"
                if item.schedule.get("leader_finalize") is True
                else "subgroup"
                if int(item.schedule.get("block_threads", 64)) == 32
                else "workgroup"
            )
            if item.schedule.get("approximate_div") is True:
                numeric = "approximate_div"
            elif item.schedule.get("fuse_fma") is True:
                numeric = "fma"
            else:
                numeric = "plain"
            if item.schedule.get("materialize_shared_row_output") is False:
                return f"row_reduction.{topology}.{numeric}.elide.{local}"
            return f"row_reduction.{topology}.{numeric}.{local}"
        if item.lowering_path_kind == "elementwise":
            paired_head_reuse = bool(
                item.schedule.get("paired_head_reuse", False)
            )
            paired_direct_plane = bool(
                item.schedule.get("paired_direct_plane", False)
            )
            if paired_head_reuse or paired_direct_plane:
                paired_state = (
                    "head-reuse+direct-plane"
                    if paired_head_reuse and paired_direct_plane
                    else "head-reuse"
                    if paired_head_reuse
                    else "direct-plane"
                )
                return f"elementwise.paired-{paired_state}.{local}"
            approximation = str(
                item.schedule.get("tanh_approximation", "stable-ex2")
            )
            return f"elementwise.{approximation}.{local}"
        if item.lowering_path_kind == "padded_row_reduction":
            topology = str(item.schedule.get("reduction_topology", "serial-row"))
            packet = (
                "packet"
                if item.schedule.get("vectorize_full_features") is True
                else "scalar"
            )
            lanes = int(item.schedule.get("lanes_per_row", 32))
            return (
                f"{item.lowering_path_kind}.{topology}.{packet}."
                f"lanes{lanes}.{local}"
            )
        return item.lowering_path_kind

    known_lane_order = (
        "elementwise.stable-ex2.base",
        "row_reduction.workgroup.plain.base",
        "padded_row_reduction.serial-row.scalar.lanes32.base",
        "elementwise.paired-head-reuse.base",
        "elementwise.paired-direct-plane.base",
        "elementwise.paired-head-reuse+direct-plane.base",
        "elementwise.shifted-exp-cancellation.base",
        "contraction.split-k.base",
        "padded_row_reduction.serial-row.packet.lanes32.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes32.base",
        "padded_row_reduction.subgroup-feature.packet.lanes16.base",
        "padded_row_reduction.subgroup-feature.packet.lanes8.base",
        "padded_row_reduction.subgroup-feature.packet.lanes4.base",
        "padded_row_reduction.subgroup-feature.packet.lanes32.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes16.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes8.base",
        "padded_row_reduction.subgroup-feature.scalar.lanes4.base",
        "row_reduction.cross_row_finalize.plain.base",
        "row_reduction.cross_row_finalize.fma.base",
        "row_reduction.raw_moment.plain.base",
        "row_reduction.raw_moment.fma.base",
        "row_reduction.raw_moment.approximate_div.base",
        "row_reduction.multi_accumulator2.plain.base",
        "row_reduction.multi_accumulator4.plain.base",
        "row_reduction.multi_stage.plain.base",
        "elementwise.native-ptx.base",
        "row_reduction.subgroup_rows.plain.base",
        "row_reduction.subgroup.plain.base",
        "row_reduction.workgroup.fma.base",
        "row_reduction.subgroup.fma.base",
        "row_reduction.workgroup.approximate_div.base",
        "row_reduction.subgroup_rows.approximate_div.base",
        "row_reduction.subgroup.approximate_div.base",
        "row_reduction.workgroup_leader.plain.base",
        "row_reduction.workgroup_leader.fma.base",
        "row_reduction.workgroup_leader.approximate_div.base",
        "row_reduction.workgroup.plain.elide.base",
        "row_reduction.workgroup.fma.elide.base",
        "row_reduction.workgroup.approximate_div.elide.base",
        "row_reduction.workgroup_leader.plain.elide.base",
        "row_reduction.workgroup_leader.fma.elide.base",
        "row_reduction.workgroup_leader.approximate_div.elide.base",
        "elementwise.stable-ex2.local",
        "elementwise.native-ptx.local",
        "row_reduction.workgroup.plain.local",
        "row_reduction.subgroup_rows.plain.local",
        "row_reduction.subgroup.plain.local",
        "row_reduction.workgroup.fma.local",
        "row_reduction.subgroup.fma.local",
        "row_reduction.workgroup.approximate_div.local",
        "row_reduction.subgroup_rows.approximate_div.local",
        "row_reduction.subgroup.approximate_div.local",
        "row_reduction.workgroup.plain.elide.local",
        "row_reduction.workgroup.fma.elide.local",
        "row_reduction.workgroup.approximate_div.elide.local",
        "row_reduction.cross_row_finalize.plain.local",
        "row_reduction.cross_row_finalize.fma.local",
        "padded_row_reduction.serial-row.scalar.lanes32.local",
        "padded_row_reduction.serial-row.packet.lanes32.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes32.local",
        "padded_row_reduction.subgroup-feature.packet.lanes16.local",
        "padded_row_reduction.subgroup-feature.packet.lanes8.local",
        "padded_row_reduction.subgroup-feature.packet.lanes4.local",
        "padded_row_reduction.subgroup-feature.packet.lanes32.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes16.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes8.local",
        "padded_row_reduction.subgroup-feature.scalar.lanes4.local",
    )
    lanes = {search_lane(item) for item in specialized}
    path_order = (
        *(lane for lane in known_lane_order if lane in lanes),
        *sorted(lanes - set(known_lane_order)),
    )
    by_path = {
        lane: [item for item in specialized if search_lane(item) == lane]
        for lane in path_order
    }
    for candidates in by_path.values():
        candidates.sort(key=lambda item: item.c_order_score)
    specialized = []
    rank = 0
    while any(rank < len(by_path[path]) for path in path_order):
        for path in path_order:
            candidates = by_path[path]
            if rank < len(candidates):
                specialized.append(candidates[rank])
        rank += 1
    generic = _generic_cuda_candidate()
    if fixed_fast_path_id is not None:
        return tuple(specialized)
    if fixed_lowering_path_kind is not None:
        if fixed_lowering_path_kind == "generic":
            return (generic,)
        return tuple(specialized)
    expanded = (generic, *specialized)
    if axes is None or not axes.search_structural_states:
        return expanded

    # IntentIR expands the concrete state space, but its fixed budget must retain
    # the complete Schedule-Only prefix.  Build that prefix through this exact
    # enumerator with structural/numeric/local expansion disabled.
    canonical_kind, canonical_fast_path_id = schedule_only_cuda_path_identity(
        state,
        intent_module=intent_module,
    )
    schedule_axes = replace(
        axes,
        search_structural_states=False,
        enable_local_b_transformations=False,
        search_reduction_topology=False,
        search_fma=False,
        search_precision=False,
        search_approximation=False,
    )
    schedule_spine = enumerate_cuda_candidates(
        state,
        enable_b_transformations=False,
        axes=schedule_axes,
        fixed_lowering_path_kind=(
            canonical_kind
        ),
        fixed_fast_path_id=canonical_fast_path_id,
        intent_module=intent_module,
    )
    spine_ids = {item.candidate_id for item in schedule_spine}
    expanded_remainder = [
        item for item in expanded if item.candidate_id not in spine_ids
    ]

    def schedule_coordinate(
        item: CudaSearchCandidate,
    ) -> tuple[object, ...] | None:
        if item.lowering_path_kind == "row_reduction":
            return (
                item.schedule.get("block_threads"),
                item.schedule.get("tile_cols"),
                item.schedule.get("vector_width"),
            )
        if item.lowering_path_kind == "elementwise":
            return (
                item.schedule.get("block_threads"),
                item.schedule.get("flatten_contiguous"),
                item.schedule.get("vector_width"),
                item.schedule.get("grid_stride_unroll"),
                item.schedule.get("logical_tile_elements"),
            )
        return None

    def same_path_lowering_choice(
        item: CudaSearchCandidate,
    ) -> tuple[object, ...] | None:
        if (
            item.fast_path_id != canonical_fast_path_id
            or item.b_transformations
            or schedule_coordinate(item) is None
        ):
            return None
        if item.lowering_path_kind == "row_reduction":
            # Topology-changing states own different schedule domains and stay
            # in the ordinary lowering-path order below.  Here we cross the
            # same workgroup schedules with A-visible numeric evaluation and
            # intermediate-materialization choices.
            if (
                int(item.schedule.get("rows_per_block", 1)) != 1
                or int(item.schedule.get("reduction_parts", 1)) != 1
                or int(item.schedule.get("block_threads", 64)) == 32
            ):
                return None
            choice = (
                bool(item.schedule.get("compact_reduction_topology", True)),
                bool(item.schedule.get("fuse_fma", False)),
                bool(item.schedule.get("approximate_div", False)),
                bool(item.schedule.get("leader_finalize", False)),
                item.schedule.get("materialize_shared_row_output", True) is False,
                bool(item.schedule.get("raw_moment_variance", False)),
                int(item.schedule.get("accumulator_chains", 1)) - 1,
            )
            return ("row_reduction", *choice) if any(choice) else None
        if item.lowering_path_kind == "elementwise":
            paired_head_reuse = bool(
                item.schedule.get("paired_head_reuse", False)
            )
            paired_direct_plane = bool(
                item.schedule.get("paired_direct_plane", False)
            )
            if paired_head_reuse or paired_direct_plane:
                return (
                    "elementwise",
                    "paired_rotation",
                    paired_head_reuse,
                    paired_direct_plane,
                )
            approximation = str(
                item.schedule.get("tanh_approximation", "stable-ex2")
            )
            if approximation != "stable-ex2":
                return ("elementwise", approximation)
        return None

    # A finite IntentIR search must exercise each admitted same-path lowering
    # choice at more than one schedule without crowding out other paths.  Take
    # the first three distinct thread counts in the C-ranked Schedule-Only
    # order and cross both individual changes and their interactions with all
    # three.  A's opportunity gates make materialization mutually exclusive
    # with multi-reduction finalization, so at most three independent axes are
    # active here and the complete three-schedule cross fits budget 32 after the
    # fixed Schedule-Only spine.  Remaining lowering paths retain their ordinary
    # path-fair order afterwards.
    by_lowering_choice: dict[
        tuple[object, ...], list[CudaSearchCandidate]
    ] = {}
    for item in expanded_remainder:
        choice = same_path_lowering_choice(item)
        if choice is not None:
            by_lowering_choice.setdefault(choice, []).append(item)

    single_choices = [
        choice
        for choice in by_lowering_choice
        if choice[0] != "row_reduction"
        or sum(bool(item) for item in choice[1:]) == 1
    ]
    interaction_choices = [
        choice
        for choice in by_lowering_choice
        if choice[0] == "row_reduction"
        and sum(bool(item) for item in choice[1:]) > 1
    ]
    representative_spine: list[CudaSearchCandidate] = []
    observed_thread_counts: set[object] = set()
    for item in schedule_spine:
        block_threads = item.schedule.get("block_threads")
        if block_threads in observed_thread_counts:
            continue
        observed_thread_counts.add(block_threads)
        representative_spine.append(item)
        if len(representative_spine) == 3:
            break

    prioritized: list[CudaSearchCandidate] = []
    prioritized_ids: set[str] = set()
    for spine_item in representative_spine:
        coordinate = schedule_coordinate(spine_item)
        if coordinate is None:
            continue
        choices = (*single_choices, *interaction_choices)
        for choice in choices:
            candidates = by_lowering_choice[choice]
            match = next(
                (
                    item
                    for item in candidates
                    if schedule_coordinate(item) == coordinate
                    and item.candidate_id not in prioritized_ids
                ),
                None,
            )
            if match is not None:
                prioritized.append(match)
                prioritized_ids.add(match.candidate_id)

    ordered_expanded = (
        *prioritized,
        *(
            item
            for item in expanded_remainder
            if item.candidate_id not in prioritized_ids
        ),
    )

    return (
        *(
            replace(
                item,
                search_partition=SEARCH_PARTITION_SCHEDULE_SPINE,
            )
            for item in schedule_spine
        ),
        *(
            replace(item, search_partition=SEARCH_PARTITION_EXPANDED)
            for item in ordered_expanded
        ),
    )


def admit_cuda_candidate_paths(
    state: BackendSearchState,
    candidates: tuple[CudaSearchCandidate, ...],
    *,
    intent_module: str | None = None,
) -> tuple[tuple[CudaSearchCandidate, ...], tuple[Mapping[str, Any], ...]]:
    """Run source-blind structural probes before spending tuning budget.

    The optional module is a target-lowering view derived from the same neutral
    M0.  It is used only for the C++ probe; contract and candidate facts still
    come from ``state``.
    """

    grouped: dict[str, list[CudaSearchCandidate]] = {}
    generic_candidates: list[CudaSearchCandidate] = []
    for candidate in candidates:
        if candidate.generic:
            generic_candidates.append(candidate)
            continue
        assert candidate.fast_path_id is not None
        grouped.setdefault(candidate.fast_path_id, []).append(candidate)

    target = LoweringTarget(
        backend="cuda",
        chip=state.target.architecture,
        features=frozenset(state.target.features),
    )
    source = read_intent_module(intent_module or state.base_ir)
    has_ragged = backend_compile._has_ragged_ops(source)
    admitted_ids: set[str] = set()
    records: list[Mapping[str, Any]] = []
    generic_admitted = False
    if generic_candidates:
        if has_ragged:
            try:
                selection = select_variant(
                    source,
                    target=target,
                    only_fast_path=RAGGED_FAST_PATH_ID,
                )
            except Exception as error:
                records.append(
                    {
                        "path_id": RAGGED_FAST_PATH_ID,
                        "candidate_count": len(generic_candidates),
                        "status": "probe_failed",
                        "admitted": False,
                        "reason": f"{type(error).__name__}: {error}",
                        "probe": {
                            "fast_path_id": RAGGED_FAST_PATH_ID,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    }
                )
            else:
                ragged_attempts = tuple(
                    attempt
                    for attempt in selection.attempts
                    if attempt.fast_path_id == RAGGED_FAST_PATH_ID
                )
                if len(ragged_attempts) != 1:
                    raise CudaContractError(
                        "Ragged selection did not return exactly one "
                        "descriptor-aware probe attempt"
                    )
                probe = ragged_attempts[0]
                generic_admitted = probe.direct_hit
                records.append(
                    {
                        "path_id": RAGGED_FAST_PATH_ID,
                        "candidate_count": len(generic_candidates),
                        "status": "admitted" if probe.direct_hit else probe.status,
                        "admitted": probe.direct_hit,
                        "reason": probe.reason,
                        "probe": selection.to_dict(),
                    }
                )
        else:
            generic_admitted = True
            records.append(
                {
                    "path_id": "cuda.pipeline.linalg-materialized.v1",
                    "candidate_count": len(generic_candidates),
                    "status": "admitted_by_registered_generic_pipeline",
                    "admitted": True,
                    "probe": None,
                }
            )
    for fast_path_id, path_candidates in grouped.items():
        if fast_path_id == LINALG_FUSED_FAST_PATH_ID:
            admitted_ids.add(fast_path_id)
            records.append(
                {
                    "path_id": fast_path_id,
                    "candidate_count": len(path_candidates),
                    "status": "deferred_to_exact_upstream_mlir_pipeline",
                    "admitted": True,
                    "probe": None,
                }
            )
            continue
        try:
            selection = select_variant(
                source,
                target=target,
                only_fast_path=fast_path_id,
            )
        except Exception as error:
            records.append(
                {
                    "path_id": fast_path_id,
                    "candidate_count": len(path_candidates),
                    "status": "probe_failed",
                    "admitted": False,
                    "reason": f"{type(error).__name__}: {error}",
                    "probe": None,
                }
            )
            continue
        admitted = (
            selection.pipeline_kind == "specialized"
            and selection.fast_path_id == fast_path_id
        )
        if admitted:
            admitted_ids.add(fast_path_id)
        records.append(
            {
                "path_id": fast_path_id,
                "candidate_count": len(path_candidates),
                "status": "admitted" if admitted else "not_matched",
                "admitted": admitted,
                "probe": selection.to_dict(),
            }
        )

    admitted_candidates = tuple(
        candidate
        for candidate in candidates
        if (candidate.generic and generic_admitted)
        or candidate.fast_path_id in admitted_ids
    )
    return admitted_candidates, tuple(records)


def compile_cuda_candidate(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
    *,
    intent_module: str | None = None,
) -> LoweredProgram:
    """Compile one path-admitted request through its registered MLIR pipeline.

    Search, portable export, and any future offline compiler must share this
    dispatcher.  Keeping the generic, upstream-fused, and direct C++ paths in
    one place prevents a transport tool from silently treating a requested
    path as an unknown direct pass or from counting a different compiler state.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA candidate compilation requires BackendSearchState")
    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA candidate compilation requires CudaSearchCandidate")
    if state.target.family != "cuda":
        raise CudaContractError("CUDA compiler requires a cuda BackendTarget")
    source = intent_module or state.base_ir
    target = LoweringTarget(
        backend="cuda",
        chip=state.target.architecture,
        features=frozenset(state.target.features),
    )
    if candidate.generic:
        return lower_to_cuda(
            source,
            target=target,
            allow_specialized=False,
        )
    if candidate.upstream_fused:
        tile = tuple(
            int(item)
            for item in str(candidate.schedule["tile_shape"]).split("x")
        )
        return lower_to_cuda_fused(
            source,
            tile=tile,
            target=target,
            # A reduction barrier can legitimately leave two or more fused
            # stages.  That compiler-emitted graph is the production canonical
            # CF/E2 structure; requiring one launch would silently exclude the
            # very materialization boundary that Full is intended to remove.
            require_single_kernel=False,
        )
    assert candidate.fast_path_id is not None
    assert candidate.pass_argument is not None
    return lower_to_cuda(
        source,
        target=target,
        only_fast_path=candidate.fast_path_id,
        pass_overrides={candidate.fast_path_id: candidate.pass_argument},
        require_specialized=True,
    )


def compile_cuda_target_plugin(
    state: BackendSearchState,
    candidate: CudaSearchCandidate,
) -> LoweredProgram:
    """Compile one candidate through the target Layer-C production boundary.

    Search may carry a legacy Source-C prior solely for ordering.  The actual
    target compilation always rebuilds a neutral M0 state, constructs a closed
    CUDA plugin from that state, and lets the public J-facing lowering validate
    the plugin before dispatching to the registered compiler path.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("CUDA target compilation requires BackendSearchState")
    if not isinstance(candidate, CudaSearchCandidate):
        raise TypeError("CUDA target compilation requires CudaSearchCandidate")
    neutral_state = BackendSearchState(
        base=state.base,
        target=state.target,
        priors=(),
        ignored_slots=(),
    )
    artifact = build_cuda_target_artifact(neutral_state, candidate)
    from intentir.backend.protocol import LoweringRequest
    from .program import lower_j_to_cuda

    lowering = lower_j_to_cuda(
        LoweringRequest(
            j=neutral_state.base.j,
            target=neutral_state.target,
            variant="generic" if candidate.generic else "specialized",
            c=artifact,
            constexpr_bindings=neutral_state.base.constexpr_bindings,
        ),
        neutral_state.base.concrete_extents_map,
    )
    return lowering.artifact


def search_cuda(
    state: BackendSearchState,
    *,
    benchmark: Benchmark | None = None,
    benchmark_key: str | None = None,
    budget: int = 96,
    include_generic: bool = True,
    finalist_count: int = 16,
    finalist_remeasurements: int = 2,
    enable_b_transformations: bool = True,
    axes: SearchAxes | None = None,
    fixed_lowering_path_kind: str | None = None,
    fixed_fast_path_id: str | None = None,
    qd_controller: SearchAndCloseController | None = None,
    require_backend_closure: bool = False,
) -> CudaSearchResult:
    """Compile and optionally measure a bounded, already-legal candidate set."""

    if budget <= 0:
        raise ValueError("CUDA search budget must be positive")
    if (benchmark is None) != (benchmark_key is None):
        raise ValueError("benchmark and benchmark_key must be provided together")
    if finalist_count <= 0:
        raise ValueError("CUDA search finalist_count must be positive")
    if finalist_remeasurements < 0:
        raise ValueError("CUDA search finalist_remeasurements cannot be negative")
    if axes is not None and budget != axes.candidate_budget:
        raise ValueError("CUDA search budget differs from configured ablation arm")
    if axes is not None:
        enable_b_transformations = axes.enable_local_b_transformations
    if require_backend_closure and qd_controller is None:
        raise ValueError("formal backend closure requires SearchAndClose")
    started = time.perf_counter_ns()
    contract_view = derive_cuda_contract_view(state)
    if not include_generic and backend_compile._has_ragged_ops(
        read_intent_module(state.base_ir)
    ):
        raise CudaContractError(
            "CUDA Ragged lowering requires the generic descriptor-aware "
            "candidate; include_generic=False cannot close this graph"
        )
    if (
        axes is not None
        and not axes.search_structural_states
        and fixed_lowering_path_kind is None
        and fixed_fast_path_id is None
    ):
        fixed_lowering_path_kind, fixed_fast_path_id = (
            schedule_only_cuda_path_identity(state)
        )
        include_generic = fixed_lowering_path_kind == "generic"
    planned = enumerate_cuda_candidates(
        state,
        enable_b_transformations=enable_b_transformations,
        axes=axes,
        fixed_lowering_path_kind=fixed_lowering_path_kind,
        fixed_fast_path_id=fixed_fast_path_id,
    )
    candidate_universe = tuple(
        candidate
        for candidate in planned
        if include_generic or not candidate.generic
    )
    admitted_candidates, hard_admission = admit_cuda_candidate_paths(
        state,
        tuple(candidate_universe),
    )
    # C-Fixed is not "the first candidate under a clever C-aware order".  It is
    # one mechanical projection inside the same canonical-path schedule space
    # used by Schedule-Only.  Expanded searches carry that spine as an explicit
    # partition; fixed-path searches are already the spine.
    schedule_candidates = tuple(
        item
        for item in admitted_candidates
        if (
            item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
            if axes is not None and axes.search_structural_states
            else True
        )
    )
    c_fixed_candidate: CudaSearchCandidate | None = None
    if schedule_candidates:
        c_fixed_candidate, _c_fixed_projection = project_cuda_c_fixed_candidate(
            state,
            schedule_candidates,
        )
    c_fixed_candidate_id = (
        c_fixed_candidate.candidate_id if c_fixed_candidate is not None else None
    )

    # Budget counts the explicit, directly readable candidate list.  The CF arm
    # contains exactly the projected state; it does not enumerate then truncate
    # a target-aware order at budget one.
    candidates = list(admitted_candidates)
    if axes is not None and axes.arm is ExperimentArm.C_FIXED:
        if c_fixed_candidate is None:
            raise CudaContractError("C-Fixed has no legal canonical-path projection")
        candidates = [c_fixed_candidate]
    schedule_spine_candidate_count = sum(
        item.search_partition == SEARCH_PARTITION_SCHEDULE_SPINE
        for item in candidates
    )
    expanded_candidate_count = sum(
        item.search_partition == SEARCH_PARTITION_EXPANDED
        for item in candidates
    )

    attempts: list[SearchAttempt] = []
    winner: CudaSearchCandidate | None = None
    winner_program: LoweredProgram | None = None
    winner_latency: float | None = None
    time_to_best: float | None = None
    measured_programs: dict[str, LoweredProgram] = {}
    concrete_states: dict[str, ConcreteLoweringState] = {}
    budget_consumed = 0

    for candidate in candidates:
        if budget_consumed >= budget:
            break
        compile_started = time.perf_counter_ns()
        try:
            program = compile_cuda_target_plugin(state, candidate)
        except ValueError as error:
            budget_consumed += 1
            attempts.append(
                SearchAttempt(
                    candidate,
                    "rejected",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        except Exception as error:  # compiler diagnostics belong in the attempt
            budget_consumed += 1
            attempts.append(
                SearchAttempt(
                    candidate,
                    "compile_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        concrete_state = candidate.concrete_state(program)
        budget_consumed += 1
        concrete_states[candidate.candidate_id] = concrete_state

        if (
            benchmark is not None
            and concrete_state.observation_class
            is ObservationClass.OBSERVATION_CHANGING
            and qd_controller is None
        ):
            attempts.append(
                SearchAttempt(
                    candidate,
                    "missing_qd",
                    "observation-changing state requires target-to-RefJ Q/D "
                    "before performance measurement",
                    compile_ms=compile_ms,
                    lowering_record=program.to_record(),
                    concrete_state=concrete_state,
                )
            )
            continue

        qd_search: CandidateQDResult | None = None
        if qd_controller is not None:
            qd_search = qd_controller.filter_candidate(program, concrete_state)
            if not qd_search.passed:
                attempts.append(
                    SearchAttempt(
                        candidate,
                        "qd_rejected",
                        "direct target-to-RefJ comparison failed on D_search",
                        compile_ms=compile_ms,
                        lowering_record=program.to_record(),
                        concrete_state=concrete_state,
                        qd_search=qd_search,
                    )
                )
                continue

        latency: float | None = None
        status = "compiled"
        reason = (
            "complete Layer-B audit, C++ whole-graph admission, and cubin "
            "compilation passed"
        )
        if benchmark is not None:
            assert benchmark_key is not None
            try:
                latency = float(benchmark(program, candidate))
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError("benchmark returned a non-positive latency")
            except Exception as error:
                attempts.append(
                    SearchAttempt(
                        candidate,
                        "benchmark_failed",
                        f"{type(error).__name__}: {error}",
                        compile_ms=compile_ms,
                        lowering_record=program.to_record(),
                        concrete_state=concrete_state,
                        qd_search=qd_search,
                    )
                )
                continue
            status = "measured"
            reason = "performance-only selection over admitted candidate"
            measured_programs[candidate.candidate_id] = program

        attempts.append(
            SearchAttempt(
                candidate,
                status,
                reason,
                compile_ms=compile_ms,
                latency_us=latency,
                lowering_record=program.to_record(),
                concrete_state=concrete_state,
                qd_search=qd_search,
            )
        )
        if latency is not None and (
            winner_latency is None or latency < winner_latency
        ):
            winner = candidate
            winner_program = program
            winner_latency = latency
            time_to_best = (
                time.perf_counter_ns() - started
            ) / 1_000_000.0

    finalist_actual_count = 0
    if benchmark is not None and finalist_remeasurements:
        initial_ranked = sorted(
            (
                (index, attempt)
                for index, attempt in enumerate(attempts)
                if attempt.latency_us is not None
            ),
            key=lambda item: item[1].latency_us,
        )

        def provisional_best_index(
            predicate: Callable[[SearchAttempt], bool],
        ) -> int | None:
            eligible = [
                (index, attempt)
                for index, attempt in initial_ranked
                if predicate(attempt)
            ]
            return eligible[0][0] if eligible else None

        # Preserve the causal contrasts even when a small finalist_count would
        # otherwise exclude a slow C projection or one side of the nested
        # Schedule-Only/IntentIR comparison.
        mandatory_indices = [
            index
            for index in (
                provisional_best_index(
                    lambda item: item.candidate.candidate_id
                    == c_fixed_candidate_id
                ),
                provisional_best_index(
                    lambda item: item.candidate.search_partition
                    == SEARCH_PARTITION_SCHEDULE_SPINE
                ),
                provisional_best_index(
                    lambda item: item.candidate.search_partition
                    == SEARCH_PARTITION_EXPANDED
                ),
            )
            if index is not None
        ]
        ranked_indices: list[int] = []
        for index in (
            *mandatory_indices,
            *(index for index, _attempt in initial_ranked),
        ):
            if index not in ranked_indices:
                ranked_indices.append(index)
            if (
                len(ranked_indices) >= finalist_count
                and all(item in ranked_indices for item in mandatory_indices)
            ):
                break
        ranked = [(index, attempts[index]) for index in ranked_indices]
        fresh_samples: dict[int, list[float]] = {
            index: [] for index, _attempt in ranked
        }
        finalist_errors: dict[int, str] = {}
        for round_index in range(finalist_remeasurements):
            round_order = ranked if round_index % 2 == 0 else list(reversed(ranked))
            for index, attempt in round_order:
                if index in finalist_errors:
                    continue
                candidate = attempt.candidate
                program = measured_programs[candidate.candidate_id]
                try:
                    latency = float(benchmark(program, candidate))
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError(
                            "finalist benchmark returned a non-positive latency"
                        )
                    fresh_samples[index].append(latency)
                except Exception as error:
                    finalist_errors[index] = f"{type(error).__name__}: {error}"

        for index, attempt in ranked:
            candidate = attempt.candidate
            if index in finalist_errors:
                attempts[index] = replace(
                    attempt,
                    reason=(
                        attempt.reason
                        + "; finalist remeasurement failed: "
                        + finalist_errors[index]
                    ),
                )
                continue
            samples = fresh_samples[index]
            if not samples:
                continue
            robust_latency = float(statistics.median(samples))
            attempts[index] = replace(
                attempt,
                latency_us=robust_latency,
                reason=(
                    "performance-only selection over admitted candidate; "
                    f"finalist median of {len(samples)} fresh round-robin "
                    "measurements"
                ),
                finalist_samples_us=tuple(samples),
            )

        freshly_remeasured = [
            attempts[index]
            for index, _attempt in ranked
            if attempts[index].finalist_samples_us
        ]
        finalist_actual_count = len(freshly_remeasured)
        if freshly_remeasured:
            robust_winner = min(
                freshly_remeasured,
                key=lambda item: item.latency_us,
            )
            winner = robust_winner.candidate
            winner_latency = robust_winner.latency_us
            winner_program = measured_programs[winner.candidate_id]

    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    winner_state = (
        concrete_states.get(winner.candidate_id)
        if winner is not None
        else None
    )
    backend_closure: BackendClosureResult | None = None
    backend_closure_error: str | None = None
    if (
        qd_controller is not None
        and winner_program is not None
        and winner_state is not None
    ):
        try:
            backend_closure = qd_controller.close_winner(
                winner_program,
                winner_state,
            )
            if not backend_closure.passed:
                backend_closure_error = (
                    "fixed winner failed a direct edge on held-out D_cert"
                )
        except BackendClosureError as error:
            backend_closure_error = f"{type(error).__name__}: {error}"
    if backend_closure is not None and not backend_closure.passed:
        raise ValueError(
            "fixed CUDA winner failed held-out D_cert; D_cert is not reused "
            "to select a replacement candidate"
        )
    if require_backend_closure and (
        backend_closure is None or not backend_closure.passed
    ):
        raise ValueError(
            "fixed CUDA winner lacks backend closure: "
            + (backend_closure_error or "no executable winner")
        )
    return CudaSearchResult(
        state=state,
        contract_view=contract_view,
        attempts=tuple(attempts),
        winner=winner,
        winner_program=winner_program,
        winner_latency_us=winner_latency,
        benchmark_key=benchmark_key,
        elapsed_ms=elapsed,
        time_to_best_ms=time_to_best,
        finalist_count=finalist_count if benchmark is not None else 0,
        finalist_remeasurements=(
            finalist_remeasurements if benchmark is not None else 0
        ),
        b_transformations_enabled=enable_b_transformations,
        winner_state=winner_state,
        backend_closure=backend_closure,
        backend_closure_error=backend_closure_error,
        ablation_axes=axes,
        fixed_lowering_path_kind=fixed_lowering_path_kind,
        fixed_fast_path_id=fixed_fast_path_id,
        hard_admission=hard_admission,
        candidate_universe_count=len(candidate_universe),
        candidate_budget=budget,
        budget_consumed=budget_consumed,
        c_fixed_candidate_id=c_fixed_candidate_id,
        finalist_actual_count=finalist_actual_count,
        schedule_spine_candidate_count=schedule_spine_candidate_count,
        expanded_candidate_count=expanded_candidate_count,
    )


__all__ = [
    "B_CAP_RANK0_HOST_CAPTURE",
    "B_CAP_READONLY_INPUTS",
    "B_TRANSFORM_EPILOGUE_LOAD_MOTION",
    "B_TRANSFORM_RANK0_HOST_CAPTURE",
    "B_TRANSFORM_UNIFORM_HOST_CAPTURE",
    "CONTRACT_VIEW_VERSION",
    "CUDA_LAYER_C_BACKEND_KIND",
    "LINALG_FUSED_FAST_PATH_ID",
    "SEARCH_PROTOCOL",
    "CudaContractError",
    "CudaContractView",
    "CudaSearchCandidate",
    "CudaSearchResult",
    "SEARCH_PARTITION_EXPANDED",
    "SEARCH_PARTITION_SCHEDULE_SPINE",
    "SEARCH_PARTITION_UNPARTITIONED",
    "SearchAttempt",
    "admit_cuda_candidate_paths",
    "build_cuda_target_plugin",
    "build_cuda_target_artifact",
    "canonical_cuda_candidate",
    "compile_cuda_candidate",
    "compile_cuda_target_plugin",
    "cuda_allowed_authorizations",
    "cuda_candidate_digest",
    "cuda_candidate_key",
    "cuda_layer_c_interface",
    "cuda_plugin_loop_schedule_tile",
    "enumerate_cuda_candidates",
    "derive_cuda_contract_view",
    "derive_default_fast_path_id",
    "derive_default_lowering_path_kind",
    "project_cuda_c_fixed_candidate",
    "resolve_cuda_candidate_key",
    "resolve_cuda_target_plugin",
    "schedule_only_cuda_path_identity",
    "search_cuda",
]
