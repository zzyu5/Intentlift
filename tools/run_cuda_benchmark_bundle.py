#!/usr/bin/env python3
"""Execute and search a precompiled production CUDA bundle on its target GPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from intentir.runtime_dtype import torch_tensor_from_numpy, torch_tensor_to_numpy
from intentir.backend.cuda.benchmark import (
    ExtractedGraphLaunch,
    ExtractedSingleLaunch,
    PortableExtractedLaunch,
    PreparedGraphLaunch,
    PreparedSingleLaunch,
    do_bench_cudagraph_us,
    do_bench_us,
    portable_launch_from_dict,
)
from intentir.backend.cuda.tilelang_source import PreparedTileLangInvocation
from intentir.backend.cuda.portable_bundle import (
    cuda_compute_capability_for_chip,
    decode_triton_spec,
    decode_values,
    logical_runtime_array,
)
from intentir.backend.portable_workload import decode_workload_recipe
from intentir.verify.gate1.model import (
    ResultRelationKind,
    ResultRelationSpec,
    SemanticArchive,
)
from intentir.backend.roofline import (
    RooflineCeilings,
    evaluate_logical_roofline,
)
from intentir.verify.gate1.compiler import KernelSource
from intentir.ir.value_dimension import resolve_runtime_output_shapes
from intentir.ir.canonical_j import j_elaboration
from intentir.verify.gate2.compare import compare_outputs, validate_j_inputs
from intentir.verify.gate2.model import OutputAgreement, PairwiseAgreement
from intentir.verify.gate2.policy import ComparisonPolicy, DEFAULT_GATE2_POLICY
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec
from intentir.verify.gate2.triton import PreparedTritonInvocation


REPORT_PROTOCOL = "intentir.production.cuda-remote-performance.v4"
BUNDLE_PROTOCOL = "intentir.production.cuda-search-bundle.v3"
SUPPORTED_BUNDLE_PROTOCOLS = frozenset(
    {BUNDLE_PROTOCOL, "intentir.production.cuda-search-bundle.v2"}
)


class KernelFailure(RuntimeError):
    """One kernel failed after producing useful candidate diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "bool": "bool",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    return getattr(torch, aliases[dtype.removeprefix("torch.").lower()])


def _device_record(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "l2_cache_bytes": int(getattr(properties, "L2_cache_size", 0)),
    }


def _poison_cuda_outputs(
    torch: Any,
    outputs: list[Any],
    *,
    variant: int,
    protected: list[Any] | None = None,
) -> None:
    """Overwrite every observable output before a correctness launch.

    Candidate output buffers are intentionally reused across a large search.
    Without an explicit poison, a candidate that omits some or all stores can
    inherit a prior candidate's correct bytes and falsely pass Q.  Two distinct
    variants make accidental agreement with a legitimate constant output
    insufficient: the candidate must overwrite the complete observable result
    on both launches.
    """

    protected_storages = {
        item.untyped_storage().data_ptr() for item in (protected or [])
    }
    for index, output in enumerate(outputs):
        if output.untyped_storage().data_ptr() in protected_storages:
            continue
        salt = variant * 17 + index * 3
        if output.dtype == torch.bool:
            output.fill_(bool((salt + 1) & 1))
        elif output.dtype.is_floating_point:
            output.fill_(float(91 + salt))
        else:
            info = torch.iinfo(output.dtype)
            value = info.max - min(salt, max(0, info.max - info.min))
            output.fill_(value)


def _select_full_record(
    expanded: Mapping[str, Any] | None,
    schedule_spine: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Select Full over the measured structural and Schedule-Only union."""

    records = [item for item in (expanded, schedule_spine) if item is not None]
    return min(records, key=lambda item: item["latency_us"]) if records else None


def _measure_cold_data_us(
    torch: Any,
    thunk: Any,
    *,
    eviction: Any,
    iterations: int,
    prepare: Any | None = None,
) -> float:
    """Measure single launches after a target-side L2 eviction.

    The eviction and synchronization are deliberately outside the CUDA-event
    interval.  This is a cold-*data* protocol, not process cold-start and not a
    claim about compulsory instruction-cache misses.
    """

    samples: list[float] = []
    for _ in range(iterations):
        if prepare is not None:
            prepare()
        eviction.add_(1.0)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        thunk()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) * 1000.0)
    return float(statistics.median(samples))


def _warmup_thunks(
    torch: Any,
    thunks: tuple[Any, ...],
    *,
    warmup_ms: int,
) -> int:
    """Stabilize kernel code and GPU clocks without adding timed samples.

    Cold-data measurements still evict L2 immediately before every timed launch.
    This controls process/JIT/P-state history, not the measured data-cache state.
    Multiple thunks are traversed round-robin so paired arms receive an equal
    number of untimed launches.
    """

    if warmup_ms <= 0 or not thunks:
        return 0
    deadline = time.perf_counter() + warmup_ms / 1000.0
    rounds = 0
    while True:
        for thunk in thunks:
            thunk()
        rounds += 1
        if rounds % 32 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() >= deadline:
                break
    torch.cuda.synchronize()
    return rounds


def _measure_cuda_roofs(
    torch: Any,
    *,
    memory_mib: int,
    matrix_extent: int,
) -> RooflineCeilings:
    if memory_mib <= 0 or matrix_extent <= 0:
        raise ValueError("Roofline calibration sizes must be positive")
    elements = memory_mib * 1024 * 1024 // 4
    source = torch.empty(elements, dtype=torch.float32, device="cuda").uniform_()
    destination = torch.empty_like(source)
    memory_samples = [
        do_bench_us(
            lambda: destination.copy_(source),
            warmup_ms=50,
            repeat_ms=200,
        )
        for _ in range(3)
    ]
    memory_us = float(statistics.median(memory_samples))
    memory_gbps = 2 * source.numel() * source.element_size() / memory_us / 1_000.0

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        left = torch.randn(
            (matrix_extent, matrix_extent), dtype=torch.float32, device="cuda"
        )
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        compute_samples = [
            do_bench_us(
                lambda: torch.mm(left, right, out=output),
                warmup_ms=50,
                repeat_ms=200,
            )
            for _ in range(3)
        ]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    compute_us = float(statistics.median(compute_samples))
    fp32_gflops = 2 * matrix_extent**3 / compute_us / 1_000.0
    torch.cuda.synchronize()
    return RooflineCeilings(
        memory_gbps=memory_gbps,
        fp32_gflops=fp32_gflops,
    )


def _actual_outputs(
    output_keys: list[str], outputs: list[Any]
) -> dict[str, np.ndarray]:
    import torch

    return {
        key: torch_tensor_to_numpy(torch, tensor)
        for key, tensor in zip(output_keys, outputs, strict=True)
    }


_GPU_Q_CHUNK_ELEMENTS = 1 << 25


def _gpu_close_q_output(
    torch: Any,
    lhs: Any,
    rhs: Any,
    rule: Any,
) -> tuple[bool, float]:
    """Apply the exact Gate2 ``Close_Q`` relation without copying tensors.

    The implementation deliberately mirrors ``compare.close_q`` and
    ``compare._max_abs_error``.  Chunking bounds temporary storage for natural
    LLM tensors; only boolean/error scalars cross back to the host.
    """

    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    exact_domain = lhs.dtype in {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    passed = True
    max_abs_error = 0.0
    for begin in range(0, lhs_flat.numel(), _GPU_Q_CHUNK_ELEMENTS):
        end = min(begin + _GPU_Q_CHUNK_ELEMENTS, lhs_flat.numel())
        left = lhs_flat[begin:end]
        right = rhs_flat[begin:end]
        if exact_domain:
            if not torch.equal(left, right):
                passed = False
            if left.numel():
                error = (
                    left.to(dtype=torch.int64) - right.to(dtype=torch.int64)
                ).abs().max()
                max_abs_error = max(max_abs_error, float(error.item()))
            continue

        lhs_nan = torch.isnan(left)
        rhs_nan = torch.isnan(right)
        if not torch.equal(lhs_nan, rhs_nan):
            return False, float("inf")
        has_nan = bool(torch.any(lhs_nan).item())
        if rule.exact:
            if not torch.equal(left, right):
                passed = False
        elif has_nan and not rule.tolerance.equal_nan:
            passed = False

        non_nan = ~(lhs_nan | rhs_nan)
        lhs_inf = torch.isinf(left) & non_nan
        rhs_inf = torch.isinf(right) & non_nan
        if not torch.equal(lhs_inf, rhs_inf):
            passed = False
        elif bool(torch.any(lhs_inf).item()) and not torch.equal(
            left[lhs_inf], right[rhs_inf]
        ):
            passed = False

        finite = torch.isfinite(left) & torch.isfinite(right)
        if not bool(torch.any(finite).item()):
            continue
        left_finite = left[finite].to(dtype=torch.float64)
        right_finite = right[finite].to(dtype=torch.float64)
        error = torch.abs(left_finite - right_finite)
        max_abs_error = max(max_abs_error, float(torch.max(error).item()))
        if not rule.exact:
            scale = torch.maximum(torch.abs(left_finite), torch.abs(right_finite))
            bound = rule.tolerance.atol + rule.tolerance.rtol * scale
            if not bool(torch.all(error <= bound).item()):
                passed = False
    return passed, max_abs_error


def _compare_cuda_outputs(
    torch: Any,
    *,
    name: str,
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_name: str,
    j: SemanticArchive,
    entry: Mapping[str, Any],
    policy: ComparisonPolicy,
) -> PairwiseAgreement:
    """Compare candidate outputs to the selected target-side reference."""

    output_bindings = j_elaboration(j).output_bindings
    expected_outputs = set(j_elaboration(j).output_names)
    if set(entry["output_bindings"]) != expected_outputs:
        raise ValueError("Gate2 output bindings differ from canonical J outputs")
    explicit = {item.output: item for item in j.boundary.result_relations}
    observations: list[OutputAgreement] = []
    for binding in output_bindings:
        runtime_key = entry["output_bindings"][binding.output]
        relation = explicit.get(
            binding.output,
            ResultRelationSpec(binding.output, ResultRelationKind.EXACT),
        )
        rule = policy.rule_for(relation.kind, binding.output_type.dtype)
        tolerance = rule.tolerance
        if runtime_key not in actual or runtime_key not in reference:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    "output missing from CUDA candidate or canonical J reference",
                    relation.kind.value,
                    True,
                    policy.policy_id,
                    float(tolerance.rtol) if tolerance is not None else None,
                    float(tolerance.atol) if tolerance is not None else None,
                    tolerance.equal_nan if tolerance is not None else None,
                )
            )
            continue
        lhs = actual[runtime_key]
        rhs = reference[runtime_key]
        expected_dtype = _torch_dtype(torch, binding.output_type.dtype)
        if lhs.dtype != expected_dtype or rhs.dtype != expected_dtype:
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    "output dtype differs from canonical J",
                    relation.kind.value,
                    True,
                    policy.policy_id,
                    float(tolerance.rtol) if tolerance is not None else None,
                    float(tolerance.atol) if tolerance is not None else None,
                    tolerance.equal_nan if tolerance is not None else None,
                )
            )
            continue
        if tuple(lhs.shape) != tuple(rhs.shape):
            observations.append(
                OutputAgreement(
                    runtime_key,
                    False,
                    None,
                    f"shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}",
                    relation.kind.value,
                    True,
                    policy.policy_id,
                )
            )
            continue
        passed, max_abs_error = _gpu_close_q_output(torch, lhs, rhs, rule)
        observations.append(
            OutputAgreement(
                runtime_key,
                passed,
                max_abs_error,
                "",
                relation.kind.value,
                True,
                policy.policy_id,
                float(tolerance.rtol) if tolerance is not None else None,
                float(tolerance.atol) if tolerance is not None else None,
                tolerance.equal_nan if tolerance is not None else None,
            )
        )
    return PairwiseAgreement(
        name,
        reference_name,
        all(item.passed for item in observations),
        tuple(observations),
        policy.policy_id,
    )


def _compare(
    *,
    name: str,
    actual: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    j: SemanticArchive,
    entry: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    policy: ComparisonPolicy,
    reference_name: str,
):
    return compare_outputs(
        name,
        actual,
        reference_name,
        reference,
        j=j,
        input_bindings=entry["input_bindings"],
        output_bindings=entry["output_bindings"],
        inputs=values,
        policy=policy,
    )


def _one_kernel(
    entry: Mapping[str, Any],
    *,
    timing: str,
    graph_repetitions: int,
    warmup_ms: int,
    repeat_ms: int,
    measurement_rounds: int,
    finalist_count: int,
    finalist_remeasurements: int,
    confirmation_batches: int,
    roofline_ceilings: RooflineCeilings | None,
    data_regime: str,
    cold_iterations: int,
    cache_eviction: Any | None,
) -> dict[str, Any]:
    import torch

    j = SemanticArchive.from_json(entry["j"])
    canonical_outputs = j_elaboration(j).output_bindings
    if not canonical_outputs:
        raise ValueError("accepted J has no canonical output bindings")
    policy = ComparisonPolicy.from_json(
        entry.get("comparison_policy", DEFAULT_GATE2_POLICY.to_json())
    )
    policy.validate_j(j)
    baseline_policy = str(entry.get("baseline_policy", "source_and_refj"))
    if baseline_policy not in {"source_and_refj", "target_refj_only"}:
        raise ValueError(f"unknown CUDA baseline policy {baseline_policy!r}")
    reference_name = "canonical-J-reference"
    canonical_j_reference_error: str | None = None
    if "workload_recipe" in entry:
        if "values" in entry or "reference_outputs" in entry:
            raise ValueError("CUDA bundle mixes rebuilt and embedded workloads")
        values = decode_workload_recipe(entry["workload_recipe"])
        reference_result = ScalarLoweringRunner().run(
            ScalarLoweringSpec.create(
                j,
                dims={name: int(extent) for name, extent in entry["extents"].items()},
                input_bindings=entry["input_bindings"],
                output_bindings=entry["output_bindings"],
            ),
            values,
        )
        if not reference_result.ok:
            canonical_j_reference_error = reference_result.error
            if baseline_policy != "source_and_refj":
                raise RuntimeError(
                    "target-side canonical J reference failed: "
                    f"{reference_result.error}"
                )
            if "source_triton" not in entry:
                raise RuntimeError(
                    "target-side canonical J reference failed before the bundled "
                    "TileLang tensor ABI was prepared: "
                    f"{reference_result.error}"
                )
            source_record = entry["source_triton"]
            source_program = KernelSource.from_code(
                source_record["code"],
                source_record["function"],
                trusted=True,
            )
            source_spec = decode_triton_spec(source_record["execution_spec"])
            with PreparedTritonInvocation(
                source_program, source_spec, values
            ) as source_reference:
                source_reference.reset_mutable_inputs()
                source_reference.launch()
                torch.cuda.synchronize()
                reference = dict(source_reference.copied_outputs())
            reference_name = "source-reference"
        else:
            reference = dict(reference_result.outputs)
    else:
        values = decode_values(entry["values"])
        reference = decode_values(entry["reference_outputs"])
    validate_j_inputs(
        j,
        input_bindings=entry["input_bindings"],
        values=values,
    )
    j_inputs = {item.name: item for item in j.intent.inputs}
    runtime_input_arrays = [
        logical_runtime_array(
            values[entry["input_bindings"][item.name]],
            rank=len(item.shape),
            name=item.name,
        )
        for item in j.intent.inputs
    ]
    runtime_input_arrays_by_name = {
        item.name: value
        for item, value in zip(j.intent.inputs, runtime_input_arrays, strict=True)
    }
    runtime_output_shapes = resolve_runtime_output_shapes(
        j.intent,
        [list(item.output_type.shape) for item in canonical_outputs],
        runtime_input_arrays,
        dims=entry["extents"],
        canonical_output_bindings=canonical_outputs,
    )
    specs_by_name = {
        str(item["name"]): item for item in entry["output_specs"]
    }
    runtime_output_specs: list[dict[str, Any]] = []
    for output, shape in zip(canonical_outputs, runtime_output_shapes, strict=True):
        raw_spec = specs_by_name.get(output.output)
        if raw_spec is None:
            raise ValueError(f"bundled output {output.output!r} is absent from its ABI")
        declared_shape = raw_spec.get("shape")
        if not isinstance(declared_shape, list) or len(declared_shape) != len(shape):
            raise ValueError(f"bundled output {output.output!r} has an invalid shape")
        if all(item != "?" for item in declared_shape):
            try:
                if tuple(int(item) for item in declared_shape) != shape:
                    raise ValueError(
                        f"bundled output {output.output!r} shape differs from the "
                        f"J-derived runtime shape: {declared_shape} != {list(shape)}"
                    )
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith("bundled output"):
                    raise
                raise ValueError(
                    f"bundled output {output.name!r} has a non-integer shape"
                ) from exc
        runtime_spec = dict(raw_spec)
        runtime_spec["shape"] = list(shape)
        runtime_output_specs.append(runtime_spec)
    entry = {**entry, "output_specs": runtime_output_specs}
    for output, spec in zip(canonical_outputs, runtime_output_specs, strict=True):
        output_key = entry["output_bindings"][output.output]
        if output_key not in reference:
            raise ValueError(f"canonical reference omitted output {output.output!r}")
        actual_shape = tuple(np.asarray(reference[output_key]).shape)
        if actual_shape != tuple(spec["shape"]):
            raise ValueError(
                f"canonical reference output shape for {output.output!r} differs "
                f"from the J-derived runtime shape: {actual_shape} != {spec['shape']}"
            )
    inputs = []
    for name in entry["input_order"]:
        if name not in j_inputs:
            raise ValueError(f"bundled runtime input {name!r} is absent from J")
        try:
            runtime_array = runtime_input_arrays_by_name[name]
        except KeyError as exc:
            raise ValueError(
                f"bundled runtime input {name!r} lacks its logical value"
            ) from exc
        logical = j_inputs[name]
        inputs.append(
            torch_tensor_from_numpy(
                torch,
                runtime_array,
                dtype=_torch_dtype(torch, logical.dtype),
                device="cuda",
            )
        )
    outputs = [
        torch.empty(
            tuple(item["shape"]),
            dtype=_torch_dtype(torch, item["dtype"]),
            device="cuda",
        )
        for item in entry["output_specs"]
    ]
    inout_refs = {
        parameter.logical_ref
        for parameter in j.signature.parameters
        if parameter.mode.value == "inout"
        and parameter.logical_ref is not None
    }
    input_index = {name: index for index, name in enumerate(entry["input_order"])}
    inplace_output_input_indices: list[int | None] = []
    for spec in entry["output_specs"]:
        logical_ref = str(spec["name"])
        if logical_ref in inout_refs:
            if logical_ref not in input_index:
                raise ValueError(
                    f"inout J output {logical_ref!r} lacks its runtime input"
                )
            inplace_output_input_indices.append(input_index[logical_ref])
        else:
            inplace_output_input_indices.append(None)
    single_launch_tensors = [
        *inputs,
        *(
            output
            for output, inplace_index in zip(
                outputs, inplace_output_input_indices, strict=True
            )
            if inplace_index is None
        ),
    ]
    single_launch_outputs = [
        inputs[inplace_index] if inplace_index is not None else output
        for output, inplace_index in zip(
            outputs, inplace_output_input_indices, strict=True
        )
    ]
    reference_cuda = {
        key: torch_tensor_from_numpy(
            torch,
            reference[key],
            dtype=_torch_dtype(torch, spec["dtype"]),
            device="cuda",
        )
        for key, spec in zip(
            entry["output_keys"], entry["output_specs"], strict=True
        )
    }
    tensors = [*inputs, *outputs]
    workload_cost = entry.get("workload_cost")
    logical_io_bytes = sum(item.numel() * item.element_size() for item in tensors)
    certified_io_bytes = (
        workload_cost.get("boundary_io_bytes")
        if workload_cost is not None
        else None
    )
    if (
        certified_io_bytes is not None
        and int(certified_io_bytes) != logical_io_bytes
    ):
        raise ValueError("CUDA tensors disagree with J workload-cost record")
    floating_point_ops = (
        int(workload_cost["floating_point_ops"])
        if workload_cost is not None
        else 0
    )

    pristine_inputs = [item.clone() for item in inputs]

    def reset_inputs() -> None:
        for destination, pristine in zip(inputs, pristine_inputs, strict=True):
            destination.copy_(pristine)

    def measure(thunk, *, prepare=None) -> float:
        samples = []
        for _ in range(measurement_rounds):
            if data_regime == "cold-data":
                if cache_eviction is None:
                    raise AssertionError("cold-data regime requires cache eviction")
                samples.append(
                    _measure_cold_data_us(
                        torch,
                        thunk,
                        eviction=cache_eviction,
                        iterations=cold_iterations,
                        prepare=prepare,
                    )
                )
            elif timing == "graph":
                if prepare is not None:
                    prepare()
                samples.append(
                    do_bench_cudagraph_us(
                        thunk, repetitions=graph_repetitions
                    )
                )
            else:
                if prepare is not None:
                    prepare()
                samples.append(
                    do_bench_us(
                        thunk,
                        warmup_ms=warmup_ms,
                        repeat_ms=repeat_ms,
                    )
                )
        return float(statistics.median(samples))

    candidate_records: list[dict[str, Any]] = []
    candidate_artifacts: dict[
        str, tuple[Mapping[str, Any], PortableExtractedLaunch, Any, list[Any]]
    ] = {}
    # Different requested schedules can collapse to the exact same executable
    # after the C++/MLIR pipeline.  Measure that executable once, then attach the
    # same observation to every causal-arm candidate that produced it.  This is
    # exact target-code equality (cubin + launch/host graph), not a performance
    # cache or an artifact hash, and it prevents a large requested space from
    # overweighting duplicate code in the finalist protocol.
    execution_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    execution_signature_by_id: dict[str, tuple[Any, ...]] = {}

    def execution_signature(
        extracted: PortableExtractedLaunch,
    ) -> tuple[Any, ...]:
        if isinstance(extracted, ExtractedSingleLaunch):
            record = extracted.record
            return (
                "single",
                extracted.cubin,
                extracted.kernel_name,
                extracted.binary_name,
                extracted.plan,
                record.grid,
                record.block,
                record.dynamic_smem_bytes,
            )
        if isinstance(extracted, ExtractedGraphLaunch):
            return (
                "graph",
                tuple(
                    (item.cubin, item.kernel_name, item.binary_name)
                    for item in extracted.kernels
                ),
                extracted.source_mlir,
                extracted.binary_mlir,
                extracted.record,
            )
        raise TypeError("unsupported portable CUDA launch record")

    for candidate in entry["candidates"]:
        try:
            extracted = portable_launch_from_dict(candidate["launch"])
            candidate_id = candidate["candidate"]["candidate_id"]
            signature = execution_signature(extracted)
            execution_signature_by_id[candidate_id] = signature
            cached = execution_cache.get(signature)
            if cached is not None:
                alias_record = {
                    "candidate": candidate["candidate"],
                    "status": cached["status"],
                    "correctness": cached["correctness"],
                    "concrete_state": candidate.get("concrete_state"),
                    "lowering": candidate.get("lowering"),
                    "execution_equivalent_to": cached["candidate_id"],
                }
                if cached["status"] == "measured":
                    alias_record.update(
                        {
                            "latency_us": cached["latency_us"],
                            "launch": extracted.record.to_dict(),
                            "binary_size_bytes": extracted.binary_size_bytes,
                            "cubin_size_bytes": (
                                len(extracted.cubin)
                                if isinstance(extracted, ExtractedSingleLaunch)
                                else None
                            ),
                            "logical_boundary_gbps": (
                                logical_io_bytes / cached["latency_us"] / 1_000.0
                            ),
                            "achieved_gflops": (
                                floating_point_ops / cached["latency_us"] / 1_000.0
                            ),
                            "cold_pre_measure_warmup_rounds": cached[
                                "cold_pre_measure_warmup_rounds"
                            ],
                        }
                    )
                    candidate_artifacts[candidate_id] = (
                        candidate,
                        extracted,
                        cached["thunk"],
                        cached["outputs"],
                    )
                candidate_records.append(alias_record)
                continue
            if isinstance(extracted, ExtractedSingleLaunch):
                prepared = PreparedSingleLaunch.from_extracted(extracted)
                thunk = prepared.thunk(single_launch_tensors)
                candidate_outputs = single_launch_outputs
            elif isinstance(extracted, ExtractedGraphLaunch):
                prepared = PreparedGraphLaunch.from_extracted(extracted, inputs)
                thunk = prepared.thunk()
                candidate_outputs = list(prepared.outputs.values())
                if len(candidate_outputs) != len(entry["output_keys"]):
                    raise ValueError(
                        "prepared graph output count differs from bundled J results"
                    )
            else:
                raise TypeError("unsupported portable CUDA launch record")
            agreements = []
            for poison_variant in (0, 1):
                reset_inputs()
                _poison_cuda_outputs(
                    torch,
                    candidate_outputs,
                    variant=poison_variant,
                    protected=inputs,
                )
                thunk()
                torch.cuda.synchronize()
                cuda_actual = dict(
                    zip(entry["output_keys"], candidate_outputs, strict=True)
                )
                agreements.append(
                    _compare_cuda_outputs(
                        torch,
                        name=f"mlir-cubin-canary-{poison_variant}",
                        actual=cuda_actual,
                        reference=reference_cuda,
                        reference_name=reference_name,
                        j=j,
                        entry=entry,
                        policy=policy,
                    )
                )
            agreement = agreements[-1]
            if not all(item.passed for item in agreements):
                correctness = {
                    "passed": False,
                    "canary_runs": [item.to_json() for item in agreements],
                }
                candidate_records.append(
                    {
                        "candidate": candidate["candidate"],
                        "status": "incorrect",
                        "correctness": correctness,
                        "concrete_state": candidate.get("concrete_state"),
                        "lowering": candidate.get("lowering"),
                    }
                )
                execution_cache[signature] = {
                    "candidate_id": candidate_id,
                    "status": "incorrect",
                    "correctness": correctness,
                }
                continue
            cold_warmup_rounds = (
                _warmup_thunks(torch, (thunk,), warmup_ms=warmup_ms)
                if data_regime == "cold-data"
                else 0
            )
            latency = measure(thunk, prepare=reset_inputs)
            candidate_records.append(
                {
                    "candidate": candidate["candidate"],
                    "status": "measured",
                    "latency_us": latency,
                    "correctness": agreement.to_json(),
                    "concrete_state": candidate.get("concrete_state"),
                    "lowering": candidate.get("lowering"),
                    "launch": extracted.record.to_dict(),
                    "binary_size_bytes": extracted.binary_size_bytes,
                    "cubin_size_bytes": (
                        len(extracted.cubin)
                        if isinstance(extracted, ExtractedSingleLaunch)
                        else None
                    ),
                    "logical_boundary_gbps": logical_io_bytes / latency / 1_000.0,
                    "achieved_gflops": floating_point_ops / latency / 1_000.0,
                    "cold_pre_measure_warmup_rounds": cold_warmup_rounds,
                }
            )
            candidate_artifacts[candidate_id] = (
                candidate,
                extracted,
                thunk,
                candidate_outputs,
            )
            execution_cache[signature] = {
                "candidate_id": candidate_id,
                "status": "measured",
                "correctness": agreement.to_json(),
                "latency_us": latency,
                "cold_pre_measure_warmup_rounds": cold_warmup_rounds,
                "thunk": thunk,
                "outputs": candidate_outputs,
            }
        except Exception as exc:
            candidate_records.append(
                {
                    "candidate": candidate["candidate"],
                    "status": "runtime_or_transport_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "concrete_state": candidate.get("concrete_state"),
                    "lowering": candidate.get("lowering"),
                }
            )
    measured_indices = [
        index
        for index, record in enumerate(candidate_records)
        if record["status"] == "measured"
    ]
    measured_index_by_id = {
        candidate_records[index]["candidate"]["candidate_id"]: index
        for index in measured_indices
    }
    execution_group_indices: dict[tuple[Any, ...], list[int]] = {}
    for index in measured_indices:
        candidate_id = candidate_records[index]["candidate"]["candidate_id"]
        signature = execution_signature_by_id[candidate_id]
        execution_group_indices.setdefault(signature, []).append(index)
    search_orders = entry["search_orders"]
    default_order = list(measured_index_by_id)
    with_c_order = list(
        search_orders.get("with_c_concrete_order", default_order)
    )
    without_c_order = list(
        search_orders.get("without_c_concrete_order", default_order)
    )
    c_oracle_complete = bool(
        search_orders.get("c_oracle_transport_complete", False)
    )
    with_c_full_order = list(
        search_orders.get("with_c_full_concrete_order", [])
    )
    without_c_full_order = list(
        search_orders.get("without_c_full_concrete_order", [])
    )
    if c_oracle_complete:
        if not with_c_full_order or set(with_c_full_order) != set(
            without_c_full_order
        ):
            raise ValueError("CUDA C oracle does not cover one identical universe")
    active_order_name = str(search_orders.get("active_order", "with_c"))
    if active_order_name not in {"with_c", "without_c"}:
        raise ValueError(f"unknown active CUDA search order {active_order_name!r}")
    active_order = (
        with_c_order if active_order_name == "with_c" else without_c_order
    )
    schedule_spine_order = list(
        search_orders.get("active_schedule_spine_concrete_order", [])
    )
    c_fixed_projection = search_orders.get("c_fixed_projection", {})
    c_fixed_candidate_id = (
        str(c_fixed_projection.get("selected_candidate_id"))
        if isinstance(c_fixed_projection, Mapping)
        and c_fixed_projection.get("selected_candidate_id") is not None
        else None
    )

    initial_ranked = sorted(
        measured_indices,
        key=lambda item: candidate_records[item]["latency_us"],
    )
    # Always remeasure each bounded ordering's provisional winner and each main
    # three-bar provisional winner.  Otherwise a small finalist_count could
    # accidentally give one causal arm only initial ranking samples.
    mandatory_indices: list[int] = []
    mandatory_orders = [with_c_order, without_c_order, schedule_spine_order]
    if c_fixed_candidate_id is not None:
        mandatory_orders.append([c_fixed_candidate_id])
    for order in mandatory_orders:
        arm_indices = [
            measured_index_by_id[candidate_id]
            for candidate_id in order
            if candidate_id in measured_index_by_id
        ]
        if arm_indices:
            provisional = min(
                arm_indices,
                key=lambda item: candidate_records[item]["latency_us"],
            )
            provisional_id = candidate_records[provisional]["candidate"][
                "candidate_id"
            ]
            provisional_signature = execution_signature_by_id[provisional_id]
            if not any(
                execution_signature_by_id[
                    candidate_records[item]["candidate"]["candidate_id"]
                ]
                == provisional_signature
                for item in mandatory_indices
            ):
                mandatory_indices.append(provisional)
    ranked_indices = []
    # Complete-space main bars need one initial latency for every transported
    # candidate, but that must not turn every candidate into a finalist.
    # Rank the complete initial map, then freshly remeasure only the requested
    # top set plus each formal arm's provisional winner.  Otherwise a large
    # oracle universe changes the confirmation protocol and introduces minutes
    # of target drift into Source/IntentIR paired timing.
    ranking_source = [*mandatory_indices, *initial_ranked]
    ranked_signatures: set[tuple[Any, ...]] = set()
    for index in ranking_source:
        candidate_id = candidate_records[index]["candidate"]["candidate_id"]
        signature = execution_signature_by_id[candidate_id]
        if signature in ranked_signatures:
            continue
        ranked_signatures.add(signature)
        if index not in ranked_indices:
            ranked_indices.append(index)
        if (
            len(ranked_indices) >= finalist_count
            and all(item in ranked_indices for item in mandatory_indices)
        ):
            break
    finalist_samples: dict[int, list[float]] = {
        index: [] for index in ranked_indices
    }
    for round_index in range(finalist_remeasurements):
        round_order = (
            ranked_indices
            if round_index % 2 == 0
            else list(reversed(ranked_indices))
        )
        for index in round_order:
            record = candidate_records[index]
            candidate_id = record["candidate"]["candidate_id"]
            thunk = candidate_artifacts[candidate_id][2]
            finalist_samples[index].append(
                measure(thunk, prepare=reset_inputs)
            )
    for index, samples in finalist_samples.items():
        if not samples:
            continue
        record = candidate_records[index]
        # Initial ranking samples are deliberately excluded from final
        # selection.  Fresh alternating round-robin batches remove incumbent
        # and thermal-order bias from the target-side search.
        latency = float(statistics.median(samples))
        candidate_id = record["candidate"]["candidate_id"]
        signature = execution_signature_by_id[candidate_id]
        for equivalent_index in execution_group_indices[signature]:
            equivalent = candidate_records[equivalent_index]
            equivalent["latency_us"] = latency
            equivalent["finalist_batch_us"] = samples
    if not measured_indices:
        raise KernelFailure(
            "no correct bundled MLIR candidate",
            {"candidates": candidate_records},
        )

    freshly_remeasured: set[int] = set()
    for index, samples in finalist_samples.items():
        if not samples:
            continue
        candidate_id = candidate_records[index]["candidate"]["candidate_id"]
        freshly_remeasured.update(
            execution_group_indices[execution_signature_by_id[candidate_id]]
        )

    def best_for_order(order: list[str]) -> dict[str, Any] | None:
        records = [
            candidate_records[measured_index_by_id[candidate_id]]
            for candidate_id in order
            if candidate_id in measured_index_by_id
            and (
                not freshly_remeasured
                or measured_index_by_id[candidate_id] in freshly_remeasured
            )
        ]
        return min(records, key=lambda item: item["latency_us"]) if records else None

    with_c_best = best_for_order(with_c_order)
    without_c_best = best_for_order(without_c_order)
    c_oracle_best = (
        best_for_order(with_c_full_order) if c_oracle_complete else None
    )
    schedule_spine_best = best_for_order(schedule_spine_order)
    c_fixed_same_run = (
        best_for_order([c_fixed_candidate_id])
        if c_fixed_candidate_id is not None
        else None
    )
    # The formal IntentIR bar is the complete admitted lowering-state envelope.
    # Bounded C-guided/Erase-C orders remain search-efficiency diagnostics only.
    expanded_best = (
        c_oracle_best
        if c_oracle_complete
        else best_for_order(active_order)
    )
    # Full is definitionally the union of the fixed-structure schedule spine and
    # the additional admitted structures.  A bounded C-guided order may omit a
    # late schedule-spine candidate even though that executable was transported
    # and freshly measured for the formal E2 bar.  Select over the measured union
    # so Full can never lose the Schedule-Only executable by controller ordering.
    best_record = _select_full_record(expanded_best, schedule_spine_best)
    if best_record is None:
        raise RuntimeError(
            f"active CUDA search order {active_order_name!r} has no correct candidate"
        )
    best_id = best_record["candidate"]["candidate_id"]
    winner, extracted, winner_thunk, winner_outputs = candidate_artifacts[best_id]
    reset_inputs()
    winner_thunk()
    torch.cuda.synchronize()
    winner_agreement = _compare_cuda_outputs(
        torch,
        name="mlir-cubin-final-winner",
        actual=dict(zip(entry["output_keys"], winner_outputs, strict=True)),
        reference=reference_cuda,
        reference_name=reference_name,
        j=j,
        entry=entry,
        policy=policy,
    )
    if not winner_agreement.passed:
        raise KernelFailure(
            f"selected CUDA winner differs from {reference_name}",
            {
                "candidates": candidate_records,
                "winner_correctness": winner_agreement.to_json(),
            },
        )
    best_record["correctness"] = winner_agreement.to_json()

    def candidates_to_95pct(order: list[str]) -> int | None:
        if c_oracle_best is None:
            return None
        threshold_us = float(c_oracle_best["latency_us"]) / 0.95
        best_so_far = float("inf")
        for count, candidate_id in enumerate(order, start=1):
            index = measured_index_by_id.get(candidate_id)
            if index is not None:
                best_so_far = min(
                    best_so_far,
                    float(candidate_records[index]["latency_us"]),
                )
            if best_so_far <= threshold_us:
                return count
        return None

    with_c_candidates_to_95pct = candidates_to_95pct(with_c_full_order)
    without_c_candidates_to_95pct = candidates_to_95pct(
        without_c_full_order
    )

    source_agreement = None
    source_latency: float | None = None
    source_samples: list[float] = []
    mlir_samples: list[float] = []
    paired_batches: list[dict[str, Any]] = []
    confirmation_warmup_rounds = 0
    if baseline_policy == "source_and_refj":
        if "source_tilelang" in entry:
            source_context = PreparedTileLangInvocation(
                entry["source_tilelang"], entry, inputs
            )
        else:
            source_record = entry["source_triton"]
            source_program = KernelSource.from_code(
                source_record["code"],
                source_record["function"],
                trusted=True,
            )
            source_spec = decode_triton_spec(source_record["execution_spec"])
            source_context = PreparedTritonInvocation(
                source_program, source_spec, values
            )
        with source_context as source:
            source.reset_mutable_inputs()
            source.launch()
            torch.cuda.synchronize()
            if reference_name == "canonical-J-reference":
                source_agreement = _compare(
                    name="source-triton",
                    actual=source.copied_outputs(),
                    reference=reference,
                    j=j,
                    entry=entry,
                    values=values,
                    policy=policy,
                    reference_name=reference_name,
                )
                if not source_agreement.passed:
                    raise KernelFailure(
                        "Source baseline differs from canonical J",
                        {
                            "candidates": candidate_records,
                            "source_correctness": source_agreement.to_json(),
                        },
                    )
            source_thunk = source.thunk()
            confirmation_warmup_rounds = _warmup_thunks(
                torch,
                (source_thunk, winner_thunk),
                warmup_ms=warmup_ms,
            )
            for pair_index in range(confirmation_batches):
                if pair_index % 2 == 0:
                    order = ("source", "target")
                    source_us = measure(
                        source_thunk, prepare=source.reset_mutable_inputs
                    )
                    target_us = measure(winner_thunk, prepare=reset_inputs)
                else:
                    order = ("target", "source")
                    target_us = measure(winner_thunk, prepare=reset_inputs)
                    source_us = measure(
                        source_thunk, prepare=source.reset_mutable_inputs
                    )
                source_samples.append(source_us)
                mlir_samples.append(target_us)
                paired_batches.append(
                    {
                        "pair_index": pair_index,
                        "order": list(order),
                        "source_us": source_us,
                        "target_us": target_us,
                        "speedup_source_over_target": source_us / target_us,
                    }
                )
            source_latency = float(statistics.median(source_samples))
    else:
        # The J-only large-shape profile extends the backend stress domain beyond
        # the accepted Source's one-block execution contract.  Reconfirm only the
        # already Target--RefJ-correct fixed winner and emit no Source claim.
        confirmation_warmup_rounds = _warmup_thunks(
            torch,
            (winner_thunk,),
            warmup_ms=warmup_ms,
        )
        for batch_index in range(confirmation_batches):
            target_us = measure(winner_thunk, prepare=reset_inputs)
            mlir_samples.append(target_us)
            paired_batches.append(
                {
                    "batch_index": batch_index,
                    "order": ["target"],
                    "target_us": target_us,
                }
            )

    mlir_latency = float(statistics.median(mlir_samples))
    roofline_evaluations = None
    if workload_cost is not None and roofline_ceilings is not None:
        roofline_evaluations = {
            "full": evaluate_logical_roofline(
                workload_cost, mlir_latency, roofline_ceilings
            )
        }
        if source_latency is not None:
            roofline_evaluations["source"] = evaluate_logical_roofline(
                workload_cost, source_latency, roofline_ceilings
            )
    return {
        "kernel": entry["kernel"],
        "case_id": entry["case_id"],
        "workload_profile": entry.get("workload_profile", "closed-v1"),
        "baseline_policy": baseline_policy,
        "workload_valuation": entry.get("workload_valuation"),
        "extents": entry["extents"],
        "workload_signature": {
            "inputs": entry.get("input_specs"),
            "outputs": entry["output_specs"],
            "extents": entry["extents"],
        },
        "contract_view": entry["contract_view"],
        "search_orders": entry["search_orders"],
        "candidate_count": len(entry["candidates"]),
        "candidates": candidate_records,
        "winner": {
            "candidate": winner["candidate"],
            "concrete_state": winner.get("concrete_state"),
            "lowering": winner.get("lowering"),
            "launch": extracted.record.to_dict(),
            "binary_size_bytes": extracted.binary_size_bytes,
            "cubin_size_bytes": (
                len(extracted.cubin)
                if isinstance(extracted, ExtractedSingleLaunch)
                else None
            ),
        },
        "correctness": {
            "source_j": (
                source_agreement.to_json()
                if source_agreement is not None
                else None
            ),
            "target_reference": reference_name,
            "canonical_j_reference_error": canonical_j_reference_error,
            "target_reference_measured_candidate_count": len(measured_indices),
            "formal_source_replay_closure_claimed": False,
            "policy": policy.policy_id,
        },
        "performance": {
            "c_fixed_us": (
                c_fixed_same_run["latency_us"]
                if c_fixed_same_run is not None
                else None
            ),
            "schedule_only_us": (
                schedule_spine_best["latency_us"]
                if schedule_spine_best is not None
                else None
            ),
            "full_us": best_record["latency_us"],
            "g_schedule": (
                c_fixed_same_run["latency_us"]
                / schedule_spine_best["latency_us"]
                if c_fixed_same_run is not None
                and schedule_spine_best is not None
                else None
            ),
            "g_full": (
                schedule_spine_best["latency_us"]
                / best_record["latency_us"]
                if schedule_spine_best is not None
                else None
            ),
            "selected_us": mlir_latency,
            "source_us": source_latency,
            "selected_over_source": (
                mlir_latency / source_latency
                if source_latency is not None
                else None
            ),
            "speedup_over_source": (
                source_latency / mlir_latency
                if source_latency is not None
                else None
            ),
            "meets_or_beats_source": (
                mlir_latency <= source_latency
                if source_latency is not None
                else None
            ),
            "c_ordering_ablation": {
                "budget": search_orders.get("budget"),
                "budget_unit": search_orders.get("budget_unit"),
                "same_transported_candidate_union": True,
                "active_order": active_order_name,
                "with_c_best_candidate_id": (
                    with_c_best["candidate"]["candidate_id"]
                    if with_c_best is not None
                    else None
                ),
                "without_c_best_candidate_id": (
                    without_c_best["candidate"]["candidate_id"]
                    if without_c_best is not None
                    else None
                ),
                "with_c_best_us": (
                    with_c_best["latency_us"]
                    if with_c_best is not None
                    else None
                ),
                "without_c_best_us": (
                    without_c_best["latency_us"]
                    if without_c_best is not None
                    else None
                ),
                "c_prior_over_erase_c_speedup": (
                    without_c_best["latency_us"] / with_c_best["latency_us"]
                    if with_c_best is not None and without_c_best is not None
                    else None
                ),
                "oracle_complete": c_oracle_complete,
                "oracle_best_us": (
                    c_oracle_best["latency_us"]
                    if c_oracle_best is not None
                    else None
                ),
                "with_c_candidates_to_95pct_of_final_best": (
                    with_c_candidates_to_95pct
                ),
                "without_c_candidates_to_95pct_of_final_best": (
                    without_c_candidates_to_95pct
                ),
                "unavailable_reason": (
                    None
                    if c_oracle_complete
                    else "complete_latency_map_not_transported"
                ),
            },
            "schedule_spine_ablation": (
                {
                    "same_run_measurement": True,
                    "same_total_search_visit": not c_oracle_complete,
                    "comparison_scope": (
                        "complete_candidate_space"
                        if c_oracle_complete
                        else "bounded_active_order"
                    ),
                    "schedule_spine_candidate_count": len(schedule_spine_order),
                    "schedule_spine_best_candidate_id": (
                        schedule_spine_best["candidate"]["candidate_id"]
                    ),
                    "schedule_spine_best_us": schedule_spine_best["latency_us"],
                    "expanded_best_candidate_id": best_id,
                    "expanded_best_us": best_record["latency_us"],
                    "expanded_over_schedule_spine_speedup": (
                        schedule_spine_best["latency_us"]
                        / best_record["latency_us"]
                    ),
                }
                if schedule_spine_best is not None
                else None
            ),
            "c_fixed_schedule_ablation": (
                {
                    "same_run_measurement": True,
                    "c_fixed_candidate_id": c_fixed_candidate_id,
                    "c_fixed_us": c_fixed_same_run["latency_us"],
                    "schedule_only_best_candidate_id": (
                        schedule_spine_best["candidate"]["candidate_id"]
                    ),
                    "schedule_only_best_us": schedule_spine_best["latency_us"],
                    "schedule_over_c_fixed_speedup": (
                        c_fixed_same_run["latency_us"]
                        / schedule_spine_best["latency_us"]
                    ),
                }
                if c_fixed_same_run is not None
                and schedule_spine_best is not None
                else None
            ),
            "logical_io": {
                "boundary_io_bytes": logical_io_bytes,
                "mlir_logical_boundary_gbps": (
                    logical_io_bytes / mlir_latency / 1_000.0
                ),
                "source_logical_boundary_gbps": (
                    logical_io_bytes / source_latency / 1_000.0
                    if source_latency is not None
                    else None
                ),
                "traffic_definition": (
                    "sum of certified boundary input and output payload bytes; "
                    "not measured DRAM transactions"
                ),
            },
            "roofline_point": {
                "j_workload_cost": workload_cost,
                "arithmetic_intensity_flop_per_byte": (
                    floating_point_ops / logical_io_bytes
                ),
                "mlir_achieved_gflops": (
                    floating_point_ops / mlir_latency / 1_000.0
                ),
                "source_achieved_gflops": (
                    floating_point_ops / source_latency / 1_000.0
                    if source_latency is not None
                    else None
                ),
                "hardware_counter_traffic": None,
                "interpretation": "J-derived logical/minimum-traffic point",
                "evaluations": roofline_evaluations,
            },
            "data_regime": data_regime,
            "timing": (
                "cuda-event-single-shot"
                if data_regime == "cold-data"
                else timing
            ),
            "requested_warm_timing": timing,
            "graph_repetitions": graph_repetitions,
            "warmup_ms": warmup_ms,
            "repeat_ms": repeat_ms,
            "measurement_rounds": measurement_rounds,
            "cold_iterations_per_round": (
                cold_iterations if data_regime == "cold-data" else None
            ),
            "cache_eviction_in_timed_region": False,
            "selection_confirmation": {
                "mlir_batch_us": mlir_samples,
                "source_batch_us": source_samples,
                "batch_aggregation": "median",
                "samples_per_batch": measurement_rounds,
                "finalist_count": len(ranked_indices),
                "finalist_remeasurements": finalist_remeasurements,
                "winner_selection_scope": (
                    "complete-space-ranked-fresh-finalists"
                    if c_oracle_complete and freshly_remeasured
                    else "freshly-remeasured-finalists"
                    if freshly_remeasured
                    else "initial-ranking-samples"
                ),
                "comparison_protocol": (
                    "paired-alternating.v1"
                    if source_latency is not None
                    else "target-only-fresh-batches.v1"
                ),
                "paired_batches": paired_batches,
                "warmup_pair_discarded": source_latency is not None,
                "pre_confirmation_warmup_ms": warmup_ms,
                "pre_confirmation_warmup_rounds": confirmation_warmup_rounds,
                "warmup_method": (
                    "untimed-round-robin-before-L2-evicted-samples"
                ),
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*")
    parser.add_argument(
        "--timing", choices=("graph", "event"), default="graph"
    )
    parser.add_argument(
        "--data-regime", choices=("warm", "cold-data"), default="warm"
    )
    parser.add_argument("--cold-iterations", type=int, default=9)
    parser.add_argument("--cache-flush-mib", type=int, default=256)
    parser.add_argument("--graph-repetitions", type=int, default=200)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--finalist-remeasurements", type=int, default=3)
    parser.add_argument("--confirmation-batches", type=int, default=4)
    parser.add_argument("--roofline-calibration", action="store_true")
    parser.add_argument("--roofline-memory-mib", type=int, default=256)
    parser.add_argument("--roofline-matrix-extent", type=int, default=4096)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("protocol") not in SUPPORTED_BUNDLE_PROTOCOLS:
        raise ValueError("unsupported CUDA benchmark bundle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.measurement_rounds <= 0:
        raise ValueError("measurement-rounds must be positive")
    if args.finalist_count <= 0:
        raise ValueError("finalist-count must be positive")
    if args.finalist_remeasurements < 0:
        raise ValueError("finalist-remeasurements cannot be negative")
    if args.confirmation_batches <= 0:
        raise ValueError("confirmation-batches must be positive")
    if args.cold_iterations <= 0:
        raise ValueError("cold-iterations must be positive")
    if args.cache_flush_mib <= 0:
        raise ValueError("cache-flush-mib must be positive")
    roofline_ceilings = (
        _measure_cuda_roofs(
            torch,
            memory_mib=args.roofline_memory_mib,
            matrix_extent=args.roofline_matrix_extent,
        )
        if args.roofline_calibration
        else None
    )
    capability = torch.cuda.get_device_capability()
    target = bundle["target"]["chip"]
    expected_capability = cuda_compute_capability_for_chip(target)
    declared_capability = tuple(bundle["target"].get("compute_capability", ()))
    if declared_capability != expected_capability:
        raise RuntimeError(
            "bundle CUDA target metadata is inconsistent: "
            f"chip {target} names {expected_capability}, metadata names "
            f"{declared_capability}"
        )
    if capability != expected_capability:
        raise RuntimeError(
            f"{target} bundle requires compute capability "
            f"{expected_capability}, got {capability}"
        )
    device_properties = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    )
    l2_cache_bytes = int(getattr(device_properties, "L2_cache_size", 0))
    cache_flush_bytes = args.cache_flush_mib * 1024 * 1024
    if args.data_regime == "cold-data" and (
        l2_cache_bytes <= 0 or cache_flush_bytes <= l2_cache_bytes
    ):
        raise ValueError(
            "cold-data cache-flush working set must exceed the detected L2: "
            f"flush={cache_flush_bytes}, l2={l2_cache_bytes}"
        )
    cache_eviction = None
    if args.data_regime == "cold-data":
        cache_eviction = torch.empty(
            cache_flush_bytes // 4,
            dtype=torch.float32,
            device="cuda",
        ).uniform_()
        # Initialize the eviction kernel/runtime before any measured candidate.
        cache_eviction.add_(1.0)
        torch.cuda.synchronize()
    selected = set(args.kernels or [])
    entries = [
        item
        for item in bundle["kernels"]
        if not selected or item["kernel"] in selected
    ]
    records: list[dict[str, Any]] = []
    started = time.time_ns()
    for entry in entries:
        try:
            record = _one_kernel(
                entry,
                timing=args.timing,
                graph_repetitions=args.graph_repetitions,
                warmup_ms=args.warmup_ms,
                repeat_ms=args.repeat_ms,
                measurement_rounds=args.measurement_rounds,
                finalist_count=args.finalist_count,
                finalist_remeasurements=args.finalist_remeasurements,
                confirmation_batches=args.confirmation_batches,
                roofline_ceilings=roofline_ceilings,
                data_regime=args.data_regime,
                cold_iterations=args.cold_iterations,
                cache_eviction=cache_eviction,
            )
            records.append(record)
            perf = record["performance"]
            if perf["source_us"] is None:
                print(
                    f"{entry['kernel']}: selected={perf['selected_us']:.4f} us "
                    "(Target--RefJ-only; no Source claim)",
                    flush=True,
                )
            else:
                print(
                    f"{entry['kernel']}: selected={perf['selected_us']:.4f} us vs "
                    f"Source={perf['source_us']:.4f} us "
                    f"({perf['selected_over_source']:.3f}x)",
                    flush=True,
                )
        except Exception as exc:
            failure = {
                "kernel": entry["kernel"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, KernelFailure):
                failure["diagnostics"] = exc.diagnostics
            records.append(failure)
            print(
                f"{entry['kernel']}: FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                break
    successful = [item for item in records if "performance" in item]
    c_ordering_speedups = [
        item["performance"]["c_ordering_ablation"][
            "c_prior_over_erase_c_speedup"
        ]
        for item in successful
        if item["performance"]["c_ordering_ablation"][
            "c_prior_over_erase_c_speedup"
        ]
        is not None
    ]
    report = {
        "protocol": REPORT_PROTOCOL,
        "bundle_protocol": bundle["protocol"],
        "target": bundle["target"],
        "workload_profile": bundle.get("workload_profile", "closed-v1"),
        "baseline_policy": bundle.get("baseline_policy", "source_and_refj"),
        "ablation_axes": bundle.get("ablation_axes"),
        "measurement_spec_protocol": bundle.get("measurement_spec_protocol"),
        "device": _device_record(torch),
        "roofline_ceilings": (
            roofline_ceilings.to_record()
            if roofline_ceilings is not None
            else None
        ),
        "measurement": {
            "data_regime": args.data_regime,
            "requested_warm_timing": args.timing,
            "effective_timing": (
                "cuda-event-single-shot"
                if args.data_regime == "cold-data"
                else args.timing
            ),
            "graph_repetitions": args.graph_repetitions,
            "warmup_ms": args.warmup_ms,
            "cold_candidate_warmup_ms": (
                args.warmup_ms if args.data_regime == "cold-data" else None
            ),
            "cold_candidate_warmup_scope": (
                "untimed code/P-state stabilization before L2-evicted samples"
                if args.data_regime == "cold-data"
                else None
            ),
            "repeat_ms": args.repeat_ms,
            "measurement_rounds": args.measurement_rounds,
            "cold_iterations_per_round": (
                args.cold_iterations
                if args.data_regime == "cold-data"
                else None
            ),
            "cache_flush_bytes": (
                cache_flush_bytes if args.data_regime == "cold-data" else None
            ),
            "detected_l2_cache_bytes": l2_cache_bytes,
            "cache_eviction_in_timed_region": False,
            "cold_scope": (
                "data-cache state only; process/module/JIT cold start excluded"
                if args.data_regime == "cold-data"
                else None
            ),
        },
        "created_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        "summary": {
            "requested": len(entries),
            "successful": len(successful),
            "failed": len(records) - len(successful),
            "meets_or_beats_source": sum(
                item["performance"]["meets_or_beats_source"] is True
                for item in successful
            ),
            "source_baseline_count": sum(
                item["performance"]["source_us"] is not None
                for item in successful
            ),
            "geomean_c_prior_over_erase_c_speedup": (
                float(
                    np.exp(
                        np.mean(np.log(np.asarray(c_ordering_speedups)))
                    )
                )
                if c_ordering_speedups
                else None
            ),
        },
        "kernels": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if len(successful) == len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
