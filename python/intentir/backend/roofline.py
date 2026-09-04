"""Operational Roofline evaluation over J-derived workload costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ROOFLINE_PROTOCOL = "intentir.backend.logical-roofline.v2"

# A logical-boundary Roofline is intentionally a conservative explanatory
# model.  Points faster than the calibrated DRAM lower bound normally indicate
# cache reuse or a ceiling/traffic mismatch; points farther than 2x from the
# bound do not support a strong "memory wall" claim either.
_MIN_CONSISTENT_LATENCY_RATIO = 0.95
_MAX_MEMORY_WALL_LATENCY_RATIO = 2.0


@dataclass(frozen=True)
class RooflineCeilings:
    memory_gbps: float
    fp32_gflops: float

    def __post_init__(self) -> None:
        if self.memory_gbps <= 0 or self.fp32_gflops <= 0:
            raise ValueError("Roofline ceilings must be positive")

    @property
    def ridge_point_flop_per_byte(self) -> float:
        return self.fp32_gflops / self.memory_gbps

    def to_record(self) -> dict[str, Any]:
        return {
            "memory_gbps": self.memory_gbps,
            "fp32_gflops": self.fp32_gflops,
            "ridge_point_flop_per_byte": self.ridge_point_flop_per_byte,
        }


def evaluate_logical_roofline(
    workload_cost: Mapping[str, Any],
    latency_us: float,
    ceilings: RooflineCeilings,
) -> dict[str, Any]:
    if latency_us <= 0:
        raise ValueError("Roofline latency must be positive")
    floating_ops = int(workload_cost["floating_point_ops"])
    raw_boundary_bytes = workload_cost.get("boundary_io_bytes")
    if raw_boundary_bytes is None:
        compute_time_us = floating_ops / ceilings.fp32_gflops / 1_000.0
        rejection_reasons = [
            "dynamic_boundary_bytes",
            "incomplete_j_workload_cost",
        ]
        return {
            "protocol": ROOFLINE_PROTOCOL,
            "ceilings": ceilings.to_record(),
            "arithmetic_intensity_flop_per_byte": None,
            "memory_roof_gflops": None,
            "attainable_gflops": None,
            "achieved_gflops": floating_ops / latency_us / 1_000.0,
            "predicted_bottleneck": "unknown_dynamic_boundary",
            "asymptotic_roof_bottleneck": "unknown_dynamic_boundary",
            "observed_operational_regime": "dynamic_boundary_unresolved",
            "supports_memory_wall_claim": False,
            "memory_wall_claim_rejection_reasons": rejection_reasons,
            "memory_time_lower_bound_us": None,
            "compute_time_lower_bound_us": compute_time_us,
            "roof_time_lower_bound_us": None,
            "latency_over_logical_roof_lower_bound": None,
            "roof_utilization": None,
            "caveats": [
                "J boundary bytes are dynamic for this invocation and cannot be used for a certified logical Roofline",
                "no runtime output length is guessed from a symbolic shape marker",
            ],
        }
    boundary_bytes = int(raw_boundary_bytes)
    intensity = floating_ops / boundary_bytes if boundary_bytes else float("inf")
    memory_roof_gflops = intensity * ceilings.memory_gbps
    attainable_gflops = min(memory_roof_gflops, ceilings.fp32_gflops)
    achieved_gflops = floating_ops / latency_us / 1_000.0
    memory_time_us = boundary_bytes / ceilings.memory_gbps / 1_000.0
    compute_time_us = floating_ops / ceilings.fp32_gflops / 1_000.0
    lower_bound_us = max(memory_time_us, compute_time_us)
    caveats = [
        "classification uses certified logical boundary bytes, not measured DRAM transactions"
    ]
    if int(workload_cost.get("special_function_ops", 0)):
        caveats.append(
            "special-function throughput may dominate but is not represented by the FP32 roof"
        )
    if not bool(workload_cost.get("complete", False)):
        caveats.append("J workload-cost record is incomplete for this invocation")
    predicted_bottleneck = (
        "memory"
        if intensity < ceilings.ridge_point_flop_per_byte
        else "fp32_compute"
    )
    latency_ratio = latency_us / lower_bound_us if lower_bound_us else None
    roof_utilization = (
        achieved_gflops / attainable_gflops if attainable_gflops else None
    )
    complete = bool(workload_cost.get("complete", False))
    has_unmodeled_sfu = bool(int(workload_cost.get("special_function_ops", 0)))
    roof_consistent = (
        latency_ratio is not None
        and latency_ratio >= _MIN_CONSISTENT_LATENCY_RATIO
        and roof_utilization is not None
        and roof_utilization <= 1.0 / _MIN_CONSISTENT_LATENCY_RATIO
    )
    memory_wall_supported = (
        predicted_bottleneck == "memory"
        and complete
        and not has_unmodeled_sfu
        and roof_consistent
        and latency_ratio <= _MAX_MEMORY_WALL_LATENCY_RATIO
    )
    rejection_reasons: list[str] = []
    if predicted_bottleneck != "memory":
        rejection_reasons.append("logical_roof_predicts_non_memory_bottleneck")
    if not complete:
        rejection_reasons.append("incomplete_j_workload_cost")
    if has_unmodeled_sfu:
        rejection_reasons.append("special_function_throughput_not_modeled")
    if not roof_consistent:
        rejection_reasons.append("logical_roof_inconsistent_with_observation")
    if (
        latency_ratio is not None
        and latency_ratio > _MAX_MEMORY_WALL_LATENCY_RATIO
    ):
        rejection_reasons.append("too_far_from_logical_memory_roof")

    if latency_ratio is not None and latency_ratio < _MIN_CONSISTENT_LATENCY_RATIO:
        operational_regime = "cache_reuse_or_roof_model_mismatch"
        caveats.append(
            "observed latency is faster than the calibrated logical Roofline; "
            "do not interpret this point as a DRAM memory-wall measurement"
        )
    elif (
        latency_ratio is not None
        and latency_ratio > _MAX_MEMORY_WALL_LATENCY_RATIO
    ):
        operational_regime = "latency_or_unmodeled_work_dominated"
    else:
        operational_regime = predicted_bottleneck + "_roof_approaching"
    return {
        "protocol": ROOFLINE_PROTOCOL,
        "ceilings": ceilings.to_record(),
        "arithmetic_intensity_flop_per_byte": intensity,
        "memory_roof_gflops": memory_roof_gflops,
        "attainable_gflops": attainable_gflops,
        "achieved_gflops": achieved_gflops,
        "predicted_bottleneck": predicted_bottleneck,
        "asymptotic_roof_bottleneck": predicted_bottleneck,
        "observed_operational_regime": operational_regime,
        "supports_memory_wall_claim": memory_wall_supported,
        "memory_wall_claim_rejection_reasons": rejection_reasons,
        "memory_time_lower_bound_us": memory_time_us,
        "compute_time_lower_bound_us": compute_time_us,
        "roof_time_lower_bound_us": lower_bound_us,
        "latency_over_logical_roof_lower_bound": latency_ratio,
        "roof_utilization": roof_utilization,
        "caveats": caveats,
    }


__all__ = [
    "ROOFLINE_PROTOCOL",
    "RooflineCeilings",
    "evaluate_logical_roofline",
]
