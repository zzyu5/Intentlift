"""Frontend-neutral registry for real SCRC Harness workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from intentir.verify.frontend_workloads import (
    FrontendDatasetHarness,
    FrontendDatasetHarnessDiagnostic,
    FrontendDatasetHarnessProjection,
    PreparedFrontendDatasetWorkload,
    prepare_frontend_dataset_workload,
    project_frontend_dataset_harnesses,
)
from intentir.verify.gate1.frontend import active_dataset_frontends
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    PreparedTritonWorkload,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)


RegisteredHarness = TritonWorkloadAdapter | FrontendDatasetHarness
PreparedRegisteredWorkload = (
    PreparedTritonWorkload | PreparedFrontendDatasetWorkload
)


def frontend_dataset_harness_projection(
    *,
    dataset_root: str | Path | None = None,
    strict: bool = False,
) -> FrontendDatasetHarnessProjection:
    try:
        return project_frontend_dataset_harnesses(dataset_root)
    except (OSError, ValueError) as exc:
        if strict:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        if len(reason) > 2048:
            reason = reason[:2045] + "..."
        return FrontendDatasetHarnessProjection(
            {},
            (
                FrontendDatasetHarnessDiagnostic(
                    "dataset:TianchenIR-MultiFrontend",
                    "native",
                    "case_unavailable",
                    reason,
                ),
            ),
        )


def registered_scrc_workloads(
    *, dataset_root: str | Path | None = None
) -> dict[str, RegisteredHarness]:
    native = frontend_dataset_harness_projection(dataset_root=dataset_root).harnesses
    overlap = set(TRITON_WORKLOAD_ADAPTERS) & set(native)
    if overlap:
        raise RuntimeError(
            "SCRC workload registries contain duplicate case ids: "
            + ", ".join(sorted(overlap))
        )
    return {**TRITON_WORKLOAD_ADAPTERS, **native}


def registered_scrc_case_ids(
    *,
    frontend: str | None = None,
    dataset_root: str | Path | None = None,
) -> tuple[str, ...]:
    rows = registered_scrc_workloads(dataset_root=dataset_root)
    if frontend is not None:
        supported = {
            item.value for item in active_dataset_frontends()
        }
        if frontend not in supported:
            raise ValueError(
                f"frontend {frontend!r} has no registered Dataset workflow; "
                f"supported frontends are {sorted(supported)!r}"
            )
        rows = {
            case_id: adapter
            for case_id, adapter in rows.items()
            if getattr(adapter, "frontend", "triton") == frontend
        }
    return tuple(sorted(rows))


def prepare_registered_scrc_workload(
    case_id: str,
    *,
    dataset_root: str | Path | None = None,
    workloads: Mapping[str, RegisteredHarness] | None = None,
) -> PreparedRegisteredWorkload:
    if workloads is not None:
        adapter: RegisteredHarness | None = workloads.get(case_id)
    else:
        adapter = TRITON_WORKLOAD_ADAPTERS.get(case_id)
        if adapter is None:
            adapter = frontend_dataset_harness_projection(
                dataset_root=dataset_root,
                strict=True,
            ).harnesses.get(case_id)
    if adapter is None:
        raise ValueError(f"case {case_id!r} has no registered SCRC Harness")
    if isinstance(adapter, TritonWorkloadAdapter):
        return prepare_triton_workload(adapter, dataset_root=dataset_root)
    if isinstance(adapter, FrontendDatasetHarness):
        return prepare_frontend_dataset_workload(
            adapter, dataset_root=dataset_root
        )
    raise TypeError(f"unsupported registered Harness {type(adapter).__name__}")


__all__ = [
    "PreparedRegisteredWorkload",
    "RegisteredHarness",
    "frontend_dataset_harness_projection",
    "prepare_registered_scrc_workload",
    "registered_scrc_case_ids",
    "registered_scrc_workloads",
]
