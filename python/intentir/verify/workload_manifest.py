"""Source/Harness-side SCRC projection of the 28-kernel lowering registry.

This module deliberately does not define another kernel universe.  Every entry
is constructed from :mod:`intentir.backend.cuda.capabilities`; the extra fields
only describe how the SCRC runner can locate the already-counted source.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from intentir.backend.cuda.capabilities import (
    HighPerformanceCapabilityRegistry,
    KernelLoweringCapability,
    load_high_performance_registry,
)
from intentir.verify.gate1.profile import (
    TRITON_OBSERVER_PROFILE as _TRITON_OBSERVER_PROFILE,
)
from intentir.verify.gate2.policy import DEFAULT_GATE2_POLICY


SCRC_WORKLOAD_MANIFEST_PROTOCOL = "intentir.scrc.high-performance-workloads.v1"
TRITON_OBSERVER_PROFILE = (
    f"{_TRITON_OBSERVER_PROFILE.profile_id}@"
    f"{_TRITON_OBSERVER_PROFILE.profile_version}"
)
GATE2_Q_POLICY = DEFAULT_GATE2_POLICY.policy_id


class SourceResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    CASE_UNAVAILABLE = "case_unavailable"
    MANIFEST_INVALID = "manifest_invalid"


_REPRESENTATIVES = (
    ("high_performance:liger_dyt", "core.elementwise"),
    ("high_performance:liger_swiglu", "core.masked_pointwise"),
    (
        "high_performance:flash_attn.flash_attn_triton."
        "_bwd_preprocess_do_o_dot",
        "core.single_axis_sum",
    ),
    ("high_performance:liger_softmax", "frontier.softmax"),
    ("high_performance:liger_rms_norm", "frontier.normalization"),
    (
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
        "frontier.contraction",
    ),
)
_REPRESENTATIVE_TIERS = dict(_REPRESENTATIVES)
_DECLARED_ADAPTER_IDS = frozenset(
    {
        *_REPRESENTATIVE_TIERS,
        "high_performance:flash_attn.flash_attn_triton._fwd_kernel",
        "high_performance:flash_attn.flash_attn_triton._bwd_kernel",
        "high_performance:liger_fused_add_rms_norm",
        "high_performance:liger_sparsemax",
        "high_performance:liger_qwen2vl_mrope",
        "high_performance:liger_group_norm",
        "high_performance:liger_jsd",
        "high_performance:liger_cross_entropy",
        "high_performance:liger_geglu",
        "high_performance:liger_poly_norm",
        "high_performance:liger_llama4_rope",
        "high_performance:liger_rope",
        "high_performance:liger_layer_norm",
        "high_performance:liger_kl_div",
        "high_performance:liger_tvd",
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_kernel"
        ),
        (
            "high_performance:xformers.ops._triton.rmsnorm_kernels."
            "_rms_norm_add_kernel"
        ),
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_fwd_kernel_splitK"
        ),
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs"
        ),
        (
            "high_performance:xformers.ops.fmha._triton.splitk_kernels."
            "_splitK_reduce_varargs_backward"
        ),
        (
            "high_performance:flash_attn.ops.triton.layer_norm."
            "_layer_norm_fwd_1pass_kernel"
        ),
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_kernel_one_col_block"
        ),
    }
)
_REPRESENTATIVE_ORDER = {
    case_id: index for index, (case_id, _tier) in enumerate(_REPRESENTATIVES)
}
_SELECTABLE_TIERS = frozenset(
    {"catalog", "representative", *_REPRESENTATIVE_TIERS.values()}
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    configured = os.environ.get("INTENTIR_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return _repo_root().parent / "Dataset"


def _source_module(capability: KernelLoweringCapability) -> str:
    identity = capability.kernel_id.removeprefix("high_performance:")
    if identity == capability.kernel_id:
        raise ValueError("high-performance kernel id lacks canonical prefix")
    if capability.suite == "liger":
        module = identity.removeprefix("liger_")
        if module == identity or not module:
            raise ValueError("Liger kernel id lacks the liger_ module projection")
        return f"liger_kernel.ops.{module}"
    if capability.suite in {"flash_attention", "xformers"}:
        if "." not in identity:
            raise ValueError("module-qualified kernel id lacks a symbol suffix")
        return identity.rsplit(".", 1)[0]
    raise ValueError(f"unsupported high-performance suite {capability.suite!r}")


def _dataset_candidate(
    suite: str,
    module: str,
    dataset_root: Path,
) -> Path:
    roots = {
        "liger": dataset_root / "Liger-Kernel" / "src",
        "flash_attention": dataset_root / "flash-attention",
        "xformers": dataset_root / "xformers",
    }
    return roots[suite].joinpath(*module.split(".")).with_suffix(".py")


def _installed_module_path(module: str) -> Path | None:
    """Locate a source module without importing any package in its chain."""

    relative = Path(*module.split(".")).with_suffix(".py")
    seen: set[Path] = set()
    for raw_root in sys.path:
        root = Path(raw_root or os.curdir).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _configured_external_source_roots() -> tuple[Path, ...]:
    """Return stable source roots used by the external library catalog.

    The active Dataset keeps the canonical ``dataset://`` layout, while some
    source archives retain their upstream or normalized layout.  Explicit
    roots let the resolver consume those existing archives without copying
    files into the Dataset or importing the third-party packages.  The
    workspace's existing sibling source archive is discovered by default so
    callers do not need to recreate an environment or export a transient
    shell variable for every invocation.
    """

    raw_roots = os.environ.get("INTENTIR_EXTERNAL_SOURCE_ROOTS", "")
    roots: list[Path] = []
    seen: set[Path] = set()
    configured = [value for value in raw_roots.split(os.pathsep) if value]
    configured.append(
        str(_repo_root().parent / "intent-dsl" / "source" / "triton")
    )
    for raw_root in configured:
        value = raw_root.strip()
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        roots.append(root)
    return tuple(roots)


def _external_module_paths(
    module: str,
    symbol: str,
    suite: str,
    roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Find configured source files in the requested library only.

    External archives may normalize different libraries into the same
    directory shape.  A basename-only recursive lookup can therefore select,
    for example, FlashAttention's ``layer_norm.py`` for a Liger workload.  The
    resolver keeps candidates under a suite path hint and requires the exact
    top-level entry symbol before returning a path.  Source files are still
    read-only; this function never copies or imports them.
    """

    suite_hints = {
        "liger": {"liger", "liger-kernel", "liger_kernel"},
        "flash_attention": {
            "flash-attention",
            "flash_attention",
            "flash-attn",
            "flash_attn",
        },
        "xformers": {"xformers"},
    }
    try:
        hints = suite_hints[suite]
    except KeyError as exc:
        raise ValueError(f"unsupported external source suite {suite!r}") from exc

    def has_suite_hint(path: Path) -> bool:
        parts = {part.casefold() for part in path.parts}
        return bool(parts & hints)

    def has_cuda_triton_layout(path: Path) -> bool:
        if suite != "liger":
            return True
        # The Liger repository also carries Triton-shaped implementations for
        # non-CUDA backends.  This registry is the CUDA/Triton workload catalog;
        # accepting an identically named Ascend or cuTile kernel would make a
        # syntactically valid but target-incompatible source look resolved.
        parts = {part.casefold() for part in path.parts}
        return not parts.intersection({"_ascend", "ascend", "_cutile", "cutile"})

    def has_top_level_symbol(path: Path) -> bool:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            # Preserve a path with a real source parse failure so the caller
            # can report the source as invalid instead of hiding the failure.
            return True
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol
            for node in tree.body
        ) == 1

    relative = Path(*module.split(".")).with_suffix(".py")
    basename = relative.name
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        direct = root / relative
        candidates = [direct] if direct.is_file() else []
        try:
            candidates.extend(
                path
                for path in sorted(root.rglob(basename), key=lambda item: item.as_posix())
                if path != direct
            )
        except OSError:
            candidates = [direct] if direct.is_file() else []
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved in seen
                or not resolved.is_file()
                or not has_suite_hint(resolved)
                or not has_cuda_triton_layout(resolved)
                or not has_top_level_symbol(resolved)
            ):
                continue
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


@dataclass(frozen=True)
class SourceResolution:
    status: SourceResolutionStatus
    module: str
    symbol: str
    path: Path | None = None
    diagnostic: str = ""
    location: str = ""

    @property
    def resolved(self) -> bool:
        return self.status is SourceResolutionStatus.RESOLVED

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "module": self.module,
            "symbol": self.symbol,
            "location": self.location or None,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class SCRCWorkloadEntry:
    ordinal: int
    case_id: str
    suite: str
    symbol: str
    mechanisms: tuple[str, ...]
    source_module: str
    source_alignment: str
    lowering_status: str
    representative_tier: str = "catalog"
    adapter_readiness: str = "pending"
    frontend: str = "triton"
    observer_profile: str = TRITON_OBSERVER_PROFILE
    q_policy: str = GATE2_Q_POLICY

    @classmethod
    def from_capability(
        cls, capability: KernelLoweringCapability
    ) -> "SCRCWorkloadEntry":
        return cls(
            ordinal=capability.ordinal,
            case_id=capability.kernel_id,
            suite=capability.suite,
            symbol=capability.symbol,
            mechanisms=capability.mechanisms,
            source_module=_source_module(capability),
            source_alignment=capability.source_alignment,
            lowering_status=capability.lowering_status,
            representative_tier=_REPRESENTATIVE_TIERS.get(
                capability.kernel_id, "catalog"
            ),
            adapter_readiness=(
                "declared"
                if capability.kernel_id in _DECLARED_ADAPTER_IDS
                else "pending"
            ),
        )

    def resolve_source(
        self,
        *,
        dataset_root: str | Path | None = None,
    ) -> SourceResolution:
        root = (
            Path(dataset_root).expanduser().resolve()
            if dataset_root is not None
            else default_dataset_root()
        )
        installed = _installed_module_path(self.source_module)
        fallback = _dataset_candidate(self.suite, self.source_module, root)
        candidates: list[tuple[str, Path, str]] = []
        try:
            relative = fallback.relative_to(root)
            dataset_location = f"dataset://{relative.as_posix()}"
        except ValueError:
            dataset_location = f"dataset://{fallback.name}"

        external_roots = _configured_external_source_roots()
        if external_roots:
            if fallback.is_file():
                candidates.append(("dataset", fallback, dataset_location))
            candidates.extend(
                ("external", path, dataset_location)
                for path in _external_module_paths(
                    self.source_module,
                    self.symbol,
                    self.suite,
                    external_roots,
                )
            )
            if installed is not None:
                candidates.append(("installed", installed, dataset_location))
        else:
            if installed is not None:
                candidates.append(
                    ("installed", installed, f"module://{self.source_module}")
                )
            if installed is None or fallback.resolve() != installed:
                candidates.append(("dataset", fallback, dataset_location))

        failures: list[str] = []
        found_source = False
        for origin, candidate, location in candidates:
            if not candidate.is_file():
                continue
            found_source = True
            try:
                tree = ast.parse(candidate.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeError) as exc:
                failures.append(
                    f"{origin} source cannot be parsed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            definitions = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == self.symbol
            ]
            if len(definitions) != 1:
                failures.append(
                    f"{origin} entry symbol must resolve once at module top level; "
                    f"found {len(definitions)} definitions"
                )
                continue
            return SourceResolution(
                SourceResolutionStatus.RESOLVED,
                self.source_module,
                self.symbol,
                candidate.resolve(),
                location=location,
            )

        if not found_source:
            return SourceResolution(
                SourceResolutionStatus.CASE_UNAVAILABLE,
                self.source_module,
                self.symbol,
                diagnostic=(
                    "source module is not installed and Dataset fallback is absent: "
                    f"{fallback}"
                ),
            )
        return SourceResolution(
            SourceResolutionStatus.MANIFEST_INVALID,
            self.source_module,
            self.symbol,
            installed or fallback,
            "; ".join(failures),
        )

    def to_json(
        self,
        *,
        source: SourceResolution | None = None,
    ) -> dict[str, Any]:
        result = {
            "ordinal": self.ordinal,
            "case_id": self.case_id,
            "suite": self.suite,
            "symbol": self.symbol,
            "mechanisms": list(self.mechanisms),
            "frontend": self.frontend,
            "source_module": self.source_module,
            "source_alignment": self.source_alignment,
            "lowering_status": self.lowering_status,
            "observer_profile": self.observer_profile,
            "q_policy": self.q_policy,
            "representative_tier": self.representative_tier,
            "adapter_readiness": self.adapter_readiness,
        }
        if source is not None:
            result["source_resolution"] = source.to_json()
        return result


@dataclass(frozen=True)
class ManifestValidation:
    entries: tuple[SCRCWorkloadEntry, ...]
    sources: tuple[SourceResolution, ...]
    structural_errors: tuple[str, ...] = ()

    @property
    def resolved_count(self) -> int:
        return sum(item.resolved for item in self.sources)

    @property
    def passed(self) -> bool:
        return not self.structural_errors and self.resolved_count == len(self.entries)

    def to_json(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in SourceResolutionStatus}
        for source in self.sources:
            counts[source.status.value] += 1
        return {
            "protocol": SCRC_WORKLOAD_MANIFEST_PROTOCOL,
            "passed": self.passed,
            "total": len(self.entries),
            "resolved": self.resolved_count,
            "source_status_counts": counts,
            "structural_errors": list(self.structural_errors),
            "entries": [
                entry.to_json(source=source)
                for entry, source in zip(self.entries, self.sources, strict=True)
            ],
        }


@dataclass(frozen=True)
class SCRCWorkloadManifest:
    entries: tuple[SCRCWorkloadEntry, ...]
    capability_schema: str
    scenario_29: dict[str, Any]

    @property
    def by_id(self) -> dict[str, SCRCWorkloadEntry]:
        return {entry.case_id: entry for entry in self.entries}

    def select(
        self,
        *,
        case_id: str | None = None,
        tier: str | None = None,
    ) -> tuple[SCRCWorkloadEntry, ...]:
        if tier is not None and tier not in _SELECTABLE_TIERS:
            raise KeyError(f"unknown SCRC workload tier {tier!r}")
        entries: Iterable[SCRCWorkloadEntry] = self.entries
        if case_id is not None:
            selected = self.by_id.get(case_id)
            if selected is None:
                raise KeyError(f"unknown high-performance case {case_id!r}")
            entries = (selected,)
        if tier is not None:
            if tier == "representative":
                entries = sorted(
                    (
                    entry
                    for entry in entries
                    if entry.representative_tier != "catalog"
                    ),
                    key=lambda entry: _REPRESENTATIVE_ORDER[entry.case_id],
                )
            else:
                entries = (
                    entry
                    for entry in entries
                    if entry.representative_tier == tier
                )
        selected_entries = tuple(entries)
        if not selected_entries:
            raise KeyError(
                "no SCRC workloads match the requested case/tier selection"
            )
        return selected_entries

    def validate(
        self,
        *,
        dataset_root: str | Path | None = None,
    ) -> ManifestValidation:
        errors: list[str] = []
        ordinals = [entry.ordinal for entry in self.entries]
        if ordinals != list(range(1, 29)):
            errors.append("SCRC projection ordinals must remain exactly 1..28")
        ids = [entry.case_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            errors.append("SCRC projection case ids must be unique")
        sources = tuple(
            entry.resolve_source(dataset_root=dataset_root)
            for entry in self.entries
        )
        return ManifestValidation(tuple(self.entries), sources, tuple(errors))

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": SCRC_WORKLOAD_MANIFEST_PROTOCOL,
            "capability_schema": self.capability_schema,
            "direct_kernel_count": len(self.entries),
            "scenario_29": dict(self.scenario_29),
            "entries": [entry.to_json() for entry in self.entries],
        }


def build_workload_manifest(
    registry: HighPerformanceCapabilityRegistry | None = None,
) -> SCRCWorkloadManifest:
    source = registry or load_high_performance_registry()
    return SCRCWorkloadManifest(
        tuple(SCRCWorkloadEntry.from_capability(item) for item in source.kernels),
        source.schema_version,
        dict(source.scenario_29),
    )


__all__ = [
    "GATE2_Q_POLICY",
    "ManifestValidation",
    "SCRC_WORKLOAD_MANIFEST_PROTOCOL",
    "SCRCWorkloadEntry",
    "SCRCWorkloadManifest",
    "SourceResolution",
    "SourceResolutionStatus",
    "TRITON_OBSERVER_PROFILE",
    "build_workload_manifest",
    "default_dataset_root",
]
