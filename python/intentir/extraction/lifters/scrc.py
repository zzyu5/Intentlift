"""Source-aware LLM Lift producing anonymous ``J + pure C + pi`` directly."""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from intentir.extraction.providers.client import ProviderError, call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.canonical_construction import (
    CANONICAL_AUTHORED_HANDLE_PREFIXES,
    CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES,
    CANONICAL_HANDLE_NAMESPACE_PATTERNS,
    CanonicalConstruction,
    CanonicalConstructionRecords,
    CanonicalHandleAllocator,
    SourceFacts,
    SourceTransferLedger,
    SOURCE_OWNER_TOKEN_PATTERN,
    validate_source_owner_binding,
    project_host_launch_contract,
    validate_host_launch_contract,
    canonical_authored_handle_declarations,
    canonical_boundary_records,
    canonical_operation_objects,
    canonical_operation_records,
    semantic_operation_wire_rows,
    canonical_output_records as build_canonical_output_records,
    canonical_value_type_records as build_canonical_value_type_records,
    canonical_scope_records,
    canonical_artifact_envelope_error,
    computed_guard_projection_supported,
    is_canonical_ordinal,
    require_canonical_artifact_envelope,
)
from intentir.ir.dim_expr import validate_dimension_instantiation
from intentir.ir.intent_op_contract import (
    CANONICAL_J_OPS,
    STRUCTURED_INTENT_OPS,
    is_closed_boolean_dtype,
    validate_intent_op_structure,
)
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.source_projection_contracts import (
    matching_source_semantic_obligations,
    provider_dimension_alias_map,
    semantic_obligation_diagnostic,
    validate_source_semantic_obligations,
)
from intentir.verify.gate1.model import (
    GATE1_ARTIFACT_MODEL_FIELDS,
    AliasSpec,
    Boundary,
    Gate1Artifact,
    ImplementationConfig,
    ParameterKind,
    ScheduleSlot,
    ScheduleSlotKind,
    SemanticArchive,
    SemanticArchiveCrossContractError,
    SchemaError,
    gate1_artifact_wire_diagnostics,
)
from intentir.verify.gate1.pi import (
    NORMALIZATION_CLASS_SCOPES,
    RealizationWitness,
)
from intentir.verify.typecheck import TYPECHECK_PROTOCOL, type_check

from .scrc_schema import (
    canonical_semantic_candidate_contract,
    canonical_semantic_candidate_template,
    canonical_semantic_mcp_write_protocol,
    canonical_source_wire_seed,
    canonical_source_wire_bundle,
    canonical_wire_bundle,
    render_canonical_semantic_candidate_layout_summary,
    render_canonical_semantic_operation_namespace_summary,
    scrc_lift_core_json_schema,
    scrc_semantic_candidate_text_format,
    unique_access_guard_projection_axes,
)
from .direct_agent_context import build_direct_agent_mcp_snapshot


SCRC_RELIFT_PROTOCOL = "llm_direct_pi_structured_source_aware_relift_v2"
SCRC_DECODE_PROTOCOL = "intentir.scrc.decode.v18"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
_CANDIDATE_DIAGNOSTIC_LIMIT = 20
_CANDIDATE_DIAGNOSTIC_TEXT_LIMIT = 2048
MAX_RELIFT_CANDIDATE_BYTES = 256 * 1024
# A complete semantic J body needs room for its operation/access/output rows,
# so keep one bounded Lift output budget instead of relying on a provider
# default that may truncate the JSON candidate.
SCRC_LIFT_MAX_OUTPUT_TOKENS = 4096
# Lifecycle state remains cumulative on the host, but a fresh provider turn
# receives only this bounded guidance projection.  Without a shared bound,
# Responses/Codex relifts accumulated every historical rule row and gradually
# diluted the current repair signal.
_MAX_PROVIDER_GUIDANCE_ROWS = 24
_MAX_PROVIDER_GUIDANCE_EVIDENCE = 12
_MAX_PROVIDER_DIAGNOSTIC_KEYS = 32
_MAX_PROVIDER_DIAGNOSTIC_ITEMS = 16
_MAX_PROVIDER_DIAGNOSTIC_TEXT = 2048
_MAX_PROVIDER_DIAGNOSTIC_DEPTH = 5
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DTYPE_ALIASES = {
    "float16": "f16",
    "half": "f16",
    "float32": "f32",
    "float": "f32",
    "float64": "f64",
    "double": "f64",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "i1",
}


def resolve_source_aware_lift_timeout(
    effort: str,
    requested: float | None = None,
) -> float | None:
    """Resolve one source-aware Lift provider turn.

    No default framework deadline is imposed.  ``None`` means that the caller
    intentionally selected the unbounded Lift contract; a positive finite
    ``requested`` value remains available as an explicit experiment-local
    override.
    """

    if not isinstance(effort, str) or not effort.strip():
        raise ValueError("Lift reasoning effort must be a non-empty string")
    if requested is None:
        return None
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(float(requested))
        or requested <= 0
    ):
        raise ValueError("Lift timeout_seconds must be a finite positive number")
    return float(requested)


def _canonical_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPE_ALIASES.get(spelling, spelling)


def canonical_scalar_binding_dtype(value: bool | int | float | str) -> str:
    """Return the deterministic public dtype for one trusted scalar binding."""

    if isinstance(value, bool):
        return "i1"
    if isinstance(value, int):
        return "i32" if -(2**31) <= value < 2**31 else "i64"
    if isinstance(value, float):
        return "f64"
    if isinstance(value, str):
        return "str"
    raise TypeError("Source bindings must be bool/int/float/string scalars")


@dataclass(frozen=True)
class SCRCHarnessBinding:
    """Privileged source-name sidecar; never enters J/C/pi or Replay."""

    source_to_opaque: tuple[tuple[str, str], ...]
    source_function: str
    replay_function: str = "scrc_k0"

    def opaque_name(self, source_name: str) -> str:
        return dict(self.source_to_opaque)[source_name]


@dataclass(frozen=True)
class SCRCProvenance:
    model: str
    prompt_sha256: str
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    backend: str = "responses"
    termination_status: str | None = None
    trajectory: Mapping[str, Any] | None = field(default=None, repr=False)
    deterministic_repairs: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SCRCExtraction:
    artifact: Gate1Artifact
    provenance: SCRCProvenance
    harness_binding: SCRCHarnessBinding
    source: str = field(repr=False)
    construction: CanonicalConstruction = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.construction, CanonicalConstruction):
            raise TypeError("SCRC extraction requires canonical construction")


@dataclass(frozen=True)
class SCRCLiftRequest:
    """Exact source-anonymized prompt material before the single provider call."""

    instructions: str = field(repr=False)
    user_input: str = field(repr=False)
    prompt_sha256: str
    harness_binding: SCRCHarnessBinding
    source_import_roots: tuple[str, ...] = ()
    # Exact host-only Source compilation inputs.  These remain in-process and
    # are never included in the anonymized provider prompt, artifact, or
    # Replay packet.  The Source occurrence ledger must compare compiler debug
    # locations in this same coordinate space, not in an ast.unparse rendering.
    source_text: str = field(default="", repr=False)
    source_compiler_view: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    # Host-only construction authority.  It is never serialized into the
    # provider-facing artifact or Replay packet; Source-aware ingress and all
    # deterministic repairs consume this same fact set.
    source_facts: SourceFacts = field(default=None, repr=False)
    construction: CanonicalConstruction = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_facts, SourceFacts):
            raise TypeError("SCRC Lift request requires canonical SourceFacts")
        if not isinstance(self.construction, CanonicalConstruction):
            raise TypeError("SCRC Lift request requires canonical construction")
        if self.construction.source_facts.digest != self.source_facts.digest:
            raise ValueError("SCRC Lift construction differs from SourceFacts")
        if self.construction.source_facts.to_json() != self.source_facts.to_json():
            raise ValueError("SCRC Lift construction is not the same SourceFacts authority")
        if not isinstance(self.source_text, str) or not self.source_text:
            raise ValueError("SCRC Lift request requires exact Source text")
        if not isinstance(self.source_compiler_view, Mapping):
            raise TypeError("SCRC Lift request requires the exact Source compiler view")
        try:
            canonicalized_view = SourceFacts.from_compiler_view(
                self.source_compiler_view,
                frontend=self.source_facts.frontend,
                source_kind=self.source_facts.source_kind,
                source_symbol=self.source_facts.source_symbol,
                source_to_opaque=dict(self.source_facts.source_to_opaque),
                source_digest=self.source_facts.source_digest,
                source_slice_digest=self.source_facts.source_slice_digest,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                "SCRC Lift request compiler view cannot be canonicalized"
            ) from exc
        if canonicalized_view.to_json() != self.source_facts.to_json():
            raise ValueError(
                "SCRC Lift request compiler view differs from SourceFacts authority"
            )
        json.dumps(dict(self.source_compiler_view), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class SCRCCandidateCheckContext:
    """Source-free, versioned context visible to the local CandidateCheck."""

    expected_parameters: tuple[str, ...]
    expected_tensors: tuple[str, ...]
    expected_dimensions: tuple[str, ...]
    # The Source-owned allocator is the root of the candidate allocator used
    # by the canonical construction transaction.  CandidateCheck keeps this
    # authority instead of reconstructing namespace counts from raw JSON.
    allocator: CanonicalHandleAllocator = field(repr=False)
    trusted_compiler_facts: Mapping[str, Any] = field(repr=False)
    schema_version: str = Gate1Artifact.SCHEMA
    decode_protocol: str = SCRC_DECODE_PROTOCOL
    typecheck_protocol: str = TYPECHECK_PROTOCOL

    def __post_init__(self) -> None:
        if any(not re.fullmatch(r"p[0-9]+", item) for item in self.expected_parameters):
            raise ValueError("CandidateCheck parameters must be anonymous p* handles")
        if not isinstance(self.allocator, CanonicalHandleAllocator):
            raise TypeError("CandidateCheck allocator must be canonical")
        if not isinstance(self.trusted_compiler_facts, Mapping):
            raise TypeError("CandidateCheck compiler facts must be a mapping")
        value = dict(self.trusted_compiler_facts)
        json.dumps(value, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "trusted_compiler_facts", value)


def scrc_candidate_check_context(
    request: SCRCLiftRequest,
) -> SCRCCandidateCheckContext:
    """Project a Lift request to the exact source-free checker context."""

    allocator = request.construction.allocator
    expected_parameters = allocator.declared("p")
    trusted_view = request.construction.source_facts.compiler_view()
    return SCRCCandidateCheckContext(
        expected_parameters=expected_parameters,
        expected_tensors=allocator.declared("t"),
        expected_dimensions=allocator.declared("x"),
        allocator=allocator,
        trusted_compiler_facts=trusted_view,
        typecheck_protocol=TYPECHECK_PROTOCOL,
    )


@dataclass(frozen=True)
class SCRCReliftContext:
    """Rule-level prior diagnostics visible to one fresh Lift lifecycle."""

    relift_round: int
    failure_kind: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    open_guidance: tuple[Mapping[str, Any], ...] = ()
    resolved_guidance: tuple[Mapping[str, Any], ...] = ()
    candidate_delta: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.relift_round < 1:
            raise ValueError("relift_round must be positive")
        if self.failure_kind not in {
            "lift",
            "typecheck",
            "gate1",
        }:
            raise ValueError("unsupported relift failure kind")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("relift diagnostics must be a mapping")
        diagnostics = dict(self.diagnostics)
        # The payload is sent only to Lift, but it must still be deterministic
        # JSON data rather than an executable callback or arbitrary object.
        json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "diagnostics", diagnostics)
        for field_name in ("open_guidance", "resolved_guidance"):
            value = getattr(self, field_name)
            if value is None:
                value = ()
            object.__setattr__(
                self,
                field_name,
                _guidance_rows(value, strict=True),
            )
        if self.candidate_delta is not None:
            object.__setattr__(
                self,
                "candidate_delta",
                normalize_scrc_candidate_delta(self.candidate_delta),
            )

    def to_json(self) -> dict[str, Any]:
        # Keep the complete evidence in this lifecycle object, while presenting
        # repeated Source-transfer diagnostics to a provider as one coordinated
        # repair obligation. The projection changes no acceptance evidence and
        # never omits an individual observed fact.
        provider_open_guidance = _provider_guidance_projection(self.open_guidance)
        provider_resolved_guidance = _provider_guidance_projection(
            self.resolved_guidance
        )
        return {
            "relift_round": self.relift_round,
            "failure_kind": self.failure_kind,
            # Keep complete diagnostics in the host-side context, but expose
            # only the bounded provider projection on every backend wire.
            "diagnostics": _provider_diagnostics_projection(self.diagnostics),
            "candidate_delta": (
                dict(self.candidate_delta)
                if self.candidate_delta is not None else None
            ),
            "open_guidance": [dict(item) for item in provider_open_guidance],
            "resolved_guidance": [dict(item) for item in provider_resolved_guidance],
            "deterministic_repair_checklist": _relift_repair_checklist(
                self.diagnostics,
                provider_open_guidance,
            ),
        }


def _relift_feedback_value(feedback: Any, name: str, default: Any = None) -> Any:
    """Read one transport-neutral relift field from an object or mapping."""

    if isinstance(feedback, Mapping):
        return feedback.get(name, default)
    return getattr(feedback, name, default)


def _guidance_rows(
    value: Any,
    *,
    strict: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Copy one transport-provided guidance sequence without interpreting it."""

    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        if strict:
            raise TypeError("relift guidance must be a sequence")
        return ()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    row_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            if strict:
                raise TypeError("relift guidance items must be mappings")
            continue
        row = dict(item)
        key = (str(row.get("rule_code", "")), str(row.get("path", "")))
        if key in seen:
            existing = row_by_key[key]
            def locator_items(raw: Any) -> list[Any]:
                if isinstance(raw, (list, tuple)):
                    return list(raw)
                return [raw] if raw is not None else []

            locators = safe_source_repair_locators(
                [
                    *locator_items(existing.get("repair_locators")),
                    *locator_items(existing.get("repair_locator")),
                    *locator_items(row.get("repair_locators")),
                    *locator_items(row.get("repair_locator")),
                ]
            )
            existing.pop("repair_locator", None)
            if locators:
                existing["repair_locators"] = locators
            constraints = safe_expected_constraints(
                [
                    *(existing.get("expected_constraints", ())),
                    *(row.get("expected_constraints", ())),
                    *(existing.get("observed_refs_or_types", ())),
                    *(row.get("observed_refs_or_types", ())),
                ],
                accept_unmarked=True,
            )
            if constraints:
                existing["expected_constraints"] = constraints
            continue
        seen.add(key)
        json.dumps(row, sort_keys=True, allow_nan=False)
        rows.append(row)
        row_by_key[key] = row
    return tuple(rows)


def _merge_guidance_rows(
    previous: Any,
    current: Any,
) -> tuple[dict[str, Any], ...]:
    """Append transport state without allowing a backend to forget it."""

    return _guidance_rows([*_guidance_rows(previous), *_guidance_rows(current)])


_SOURCE_TRANSFER_GUIDANCE_CODES = frozenset(
    {
        "compiled_transfer_direct_count",
        "j_masked_load_presence",
        "j_masked_store_presence",
        "source_access_fill_occurrence_unproved",
        "source_access_fill_presence_unproved",
        "source_access_fill_presence_mismatch",
        "source_boundary_effect_order",
    }
)


def _provider_guidance_projection(
    guidance: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Compress repeated Source-transfer facts without discarding evidence.

    SourceReconcile may report transfer count, masking, fill availability, and
    effect ordering independently. They all constrain one ordered J boundary
    ledger. Keeping them as separate provider tasks encourages piecemeal edits;
    this projection retains every row in ``evidence`` but exposes one repair
    obligation. The unprojected lifecycle state remains available to runner
    diagnostics and evidence serialization.
    """

    def rule_path_projection(item: Mapping[str, Any]) -> dict[str, Any]:
        # Guidance and diagnostics are the same host-owned reject-only
        # evidence class.  Reusing the diagnostics projector prevents a
        # second, permanently incomplete whitelist from silently dropping
        # safe typed relations (for example expected shape/dtype) during
        # relift.  The shared projector still removes candidate/source/row
        # payloads and applies the same bounded depth/key/item rules.
        projected = _provider_diagnostics_projection(item)
        return projected if isinstance(projected, dict) else {}

    ordinary: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for raw in guidance:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        if item.get("rule_code") in _SOURCE_TRANSFER_GUIDANCE_CODES:
            transfer_rows.append(item)
        else:
            projected = rule_path_projection(item)
            if projected:
                ordinary.append(projected)

    def bound_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        if len(rows) <= _MAX_PROVIDER_GUIDANCE_ROWS:
            return tuple(rows)
        # Keep the earliest root/wire guard when present and the newest
        # guidance rows, which are the rows most relevant to the current
        # relift. The complete cumulative sequence remains host-side.
        head = rows[:1]
        tail_count = max(0, _MAX_PROVIDER_GUIDANCE_ROWS - len(head) - 1)
        tail = rows[-tail_count:] if tail_count else []
        omitted = len(rows) - len(head) - len(tail)
        return tuple(
            [
                {
                    "rule_code": "relift_guidance_history_compacted",
                    "path": "relift.open_guidance/resolved_guidance",
                    "omitted_count": omitted,
                },
                *head,
                *tail,
            ][: _MAX_PROVIDER_GUIDANCE_ROWS]
        )

    if not transfer_rows:
        return bound_rows(ordinary)

    evidence: list[dict[str, Any]] = []
    for item in transfer_rows[:_MAX_PROVIDER_GUIDANCE_EVIDENCE]:
        projected = rule_path_projection(item)
        if projected:
            evidence.append(projected)
    omitted_transfer_rows = max(
        0, len(transfer_rows) - _MAX_PROVIDER_GUIDANCE_EVIDENCE
    )
    if omitted_transfer_rows:
        evidence.append(
            {
                "rule_code": "source_transfer_ledger_history_compacted",
                "path": "j.boundary.accesses+effects",
                "omitted_count": omitted_transfer_rows,
            }
        )
    ordinary.append(
        {
            "rule_code": "source_transfer_ledger_reconcile",
            "path": "j.boundary.accesses+effects",
            "evidence": evidence,
        }
    )
    return bound_rows(ordinary)


_PROVIDER_DIAGNOSTIC_OMIT_KEYS = frozenset(
    {
        # These are lifecycle/provenance or recursive observer payloads.  The
        # host keeps them in SCRCReliftContext.diagnostics; a fresh provider
        # gets the bounded category/frontier views instead.
        "artifact",
        "candidate",
        "candidate_draft",
        "candidate_text",
        "decoded_candidate",
        "draft",
        "previous_artifact",
        "raw_response",
        "raw_response_text",
        "repair_base",
        "repair_base_candidate_check_digest",
        "response_text",
        "replay_counterexample",
        "trajectory",
        "node_witnesses",
        "observer_hashes",
        "ttir",
        "source_ttir",
        "source_code",
        "source_program",
        "source_text",
        "full_source",
        "candidate_history",
        "observed_refs_or_types",
        "anonymous_handles",
        "expected",
        "actual",
        "value",
        "row",
        "rows",
        "operation_row",
        "access_row",
        "effect_row",
        "declaration",
        "body",
        "operands",
        "attrs",
        "examples",
        "effect_order_witnesses",
        "local_frontier",
    }
)


_SAFE_EXPECTED_CONSTRAINT = re.compile(r"[A-Za-z0-9_.*\[\](),:=<>+ /\-]{1,256}\Z")


def safe_expected_constraints(
    value: Any, *, accept_unmarked: bool = False
) -> list[str]:
    """Extract bounded host-authored constraints from generic reject diagnostics.

    ``observed_refs_or_types`` mixes checker-authored expectations with
    candidate-derived observations. Only the explicitly marked expected
    clauses cross the provider boundary; observed values and candidate text
    remain private. This is rule-agnostic so new validators do not need a
    parallel allowlist to make useful repair guidance available.
    """

    if not isinstance(value, (list, tuple)):
        return []
    constraints: list[str] = []
    for item in value[:_MAX_PROVIDER_DIAGNOSTIC_ITEMS]:
        if not isinstance(item, str):
            continue
        if item.startswith("expected:"):
            clause = item[len("expected:") :].strip()
        elif accept_unmarked:
            clause = item.strip()
        else:
            continue
        if not _SAFE_EXPECTED_CONSTRAINT.fullmatch(clause):
            continue
        if clause not in constraints:
            constraints.append(clause)
    return constraints


def safe_source_repair_locator(value: Any) -> dict[str, Any] | None:
    """Keep only Source-issued locators needed to repair a rejected relation.

    Candidate-local ids and operation rows are intentionally not accepted
    here. A fresh Lift maps these stable Source graph positions onto its own
    newly authored local labels.
    """

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    access_label = value.get("access_label")
    ordinal = value.get("ordinal")
    if isinstance(access_label, str) and re.fullmatch(r"access[0-9]+", access_label):
        result["access_label"] = access_label
    if type(ordinal) is int and ordinal >= 0:
        result["ordinal"] = ordinal
    raw_requirements = value.get("owner_broadcast_requirements")
    if isinstance(raw_requirements, list):
        requirements: list[dict[str, Any]] = []
        for row in raw_requirements[:_MAX_PROVIDER_DIAGNOSTIC_ITEMS]:
            if not isinstance(row, Mapping):
                continue
            consumer = row.get("consumer_graph_node")
            operand = row.get("operand_position")
            axes = row.get("target_axes")
            if (
                type(consumer) is not int or consumer < 0
                or type(operand) is not int or operand < 0
                or not isinstance(axes, list)
                or any(type(axis) is not int or axis < 0 for axis in axes)
                or len(set(axes)) != len(axes)
            ):
                continue
            requirements.append(
                {
                    "consumer_graph_node": consumer,
                    "operand_position": operand,
                    "target_axes": list(axes),
                }
            )
        result["owner_broadcast_requirements"] = requirements
    predicate_requirement = value.get("predicate_requirement")
    if isinstance(predicate_requirement, str) and len(predicate_requirement) <= 512:
        result["predicate_requirement"] = predicate_requirement
    return result or None


def safe_source_repair_locators(value: Any) -> list[dict[str, Any]]:
    """Validate and de-duplicate one or more Source-only repair locators."""

    candidates = value if isinstance(value, (list, tuple)) else (value,)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        locator = safe_source_repair_locator(candidate)
        if locator is None:
            continue
        key = json.dumps(locator, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        result.append(locator)
        if len(result) >= _MAX_PROVIDER_DIAGNOSTIC_ITEMS:
            break
    return result


def relift_repair_instruction(rule_code: Any, path: Any) -> str | None:
    """Map a closed diagnostic code to one safe, field-specific repair action."""

    if not isinstance(rule_code, str):
        return None
    if rule_code == "source_owner_broadcast_relation_missing":
        return (
            "For each Source locator in this access row's owner_broadcast_requirements, "
            "map consumer_graph_node and operand_position onto the fresh graph's local "
            "consumer result and operand result. Add exactly one j.boundary.broadcasts "
            "row per locator, with axes copied exactly from target_axes; do not insert "
            "a broadcast operation or infer a new relation."
        )
    if rule_code == "source_owner_broadcast_relation_ambiguous":
        return (
            "Use the Source owner_broadcast_requirements locator to identify the one "
            "matching consumer/operand edge. Keep exactly one j.boundary.broadcasts "
            "row for that fresh local edge and copy its target_axes exactly; do not "
            "choose among candidate rows by shape alone."
        )
    if rule_code.startswith("source_owner_broadcast_"):
        return (
            "Repair only the j.boundary.broadcasts relation identified by the Source "
            "owner_broadcast_requirements locator: bind its fresh local consumer and "
            "operand, and use the exact Source target_axes. Keep the original pointwise "
            "operand and fail closed if that mapping is not exact."
        )
    if rule_code == "source_owner_computed_guard_relation_missing":
        return (
            "For the Source access_label in this locator, add exactly one "
            "j.boundary.access_guard_relations row: access_ref is that access label, "
            "value_ref is its fresh local Source-compatible i1 result, polarity is "
            "positive, and axes come only from the Source-proven target-axis relation. "
            "Set that access row's predicate to the literal string 'true'; put the i1 "
            "result only in value_ref."
        )
    if rule_code == "source_owner_computed_guard_value_unproved":
        return (
            "For the named Source access, set j.boundary.accesses[].predicate to the "
            "literal string 'true'. Keep the computed i1 result only in the matching "
            "access_guard_relations.value_ref; do not place the local value label in "
            "predicate. Preserve the issued Source guard authorization."
        )
    if rule_code == "source_owner_computed_guard_projection_unproved":
        return (
            "Repair the matching j.boundary.access_guard_relations row using the "
            "Source-proven access target-axis insertion exactly; do not reuse the "
            "guard's retained axis or axes from a separate broadcast relation."
        )
    if rule_code.startswith("source_owner_computed_guard_"):
        return (
            "Use only the matching transfer row's predicate_requirement and existing "
            "semantic_guard_authorization. Bind the required fresh local i1 producer "
            "through one access_guard_relations row, with the access predicate set to "
            "'true'; if Source facts do not prove the relation, leave it unclaimed."
        )
    if rule_code.startswith("gate1."):
        category = rule_code.removeprefix("gate1.")
        gate1_actions = {
            "interface": "Check J.signature, output bindings, and region/result interfaces first; repair the first missing or extra interface edge before downstream values.",
            "predicate_effect": "Check Source-visible predicates, masks, effect/value arms, and scope effect membership; preserve the complete control/effect construct and ordering.",
            "op_multiset": "Compare operation and memory-transfer multiplicity in Source order; do not merge, duplicate, or remove an operation merely to match Replay.",
            "dataflow": "Start at the first causal edge in the bounded Gate1 frontier; audit the complete producer/consumer and effect chain before downstream arithmetic.",
            "reduction": "Check reduction axes, keepdims, associativity, identity, and recurrence together; do not replace the recurrence with an algebraically similar result.",
        }
        return gate1_actions.get(category)
    if rule_code.startswith("candidate_decode_j_boundary_cross_contract_"):
        return (
            "Repair only the named Boundary-to-J cross-contract at the reported path. "
            "Use the mounted canonical wire and current Source facts; preserve the "
            "operation graph and do not copy or reconstruct the rejected artifact."
        )
    if rule_code == "candidate_decode_j_boundary_wire_model_rejected":
        return (
            "Repair the Boundary row grammar at the reported path using the mounted "
            "schema. Do not add host-owned ids or fields, and resubmit a complete fresh J."
        )
    return None


def _provider_diagnostics_projection(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded provider view of one host-side diagnostic object.

    Relift diagnostics are evidence, not a second artifact authority.  The
    complete object stays on the host for closure records; a fresh provider
    only needs the current typed rows and bounded local context.  In
    particular, recursive observer witnesses and raw response/source payloads
    must not be reintroduced through ``SCRCReliftContext.to_json`` after the
    guidance projection was compacted.
    """

    def compact(value: Any, depth: int = 0) -> Any:
        if isinstance(value, str):
            if len(value) <= _MAX_PROVIDER_DIAGNOSTIC_TEXT:
                return value
            marker = "...[provider diagnostic text truncated]"
            return value[: _MAX_PROVIDER_DIAGNOSTIC_TEXT - len(marker)] + marker
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= _MAX_PROVIDER_DIAGNOSTIC_DEPTH:
            return "...[provider diagnostic depth truncated]"
        if isinstance(value, Mapping):
            mapping_keys = {str(key) for key in value}
            if {"j", "c", "pi"} <= mapping_keys:
                # Do not rely on the surrounding key spelling.  A backend may
                # place a rejected canonical body under a generic ``details``
                # field; recognize the artifact shape itself and replace it
                # with a bounded omission receipt before provider transport.
                return {"_provider_omitted_artifact": True}
            if (
                "op" in mapping_keys
                and mapping_keys
                & {"operands", "result", "results", "type", "attrs"}
            ):
                return {"_provider_omitted_operation_row": True}
            if "access_id" in mapping_keys or "effect_id" in mapping_keys:
                return {"_provider_omitted_boundary_row": True}
            result: dict[str, Any] = {}
            omitted = 0
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if key == "observed_refs_or_types":
                    constraints = safe_expected_constraints(raw_value)
                    if constraints:
                        result["expected_constraints"] = constraints
                    else:
                        omitted += 1
                    continue
                if key == "expected_constraints":
                    constraints = safe_expected_constraints(
                        raw_value, accept_unmarked=True
                    )
                    if constraints:
                        result[key] = constraints
                    else:
                        omitted += 1
                    continue
                if key in _PROVIDER_DIAGNOSTIC_OMIT_KEYS:
                    omitted += 1
                    continue
                if key == "repair_locator":
                    safe_locator = safe_source_repair_locator(raw_value)
                    if safe_locator is not None:
                        result[key] = safe_locator
                    else:
                        omitted += 1
                    continue
                if key == "repair_locators":
                    safe_locators = safe_source_repair_locators(raw_value)
                    if safe_locators:
                        result[key] = safe_locators
                    else:
                        omitted += 1
                    continue
                if key == "diagnostic" and (
                    not isinstance(raw_value, str)
                    or re.fullmatch(r"[A-Za-z0-9_.:+\-/]+", raw_value) is None
                ):
                    omitted += 1
                    continue
                if len(result) >= _MAX_PROVIDER_DIAGNOSTIC_KEYS:
                    omitted += 1
                    continue
                result[key] = compact(raw_value, depth + 1)
            instruction = relift_repair_instruction(
                result.get("rule_code"), result.get("path")
            )
            if instruction is not None and "repair_instruction" not in result:
                result["repair_instruction"] = instruction
            if omitted:
                result["_provider_omitted_fields"] = omitted
            return result
        if isinstance(value, (list, tuple)):
            bounded = [
                compact(item, depth + 1)
                for item in value[:_MAX_PROVIDER_DIAGNOSTIC_ITEMS]
            ]
            omitted = len(value) - len(bounded)
            if omitted:
                bounded.append(
                    f"...[{omitted} provider diagnostic item(s) omitted]"
                )
            return bounded
        return str(value)

    projected = compact(dict(diagnostics))
    return projected if isinstance(projected, dict) else {}


_CANDIDATE_DELTA_LIST_FIELDS = frozenset(
    {"root_keys", "missing_root_keys", "extra_root_keys"}
)
_MAX_CANDIDATE_DELTA_LIST_ITEMS = 48
_MAX_CANDIDATE_DELTA_TEXT = 256


def normalize_scrc_candidate_delta(
    value: Any,
) -> dict[str, Any] | None:
    """Keep only the bounded reject-only candidate-delta receipt.

    ``candidate_delta`` is lifecycle diagnostics, never an alternate artifact
    channel.  Normalizing it here gives Responses, Codex, MiniSWE, and the
    runner the same wire boundary: candidate bodies, nested objects, and
    backend-specific fields cannot be smuggled into the next provider turn.
    """

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    authority = value.get("authority")
    if isinstance(authority, str):
        result["authority"] = authority[:_MAX_CANDIDATE_DELTA_TEXT]
    for key in _CANDIDATE_DELTA_LIST_FIELDS:
        item = value.get(key)
        if not isinstance(item, (list, tuple)):
            continue
        result[key] = [
            item_value[:_MAX_CANDIDATE_DELTA_TEXT]
            for item_value in item[:_MAX_CANDIDATE_DELTA_LIST_ITEMS]
            if isinstance(item_value, str)
        ]
    if not result:
        return None
    return result


def scrc_candidate_delta_receipt(
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce candidate bytes immediately to a source-free root-shape receipt."""

    if not isinstance(candidate, Mapping):
        return None
    root_keys = {str(key) for key in candidate}
    expected_root = set(GATE1_ARTIFACT_MODEL_FIELDS)
    return normalize_scrc_candidate_delta(
        {
            "root_keys": sorted(root_keys),
            "missing_root_keys": sorted(expected_root - root_keys),
            "extra_root_keys": sorted(root_keys - expected_root),
            "authority": (
                "diagnostic-only structural receipt; candidate bytes are "
                "discarded before crossing the Lift backend boundary"
            ),
        }
    )


def _relift_repair_checklist(
    diagnostics: Mapping[str, Any],
    open_guidance: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    """Render a bounded, deterministic repair order for the next Lift turn.

    Raw candidates remain diagnostic-only.  The next provider turn still needs
    one compact rendering of the current reject-only observations. Rows
    absent from the current CandidateCheck are retained as ``preserve``
    constraints rather than being presented as fresh repairs.
    """

    def rows_from(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            code = item.get("rule_code")
            path = item.get("path")
            if not isinstance(code, str) or not code:
                continue
            if not isinstance(path, str) or not path:
                path = "<candidate>"
            row = {"rule_code": code, "path": path}
            constraints = safe_expected_constraints(
                item.get("expected_constraints", ()),
                accept_unmarked=True,
            )
            constraints.extend(
                safe_expected_constraints(item.get("observed_refs_or_types"))
            )
            if constraints:
                row["expected_constraints"] = constraints
            locators = safe_source_repair_locators(
                item.get("repair_locators", item.get("repair_locator"))
            )
            if locators:
                row["repair_locators"] = locators
            instruction = item.get("repair_instruction")
            if isinstance(instruction, str) and len(instruction) <= 1024:
                row["repair_instruction"] = instruction
            result.append(row)
        return result

    candidate_check = diagnostics.get("candidate_check")
    current_rows = rows_from(
        candidate_check.get("diagnostics")
        if isinstance(candidate_check, Mapping)
        else None
    )
    current_keys = {
        (item["rule_code"], item["path"])
        for item in current_rows
    }

    # Relift uses the same semantic submission surface as the first Lift.
    # The former checklist below was written for a provider editing a complete
    # canonical artifact; it therefore instructed a fresh worker to repair
    # host-owned schema/C/node/scope/access/effect-identity/pi fields. Keep only a
    # compact semantic responsibility per diagnostic here.  The host rebuilds
    # every omitted sibling again from SourceFacts and the transfer ledger.
    def semantic_instruction(item: Mapping[str, Any]) -> str:
        code = str(item["rule_code"])
        specific = item.get("repair_instruction") or relift_repair_instruction(
            code, item.get("path")
        )
        if isinstance(specific, str):
            return specific
        if code in _SOURCE_TRANSFER_GUIDANCE_CODES or code.startswith("source_"):
            return (
                "Repair the named semantic boundary access occurrence: preserve its "
                "Source load/store kind, predicate, fill, and paired effect "
                "kind/target/node semantics. Keep one unique local accessN label; "
                "host derives only canonical access/effect identity, C owner, and "
                "π binding from the transfer ledger."
            )
        if "gather" in code or "iota" in code or "index_bound" in code:
            return (
                "Repair only the named typed semantic operation/access relation using "
                "the schema-derived op/attrs contract. Do not add physical-capacity "
                "iota/gather semantics or any host-generated index/effect identifier."
            )
        if "output" in code or "return" in code:
            return (
                "Repair the ordered semantic output value selection so each chosen "
                "local result or Source t* has the Source-declared output type. The "
                "host derives output names, producer attachment, and returns."
            )
        if "scope" in code or "handle" in code or "node" in code:
            return (
                "Repair the semantic operation/region wiring with provider-local "
                "labels and visible SSA only. Do not author canonical node/value, "
                "scope, access, or effect identities; host construction rebuilds them."
            )
        return (
            "Repair only the named semantic J field according to the mounted "
            "canonical provider wire. Do not reconstruct schema, C, ABI, canonical "
            "handles, scopes, effect identities, returns, or derived π bindings."
        )

    checklist: list[dict[str, Any]] = [
        {
            "priority": 0,
            "mode": "repair_or_preserve",
            "rule_code": "semantic_provider_wire",
            "path": "j + optional pi.normalization_classes",
            "instruction": (
                "Submit only the semantic candidate surface: J operation/boundary "
                "meaning plus optional J-proven normalization_classes. The complete "
                "canonical artifact is rebuilt and checked by the host."
            ),
        }
    ]
    for item in current_rows:
        checklist_item = {
            "priority": 10,
            "mode": "repair",
            "rule_code": item["rule_code"],
            "path": item["path"],
            "instruction": semantic_instruction(item),
        }
        if "expected_constraints" in item:
            checklist_item["expected_constraints"] = item[
                "expected_constraints"
            ]
        if "repair_locators" in item:
            checklist_item["repair_locators"] = item["repair_locators"]
        checklist.append(checklist_item)
    seen_preserve: set[tuple[str, str]] = set(current_keys)
    for raw in open_guidance:
        if not isinstance(raw, Mapping):
            continue
        code = raw.get("rule_code")
        path = raw.get("path")
        if not isinstance(code, str) or not isinstance(path, str):
            continue
        key = (code, path)
        if key in seen_preserve:
            continue
        seen_preserve.add(key)
        checklist.append(
            {
                "priority": 50,
                "mode": "preserve",
                "rule_code": code,
                "path": path,
                "instruction": (
                    str(raw.get("repair_instruction"))
                    if isinstance(raw.get("repair_instruction"), str)
                    else "Preserve this earlier semantic obligation unless the "
                    "current diagnostic names the same semantic path."
                ),
            }
        )
        locators = safe_source_repair_locators(
            raw.get("repair_locators", raw.get("repair_locator"))
        )
        constraints = safe_expected_constraints(
            raw.get("expected_constraints", ()), accept_unmarked=True
        )
        constraints.extend(
            safe_expected_constraints(raw.get("observed_refs_or_types"))
        )
        if constraints:
            checklist[-1]["expected_constraints"] = constraints
        if locators:
            checklist[-1]["repair_locators"] = locators
        if len(checklist) >= 13:
            break
    return checklist[:13]


def scrc_relift_context_from_feedback(
    feedback: Any,
    *,
    relift_round: int,
    default_failure_kind: str = "lift",
) -> SCRCReliftContext | None:
    """Normalize feedback into rule-level guidance for a fresh lifecycle.

    Neither a complete candidate nor a rejected draft is copied into this
    provider-facing lifecycle context.
    """

    if feedback is None:
        return None
    kind = _relift_feedback_value(feedback, "kind")
    kind = getattr(kind, "value", kind)
    if not isinstance(kind, str) or kind not in {"lift", "typecheck", "gate1", "gate2"}:
        kind = default_failure_kind
    diagnostics = _relift_feedback_value(feedback, "diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}

    open_guidance = _relift_feedback_value(feedback, "open_guidance", None)
    if not open_guidance:
        open_guidance = diagnostics.get("open_guidance", ())
    resolved_guidance = _relift_feedback_value(
        feedback, "resolved_guidance", None
    )
    if not resolved_guidance:
        resolved_guidance = diagnostics.get("resolved_guidance", ())
    return SCRCReliftContext(
        relift_round=max(1, int(relift_round)),
        failure_kind=kind,
        diagnostics=dict(diagnostics),
        open_guidance=_guidance_rows(open_guidance),
        resolved_guidance=_guidance_rows(resolved_guidance),
        candidate_delta=normalize_scrc_candidate_delta(
            _relift_feedback_value(feedback, "candidate_delta")
        ),
    )


@dataclass(frozen=True)
class SCRCReliftRecord:
    relift_round: int
    extraction: SCRCExtraction | None = None
    diagnostic: str | None = None
    backend_evidence: Mapping[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SCRCCompilerArgument:
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...] = ()
    value: bool | int | float | None = None
    mode: str | None = None
    shape_contract: tuple[int | str, ...] | None = None
    shape_coordinate: str | None = None
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("compiler argument name must be an identifier")
        if self.kind not in {"tensor", "scalar", "none", "opaque"}:
            raise ValueError(
                "compiler argument kind must be tensor/scalar/none/opaque"
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("compiler argument dtype must be non-empty")
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("compiler argument dimensions must be non-negative")
        strides = (
            tuple(int(stride) for stride in self.strides)
            if self.strides is not None
            else None
        )
        if strides is not None:
            if self.kind != "tensor":
                raise ValueError("only tensor compiler arguments have physical strides")
            if len(strides) != len(shape) or any(stride < 0 for stride in strides):
                raise ValueError(
                    "compiler argument strides must match tensor rank and be non-negative"
                )
        shape_contract = (
            tuple(self.shape_contract)
            if self.shape_contract is not None
            else None
        )
        if self.kind != "tensor" and shape_contract is not None:
            raise ValueError("only tensor compiler arguments have a shape contract")
        if shape_contract is not None:
            if len(shape_contract) != len(shape):
                raise ValueError(
                    "compiler argument shape contract rank differs from valuation"
                )
            for axis, (contract, extent) in enumerate(
                zip(shape_contract, shape, strict=True)
            ):
                if isinstance(contract, bool) or not isinstance(contract, (int, str)):
                    raise TypeError(
                        "shape contract entries must be positive integers or h* coordinates"
                    )
                if isinstance(contract, int):
                    if contract <= 0 or contract != extent:
                        raise ValueError(
                            "static shape contract extent differs from its valuation"
                        )
                elif not re.fullmatch(r"h[0-9]+", contract):
                    raise ValueError(
                        f"shape contract axis {axis} must use an anonymous h* coordinate"
                    )
        shape_coordinate = self.shape_coordinate
        if shape_coordinate is not None:
            if self.kind != "scalar":
                raise ValueError(
                    "only scalar compiler arguments bind a shape coordinate"
                )
            if not re.fullmatch(r"h[0-9]+", shape_coordinate):
                raise ValueError(
                    "scalar shape-coordinate binding must use an anonymous h* coordinate"
                )
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer valuation"
                )
            if self.dtype.casefold() != "index" and re.fullmatch(
                r"[iu][0-9]+", self.dtype.casefold()
            ) is None:
                raise ValueError(
                    "scalar shape-coordinate binding requires an integer dtype"
                )
        if self.kind == "tensor" and self.value is not None:
            raise ValueError("tensor compiler argument forbids scalar value")
        if self.kind == "scalar" and (shape or self.value is None):
            raise ValueError("scalar compiler argument requires value only")
        if self.kind == "none" and (shape or self.value is not None):
            raise ValueError("none compiler argument forbids shape/value")
        if self.kind == "opaque" and (shape or self.value is not None):
            raise ValueError("opaque compiler argument exposes only its public type")
        if self.mode is not None:
            allowed_modes = {
                "tensor": {"input", "output", "inout"},
                "scalar": {"value"},
                "none": {"value"},
                "opaque": {"value"},
            }[self.kind]
            if self.mode not in allowed_modes:
                raise ValueError(
                    f"compiler argument mode for {self.kind} must be one of "
                    + ", ".join(sorted(allowed_modes))
                )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "shape_contract", shape_contract)
        object.__setattr__(self, "shape_coordinate", shape_coordinate)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
        }
        if self.kind == "tensor":
            result["shape"] = list(self.shape)
            if self.shape_contract is not None:
                result["shape_contract"] = list(self.shape_contract)
            if self.strides is not None:
                result["strides"] = list(self.strides)
        elif self.kind == "scalar":
            result["value"] = self.value
            if self.shape_coordinate is not None:
                result["shape_coordinate"] = self.shape_coordinate
        if self.mode is not None:
            result["mode"] = self.mode
        return result


@dataclass(frozen=True)
class SCRCCompilerView:
    """Trusted invocation facts supplied to Lift."""

    schedule_values: Mapping[str, bool | int | float | str]
    schedule_value_types: Mapping[str, str] = field(default_factory=dict)
    bound_constexpr_values: Mapping[str, bool | int | float | str] = field(
        default_factory=dict
    )
    bound_constexpr_shape_coordinates: Mapping[str, str] = field(
        default_factory=dict
    )
    compiler_slots: tuple[str, ...] = ()
    aliases: tuple[AliasSpec, ...] = ()
    arguments: tuple[SCRCCompilerArgument, ...] = ()
    grid: tuple[int, ...] = ()
    grid_contract: tuple[str, ...] = ()
    grid_shape_coordinates: tuple[str | None, ...] = ()
    parameter_order: tuple[str, ...] = ()
    constexpr_parameters: tuple[str, ...] = ()
    # Kept after the established positional fields so existing callers that
    # construct this view positionally do not reinterpret a shape-coordinate
    # map as a type map.
    bound_constexpr_value_types: Mapping[str, str] = field(default_factory=dict)
    # Closed physical C roles. Callers may prove a Source constexpr to be a
    # tile carrier; all unproved Source constexprs remain constexpr_parameter.
    # Compiler-only options retain their public option name as role.
    schedule_roles: Mapping[str, str] = field(default_factory=dict)
    schedule_tile_coordinates: Mapping[str, str] = field(default_factory=dict)
    host_launch_contract: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.schedule_values)
        value_types = dict(self.schedule_value_types)
        bound_constexpr_values = dict(self.bound_constexpr_values)
        bound_constexpr_value_types = dict(self.bound_constexpr_value_types)
        bound_constexpr_shape_coordinates = dict(
            self.bound_constexpr_shape_coordinates
        )
        supplied_roles = dict(self.schedule_roles)
        tile_coordinates = dict(self.schedule_tile_coordinates)
        for collection, label in (
            (values, "schedule"),
            (bound_constexpr_values, "bound constexpr"),
        ):
            for name, value in collection.items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"{label} names must be identifiers")
                if not isinstance(value, (bool, int, float, str)):
                    raise TypeError(f"{label} values must be JSON scalars")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{label} float values must be finite")
        if set(values) & set(bound_constexpr_values):
            raise ValueError(
                "bound constexpr and schedule parameters must be disjoint"
            )
        if set(value_types) - set(values):
            raise ValueError(
                "schedule value types may name only declared schedule values"
            )
        if set(bound_constexpr_value_types) - set(bound_constexpr_values):
            raise ValueError(
                "bound constexpr value types may name only bound constexpr values"
            )
        ordered_value_types: dict[str, str] = {}
        for name, value in values.items():
            value_type = value_types.get(name, canonical_scalar_binding_dtype(value))
            if not isinstance(value_type, str) or not value_type:
                raise TypeError("schedule value types must be non-empty strings")
            ordered_value_types[name] = value_type
        value_types = ordered_value_types
        slots = tuple(self.compiler_slots)
        if len(slots) != len(set(slots)) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in slots
        ):
            raise ValueError("compiler slots must contain unique identifiers")
        if set(slots) - set(values):
            raise ValueError("compiler slots require schedule values")
        if set(supplied_roles) - set(values):
            raise ValueError("schedule roles may name only schedule values")
        if set(tile_coordinates) - set(values) or any(
            not isinstance(coordinate, str)
            or re.fullmatch(r"h[0-9]+", coordinate) is None
            for coordinate in tile_coordinates.values()
        ):
            raise ValueError(
                "schedule tile coordinates must bind schedule values to h*"
            )
        roles = {
            name: supplied_roles.get(
                name,
                name if name in slots else "constexpr_parameter",
            )
            for name in values
        }
        if any(
            not isinstance(role, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", role) is None
            for role in roles.values()
        ):
            raise ValueError("schedule roles must be closed lowercase identifiers")
        tile_role_names = {
            name
            for name, role in roles.items()
            if role in {"tile_extent", "reduction_tile_extent"}
        }
        if tile_role_names != set(tile_coordinates):
            raise ValueError(
                "closed tile roles and schedule tile coordinates must correspond"
            )
        object.__setattr__(self, "schedule_values", values)
        object.__setattr__(self, "schedule_value_types", value_types)
        object.__setattr__(
            self, "bound_constexpr_values", bound_constexpr_values
        )
        object.__setattr__(
            self, "bound_constexpr_value_types", bound_constexpr_value_types
        )
        object.__setattr__(
            self,
            "bound_constexpr_shape_coordinates",
            bound_constexpr_shape_coordinates,
        )
        object.__setattr__(self, "compiler_slots", slots)
        object.__setattr__(self, "schedule_roles", roles)
        object.__setattr__(self, "schedule_tile_coordinates", tile_coordinates)
        object.__setattr__(
            self,
            "aliases",
            tuple(
                item if isinstance(item, AliasSpec) else AliasSpec.from_json(item)
                for item in self.aliases
            ),
        )
        object.__setattr__(
            self,
            "arguments",
            tuple(
                item if isinstance(item, SCRCCompilerArgument)
                else SCRCCompilerArgument(**item)
                for item in self.arguments
            ),
        )
        if len({item.name for item in self.arguments}) != len(self.arguments):
            raise ValueError("compiler arguments must have unique names")
        shape_coordinates: dict[str, int] = {}
        coordinate_order: list[str] = []
        for argument in self.arguments:
            if argument.shape_contract is None:
                continue
            for contract, extent in zip(
                argument.shape_contract, argument.shape, strict=True
            ):
                if not isinstance(contract, str):
                    continue
                if contract not in shape_coordinates:
                    shape_coordinates[contract] = extent
                    coordinate_order.append(contract)
                elif shape_coordinates[contract] != extent:
                    raise ValueError(
                        f"shape coordinate {contract!r} has inconsistent valuations"
                    )
        expected_coordinates = [
            f"h{index}" for index in range(len(coordinate_order))
        ]
        if coordinate_order != expected_coordinates:
            raise ValueError(
                "shape coordinates must use gap-free first-declaration order h0,h1,..."
            )
        for argument in self.arguments:
            coordinate = argument.shape_coordinate
            if coordinate is None:
                continue
            if coordinate not in shape_coordinates:
                raise ValueError(
                    "scalar shape-coordinate binding names an undeclared coordinate"
                )
            if argument.value != shape_coordinates[coordinate]:
                raise ValueError(
                    "scalar and tensor shape-coordinate valuations differ"
                )
        for name, coordinate in bound_constexpr_shape_coordinates.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    "bound constexpr shape-coordinate names must be identifiers"
                )
            if name not in bound_constexpr_values:
                raise ValueError(
                    "bound constexpr shape-coordinate binding requires a bound value"
                )
            if not isinstance(coordinate, str) or coordinate not in shape_coordinates:
                raise ValueError(
                    "bound constexpr shape-coordinate binding names an undeclared coordinate"
                )
            value = bound_constexpr_values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != shape_coordinates[coordinate]
            ):
                raise ValueError(
                    "bound constexpr and tensor shape-coordinate valuations differ"
                )
        grid = tuple(int(dim) for dim in self.grid)
        if grid and (len(grid) > 3 or any(dim <= 0 for dim in grid)):
            raise ValueError("compiler grid must have one to three positive dimensions")
        grid_contract = tuple(self.grid_contract)
        if any(not isinstance(item, str) or not item for item in grid_contract):
            raise ValueError("grid contract must contain non-empty expressions")
        if not grid and grid_contract:
            raise ValueError("grid contract requires a concrete compiler grid")
        if grid and grid_contract and len(grid) != len(grid_contract):
            raise ValueError("concrete grid and grid contract ranks differ")
        grid_shape_coordinates = tuple(self.grid_shape_coordinates)
        if grid_shape_coordinates and len(grid_shape_coordinates) != len(grid):
            raise ValueError(
                "grid shape-coordinate bindings and concrete grid ranks differ"
            )
        for axis, coordinate in enumerate(grid_shape_coordinates):
            if coordinate is None:
                continue
            if not isinstance(coordinate, str) or coordinate not in shape_coordinates:
                raise ValueError(
                    "grid shape-coordinate binding must name a declared h* coordinate"
                )
            if grid[axis] != shape_coordinates[coordinate]:
                raise ValueError(
                    "grid extent differs from its shape-coordinate valuation"
                )
        parameter_order = tuple(self.parameter_order)
        constexpr_parameters = tuple(self.constexpr_parameters)
        for names, label in (
            (parameter_order, "parameter order"),
            (constexpr_parameters, "constexpr parameters"),
        ):
            if len(names) != len(set(names)) or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in names
            ):
                raise ValueError(f"{label} must contain unique identifiers")
        runtime_names = {item.name for item in self.arguments}
        constexpr_names = set(constexpr_parameters)
        if runtime_names & constexpr_names:
            raise ValueError("runtime and constexpr parameters overlap")
        if parameter_order and set(parameter_order) != runtime_names | constexpr_names:
            raise ValueError("parameter order must cover runtime and constexpr inputs")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_contract", grid_contract)
        object.__setattr__(
            self, "grid_shape_coordinates", grid_shape_coordinates
        )
        object.__setattr__(self, "_shape_coordinates", shape_coordinates)
        object.__setattr__(self, "parameter_order", parameter_order)
        object.__setattr__(
            self, "constexpr_parameters", constexpr_parameters
        )
        bound_names = set(bound_constexpr_values)
        if bound_names - constexpr_names:
            raise ValueError(
                "bound constexpr values must bind constexpr parameters"
            )
        ordered_bound_constexpr_values = {
            name: bound_constexpr_values[name]
            for name in constexpr_parameters
            if name in bound_names
        }
        ordered_bound_constexpr_value_types: dict[str, str] = {}
        for name, value in ordered_bound_constexpr_values.items():
            value_type = bound_constexpr_value_types.get(
                name, canonical_scalar_binding_dtype(value)
            )
            if not isinstance(value_type, str) or not value_type:
                raise TypeError(
                    "bound constexpr value types must be non-empty strings"
                )
            ordered_bound_constexpr_value_types[name] = value_type
        object.__setattr__(
            self, "bound_constexpr_values", ordered_bound_constexpr_values
        )
        object.__setattr__(
            self,
            "bound_constexpr_value_types",
            ordered_bound_constexpr_value_types,
        )
        ordered_bound_constexpr_shape_coordinates = {
            name: bound_constexpr_shape_coordinates[name]
            for name in constexpr_parameters
            if name in bound_constexpr_shape_coordinates
        }
        object.__setattr__(
            self,
            "bound_constexpr_shape_coordinates",
            ordered_bound_constexpr_shape_coordinates,
        )
        schedule_constexprs = constexpr_names - bound_names
        if schedule_constexprs - set(values):
            raise ValueError("schedule constexpr parameters require C values")
        launch_contract = copy.deepcopy(dict(self.host_launch_contract))
        validate_host_launch_contract(
            launch_contract, arguments=[item.to_json() for item in self.arguments],
            schedule_values=values, shape_coordinates=shape_coordinates,
            tile_coordinates=tile_coordinates,
        )
        object.__setattr__(self, "host_launch_contract", launch_contract)

    def to_json(self) -> dict[str, Any]:
        return {
            "schedule_values": dict(self.schedule_values),
            "schedule_value_types": dict(self.schedule_value_types),
            "bound_constexpr_values": dict(self.bound_constexpr_values),
            "bound_constexpr_value_types": dict(
                self.bound_constexpr_value_types
            ),
            "bound_constexpr_shape_coordinates": dict(
                self.bound_constexpr_shape_coordinates
            ),
            "compiler_slots": list(self.compiler_slots),
            "compiler_roles": dict(self.schedule_roles),
            "schedule_tile_coordinates": dict(self.schedule_tile_coordinates),
            "host_launch_contract": copy.deepcopy(dict(self.host_launch_contract)),
            "aliases": [item.to_json() for item in self.aliases],
            "arguments": [item.to_json() for item in self.arguments],
            "shape_coordinates": dict(self.shape_coordinates),
            "grid": list(self.grid),
            "grid_contract": list(self.grid_contract),
            "grid_shape_coordinates": list(self.grid_shape_coordinates),
            "parameter_order": list(self.parameter_order),
            "constexpr_parameters": list(self.constexpr_parameters),
        }

    @property
    def shape_coordinates(self) -> Mapping[str, int]:
        """Current valuations for source-grounded anonymous shape coordinates."""

        return dict(getattr(self, "_shape_coordinates", {}))


@dataclass(frozen=True)
class _SCRCLiftFailureReceipt:
    diagnostic: str
    response_text: str = field(repr=False)
    attempts: int = 1
    candidate_check: Mapping[str, Any] | None = None
    termination_status: str | None = None
    source_reconcile: Mapping[str, Any] | None = None
    # Source preparation reaches this receipt before a provider is called.
    # Preserve only its closed rule code for the outer lifecycle report: the
    # descriptive diagnostic can contain source-facing context and must stay
    # inside this host-local exception boundary.
    source_failure_reason: str | None = None
    provider_failure_origin: str | None = None
    provider_error: Mapping[str, Any] | None = field(default=None, repr=False)


class SCRCLiftError(RuntimeError):
    def __init__(
        self,
        attempt: _SCRCLiftFailureReceipt,
        *,
        candidate_delta: Mapping[str, Any] | None = None,
    ) -> None:
        self.diagnostic = attempt.diagnostic
        self.physical_calls = max(0, int(attempt.attempts))
        self.source_failure_class = _source_preparation_failure(attempt)
        reason = attempt.source_failure_reason
        self.source_failure_reason = (
            reason
            if isinstance(reason, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason) is not None
            else None
        )
        self.provider_failure_class = _provider_preparation_failure(attempt)
        self.provider_failure_origin = _provider_preparation_failure_origin(attempt)
        self.provider_error = (
            dict(attempt.provider_error)
            if isinstance(attempt.provider_error, Mapping)
            else None
        )
        self.source_reconcile = (
            _provider_diagnostics_projection(attempt.source_reconcile)
            if isinstance(attempt.source_reconcile, Mapping)
            else None
        )
        self.candidate_check = (
            _provider_diagnostics_projection(attempt.candidate_check)
            if isinstance(attempt.candidate_check, Mapping)
            else None
        )
        self.candidate_delta = normalize_scrc_candidate_delta(candidate_delta)
        super().__init__(f"SCRC Lift response rejected: {attempt.diagnostic}")


class CanonicalIngressError(ValueError):
    """A semantic candidate cannot form the host-owned canonical artifact.

    This is deliberately distinct from CandidateCheck: the provider returned
    parseable J semantics, but the one constructor could not commit its
    Source-owned J/C/pi projections as a complete atomic bundle. Only bounded
    rule/path diagnostics and a structural receipt may cross into relift; the
    raw candidate is never a repair base or fallback artifact.
    """

    def __init__(
        self,
        message: str,
        *,
        candidate_check_diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.candidate_check_diagnostics = _validated_ingress_diagnostics(
            candidate_check_diagnostics
        )
        super().__init__(message)


class _CandidateDecodeStageError(ValueError):
    """Carry one private decoder stage without exporting the decoder detail.

    The underlying strict decoder may include an authored label, literal, or
    row rendering in its exception text.  CandidateCheck needs the stage to
    produce actionable fresh-attempt guidance, but neither that text nor the
    rejected candidate may cross the diagnostic boundary.  Keep the detail
    process-local solely for finite rule classification.
    """

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self._detail = str(cause)
        super().__init__(f"strict candidate decoder rejected stage {stage}")

    @property
    def detail(self) -> str:
        """Return private decoder detail for local rule selection only."""

        return self._detail


_SEMANTIC_HANDLE_REFERENCE = re.compile(r"^(?P<prefix>[cp])[0-9]+$")
_SEMANTIC_LOCAL_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_SEMANTIC_PHYSICAL_HANDLE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:p|c)[0-9]+(?![A-Za-z0-9_])"
)
_PROVIDER_CANONICAL_HANDLE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:n|v|x|p|c|a|e)[0-9]+(?![A-Za-z0-9_])"
)
_PROVIDER_DIMENSION_ALIAS_TOKEN = re.compile(r"(?<![A-Za-z0-9_])dim([0-9]+)(?![A-Za-z0-9_])")


def _rewrite_provider_local_value_references(
    expression: Any,
    value_map: Mapping[str, str],
) -> Any:
    """Rewrite only known local SSA tokens in one provider-authored expression.

    Boundary index expressions are provider-authored semantic text.  Once a
    completed allocator transaction assigns those already-declared local
    results their canonical ``v*`` identities, this is the mechanical spelling
    projection that keeps the boundary in the same SSA namespace.  Unknown
    tokens deliberately survive for strict parsing/reference validation; this
    helper never supplies an index expression or chooses a value for one.
    """

    if not isinstance(expression, str) or not value_map:
        return expression
    token = re.compile(r"(?<![A-Za-z0-9_.:-])[A-Za-z_][A-Za-z0-9_.:-]*(?![A-Za-z0-9_.:-])")
    return token.sub(lambda match: value_map.get(match.group(0), match.group(0)), expression)


def _canonical_handle_prefix(value: Any) -> str | None:
    """Return the canonical namespace of one exact handle spelling."""

    if not isinstance(value, str):
        return None
    for prefix, pattern in CANONICAL_HANDLE_NAMESPACE_PATTERNS.items():
        if re.fullmatch(pattern, value):
            return prefix
    return None


def _provider_semantic_namespace_diagnostics(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Reject canonical host handles on the provider semantic wire.

    This check runs before the host maps provider-local ``dim*`` aliases to
    canonical ``x*`` dimensions.  It is intentionally not reused on the
    completed artifact: the final J must contain dense canonical handles.
    """

    diagnostics: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(key, str):
                    visit(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        match = _PROVIDER_CANONICAL_HANDLE_TOKEN.search(value)
        if match is None:
            return
        token = match.group(0)
        diagnostics.append(
            {
                "rule_code": "provider_canonical_handle_is_host_owned",
                "path": path,
                "observed_refs_or_types": [
                    f"omit:{token}; use a provider-local label or Source t*"
                ],
            }
        )

    visit(payload.get("j"), "j")
    visit(payload.get("pi"), "pi")
    return _validated_ingress_diagnostics(diagnostics)


def _rewrite_provider_dimension_aliases(
    payload: dict[str, Any],
    construction: CanonicalConstruction,
) -> None:
    """Map provider-local dim* labels to Source-owned x* handles once.

    Dimension aliases are ordinal Source facts mounted in the semantic entry.
    The provider never submits x*; this host pass is the only place where the
    local-to-canonical dimension mapping is materialized.  Operation/result
    labels are excluded so a provider-local label named ``dim0`` cannot be
    silently changed into a dimension handle.
    """

    allocator = construction.allocator
    canonical = tuple(allocator.declared("x"))
    provider_to_canonical = {
        local: source
        for source, local in provider_dimension_alias_map(
            [
                {"ordinal": index, "local": f"dim{index}"}
                for index, _handle in enumerate(canonical)
            ],
            canonical_dimensions=canonical,
        ).items()
    }

    def rewrite(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            # ``id``/``result``/``operands`` are local labels only on an
            # operation wire.  Treating those spellings as protected in every
            # mapping left ``dimension_relations[].result`` in the provider
            # dim* namespace while every other dimension field had been
            # rewritten to x*.  The operation object is the sole place where
            # those three keys name local SSA labels, so scope the exemption
            # to it rather than to a field spelling globally.
            operation_wire = isinstance(value.get("op"), str)
            return {
                name: (
                    item
                    if operation_wire and name in {"id", "result", "operands"}
                    else rewrite(item, name)
                )
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item, key) for item in value]
        if not isinstance(value, str):
            return value
        if key in {"id", "result", "operands"}:
            return value

        def replace(match: re.Match[str]) -> str:
            alias = match.group(0)
            mapped = provider_to_canonical.get(alias)
            if mapped is None:
                raise CanonicalIngressError(
                    "provider dimension alias is not Source-declared",
                    candidate_check_diagnostics=(
                        {
                            "rule_code": "provider_dimension_alias_unknown",
                            "path": "j",
                            "observed_refs_or_types": [
                                f"expected:dim0..dim{max(len(canonical) - 1, 0)}"
                            ],
                        },
                    ),
                )
            return mapped

        return _PROVIDER_DIMENSION_ALIAS_TOKEN.sub(replace, value)

    rewritten = rewrite(payload)
    if not isinstance(rewritten, dict):
        raise TypeError("provider dimension alias rewrite lost candidate object")
    payload.clear()
    payload.update(rewritten)


def _validated_ingress_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Keep constructor diagnostics closed, bounded, and source-free."""

    accepted: list[dict[str, Any]] = []
    for item in diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        if not isinstance(item, Mapping):
            continue
        rule_code = item.get("rule_code")
        path = item.get("path")
        observed = item.get("observed_refs_or_types")
        if (
            not isinstance(rule_code, str)
            or not rule_code
            or len(rule_code) > 128
            or not isinstance(path, str)
            or not path
            or len(path) > _CANDIDATE_DIAGNOSTIC_TEXT_LIMIT
            or not isinstance(observed, (list, tuple))
            or len(observed) > _CANDIDATE_DIAGNOSTIC_LIMIT
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > 128
                for value in observed
            )
        ):
            continue
        safe_observed = [
            value
            for value in observed
            if re.fullmatch(r"[A-Za-z0-9_.:=,+\-]+", value) is not None
        ]
        accepted.append(
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": safe_observed or ["closed=0"],
            }
        )
    return tuple(accepted)


def _closed_projection_diagnostics(
    projection: object,
    *,
    fallback_rule_code: str,
    fallback_path: str,
    fallback_observation: str,
) -> tuple[dict[str, Any], ...]:
    """Carry bounded constructor diagnostics without exposing payload bytes."""

    raw = (
        projection.get("diagnostics")
        if isinstance(projection, Mapping)
        else getattr(projection, "diagnostics", None)
    )
    diagnostics = tuple(
        dict(item)
        for item in raw
        if isinstance(item, Mapping)
    ) if isinstance(raw, (list, tuple)) else ()
    if diagnostics:
        validated = _validated_ingress_diagnostics(diagnostics)
        if validated:
            return validated
    return (
        {
            "rule_code": fallback_rule_code,
            "path": fallback_path,
            "observed_refs_or_types": [fallback_observation],
        },
    )


def _json_container_kind(value: Any) -> str:
    """Return the JSON Schema primitive/container type without Python aliases."""

    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _semantic_candidate_surface_diagnostics(
    payload: Any,
) -> tuple[dict[str, Any], ...]:
    """Check only the provider surface before host construction.

    This is intentionally narrower than CandidateCheck.  It shares the
    canonical profile declaration and catches deterministic ownership/wire
    errors early, but never infers operations, rewrites fields, or accepts an
    artifact.
    """

    if not isinstance(payload, Mapping):
        return (
            {
                "rule_code": "semantic_root_not_object",
                "path": "<candidate>",
                "observed_refs_or_types": [type(payload).__name__],
            },
        )
    contract = canonical_semantic_candidate_contract()
    layout = render_canonical_semantic_candidate_layout_summary()
    operation_namespace = render_canonical_semantic_operation_namespace_summary()
    root_fields = frozenset(contract["root_fields"])
    j_fields = frozenset(contract["j_fields"])
    j_field_types = contract["j_field_types"]
    boundary_field_types = contract["boundary_field_types"]
    output_selection = contract["output_selection"]
    mcp_write_protocol = canonical_semantic_mcp_write_protocol()
    provider_wire = canonical_wire_bundle()["profile"].get(
        "semantic_provider_wire"
    )
    if not isinstance(j_field_types, Mapping) or not isinstance(
        boundary_field_types, Mapping
    ):
        raise AssertionError("canonical semantic candidate type projection is malformed")
    if (
        not isinstance(output_selection, Mapping)
        or not isinstance(output_selection.get("item_fields"), list)
        or any(
            not isinstance(field, str) or not field
            for field in output_selection["item_fields"]
        )
        or not isinstance(mcp_write_protocol, Mapping)
        or not isinstance(mcp_write_protocol.get("writes"), Mapping)
        or not isinstance(provider_wire, Mapping)
    ):
        raise AssertionError("canonical semantic provider projection is malformed")
    provider_pi = provider_wire.get("pi")
    provider_pi_fields = (
        provider_pi.get("provider_fields") if isinstance(provider_pi, Mapping) else None
    )
    if not isinstance(provider_pi_fields, Mapping):
        raise AssertionError("canonical semantic provider pi projection is malformed")
    provider_pi_field_names = frozenset(provider_pi_fields)
    writes = mcp_write_protocol["writes"]
    declared_boundary_fields = writes.get("boundary_field")
    if not isinstance(declared_boundary_fields, list) or any(
        not isinstance(field, str) for field in declared_boundary_fields
    ):
        raise AssertionError("canonical semantic boundary write projection is malformed")
    provider_boundary_fields = set(declared_boundary_fields)
    for write_name in ("access", "output_mapping"):
        path = writes.get(write_name)
        if not isinstance(path, str) or not path.startswith("j.boundary."):
            raise AssertionError("canonical semantic boundary write path is malformed")
        provider_boundary_fields.add(path.removeprefix("j.boundary.").removesuffix("[]"))
    host_derived_boundary_fields = frozenset(boundary_field_types) - frozenset(
        provider_boundary_fields
    )
    output_selection_fields = frozenset(output_selection["item_fields"])
    if len(output_selection_fields) != 1:
        raise AssertionError(
            "canonical semantic output selection must have one provider value field"
        )
    output_value_field = next(iter(output_selection_fields))
    diagnostics: list[dict[str, Any]] = []
    for field_name in sorted(set(payload) - root_fields):
        diagnostics.append(
            {
                "rule_code": "semantic_root_unknown_field",
                "path": field_name,
                "observed_refs_or_types": [
                    "root_field",
                    layout,
                ],
            }
        )
    candidate_j = payload.get("j")
    if not isinstance(candidate_j, Mapping):
        diagnostics.append(
            {
                "rule_code": "semantic_j_required_path_missing",
                "path": "j",
                "observed_refs_or_types": [
                    "object",
                    layout,
                ],
            }
        )
        return _validated_ingress_diagnostics(diagnostics)
    for field_name in sorted(set(candidate_j) - j_fields):
        diagnostics.append(
            {
                "rule_code": "semantic_j_unknown_field",
                "path": f"j.{field_name}",
                "observed_refs_or_types": [
                    "j_field",
                    layout,
                ],
            }
        )
    for field_name in sorted(set(candidate_j) & j_fields):
        expected = j_field_types.get(field_name)
        value = candidate_j[field_name]
        if isinstance(expected, str) and _json_container_kind(value) != expected:
            diagnostics.append(
                {
                    "rule_code": "semantic_j_field_type",
                    "path": f"j.{field_name}",
                    "observed_refs_or_types": [
                        f"expected:{expected}",
                        f"actual:{_json_container_kind(value)}",
                    ],
                }
            )
    intent = candidate_j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(function.get("body"), list):
        diagnostics.append(
            {
                "rule_code": "semantic_j_required_path_missing",
                "path": "j.intent.intent_function.body",
                "observed_refs_or_types": [
                    "array",
                    layout,
                ],
            }
        )
    if not isinstance(candidate_j.get("boundary"), Mapping):
        diagnostics.append(
            {
                "rule_code": "semantic_j_required_path_missing",
                "path": "j.boundary",
                "observed_refs_or_types": [
                    "object",
                    layout,
                ],
            }
        )
    else:
        boundary = candidate_j["boundary"]
        for field_name in sorted(set(boundary) - set(boundary_field_types)):
            diagnostics.append(
                {
                    "rule_code": "semantic_boundary_unknown_field",
                    "path": f"j.boundary.{field_name}",
                    "observed_refs_or_types": ["boundary_field"],
                }
            )
        for field_name in sorted(set(boundary) & set(boundary_field_types)):
            expected = boundary_field_types[field_name]
            value = boundary[field_name]
            if isinstance(expected, str) and _json_container_kind(value) != expected:
                diagnostics.append(
                    {
                        "rule_code": "semantic_boundary_field_type",
                        "path": f"j.boundary.{field_name}",
                        "observed_refs_or_types": [
                            f"expected:{expected}",
                            f"actual:{_json_container_kind(value)}",
                        ],
                    }
                )
    if "pi" in payload:
        if not isinstance(payload.get("pi"), Mapping):
            diagnostics.append(
                {
                    "rule_code": "semantic_pi_field_type",
                    "path": "pi",
                    "observed_refs_or_types": [
                        "expected:object",
                        f"actual:{_json_container_kind(payload.get('pi'))}",
                    ],
                }
            )
        else:
            for field_name in sorted(
                set(payload["pi"]) - provider_pi_field_names
            ):
                diagnostics.append(
                    {
                        "rule_code": "semantic_pi_host_derived_field",
                        "path": f"pi.{field_name}",
                        "observed_refs_or_types": [
                            "omit: host derives π bindings from SourceFacts, J, C, and the transfer ledger"
                        ],
                    }
                )

    # Output names, ABI order, canonical return rows and node attachments are
    # all host-owned.  At this semantic boundary the candidate supplies only
    # one local value selection per Source output.  Accepting final-wire
    # ``output``/``node_ref`` siblings here would reintroduce the second output
    # authority that the canonical constructor is meant to remove.
    if isinstance(candidate_j.get("boundary"), Mapping):
        semantic_boundary = candidate_j["boundary"]
        for field_name in sorted(host_derived_boundary_fields):
            # These rows are complete projections of the semantic access/op
            # graph.  A non-empty provider copy would be a parallel authority:
            # the constructor could either ignore it or accidentally use it
            # for a remap.  Empty legacy placeholders remain harmless and give
            # the relift a small migration path; all real content is rejected.
            value = semantic_boundary.get(field_name)
            if value not in (None, []):
                diagnostics.append(
                    {
                        "rule_code": "semantic_host_derived_boundary_field",
                        "path": f"j.boundary.{field_name}",
                        "observed_refs_or_types": [
                            "omit: host derives this collection from semantic operations/accesses"
                        ],
                    }
                )
        raw_accesses = semantic_boundary.get("accesses")
        access_labels: set[str] = set()
        if isinstance(raw_accesses, list):
            for access_index, access in enumerate(raw_accesses):
                if isinstance(access, Mapping) and "access_id" in access:
                    diagnostics.append(
                        {
                            "rule_code": "semantic_access_id_is_host_owned",
                            "path": f"j.boundary.accesses[{access_index}].access_id",
                            "observed_refs_or_types": [
                                "omit: host allocates canonical a* from access order"
                            ],
                        }
                    )
                label = access.get("access_label") if isinstance(access, Mapping) else None
                if (
                    not isinstance(label, str)
                    or re.fullmatch(r"access[0-9]+", label) is None
                    or label in access_labels
                ):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_access_label_invalid",
                            "path": f"j.boundary.accesses[{access_index}].access_label",
                            "observed_refs_or_types": [
                                "expected:unique provider-local accessN label"
                            ],
                        }
                    )
                else:
                    access_labels.add(label)

        # ``access_ref`` is the one provider-local join key shared by
        # predicate/consumer relations.  Validate it against the same local
        # access table before construction so an unknown spelling cannot turn
        # into a late allocator failure or accidentally survive a repair-base
        # projection. Final ``a*`` is still allocated only by the host.
        def validate_access_references(value: Any, path: str) -> None:
            if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    item_path = f"{path}.{key}"
                    if key == "access_ref" and isinstance(item, str):
                        if item not in access_labels:
                            diagnostics.append(
                                {
                                    "rule_code": "semantic_access_ref_invalid",
                                    "path": item_path,
                                    "observed_refs_or_types": [
                                        "expected:one declared provider-local accessN label"
                                    ],
                                }
                            )
                    else:
                        validate_access_references(item, item_path)
                return
            if isinstance(value, list):
                for item_index, item in enumerate(value):
                    validate_access_references(item, f"{path}[{item_index}]")

        validate_access_references(semantic_boundary, "j.boundary")
        output_mappings = semantic_boundary.get("output_mappings")
        if not isinstance(output_mappings, list):
            diagnostics.append(
                {
                    "rule_code": "semantic_output_selection_missing",
                    "path": "j.boundary.output_mappings",
                    "observed_refs_or_types": [
                        "expected:one ordered value selection per Source output"
                    ],
                }
            )
        else:
            for index, selection in enumerate(output_mappings):
                if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                    break
                path = f"j.boundary.output_mappings[{index}]"
                if not isinstance(selection, Mapping):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_output_selection_invalid",
                            "path": path,
                            "observed_refs_or_types": ["expected:object(value)"],
                        }
                    )
                    continue
                if set(selection) != output_selection_fields:
                    diagnostics.append(
                        {
                            "rule_code": "semantic_output_selection_fields",
                            "path": path,
                            "observed_refs_or_types": [
                                "expected:only "
                                + output_value_field
                                + "; output/node_ref are host-owned"
                            ],
                        }
                    )
                    continue
                # The schema-derived semantic output projection deliberately
                # has one selected value field.  Keep its spelling out of the
                # checker so a future canonical schema update does not leave
                # CandidateCheck enforcing a stale second row grammar.
                value = selection.get(output_value_field)
                prefix = _canonical_handle_prefix(value)
                if (
                    prefix not in {"t", None}
                    or not isinstance(value, str)
                    or (
                        prefix is None
                        and _SEMANTIC_LOCAL_LABEL.fullmatch(value) is None
                    )
                ):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_output_selection_value_invalid",
                            "path": f"{path}.value",
                            "observed_refs_or_types": [
                                "expected:local result label or direct Source t* input"
                            ],
                        }
                    )

    # The provider authors a semantic graph before the host allocator assigns
    # n*/v* identities.  Admit only source t* inputs and local semantic labels
    # at this boundary.  This catches stale canonical handles and dimensions
    # used as SSA operands before they can fall through into a generic
    # constructor failure, while leaving producer/order/type proof to the
    # shared allocator and strict CandidateCheck.
    if isinstance(function, Mapping) and isinstance(function.get("body"), list):
        rows = semantic_operation_wire_rows(function)
        if rows is None:
            diagnostics.append(
                {
                    "rule_code": "semantic_operation_tree_malformed",
                    "path": "j.intent.intent_function.body",
                    "observed_refs_or_types": ["operation_tree"],
                }
            )
        else:
            for wire_path, operation in rows:
                if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                    break
                operation_path = f"j.intent.intent_function.{wire_path}"
                operation_name = operation.get("op")
                if operation_name not in (CANONICAL_J_OPS | STRUCTURED_INTENT_OPS):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_operation_unknown",
                            "path": f"{operation_path}.op",
                            "observed_refs_or_types": [
                                f"unknown:{operation_name}",
                                operation_namespace,
                            ],
                        }
                    )
                node_label = operation.get("id")
                if _canonical_handle_prefix(node_label) == "n":
                    diagnostics.append(
                        {
                            "rule_code": "semantic_node_label_is_host_owned",
                            "path": f"{operation_path}.id",
                            "observed_refs_or_types": [
                                "use a local operation label or omit id; host allocates n*"
                            ],
                        }
                    )
                if (
                    isinstance(node_label, str)
                    and _PROVIDER_DIMENSION_ALIAS_TOKEN.fullmatch(node_label)
                ):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_local_label_collides_with_dimension_alias",
                            "path": f"{operation_path}.id",
                            "observed_refs_or_types": [
                                "use a non-dim* local operation label; dim* is reserved for Source dimensions"
                            ],
                        }
                    )
                if "scope_id" in operation:
                    diagnostics.append(
                        {
                            "rule_code": "semantic_scope_id_is_host_owned",
                            "path": f"{operation_path}.scope_id",
                            "observed_refs_or_types": [
                                "omit: host derives lexical q* scope ownership from the region tree"
                            ],
                        }
                    )
                if operation.get("op") in {"effect_if", "value_if"}:
                    for polarity in ("positive", "negative"):
                        arm = operation.get(polarity)
                        if isinstance(arm, Mapping) and "scope_id" in arm:
                            diagnostics.append(
                                {
                                    "rule_code": "semantic_scope_id_is_host_owned",
                                    "path": f"{operation_path}.{polarity}.scope_id",
                                    "observed_refs_or_types": [
                                        "omit: host derives lexical q* scope ownership from the region tree"
                                    ],
                                }
                            )
                result_label = operation.get("result")
                # A loop has one result per carry and therefore legitimately
                # uses a result list.  Earlier semantic preflight converted
                # that list to a string and rejected every structured loop
                # before the single structured allocator could construct its
                # lexical v* namespace.  Validate each authored local label
                # instead; ordinary operations and value_if still have one.
                result_labels = (
                    []
                    if result_label is None and operation.get("op") == "effect_if"
                    else (
                        result_label
                        if isinstance(result_label, list)
                        else [result_label]
                    )
                )
                if not result_labels and operation.get("op") != "effect_if":
                    diagnostics.append(
                        {
                            "rule_code": "semantic_result_label_invalid",
                            "path": f"{operation_path}.result",
                            "observed_refs_or_types": [
                                "expected:one local result label per operation result"
                            ],
                        }
                    )
                for result_index, label in enumerate(result_labels):
                    result_prefix = _canonical_handle_prefix(label)
                    if result_prefix is not None or not isinstance(label, str) or (
                        _SEMANTIC_LOCAL_LABEL.fullmatch(label) is None
                    ):
                        suffix = (
                            f"[{result_index}]"
                            if isinstance(result_label, list)
                            else ""
                        )
                        diagnostics.append(
                            {
                                "rule_code": "semantic_result_label_invalid",
                                "path": f"{operation_path}.result{suffix}",
                                "observed_refs_or_types": [
                                    "use one local semantic result label; host allocates v*"
                                ],
                            }
                        )
                    elif _PROVIDER_DIMENSION_ALIAS_TOKEN.fullmatch(label):
                        diagnostics.append(
                            {
                                "rule_code": "semantic_local_label_collides_with_dimension_alias",
                                "path": f"{operation_path}.result"
                                + (
                                    f"[{result_index}]"
                                    if isinstance(result_label, list)
                                    else ""
                                ),
                                "observed_refs_or_types": [
                                    "use a non-dim* local result label; dim* is reserved for Source dimensions"
                                ],
                            }
                        )
                operands = operation.get("operands")
                if not isinstance(operands, list):
                    continue
                for operand_index, operand in enumerate(operands):
                    if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                        break
                    prefix = _canonical_handle_prefix(operand)
                    if prefix == "t":
                        continue
                    if (
                        prefix is not None
                        or not isinstance(operand, str)
                        or not _SEMANTIC_LOCAL_LABEL.fullmatch(operand)
                        or _PROVIDER_DIMENSION_ALIAS_TOKEN.fullmatch(operand)
                    ):
                        diagnostics.append(
                            {
                                "rule_code": "semantic_operand_not_source_tensor_or_local_value",
                                "path": f"{operation_path}.operands[{operand_index}]",
                                "observed_refs_or_types": [
                                    "use source t* or a prior local semantic result; x*/v*/p*/c* are not semantic operands"
                                ],
                            }
                        )

    def semantic_expression_diagnostics(value: Any, path: str, field: str = "") -> None:
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(key, str):
                    # These identities are emitted only after semantic graph
                    # construction.  Reject their presence by field name, not
                    # merely when a stale canonical-looking value happens to
                    # be attached: otherwise a provider could create a second
                    # ownership channel with an arbitrary placeholder.
                    if key in {"scope_id", "access_id", "effect_id"}:
                        diagnostics.append(
                            {
                                "rule_code": "semantic_host_identity_is_provider_forbidden",
                                "path": f"{path}.{key}",
                                "observed_refs_or_types": [
                                    "omit: host allocates scope/access/effect identities"
                                ],
                            }
                        )
                    semantic_expression_diagnostics(child, f"{path}.{key}", key)
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                semantic_expression_diagnostics(child, f"{path}[{index}]", field)
            return
        if not isinstance(value, str):
            return
        if field in {"predicate", "index_expression"} and _SEMANTIC_PHYSICAL_HANDLE_TOKEN.search(value):
            diagnostics.append(
                {
                    "rule_code": "physical_handle_in_semantic_expression",
                    "path": path,
                    "observed_refs_or_types": [
                        "p*/c* are ABI/compiler carriers; use J tensor/value/dimension semantics"
                    ],
                }
            )
        if field in {
            "node_ref",
            "effect_node_ref",
            "consumer_ref",
            "update_node_ref",
        } and _canonical_handle_prefix(value) == "n":
            diagnostics.append(
                {
                    "rule_code": "semantic_node_reference_is_host_owned",
                    "path": path,
                    "observed_refs_or_types": [
                        "use the local operation label; host rewrites it to n*"
                    ],
                }
            )

    semantic_expression_diagnostics(candidate_j, "j")

    def visit(value: Any, path: str) -> None:
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(key, str):
                    visit(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        match = _SEMANTIC_HANDLE_REFERENCE.fullmatch(value)
        if match is None:
            return
        prefix = match.group("prefix")
        diagnostics.append(
            {
                "rule_code": (
                    "compiler_only_handle_in_j"
                    if prefix == "c"
                    else "physical_parameter_handle_in_j"
                ),
                "path": path,
                "observed_refs_or_types": [f"{prefix}*"],
            }
        )

    visit(candidate_j, "j")
    return _validated_ingress_diagnostics(diagnostics)


def _private_source_owner_token(
    access: Mapping[str, Any], construction: CanonicalConstruction,
) -> str:
    """Read existing authority by exact occurrence; never sign a claim."""
    ordinal = access.get("ordinal")
    if (
        construction.transfer_ledger.status != "pass"
        or not is_canonical_ordinal(ordinal)
        or access.get("access_label") != f"access{ordinal}"
    ):
        raise ValueError("source_owner_receipt_unissued")
    bindings = [row for row in construction.transfer_ledger.bindings if row.get("ordinal") == ordinal]
    if len(bindings) != 1:
        raise ValueError("source_owner_receipt_unissued")
    authorizations = bindings[0].get("semantic_authorizations")
    owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
    token = owner.get("token") if isinstance(owner, Mapping) else None
    if not isinstance(owner, Mapping) or owner.get("status") != "pass" or not isinstance(token, str) or re.fullmatch(SOURCE_OWNER_TOKEN_PATTERN, token) is None:
        raise ValueError("source_owner_receipt_unissued")
    return token


def _semantic_construction_submission_diagnostics(
    payload: Mapping[str, Any],
    construction: CanonicalConstruction,
) -> tuple[dict[str, Any], ...]:
    """Check the one Source-sized submission contract before construction."""

    source_wire = canonical_source_wire_bundle(construction)
    requirements = source_wire.get("semantic_construction_requirements")
    semantic_required = (
        requirements.get("semantic_required")
        if isinstance(requirements, Mapping)
        else None
    )
    access_requirements = (
        semantic_required.get("accesses")
        if isinstance(semantic_required, Mapping)
        else None
    )
    output_requirements = (
        semantic_required.get("outputs")
        if isinstance(semantic_required, Mapping)
        else None
    )
    expected_accesses = (
        access_requirements.get("rows")
        if isinstance(access_requirements, Mapping)
        else None
    )
    expected_outputs = (
        output_requirements.get("rows")
        if isinstance(output_requirements, Mapping)
        else None
    )
    obligation_rows = (
        requirements.get("obligations")
        if isinstance(requirements, Mapping)
        else None
    )
    try:
        validated_obligations = validate_source_semantic_obligations(
            obligation_rows,
            construction.source_facts.digest,
        )
    except ValueError:
        validated_obligations = []
    if not isinstance(expected_accesses, list) or not isinstance(
        expected_outputs, list
    ) or not validated_obligations:
        return (
            {
                "rule_code": "semantic_construction_contract_unavailable",
                "path": "semantic_construction_requirements",
                "observed_refs_or_types": ["expected:Source-sized construction contract"],
            },
        )
    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(boundary, Mapping):
        return ()
    diagnostics: list[dict[str, Any]] = []
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    body = function.get("body") if isinstance(function, Mapping) else None
    operation_rows = (
        semantic_operation_wire_rows({"body": body})
        if isinstance(body, list) else None
    )
    operations_by_local_id = {
        row.get("id"): row
        for _path, row in (operation_rows or ())
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    values_by_local_result: dict[str, Mapping[str, Any]] = {}
    for operation in operations_by_local_id.values():
        result = operation.get("result")
        results = (result,) if isinstance(result, str) else result
        if not isinstance(results, list | tuple):
            continue
        for value in results:
            if isinstance(value, str):
                values_by_local_result[value] = operation

    def operation_family(row: Mapping[str, Any]) -> str | None:
        operation = row.get("op")
        return (
            operation if operation in {
                "map_binary", "map_unary", "select", "cast", "reduce",
                "iota", "reshape", "transpose", "concat", "flip", "sort",
            } else None
        )

    accesses = boundary.get("accesses")
    if not isinstance(accesses, list):
        diagnostics.append(
            {
                "rule_code": "semantic_access_occurrence_arity",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": [
                    f"expected:{len(expected_accesses)} Source occurrence row(s)",
                    "actual:missing",
                ],
            }
        )
        return tuple(diagnostics)
    raw_guard_relations = boundary.get("access_guard_relations")
    guard_relations_by_access: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_guard_relations, list):
        for relation in raw_guard_relations:
            if not isinstance(relation, Mapping):
                continue
            access_ref = relation.get("access_ref")
            if isinstance(access_ref, str) and access_ref not in guard_relations_by_access:
                guard_relations_by_access[access_ref] = relation
    if len(accesses) != len(expected_accesses):
        diagnostics.append(
            {
                "rule_code": "semantic_access_occurrence_arity",
                "path": "j.boundary.accesses",
                "observed_refs_or_types": [
                    f"expected:{len(expected_accesses)} Source occurrence row(s)",
                    f"actual:{len(accesses)}",
                ],
            }
        )
    for index, expected in enumerate(expected_accesses):
        if index >= len(accesses) or not isinstance(expected, Mapping):
            continue
        actual = accesses[index]
        if not isinstance(actual, Mapping):
            continue
        if "semantic_owner_authorization" in actual:
            diagnostics.append({
                "rule_code": "semantic_owner_authorization_is_host_owned",
                "path": f"j.boundary.accesses[{index}].semantic_owner_authorization",
                "observed_refs_or_types": ["omit:host verifies the declared node_ref"],
            })
        expected_label = expected.get("access_label")
        if actual.get("access_label") != expected_label:
            diagnostics.append(
                {
                    "rule_code": "semantic_access_occurrence_order",
                    "path": f"j.boundary.accesses[{index}].access_label",
                    "observed_refs_or_types": [
                        f"expected:{expected_label}",
                        f"actual:{actual.get('access_label')}",
                    ],
                }
            )
        expected_ordinal = expected.get("ordinal")
        if actual.get("ordinal") != expected_ordinal:
            diagnostics.append(
                {
                    "rule_code": "semantic_access_occurrence_ordinal",
                    "path": f"j.boundary.accesses[{index}].ordinal",
                    "observed_refs_or_types": [
                        f"expected:{expected_ordinal}",
                        f"actual:{actual.get('ordinal')}",
                    ],
                }
            )
        if "id" in actual:
            diagnostics.append(
                {
                    "rule_code": "semantic_access_id_alias_forbidden",
                    "path": f"j.boundary.accesses[{index}].id",
                    "observed_refs_or_types": [
                        "use:access_label plus ordinal; access_id is host-owned",
                    ],
                }
            )
        required_fields = expected.get("required_fields")
        if isinstance(required_fields, list):
            missing = [field for field in required_fields if field not in actual]
            if missing:
                diagnostics.append(
                    {
                        "rule_code": "semantic_access_occurrence_fields",
                        "path": f"j.boundary.accesses[{index}]",
                        "observed_refs_or_types": [
                            "missing:" + ",".join(str(field) for field in missing)
                        ],
                    }
                )
            template = expected.get("wire_template")
            provider_owned = expected.get("provider_owned_fields", ())
            provider_owned_expressions = expected.get(
                "provider_owned_expression_fields", ()
            )
            if not isinstance(template, Mapping) or not isinstance(
                provider_owned, (list, tuple)
            ) or not isinstance(provider_owned_expressions, (list, tuple)):
                diagnostics.append(
                    {
                        "rule_code": "semantic_construction_contract_unavailable",
                        "path": f"semantic_construction_requirements.accesses[{index}]",
                        "observed_refs_or_types": ["expected:canonical Source access wire"],
                    }
                )
                continue
            provider_owned_fields = {
                name for name in provider_owned if isinstance(name, str)
            }
            provider_owned_expression_field_names = {
                name
                for name in provider_owned_expressions
                if isinstance(name, str)
            }
            for field_name in (
                "kind",
                "tensor",
                "index_expression",
                "predicate",
                "fill_value",
                "node_ref",
                "semantic_access_authorization",
                "semantic_owner_authorization",
                "semantic_guard_authorization",
                "effect_kind",
                "effect_target",
                "effect_node_ref",
            ):
                if field_name not in required_fields:
                    continue
                if field_name in provider_owned_expression_field_names:
                    value = actual.get(field_name)
                    if not isinstance(value, str) or value.startswith("<"):
                        diagnostics.append(
                            {
                                "rule_code": "semantic_access_provider_owned_expression_invalid",
                                "path": f"j.boundary.accesses[{index}].{field_name}",
                                "observed_refs_or_types": [
                                    "expected:provider-authored logical expression"
                                ],
                            }
                        )
                elif field_name in provider_owned_fields:
                    value = actual.get(field_name)
                    if (
                        not isinstance(value, str)
                        or _SEMANTIC_LOCAL_LABEL.fullmatch(value) is None
                        or value.startswith("<")
                    ):
                        diagnostics.append(
                            {
                                "rule_code": "semantic_access_provider_owned_field_invalid",
                                "path": f"j.boundary.accesses[{index}].{field_name}",
                                "observed_refs_or_types": [
                                    "expected:provider-local label"
                                ],
                            }
                        )
                elif field_name in actual and actual.get(field_name) != template.get(
                    field_name
                ):
                    # Do not include either literal in the diagnostic.  The
                    # initial authoring brief is the sole value-bearing Source
                    # wire; rejection feedback remains rule/path/category only.
                    diagnostics.append(
                        {
                            "rule_code": "semantic_access_source_field_mismatch",
                            "path": f"j.boundary.accesses[{index}].{field_name}",
                            "observed_refs_or_types": [
                                "category:source_wire_value_mismatch"
                            ],
                        }
                    )
            if (
                "node_ref" in provider_owned_fields
                and "effect_node_ref" in provider_owned_fields
                and actual.get("node_ref") != actual.get("effect_node_ref")
            ):
                diagnostics.append(
                    {
                        "rule_code": "semantic_access_effect_node_owner_mismatch",
                        "path": f"j.boundary.accesses[{index}].effect_node_ref",
                        "observed_refs_or_types": [
                            "expected:same provider-local operation as node_ref"
                        ],
                    }
                )
            source_semantics = expected.get("source_semantics")
            expected_authorizations = (
                source_semantics.get("semantic_authorizations")
                if isinstance(source_semantics, Mapping) else None
            )
            expected_access_view = (
                expected_authorizations.get("access_view")
                if isinstance(expected_authorizations, Mapping) else None
            )
            expected_access_view_token = (
                expected_access_view.get("token")
                if isinstance(expected_access_view, Mapping) else None
            )
            if actual.get("node_ref") is None and actual.get("kind") == "load":
                if actual.get("semantic_access_authorization") != expected_access_view_token:
                    diagnostics.append(
                        {
                            "rule_code": "semantic_access_authorization_missing",
                            "path": f"j.boundary.accesses[{index}].semantic_access_authorization",
                            "observed_refs_or_types": ["expected:ledger-issued exact Source access receipt"],
                        }
                    )
            expected_guard = (
                expected_authorizations.get("computed_guard")
                if isinstance(expected_authorizations, Mapping) else None
            )
            expected_guard_family = (
                expected_guard.get("operation_family")
                if isinstance(expected_guard, Mapping) else None
            )
            expected_guard_token = (
                expected_guard.get("token")
                if isinstance(expected_guard, Mapping) else None
            )
            node_ref = actual.get("node_ref")
            if node_ref is not None:
                try:
                    owner_token = _private_source_owner_token(actual, construction)
                    seed_j = source_wire["artifact_seed"]["j"]
                    seed_function = seed_j["intent"]["intent_function"]
                    # Preflight runs before canonical ingress. Compare a
                    # private alias-normalized view with the canonical Source
                    # seed; never rewrite the submitted body as a repair.
                    proof_view = copy.deepcopy(source_wire["artifact_seed"])
                    proof_view["j"]["intent"]["intent_function"] = {
                        **seed_function, "body": copy.deepcopy(body),
                    }
                    # Match the formal Source-seeded constructor: omitted
                    # collections retain their Source seed defaults, whereas
                    # every explicitly submitted field keeps its exact value.
                    proof_view["j"]["boundary"].update(copy.deepcopy(dict(boundary)))
                    candidate_pi = payload.get("pi")
                    if isinstance(candidate_pi, Mapping):
                        proof_view["pi"].update(copy.deepcopy(dict(candidate_pi)))
                    _rewrite_provider_dimension_aliases(proof_view, construction)
                    # The public contract permits omitting every domain so
                    # Source can project the whole launch map atomically.
                    # Owner preflight must see that same proved map; it must
                    # not reject before the later construction projection.
                    from .compiler_observation import materialize_source_program_domains
                    projected = materialize_source_program_domains(
                        proof_view,
                        source_facts=construction.source_facts,
                        ledger=construction.transfer_ledger,
                    )
                    if projected.get("status") != "pass":
                        # The projector emits a closed reason code. Keep only
                        # that bounded code, never candidate domain text.
                        reason = projected.get("reason")
                        safe_reason = (
                            reason if isinstance(reason, str)
                            and re.fullmatch(r"[a-z0-9_]{1,48}", reason)
                            else "unproved"
                        )
                        raise ValueError(f"source_owner_domain_{safe_reason}")
                    proof_view = projected["artifact"]
                    proof_boundary = proof_view["j"]["boundary"]
                    # Host-only proof view: do not change the submitted bytes
                    # or author any producer, predicate, coordinate or value.
                    proof_boundary["accesses"][index]["semantic_owner_authorization"] = owner_token
                    validate_source_owner_binding(
                        proof_view["j"]["intent"]["intent_function"],
                        seed_j["signature"]["parameters"],
                        proof_boundary["accesses"][index], proof_boundary,
                        value_dimension_relations=proof_view["j"].get("value_dimension_relations"),
                    )
                except (ValueError, TypeError, KeyError, RecursionError) as exc:
                    # The owner validator raises closed rule identifiers for
                    # its semantic obligations. Preserve only that bounded
                    # identifier, never the rejected J row or exception text
                    # from an arbitrary parser/library failure.
                    owner_rule = (
                        exc.args[0]
                        if type(exc) is ValueError
                        and len(exc.args) == 1
                        and isinstance(exc.args[0], str)
                        and re.fullmatch(r"source_owner_[a-z0-9_]{1,80}", exc.args[0])
                        else "source_exact_owner_binding_failed"
                    )
                    owner_path = f"j.boundary.accesses[{index}].node_ref"
                    if owner_rule.startswith((
                        "source_owner_broadcast_",
                        "source_owner_computed_guard_broadcast_",
                    )):
                        owner_path = "j.boundary.broadcasts"
                    elif owner_rule == "source_owner_computed_guard_projection_unproved":
                        owner_path = "j.boundary.access_guard_relations"
                    elif owner_rule == "source_owner_reduction_boundary_mismatch":
                        owner_path = "j.boundary.reductions"
                    elif owner_rule.startswith("source_owner_computed_guard_coordinate_"):
                        owner_path = "j.intent.intent_function.body"
                    elif owner_rule.startswith("source_owner_scalar_load_index_") or owner_rule in {
                        "source_owner_const_contract_unproved",
                        "source_owner_explicit_broadcast_unsupported",
                        "source_owner_pointwise_contract_unproved",
                    }:
                        owner_path = "j.intent.intent_function.body"
                    elif owner_rule.startswith("source_owner_computed_guard_comparison_"):
                        owner_path = "j.intent.intent_function.body"
                    owner_expected = "expected:unique exact typed Source producer DAG"
                    if owner_rule.endswith("broadcast_axis_type_unproved"):
                        owner_expected = "expected:unquoted JSON integer target-axis positions"
                    elif owner_rule == "source_owner_broadcast_relation_missing":
                        owner_expected = "expected:one boundary.broadcasts row per outlined consumer/operand"
                    elif owner_rule == "source_owner_computed_guard_projection_unproved":
                        owner_expected = "expected:positive guard relation with axes complementary to the guarded tensor axis"
                    elif owner_rule == "source_owner_reduction_boundary_mismatch":
                        owner_expected = "expected:one reduction boundary with proved local tile axis and typed identity"
                    elif owner_rule == "source_owner_scalar_load_index_producer_missing":
                        owner_expected = "expected:prior local result of a scalar index const"
                    elif owner_rule == "source_owner_scalar_load_index_contract_unproved":
                        owner_expected = "expected:const operands=[]; attrs only value; type i32/i64 with shape=[]"
                    elif owner_rule == "source_owner_const_contract_unproved":
                        owner_expected = "expected:owner const with attrs only value and no operands"
                    elif owner_rule == "source_owner_explicit_broadcast_unsupported":
                        owner_expected = "expected:outlined owner broadcast on the consuming boundary relation"
                    elif owner_rule == "source_owner_pointwise_contract_unproved":
                        owner_expected = "expected:owner-supported map operation and exact attrs"
                    elif owner_rule.startswith("source_owner_computed_guard_coordinate_"):
                        owner_expected = "expected:prior iota with operands=[]; type i32[Source-bound dimension]; attrs n=that dimension"
                    obligation_kind = (
                        "source_owner_broadcast"
                        if owner_rule.startswith((
                            "source_owner_broadcast_",
                            "source_owner_computed_guard_broadcast_",
                        ))
                        else "source_computed_guard"
                        if owner_rule.startswith("source_owner_computed_guard_")
                        else "program_domain"
                        if owner_rule.startswith("source_owner_domain_")
                        else "source_owner_producer"
                    )
                    locator = (
                        {}
                        if obligation_kind == "program_domain"
                        else {"ordinal": expected_ordinal}
                    )
                    matching_obligations = matching_source_semantic_obligations(
                        validated_obligations,
                        construction.source_facts.digest,
                        kind=obligation_kind,
                        source_locator=locator,
                    )
                    if matching_obligations:
                        for obligation in matching_obligations:
                            diagnostic = semantic_obligation_diagnostic(
                                obligation,
                                construction.source_facts.digest,
                                status="rejected",
                            )
                            diagnostic["obligation_rule_code"] = diagnostic[
                                "rule_code"
                            ]
                            diagnostic["rule_code"] = owner_rule
                            diagnostic["path"] = owner_path
                            diagnostic["expected_constraints"] = [
                                owner_expected.removeprefix("expected:")
                            ]
                            diagnostics.append(diagnostic)
                    else:
                        # Unknown rules stay fail-closed, but do not fabricate
                        # an obligation identity which Source construction did
                        # not issue.
                        diagnostics.append({
                            "rule_code": owner_rule,
                            "path": owner_path,
                            "observed_refs_or_types": [owner_expected],
                            "repair_locator": {
                                "access_label": expected_label,
                                "ordinal": expected_ordinal,
                            },
                        })
            actual_guard_token = actual.get("semantic_guard_authorization")
            if actual_guard_token is not None and actual_guard_token != expected_guard_token:
                diagnostics.append(
                    {
                        "rule_code": "semantic_guard_authorization_unissued",
                        "path": f"j.boundary.accesses[{index}].semantic_guard_authorization",
                        "observed_refs_or_types": [
                            "expected:ledger-issued opaque computed-guard authorization"
                        ],
                    }
                )
            relation = guard_relations_by_access.get(actual.get("access_label"))
            if relation is not None:
                value_ref = relation.get("value_ref")
                guard_operation = (
                    values_by_local_result.get(value_ref)
                    if isinstance(value_ref, str) else None
                )
                if (
                    not isinstance(expected_guard_token, str)
                    or actual_guard_token != expected_guard_token
                ):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_guard_authorization_missing",
                            "path": f"j.boundary.accesses[{index}].semantic_guard_authorization",
                            "observed_refs_or_types": [
                                "expected:ledger-issued computed-guard authorization"
                            ],
                        }
                    )
                if (
                    guard_operation is None
                    or operation_family(guard_operation) != expected_guard_family
                ):
                    diagnostics.append(
                        {
                            "rule_code": "semantic_guard_operation_family_mismatch",
                            "path": "j.boundary.access_guard_relations",
                            "observed_refs_or_types": [
                                "expected:ledger-authorized local boolean operation family"
                            ],
                        }
                    )
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            break
    output_mappings = boundary.get("output_mappings")
    if not isinstance(output_mappings, list):
        return tuple(diagnostics)
    if len(output_mappings) != len(expected_outputs):
        diagnostics.append(
            {
                "rule_code": "semantic_output_selection_arity",
                "path": "j.boundary.output_mappings",
                "observed_refs_or_types": [
                    f"expected:{len(expected_outputs)} Source output row(s)",
                    f"actual:{len(output_mappings)}",
                ],
            }
        )
    return tuple(diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT])


def check_scrc_semantic_candidate_surface(
    candidate_text: str,
    *,
    construction: CanonicalConstruction | None = None,
) -> dict[str, Any]:
    """Return a CandidateCheck-shaped deterministic semantic-surface result."""

    try:
        payload = _json_object(candidate_text)
    except Exception:
        diagnostics = (
            {
                "rule_code": "not_one_json_object",
                "path": "<candidate>",
                "observed_refs_or_types": ["json_object"],
            },
        )
    else:
        diagnostics = _semantic_candidate_surface_diagnostics(payload)
        if not diagnostics and construction is not None:
            diagnostics = _semantic_construction_submission_diagnostics(
                payload,
                construction,
            )
    return {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass" if not diagnostics else "fail",
        "diagnostics": list(diagnostics),
    }


def _provider_preparation_failure(
    attempt: _SCRCLiftFailureReceipt | None,
) -> str | None:
    """Return the provider failure class carried by an empty Lift attempt."""

    if attempt is None:
        return None
    failure_class = attempt.termination_status
    if isinstance(failure_class, str) and failure_class.startswith("provider_"):
        return failure_class
    return None


def _provider_preparation_failure_origin(
    attempt: _SCRCLiftFailureReceipt | None,
) -> str | None:
    """Return the closed transport/event origin carried by an empty Lift."""

    if attempt is None:
        return None
    origin = attempt.provider_failure_origin
    if isinstance(origin, str) and re.fullmatch(
        r"[a-z][a-z0-9_]{0,127}", origin
    ) is not None:
        return origin
    return None


def _source_preparation_failure(
    attempt: _SCRCLiftFailureReceipt | None,
) -> str | None:
    """Return a deterministic Source-ingress failure carried by an attempt."""

    if attempt is None:
        return None
    failure_class = attempt.termination_status
    if isinstance(failure_class, str) and failure_class.startswith(
        "source_preparation_"
    ):
        return failure_class
    return None


class SourceIdentifierPrivacyError(ValueError):
    """A submitted artifact retained a private Source spelling at one JSON path.

    The diagnostic deliberately exposes only the artifact path.  Returning the
    original spelling to a source-aware owner would defeat the anonymous naming
    boundary that this check protects.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"Lift response retained a source identifier at {path}"
        )


def _annotation_is_constexpr(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return ast.unparse(annotation).endswith("constexpr")


def anonymize_source(
    source: str,
    *,
    source_symbol: str | None = None,
    bound_constexpr_parameters: tuple[str, ...] = (),
    additional_compiler_slots: tuple[str, ...] = (),
) -> tuple[str, SCRCHarnessBinding, dict[str, str]]:
    """Alpha-rename source identifiers before the Lift provider sees them.

    ``source_symbol`` is a privileged source-selection coordinate.  It chooses
    the callable whose positional interface becomes ``scrc_k0`` but is never
    placed in J, C, pi, or the provider prompt.  This is required for real
    library modules that contain helpers and several kernels; declaration order
    in a Python file is not a semantic kernel identity.
    """

    tree = ast.parse(source)
    functions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    if not functions:
        raise ValueError("Lift source contains no function")
    if source_symbol is not None:
        if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
            raise ValueError("Lift source_symbol must be a Python identifier")
        selected = [
            function for function in functions if function.name == source_symbol
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Lift source_symbol {source_symbol!r} must resolve exactly once; "
                f"found {len(selected)} definitions"
            )
        primary = selected[0]
        ordered_functions = [
            primary, *(item for item in functions if item is not primary)
        ]
    else:
        primary = functions[0]
        ordered_functions = functions
    primary_source_name = primary.name
    mapping: dict[str, str] = {}
    for index, function in enumerate(ordered_functions):
        mapping.setdefault(function.name, f"scrc_k{index}")

    bound_constexprs = set(bound_constexpr_parameters)
    if len(bound_constexprs) != len(bound_constexpr_parameters) or any(
        not isinstance(name, str) or not name.isidentifier()
        for name in bound_constexpr_parameters
    ):
        raise ValueError(
            "bound constexpr parameters must be unique identifiers"
        )
    annotated_constexprs = {
        argument.arg
        for argument in primary.args.args
        if _annotation_is_constexpr(argument.annotation)
    }
    unknown_bound_constexprs = bound_constexprs - annotated_constexprs
    if unknown_bound_constexprs:
        raise ValueError(
            "bound constexpr parameters are absent from the selected ABI: "
            + ", ".join(sorted(unknown_bound_constexprs))
        )
    if (
        len(additional_compiler_slots) != len(set(additional_compiler_slots))
        or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in additional_compiler_slots
        )
    ):
        raise ValueError(
            "additional compiler slots must be unique identifiers"
        )

    runtime_parameters = tuple(
        argument.arg
        for argument in primary.args.args
        if not _annotation_is_constexpr(argument.annotation)
        or argument.arg in bound_constexprs
    )
    source_compiler_slots = tuple(
        argument.arg
        for argument in primary.args.args
        if _annotation_is_constexpr(argument.annotation)
        and argument.arg not in bound_constexprs
    )
    if set(source_compiler_slots) & set(additional_compiler_slots):
        raise ValueError(
            "additional compiler slots overlap Source constexpr compiler slots"
        )
    compiler_slots = (*source_compiler_slots, *additional_compiler_slots)
    # The selected source ABI is allocated by the shared authority.  Source
    # locals deliberately do not use a canonical J handle namespace: ``v*``
    # belongs exclusively to J SSA values, and using it in anonymous Source
    # creates a second, visually identical value namespace for Lift.
    mapping.update(
        CanonicalHandleAllocator.abi_mapping(runtime_parameters, compiler_slots)
    )

    source_identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.id, str)
    }
    source_identifiers.update(
        argument.arg
        for function in functions
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if isinstance(argument.arg, str)
    )
    source_identifiers.update(
        function.name for function in functions if isinstance(function.name, str)
    )
    # Adapter-only compiler options are ABI/C declarations, not Source lexical
    # values.  Mapping one onto a local identifier would make a ``c*`` carrier
    # appear in the anonymous Source program, so reject that ambiguous ingress
    # instead of silently rewriting the local as a compiler slot.
    external_slot_names = set(additional_compiler_slots)
    if external_slot_names & source_identifiers:
        raise ValueError(
            "additional compiler slots must not shadow Source identifiers"
        )
    local_serial = 0

    def allocate_local_value() -> str:
        """Allocate a privacy-safe Source-local name outside J namespaces."""

        nonlocal local_serial
        while True:
            candidate = f"src_local_{local_serial}"
            local_serial += 1
            if candidate not in source_identifiers and candidate not in mapping.values():
                return candidate

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
    reserved = (
        set(keyword.kwlist)
        | set(dir(builtins))
        | imported
        | {"triton", "tl", "libdevice", "torch", "math"}
    )
    local_nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id not in reserved
            and node.id not in mapping
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in local_nodes:
        if node.id not in mapping:
            mapping[node.id] = allocate_local_value()

    # Keyword names are identifiers only for the statically selected local
    # callee; they are not ordinary Name nodes.  Record each local function's
    # own formal mapping so calls can be renamed without touching tl.*, library,
    # attribute, dynamic, or unresolved call keywords.
    local_call_keyword_mappings: dict[str, dict[str, str]] = {}
    plain_local_functions = {
        function.name
        for function in functions
        if not function.decorator_list
    }
    for function in functions:
        parameters = (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        for argument in parameters:
            if argument.arg not in mapping:
                mapping[argument.arg] = allocate_local_value()
        local_call_keyword_mappings[function.name] = {
            argument.arg: mapping[argument.arg] for argument in parameters
        }

    class _AlphaRenamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = mapping.get(node.arg, node.arg)
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            plain_callee = (
                isinstance(node.func, ast.Name)
                and node.func.id in plain_local_functions
            )
            formal_mapping = (
                local_call_keyword_mappings.get(node.func.id)
                if isinstance(node.func, ast.Name)
                else None
            )
            if formal_mapping is None:
                return self.generic_visit(node)
            node.func = self.visit(node.func)
            node.args = [
                argument
                if plain_callee
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                else self.visit(argument)
                for argument in node.args
            ]
            for item in node.keywords:
                if item.arg is not None:
                    item.arg = formal_mapping.get(item.arg, item.arg)
                if not (
                    plain_callee
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    item.value = self.visit(item.value)
            return node

        def visit_keyword(self, node: ast.keyword) -> ast.AST:
            # Unknown and attribute/library call keyword spellings are public
            # API, not source identifiers. Local calls are handled by
            # visit_Call with their statically resolved callee formal map.
            return self.generic_visit(node)

        def visit_Global(self, node: ast.Global) -> ast.AST:
            node.names = [mapping.get(name, name) for name in node.names]
            return node

        def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
            node.names = [mapping.get(name, name) for name in node.names]
            return node

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
            if node.name is not None:
                node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                node.value = _replace_names(node.value, mapping)
            return node

    anonymous_tree = _AlphaRenamer().visit(tree)
    ast.fix_missing_locations(anonymous_tree)
    anonymous = ast.unparse(anonymous_tree) + "\n"
    binding = SCRCHarnessBinding(
        source_to_opaque=tuple(mapping.items()),
        source_function=primary_source_name,
    )
    return anonymous, binding, mapping


def _mask_native_noncode(source: str) -> str:
    """Mask comments and quoted literals before native identifier rewriting."""

    result = list(source)
    index = 0
    state = "normal"
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "normal":
            if character == "/" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if character == "/" and following == "*":
                result[index] = result[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if character in {"\"", "'"}:
                result[index] = " "
                state = character
            index += 1
            continue
        if state == "line_comment":
            if character == "\n":
                state = "normal"
            elif character not in {"\r"}:
                result[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if character == "*" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "normal"
                continue
            if character not in {"\n", "\r"}:
                result[index] = " "
            index += 1
            continue
        if character == "\\":
            result[index] = " "
            if index + 1 < len(source) and source[index + 1] not in {"\n", "\r"}:
                result[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        if character == state:
            result[index] = " "
            state = "normal"
        elif character not in {"\n", "\r"}:
            result[index] = " "
        index += 1
    return "".join(result)


def anonymize_native_source(
    source: str,
    *,
    source_symbol: str,
    source_parameter_order: tuple[str, ...],
    source_to_opaque: Mapping[str, str],
) -> tuple[str, SCRCHarnessBinding, dict[str, str]]:
    """Alpha-rename one verified native device projection without Python parsing."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("native Lift source must be non-empty")
    if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
        raise ValueError("native Lift source symbol must be an identifier")
    parameter_order = tuple(source_parameter_order)
    if len(parameter_order) != len(set(parameter_order)) or any(
        not isinstance(name, str) or not name.isidentifier()
        for name in parameter_order
    ):
        raise ValueError("native Lift source parameter order must be unique identifiers")
    mapping = dict(source_to_opaque)
    if not set(parameter_order) <= set(mapping):
        raise ValueError(
            "native Lift source mapping must cover every selected ABI parameter"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("native Lift source mapping must be one-to-one")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"(?:p|c)[0-9]+", value) is None
        for value in mapping.values()
    ):
        raise ValueError("native Lift source mapping must use p*/c* handles")
    masked = _mask_native_noncode(source)
    symbol_matches = tuple(
        re.finditer(
            rf"(?<![A-Za-z0-9_]){re.escape(source_symbol)}(?![A-Za-z0-9_])",
            masked,
        )
    )
    if len(symbol_matches) != 1:
        raise ValueError(
            "native Lift source must contain exactly one selected device entry symbol"
        )
    replacements = {source_symbol: "scrc_k0", **mapping}
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", masked):
        replacement = replacements.get(match.group(0))
        if replacement is None:
            continue
        pieces.append(source[cursor:match.start()])
        pieces.append(replacement)
        cursor = match.end()
    pieces.append(source[cursor:])
    anonymous = "".join(pieces)
    binding = SCRCHarnessBinding(
        source_to_opaque=tuple(mapping.items()),
        source_function=source_symbol,
    )
    return anonymous, binding, mapping


def _replace_names(expression: str, mapping: Mapping[str, str]) -> str:
    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: mapping.get(match.group(0), match.group(0)),
        expression,
    )


def _anonymous_compiler_view(
    view: SCRCCompilerView | None,
    mapping: dict[str, str],
    *,
    source_facts: SourceFacts | None = None,
) -> dict[str, Any] | None:
    if view is None:
        return None
    if not isinstance(source_facts, SourceFacts):
        raise ValueError(
            "anonymous compiler projection requires canonical SourceFacts"
        )
    names = list(
        dict.fromkeys(
            name
            for name in [*view.schedule_values, *view.compiler_slots]
            if not (name in mapping and mapping[name].startswith("c"))
        )
    )
    authority_mapping = dict(source_facts.source_to_opaque)
    canonical_compiler_slots = set(source_facts.compiler_slots)
    for name in names:
        # The caller may provide either the original Harness spelling or the
        # already-anonymized compiler view.  Both must resolve through the
        # SourceFacts-owned ABI map; no prompt/digest path may allocate a
        # second c* namespace.
        opaque = authority_mapping.get(name)
        if opaque is None and name in canonical_compiler_slots:
            opaque = name
        if not isinstance(opaque, str) or not opaque.startswith("c"):
            raise ValueError(
                "canonical SourceFacts lacks a compiler-slot binding "
                f"for {name!r}"
            )
        existing = mapping.get(name)
        if existing is not None and existing != opaque:
            raise ValueError(
                "source anonymization differs from canonical SourceFacts"
            )
        mapping[name] = opaque
    return {
        "schedule_values": {
            mapping[name]: value for name, value in view.schedule_values.items()
        },
        "schedule_value_types": {
            mapping[name]: view.schedule_value_types[name]
            for name in view.schedule_values
        },
        "bound_constexpr_values": {
            mapping[name]: value
            for name, value in view.bound_constexpr_values.items()
        },
        "bound_constexpr_value_types": {
            mapping[name]: view.bound_constexpr_value_types[name]
            for name in view.bound_constexpr_values
        },
        "bound_constexpr_shape_coordinates": {
            mapping[name]: coordinate
            for name, coordinate in view.bound_constexpr_shape_coordinates.items()
        },
        "compiler_slots": [mapping[name] for name in view.compiler_slots],
        "compiler_roles": {
            mapping[name]: view.schedule_roles[name]
            for name in view.schedule_values
        },
        "schedule_tile_coordinates": {
            mapping[name]: coordinate
            for name, coordinate in view.schedule_tile_coordinates.items()
        },
        "host_launch_contract": project_host_launch_contract(view.host_launch_contract, mapping),
        "aliases": [
            {
                "left": mapping[item.left],
                "right": mapping[item.right],
                "kind": item.kind.value,
            }
            for item in view.aliases
        ],
        "arguments": [
            {**item.to_json(), "name": mapping[item.name]}
            for item in view.arguments
        ],
        "shape_coordinates": dict(view.shape_coordinates),
        "grid": list(view.grid),
        "grid_contract": [
            _replace_names(item, mapping) for item in view.grid_contract
        ],
        "grid_shape_coordinates": list(view.grid_shape_coordinates),
        "parameter_order": [mapping[name] for name in view.parameter_order],
        "constexpr_parameters": [
            mapping[name] for name in view.constexpr_parameters
        ],
    }


def _frontend_lift_guidance(frontend: str | None) -> dict[str, str]:
    """Return compiler-observer wording for one isolated frontend branch.

    The Source-to-J contract is shared, but compiler evidence is not.  Only
    Triton owns the TTIR observer; CuTe, TileLang, and CUDA retain their own
    compiler/source observations.
    """

    selected = str(frontend or "triton").casefold()
    if selected == "triton":
        return {
            "layout_reference": "TTIR layout",
            "numeric_query": "the exact TTIR def-use",
            "fused_query": "the exact Source-TTIR operand/result def-use",
            "index_query": "the exact TTIR def-use",
            "index_width": "Source-TTIR",
            "pi_query": "the exact pi schema leaves and raw Triton realization ledger",
        }
    if selected == "cuda":
        return {
            "layout_reference": "the selected CUDA compiler/Harness layout",
            "numeric_query": "the selected CUDA compiler observation",
            "fused_query": "the selected CUDA compiler/source observation",
            "index_query": "the selected CUDA compiler observation",
            "index_width": "selected CUDA compiler",
            "pi_query": "the exact pi schema leaves and native CUDA source/compiler facts",
        }
    if selected == "cute":
        return {
            "layout_reference": "the selected CuTe compiler layout",
            "numeric_query": "the selected CuTe compiler observation",
            "fused_query": "the selected CuTe MLIR/PTX observation",
            "index_query": "the selected CuTe compiler observation",
            "index_width": "selected CuTe compiler",
            "pi_query": "the exact pi schema leaves and native CuTe compiler facts",
        }
    if selected == "tilelang":
        return {
            "layout_reference": "the selected TileLang/TIR compiler layout",
            "numeric_query": "the selected TileLang/TIR compiler observation",
            "fused_query": "the selected TileLang/TIR compiler observation",
            "index_query": "the selected TileLang/TIR compiler observation",
            "index_width": "selected TileLang/TIR compiler",
            "pi_query": "the exact pi schema leaves and native TileLang/TIR compiler facts",
        }
    return {
        "layout_reference": "the selected frontend compiler layout",
        "numeric_query": "the selected frontend compiler observation",
        "fused_query": "the selected frontend compiler/source observation",
        "index_query": "the selected frontend compiler observation",
        "index_width": "selected frontend compiler",
        "pi_query": "the exact pi schema leaves and selected frontend source/compiler facts",
    }


def _source_aware_build_scrc_lift_instructions(
    *,
    frontend: str | None = "triton",
    authoring_transport: str = "mcp",
) -> str:
    """Route one fresh author to its transport's single schema authority."""

    if authoring_transport not in {"mcp", "responses"}:
        raise ValueError("Lift authoring transport must be 'mcp' or 'responses'")
    guidance = _frontend_lift_guidance(frontend)
    if authoring_transport == "responses":
        entry = """Return one JSON object, no prose. Read the read-only
`source_input_channel` first, then copy
`semantic_authoring_entry.candidate_template`; complete J and optional pi using
that entry's `compact_provider_wire` and `semantic_construction_requirements`.
No MCP calls. Keep pi empty unless every row has a closed class and scope. Do
not copy host-owned c or schema into the response; the host closes those fields
from the input channel."""
    else:
        entry = """Use only the exposed MCP action. Read brief, then ledger;
read each exposed missing `op:NAME` once before begin_candidate. Sequence:
begin, graph, post-graph ledger read, accesses, outputs, finalize.
Each declaration is one JSON value in its named *_text; no free JSON or patches.
Brief owns operation/control/returns; ledger owns ordered boundary facts."""
    return f"""One fresh IntentIR Lift; no prior candidate/response/draft/trajectory reuse.
{entry}

Before returning, apply this compact admission gate. If any item cannot be
proved from the supplied Source/ledger/outline, leave that claim unproved;
never invent a node, owner, producer, predicate, coordinate, identity or
partition:
1. Every owner-required access has exactly one Source-matching local `node_ref`
   and producer; `effect_node_ref` is the required same local op, or the exact
   template value for non-owner rows.
2. Every outlined operation, operand order, load occurrence, cast, reduction,
   store value, shape and dtype is mirrored once in J. No duplicate roots or
   ambiguous producer IDs are allowed.
3. Every broadcast consumer/operand has exactly one boundary relation with the
   Source target-axis spelling; scalar broadcasts still require their row.
4. Follow each transfer row's `predicate_requirement` and
   `semantic_guard_authorization` exactly. A computed guard needs one
   Source-compatible i1 `map_binary(lt)` producer and one positive access
   relation; never infer missing axis/domain facts or claim an unproved guard.
5. Every load coordinate, domain, predicate, fill, reduction axis/combiner and
   typed dataflow is Source-proven; no numeric run-parameter equality is proof.
6. Each logical output has one exact final binding. Multiple physical stores
   require a complete, disjoint, gap-free Source-proven partition with matching
   effect/target/predicate/dtype/shape; one store is never promoted to owner.
7. Run a final reference check: all IDs resolve, all operands dominate, all
   required rows are present, and every listed Source output has exactly one
   matching `j.boundary.output_mappings` row; when outputs are listed, an
   empty output-mapping array is invalid. No host-owned token/ABI/C/handle
   field is submitted. Return JSON only.

If `relift` is present, follow each current diagnostic's
`repair_instruction` and `deterministic_repair_checklist`. Treat
`repair_locators` as Source facts and map them only onto this fresh graph's
local labels. Preserve every `open_guidance` obligation unless explicitly
resolved; absence from a newer diagnostic is not proof of repair.
`resolved_guidance` is history, not permission to reuse prior content. Build a
wholly fresh J from current Source/ledger/outline; never copy a prior
candidate, draft, response, trajectory, Replay source or repair base.

Complete typed, producer-ordered SSA and loop/branch wiring; use admitted ops,
listed t* inputs/prior results and local op*/r*/dim*/access* names. No host
schema/C/ABI/signatures or canonical n*/v*/x*/p*/c*/a*/e* handles.
Source and {guidance["layout_reference"]} are evidence, not output/Replay templates.

Emit required access/output rows in order. Copy only the fixed fields, labels,
ordinals, nulls and provider-visible access/guard receipts in wire_template;
fill only provider-owned fields. For an owner-required store or update,
node_ref and effect_node_ref are the same one local op* id; for every other
row preserve the template's effect_node_ref, including null. Do not submit
boundary.effects: the host builds that table from these checked row fields.
Omit semantic_owner_authorization entirely:
the host checks node_ref against its private Source ledger and attaches the
owner receipt only to the completed internal artifact.
For an issued owner authoring_outline, mirror its logical typed DAG in J:
preserve operation/operand order, shared producers, shapes, load occurrences,
casts, broadcasts, reductions and store value; bind node_ref to its unique root.
When several outlined stores target one output, each root is a physical slice
producer, not the final logical output; preserve every outlined coordinate and
let the canonical functional partition proof establish the final binding.
When `source_logical_output_assembly_requirements` is present, author exactly
one top-level canonical `concat` using its Source-proven axis, ordered slice
operands, operand type and result type. The concat is the logical-output
assembly; a single slice or output mapping is never sufficient.
In the transfer ledger this appears as source_owner_authoring_outline. The
begin_candidate receipt may also expose the same outline under the matching
output ordinal. The outline is guidance, not a J row or permission to infer
missing semantics.
Every outlined broadcast needs its exact boundary.broadcasts relation on the
consuming operation and operand; implicit pointwise broadcasting is insufficient
for Source owner proof. Keep the original operand on that consuming operation;
an explicit broadcast_in_dim operation is not this owner profile's proof.
For rank-bearing operands, its TARGET axes are the
complement of the outline's logical broadcast source-axis positions. For a
scalar operand, keep the relation row even though J accepts implicit scalar
broadcast; the scalar has an empty input map regardless of [] or listed axes.
Copy each outline target_axes array into the corresponding
`j.boundary.broadcasts[]` row's `axes` field as unquoted JSON integer
positions. `target_axes` is an outline field, not a J boundary field; never
write quoted numerals, dim* or x* dimension labels as axes.
Each transfer-row owner_broadcast_requirements entry locates the consuming
outline graph node and operand position; author a boundary.broadcasts row with
your corresponding local consumer id and operand result. Its target_axes are
the Source-proven rank-insertion spelling, not host-authored J content.
index_expression is a Python list literal of exact tensor rank: each axis is
one logical coordinate name from program_domain_requirements.domain_coordinates
or an in-range integer for a proved static slice. A dim* label is an extent,
not a coordinate; do not use arithmetic or a flat pointer offset here.
When an access row has index_expression_requirement, use that exact proven
string as its index_expression; keep authoring the field in the access row.
In an issued owner outline, store.coordinates axisN means the proved coordinate
for store.shape[N]; integer coordinates remain that exact integer slice.
Never invent owners or transfer receipts. Predicate placeholders require prior
i1 results except relation-backed computed guards, whose access predicate is
literal "true".
For an owner outline's symbolic-axis lt guard, author map_binary(fn=lt) with
an i1 result shaped ONLY by that one symbolic axis. Its ordered operands are
an i32 iota of that exact dimension and the Source-bound scalar i32 value
whose value_dimension_relations row equates it to that dimension. The iota
has operands=[], type dtype=i32 with shape=[that dimension], and attrs
containing n=that dimension (optionally dtype=i32); n is not the tile
capacity or an input value label. A declared
broadcast on this comparison may expand only that scalar operand (either []
or [0]); a coordinate-operand row may only restate identity with axes=[].
Bind the i1 result to the same access with a positive
`access_guard_relations` row. Its `axes` are the access TARGET axes inserted
when expanding the one-axis i1 guard to the full tensor: for a two-axis
access guarded along axis 1 use [0], and along axis 0 use [1]. They are not
the guard's retained axis or the `boundary.broadcasts` operand axes;
do not compare an already expanded two-dimensional guard. With a relation,
the boundary access `predicate` is the literal string "true"; the i1 value
belongs only in `access_guard_relations.value_ref`, not also in `predicate`.
If those existing facts cannot be expressed, leave the claim unproved.
For an outlined scalar load from a rank-one cache at a fixed coordinate,
author a scalar `gather` with attrs `axis=0,mode=take` and a prior scalar
i32/i64 `const` index equal to that Source-proven coordinate. The index
`const` must have operands=[], type shape=[], and attrs containing only the
integer `value` (omit optional attrs dtype/shape); pass its result as the
gather's second operand. Do not use a bare input value or infer an index not
present in the outline.
Supply every consumer_extensions edge and matching reduction contract. In
the supported owner-padding profile their frontend-local `tile_axis` is 0
on both rows, even when Layer-A `reduce.attrs.axes` names logical axis 1;
the reduction identity and typed outside value must match the existing
Source owner obligation exactly.

If program_domain_requirements.status is pass, omit domains entirely: the host
projects the complete Source map. Never submit a partial domain map.
Every scatter needs its exact update row. Host derives index_bounds, ordinary
exact/tolerance result relations, returns and output attachments from the
closed graph; leave result_relations empty unless an actual one_of update
requires its explicit one_of_updates ownership relation.
When multiple Source-owned stores jointly realize one output-only scatter,
the scatter base, values, and result must each have the exact full logical
output dtype and symbolic shape; indices have that same shape and an integer
dtype. Form values from the already authored, typed store value slices in
their Source-proven coordinates. A row slice or current Harness shape is not
the complete logical output. If this full relation cannot be expressed,
leave the partition unproved.
Preserve control, transfer/effect order, predicates, fills, outputs and
value/dimension relations."""


def build_scrc_lift_instructions(
    *,
    frontend: str | None = "triton",
    authoring_transport: str = "mcp",
) -> str:
    """Return the source-aware authoring contract for the selected frontend."""

    return _source_aware_build_scrc_lift_instructions(
        frontend=frontend,
        authoring_transport=authoring_transport,
    )


def _build_user_input(
    anonymous_source: str,
    compiler_view: dict[str, Any] | None,
    relift_context: SCRCReliftContext | None,
    source_kind: str,
    frontend: str | None,
    construction: CanonicalConstruction,
) -> str:
    payload: dict[str, Any] = {
        "anonymous_source": anonymous_source,
        "response_mode": "json_object",
        "source_kind": source_kind,
    }
    if frontend is not None:
        payload["frontend"] = str(frontend)
    # Responses Lift has no local MCP resource reader. Mount the same
    # schema-derived semantic authoring entry as MCP, never the mechanical
    # Source-bound seed. Unresolved operation semantics remain provider-owned;
    # host construction derives the ABI/C/handle/scope/output projections.
    source_wire = canonical_source_wire_bundle(construction)
    profile = source_wire["profile"]
    first_turn = profile["first_turn"]
    operation_names = first_turn["operation_names"]
    boundary_row_names = first_turn["boundary_rows"]
    if (
        not isinstance(operation_names, list)
        or not isinstance(boundary_row_names, list)
        or any(
            not isinstance(name, str)
            or name not in profile["operation_contracts"]
            for name in operation_names
        )
        or any(
            not isinstance(name, str) or name not in profile["boundary_rows"]
            for name in boundary_row_names
        )
    ):
        raise TypeError("canonical first-turn semantic wire is malformed")
    compact_wire = profile.get("semantic_provider_wire_compact")
    semantic_candidate = profile.get("semantic_candidate")
    if not isinstance(compact_wire, Mapping) or not isinstance(
        semantic_candidate, Mapping
    ):
        raise TypeError("canonical profile lacks compact semantic provider wire")
    # MCP and direct Responses receive this same schema-derived compact
    # projection. The completed artifact schema remains local authority; the
    # compact wire already contains operation fields, so do not attach the
    # larger semantic_operation_contracts duplicate.
    payload["semantic_authoring_entry"] = {
        "candidate_template": canonical_semantic_candidate_template(),
        "semantic_construction_requirements": source_wire[
            "semantic_construction_requirements"
        ],
        # Anonymous Source argument positions are locators, not J operands.
        # Mount the sole SourceFacts/allocator-derived bridge so direct
        # Responses cannot mistake visible ABI names for legal J values.
        "semantic_operand_inputs": copy.deepcopy(
            source_wire["semantic_operand_inputs"]
        ),
        "program_domain_requirements": copy.deepcopy(
            source_wire["program_domain_requirements"]
        ),
        "provider_dimension_aliases": source_wire[
            "provider_dimension_aliases"
        ],
        "compact_provider_wire": copy.deepcopy(dict(compact_wire)),
    }
    # Direct Responses has no resource-read round trip.  Give it one snapshot
    # of the same provider-visible MCP Source/Harness views instead of keeping
    # a second, reduced prompt-only fact list.  The snapshot is context only;
    # the provider still authors only semantic J/pi.
    payload["source_input_channel"] = build_direct_agent_mcp_snapshot(
        source_wire=source_wire,
        anonymous_source=anonymous_source,
        compiler_view=compiler_view,
    )
    if compiler_view is not None:
        # Compiler/Harness ABI rows are host-owned construction facts.  Direct
        # Responses sees only the selected frontend label; publishing the full
        # view here would recreate the p*/c* authoring surface removed from
        # MCP and let the two transports drift.
        payload["compiler_realization"] = {
            "source_kind": source_kind,
            "selected_frontend": str(frontend or "unknown"),
        }
    if relift_context is not None:
        payload["relift"] = relift_context.to_json()
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _prompt_material(
    source: str,
    compiler_view: SCRCCompilerView | None,
    relift_context: SCRCReliftContext | None,
    construction: CanonicalConstruction,
    source_symbol: str | None = None,
    *,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    frontend: str | None = "triton",
    source_facts: SourceFacts | None = None,
    authoring_transport: str = "mcp",
) -> tuple[
    str,
    str,
    SCRCHarnessBinding,
    dict[str, Any] | None,
]:
    if not isinstance(construction, CanonicalConstruction):
        raise TypeError("Lift prompt preparation requires canonical construction")
    if source_kind == "python":
        anonymous_source, binding, mapping = anonymize_source(
            source,
            source_symbol=source_symbol,
            bound_constexpr_parameters=(
                tuple(compiler_view.bound_constexpr_values)
                if compiler_view is not None
                else ()
            ),
        )
    elif source_kind == "cuda":
        if source_symbol is None or source_to_opaque is None:
            raise ValueError(
                "CUDA Lift preparation requires an explicit source symbol and ABI mapping"
            )
        anonymous_source, binding, mapping = anonymize_native_source(
            source,
            source_symbol=source_symbol,
            source_parameter_order=source_parameter_order,
            source_to_opaque=source_to_opaque,
        )
    else:
        raise ValueError(f"unsupported Lift source kind: {source_kind!r}")
    anonymous_view = _anonymous_compiler_view(
        compiler_view,
        mapping,
        source_facts=source_facts,
    )
    # Re-freeze the binding after Source and compiler projections have joined.
    # Both are required to agree with the SourceFacts-owned ABI map; this path
    # never allocates a preview namespace.
    binding = SCRCHarnessBinding(
        source_to_opaque=tuple(mapping.items()),
        source_function=binding.source_function,
        replay_function=binding.replay_function,
    )
    instructions = build_scrc_lift_instructions(
        frontend=frontend,
        authoring_transport=authoring_transport,
    )
    user = _build_user_input(
        anonymous_source,
        (
            compiler_view.to_json()
            if isinstance(compiler_view, SCRCCompilerView)
            else dict(compiler_view)
            if isinstance(compiler_view, Mapping)
            else None
        ),
        relift_context,
        source_kind,
        frontend,
        construction,
    )
    return instructions, user, binding, anonymous_view


def prepare_scrc_lift(
    source: str,
    *,
    compiler_view: SCRCCompilerView | None = None,
    relift_context: SCRCReliftContext | None = None,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    source_import_roots: tuple[str, ...] = (),
    frontend: str | None = "triton",
    source_facts: SourceFacts | None = None,
    construction: CanonicalConstruction | None = None,
    authoring_transport: str = "mcp",
) -> SCRCLiftRequest:
    if not isinstance(source_facts, SourceFacts):
        raise TypeError("Lift preparation requires canonical SourceFacts")
    if not isinstance(construction, CanonicalConstruction):
        raise TypeError("Lift preparation requires canonical construction")
    if construction.source_facts.digest != source_facts.digest:
        raise ValueError("Lift construction differs from the owning SourceFacts")
    actual_source_slice_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        isinstance(source_facts, SourceFacts)
        and source_facts.source_slice_digest is not None
        and source_facts.source_slice_digest != actual_source_slice_digest
    ):
        raise ValueError("Lift Source text differs from the owning SourceFacts slice")
    instructions, user, binding, anonymous_view = _prompt_material(
        source,
        compiler_view,
        relift_context,
        construction,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        frontend=frontend,
        source_facts=source_facts,
        authoring_transport=authoring_transport,
    )
    if not isinstance(anonymous_view, Mapping):
        raise ValueError("Lift preparation lacks the canonical compiler projection")
    if dict(anonymous_view) != source_facts.compiler_view():
        raise ValueError(
            "prepared anonymous Source facts differ from the owning SCRC request"
        )
    expected_abi_mapping = dict(source_facts.source_to_opaque)
    prepared_abi_mapping = {
        source_name: opaque
        for source_name, opaque in binding.source_to_opaque
        if isinstance(opaque, str) and re.fullmatch(r"(?:p|c)[0-9]+", opaque)
    }
    if prepared_abi_mapping != expected_abi_mapping:
        raise ValueError("prepared anonymous ABI differs from the owning SourceFacts")
    # Construction originates once at Source/Harness ingress.  Lift preparation
    # only validates that its anonymous view is an exact projection of that
    # authority; it never rebuilds facts, the ledger, or the allocator.
    return SCRCLiftRequest(
        instructions,
        user,
        hashlib.sha256((instructions + "\0" + user).encode()).hexdigest(),
        binding,
        tuple(source_import_roots),
        source,
        (
            compiler_view.to_json()
            if isinstance(compiler_view, SCRCCompilerView)
            else dict(compiler_view)
            if isinstance(compiler_view, Mapping)
            else None
        ),
        source_facts,
        construction,
    )


def scrc_lift_prompt_sha256(
    source: str,
    name_hint: str | None = None,
    *,
    compiler_view: SCRCCompilerView | None = None,
    construction: CanonicalConstruction,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    frontend: str | None = "triton",
    authoring_transport: str = "responses",
) -> str:
    del name_hint
    instructions, user, _binding, _view = _prompt_material(
        source,
        compiler_view,
        None,
        construction,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        frontend=frontend,
        source_facts=construction.source_facts,
        authoring_transport=authoring_transport,
    )
    return hashlib.sha256((instructions + "\0" + user).encode()).hexdigest()


def scrc_relift_prompt_sha256(
    source: str,
    context: SCRCReliftContext,
    *,
    compiler_view: SCRCCompilerView | None = None,
    construction: CanonicalConstruction,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    frontend: str | None = "triton",
    authoring_transport: str = "responses",
) -> str:
    instructions, user, _binding, _view = _prompt_material(
        source,
        compiler_view,
        context,
        construction,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        frontend=frontend,
        source_facts=construction.source_facts,
        authoring_transport=authoring_transport,
    )
    return hashlib.sha256((instructions + "\0" + user).encode()).hexdigest()


def _json_object(text: str) -> dict[str, Any]:
    match = _FENCE.search(text)
    if match:
        text = match.group(1)

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Lift response contains non-JSON number {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Lift response contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        text,
        parse_constant=reject_nonfinite,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise TypeError("Lift response must be a JSON object")
    return parsed


def _canonical_candidate_text(payload: Mapping[str, Any]) -> str:
    """Serialize one internally repaired candidate without provider prose."""

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _record_deterministic_repair(
    repairs: list[dict[str, Any]],
    *,
    rule_code: str,
    path: str,
    action: str,
) -> None:
    repairs.append(
        {
            "rule_code": rule_code,
            "path": path,
            "action": action,
            "authority": "closed source-free ABI/schema fact",
        }
    )


def _rewrite_c_slot_references(
    payload: dict[str, Any],
    mappings: Mapping[str, str],
) -> None:
    """Remove retired J ownership aliases from the physical C ledger.

    ``c*`` owns only Source/Harness schedule values.  Any logical-axis
    association lives once in ``pi.iteration_bindings``.  Keep the historic
    function boundary because the canonical allocator calls it while it
    rewrites J handles, but C must never participate in that rewrite.
    """

    c = payload.get("c")
    slots = c.get("slots") if isinstance(c, Mapping) else None
    if not isinstance(slots, list):
        return
    for slot in slots:
        if isinstance(slot, dict):
            slot.pop("subject_ref", None)
            slot.pop("axis_ref", None)


def _valid_schedule_role(value: Any) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        return False
    return not any(
        term in value
        for term in {
            "api", "combiner", "conflict", "family", "fill", "identity",
            "index_map", "mask", "operator", "predicate", "realization",
            "rule", "semantic", "source", "strategy",
        }
    )


def _repair_compiler_c(
    payload: dict[str, Any],
    view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Rebuild only C's closed declaration/value ledger from trusted ABI facts."""

    if not isinstance(view, Mapping):
        return
    values = view.get("schedule_values")
    compiler_slots = view.get("compiler_slots")
    compiler_roles = view.get("compiler_roles")
    value_types = view.get("schedule_value_types", {})
    current = payload.get("c")
    if (
        not isinstance(values, Mapping)
        or not isinstance(compiler_slots, list)
        or not isinstance(compiler_roles, Mapping)
        or not isinstance(value_types, Mapping)
        or not isinstance(current, Mapping)
        or not isinstance(current.get("slots"), list)
    ):
        return
    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    # JSON object key order is not the C ABI order. In particular, a serialized
    # object containing c10 is commonly materialized as c0,c1,c10,c2,... . The
    # sole order is the Source-owned construction allocator; deriving c* from
    # the candidate or from the number of values would establish a second C
    # namespace authority.
    names = list(allocator.declared("c"))
    if (
        set(values) != set(names)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"c[0-9]+", name) is None
            for name in compiler_slots
        )
        or len(compiler_slots) != len(set(compiler_slots))
        or set(compiler_slots) - set(names)
        or set(value_types) != set(names)
        or set(compiler_roles) != set(names)
    ):
        return
    compiler_set = set(compiler_slots)
    for name in names:
        role = compiler_roles.get(name)
        if not _valid_schedule_role(role):
            # There is no safe fallback for a public compiler-option role.
            return

    current_by_name: dict[str, Mapping[str, Any]] = {}
    for raw_slot in current["slots"]:
        if not isinstance(raw_slot, Mapping):
            continue
        name = raw_slot.get("name")
        if not isinstance(name, str):
            continue
        if name in current_by_name:
            return
        current_by_name[name] = raw_slot

    slots: list[dict[str, Any]] = []
    for name in names:
        kind = "compiler" if name in compiler_set else "constexpr"
        role = compiler_roles[name]
        value_type = value_types.get(name)
        if not isinstance(value_type, str) or not value_type:
            # SourceFacts closes this ledger before ingress repair.  Keeping
            # this guard here prevents a stale mounted view from being
            # converted into a guessed C declaration if it bypasses that
            # constructor.
            return
        row: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "role": role,
            "value_type": value_type,
            # SourceFacts closes the C ledger completely. A provider cannot
            # turn a required slot optional to sidestep a missing ownership or
            # iteration binding; optionality in J remains a J contract.
            "required": True,
        }

        try:
            ScheduleSlot.from_json(row)
        except (SchemaError, TypeError, ValueError):
            return
        slots.append(row)

    rebuilt = {
        "slots": slots,
        "values": copy.deepcopy(dict(values)),
    }
    try:
        ImplementationConfig.from_json(rebuilt)
    except (SchemaError, TypeError, ValueError):
        return
    if dict(current) == rebuilt:
        return
    payload["c"] = rebuilt
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_c_ledger_repair",
        path="c",
        action="rebuild slot order/kind/role/value ledger from trusted compiler view",
    )


def _repair_canonical_abi_layout(
    payload: dict[str, Any],
    view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
) -> None:
    """Synchronize the physical ``p*`` ABI with the trusted Harness view.

    This is the only ingress repair that is allowed to touch the callable
    signature layout. SourceFacts and the canonical seed decide which ABI
    values have logical J bindings; the provider cannot turn a compiler-only
    ``c*`` slot into a callable parameter, leave a trusted scalar without its
    dtype/rank, or permute an already closed physical parameter order. No
    logical reference is invented here.
    """

    if not isinstance(view, Mapping):
        return
    j = payload.get("j")
    signature = j.get("signature") if isinstance(j, dict) else None
    parameters = signature.get("parameters") if isinstance(signature, dict) else None
    if not isinstance(j, dict) or not isinstance(signature, dict) or not isinstance(parameters, list):
        return

    if any(
        not isinstance(row, dict) or not isinstance(row.get("name"), str)
        for row in parameters
    ):
        return
    names = [str(row["name"]) for row in parameters]
    if len(names) != len(set(names)):
        # A duplicate physical declaration is not a safe permutation.
        return

    raw_order = view.get("parameter_order")
    trusted_order = [
        name
        for name in raw_order
        if isinstance(name, str) and re.fullmatch(r"p[0-9]+", name)
    ] if isinstance(raw_order, list) else []
    if not trusted_order:
        return
    if len(trusted_order) != len(set(trusted_order)):
        return

    trusted_c_names = {
        name
        for name in view.get("schedule_values", {})
        if isinstance(name, str) and re.fullmatch(r"c[0-9]+", name)
    } if isinstance(view.get("schedule_values"), Mapping) else set()
    by_name = {str(row["name"]): row for row in parameters}

    # A c* row can be removed only when the trusted C ledger owns it and the
    # row carries no J meaning.  If it has a logical_ref, leave the candidate
    # untouched and let CandidateCheck reject the cross-layer ownership rather
    # than silently deleting a semantic value.
    removable_c: set[str] = set()
    for name in trusted_c_names:
        row = by_name.get(name)
        if row is None:
            continue
        if row.get("logical_ref") is not None:
            return
        removable_c.add(name)

    changed = False
    retained = [row for row in parameters if str(row["name"]) not in removable_c]
    if len(retained) != len(parameters):
        changed = True

    remaining_names = [str(row["name"]) for row in retained]
    if (
        trusted_order
        and set(remaining_names) == set(trusted_order)
        and remaining_names != trusted_order
    ):
        retained = [by_name[name] for name in trusted_order]
        remaining_names = list(trusted_order)
        changed = True

    # Complete only fields whose values are already explicit in the trusted
    # Source/Harness argument manifest.  Conflicting authored values remain
    # untouched and stay fail-closed.
    raw_arguments = view.get("arguments")
    trusted_arguments: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_arguments, list):
        trusted_arguments = {
            str(row.get("name")): row
            for row in raw_arguments
            if isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and re.fullmatch(r"p[0-9]+", str(row.get("name")))
        }
    bound_types = view.get("bound_constexpr_value_types", {})
    if not isinstance(bound_types, Mapping):
        bound_types = {}
    bound_values = view.get("bound_constexpr_values", {})
    if not isinstance(bound_values, Mapping):
        bound_values = {}

    for row in retained:
        name = str(row["name"])
        trusted = trusted_arguments.get(name)
        expected_dtype = None
        expected_rank: int | None = None
        expected_mode = None
        if isinstance(trusted, Mapping):
            raw_dtype = trusted.get("dtype")
            if isinstance(raw_dtype, str) and raw_dtype:
                expected_dtype = _canonical_dtype(raw_dtype)
            kind = trusted.get("kind")
            raw_shape = trusted.get("shape")
            if kind == "tensor" and isinstance(raw_shape, list):
                expected_rank = len(raw_shape)
            elif kind == "scalar":
                expected_rank = 0
            raw_mode = trusted.get("mode")
            if isinstance(raw_mode, str) and raw_mode:
                expected_mode = raw_mode
        if name in bound_types and isinstance(bound_types[name], str):
            expected_dtype = _canonical_dtype(bound_types[name])
            expected_rank = 0
        elif name in bound_values:
            expected_dtype = canonical_scalar_binding_dtype(bound_values[name])
            expected_rank = 0

        if expected_dtype is not None and row.get("dtype") is None:
            row["dtype"] = expected_dtype
            changed = True
        if expected_rank is not None and row.get("rank") is None:
            row["rank"] = expected_rank
            changed = True
        if expected_mode is not None and row.get("mode") is None:
            row["mode"] = expected_mode
            changed = True

    if not changed:
        return
    signature["parameters"] = retained
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_canonical_abi_layout",
        path="j.signature.parameters",
        action=(
            "derive the p* signature order and typed scalar carriers from the "
            "trusted Harness view, while keeping compiler-only c* slots in C"
        ),
    )


def _validate_canonical_namespace_ownership(
    payload: Mapping[str, Any],
    view: Mapping[str, Any] | None,
) -> None:
    """Reject cross-layer physical handles before the strict semantic check."""

    j = payload.get("j")
    if not isinstance(j, Mapping):
        return
    signature = j.get("signature")
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if not isinstance(parameters, list):
        return
    parameter_names = {
        str(row.get("name"))
        for row in parameters
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    compiler_names = {
        name
        for name in (
            view.get("compiler_slots", [])
            if isinstance(view, Mapping)
            else []
        )
        if isinstance(name, str) and re.fullmatch(r"c[0-9]+", name)
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                visit(item, child)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        for match in re.finditer(
            r"(?<![A-Za-z0-9_])([pc][0-9]+)(?![A-Za-z0-9_])", value
        ):
            handle = match.group(1)
            physical_parameter_path = bool(
                re.fullmatch(
                    r"j\.signature\.parameters\[\d+\]\.(?:name|alias_of)",
                    path,
                )
            )
            if handle.startswith("c"):
                raise ValueError(
                    f"compiler-only handle {handle} leaked into J at {path}"
                )
            if handle.startswith("p") and not physical_parameter_path:
                raise ValueError(
                    f"physical ABI handle {handle} leaked into J at {path}"
                )

    visit(j, "j")
    if any(
        name not in parameter_names
        or not re.fullmatch(r"p[0-9]+", name)
        for name in parameter_names
    ):
        raise ValueError("J signature contains a non-p* physical parameter")
    if compiler_names & parameter_names:
        raise ValueError("compiler-only c* slot remains in J signature")


def _repair_bound_constexpr_parameter_kinds(
    payload: dict[str, Any],
    view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
) -> None:
    """Repair only a bound constexpr's ABI kind from the trusted view.

    A bound constexpr is still a physical ``p*`` signature parameter, but its
    kind is part of the closed Harness contract.  Providers sometimes spell
    an extent-only bound value as an ordinary scalar because both carry a
    rank-zero value.  The correction is safe only when the parameter already
    exists, has the scalar carrier shape/mode, and is either unbound or tied
    to a symbolic ``x*`` dimension.  No parameter, logical reference, dtype,
    or value is invented here.
    """

    if not isinstance(view, Mapping):
        return
    bound_values = view.get("bound_constexpr_values")
    if not isinstance(bound_values, Mapping) or not bound_values:
        return
    j = payload.get("j")
    signature = j.get("signature") if isinstance(j, Mapping) else None
    parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    if not isinstance(parameters, list):
        return
    by_name: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return
        name = parameter.get("name")
        if not isinstance(name, str) or name in by_name:
            return
        by_name[name] = parameter
    value_types = view.get("bound_constexpr_value_types", {})
    if not isinstance(value_types, Mapping):
        value_types = {}
    for name in bound_values:
        parameter = by_name.get(name)
        if parameter is None or parameter.get("kind") == "constexpr":
            continue
        if parameter.get("kind") != "scalar":
            continue
        if (
            parameter.get("mode") != "value"
            or parameter.get("rank") != 0
            or parameter.get("alias_of") is not None
        ):
            continue
        logical_ref = parameter.get("logical_ref")
        if logical_ref is not None and re.fullmatch(r"x[0-9]+", str(logical_ref)) is None:
            continue
        expected_dtype = value_types.get(name)
        if (
            isinstance(expected_dtype, str)
            and _canonical_dtype(parameter.get("dtype"))
            != _canonical_dtype(expected_dtype)
        ):
            continue
        parameter["kind"] = "constexpr"
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_bound_constexpr_kind_repair",
            path=f"j.signature.parameters[{name}].kind",
            action="restore bound non-C constexpr ABI kind from trusted Harness view",
        )


def _repair_bound_constexpr_parameter_rank(
    payload: dict[str, Any],
    view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
) -> None:
    """Restore the closed rank-zero carrier shape of bound constexpr p* rows."""

    if not isinstance(view, Mapping):
        return
    expected = view.get("bound_constexpr_values")
    j = payload.get("j")
    signature = j.get("signature") if isinstance(j, Mapping) else None
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if not isinstance(expected, Mapping) or not isinstance(parameters, list):
        return
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        if name not in expected or parameter.get("rank") != 0:
            if name in expected and parameter.get("rank") is None:
                parameter["rank"] = 0
                _record_deterministic_repair(
                    repairs,
                    rule_code="deterministic_bound_constexpr_rank_repair",
                    path=f"j.signature.parameters[{name}].rank",
                    action="restore rank-0 bound constexpr carrier from trusted Harness view",
                )
            continue


def _repair_source_shape_scalar_bindings(
    payload: dict[str, Any],
    view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Restore the Source-owned logical scalar declaration block.

    Shape-coordinate scalars and Source-proven data scalars are both closed
    rank-zero J inputs.  The allocator already decided which p* carriers have
    that ownership; this pass only materializes the missing declaration and
    rewrites the dependent t* names.  Address-only carriers remain p* and are
    never promoted here.  No operation or boundary semantics are inferred.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    if not isinstance(view, Mapping):
        return
    raw_arguments = view.get("arguments")
    if not isinstance(raw_arguments, list):
        return
    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = (
        intent.get("intent_function") if isinstance(intent, Mapping) else None
    )
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if (
        not isinstance(j, dict)
        or not isinstance(function, dict)
        or not isinstance(signature, dict)
        or not isinstance(function.get("inputs"), list)
        or not isinstance(function.get("outputs"), list)
        or not isinstance(signature.get("parameters"), list)
    ):
        return

    parameters = signature["parameters"]
    parameter_by_name: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return
        name = parameter.get("name")
        if not isinstance(name, str) or name in parameter_by_name:
            return
        parameter_by_name[name] = parameter

    tensors: dict[str, dict[str, Any]] = {}
    for item in [*function["inputs"], *function["outputs"]]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return
        name = item["name"]
        if name in tensors and tensors[name] != item:
            return
        tensors[name] = item

    coordinate_to_dimension: dict[str, str] = {}
    dimension_to_coordinate: dict[str, str] = {}
    for argument in raw_arguments:
        if not isinstance(argument, Mapping) or argument.get("kind") != "tensor":
            continue
        parameter_name = argument.get("name")
        parameter = parameter_by_name.get(parameter_name)
        if parameter is None:
            continue
        logical_ref = parameter.get("logical_ref")
        tensor = tensors.get(logical_ref)
        contracts = argument.get("shape_contract")
        if (
            not isinstance(logical_ref, str)
            or tensor is None
            or not isinstance(tensor.get("shape"), list)
            or not isinstance(contracts, list)
            or len(tensor["shape"]) != len(contracts)
        ):
            continue
        for dimension, coordinate in zip(
            tensor["shape"], contracts, strict=True
        ):
            if not isinstance(dimension, str) or not re.fullmatch(
                r"x[0-9]+", dimension
            ):
                continue
            if not isinstance(coordinate, str) or not re.fullmatch(
                r"h[0-9]+", coordinate
            ):
                continue
            previous_dimension = coordinate_to_dimension.setdefault(
                coordinate, dimension
            )
            previous_coordinate = dimension_to_coordinate.setdefault(
                dimension, coordinate
            )
            if previous_dimension != dimension or previous_coordinate != coordinate:
                return

    logical_bindings = allocator.logical_binding_map()
    bound_values = view.get("bound_constexpr_values", {})
    bound_types = view.get("bound_constexpr_value_types", {})
    bound_coordinates = view.get("bound_constexpr_shape_coordinates", {})
    if not isinstance(bound_values, Mapping):
        bound_values = {}
    if not isinstance(bound_types, Mapping):
        bound_types = {}
    if not isinstance(bound_coordinates, Mapping):
        bound_coordinates = {}

    missing: list[tuple[str, dict[str, Any], str | None]] = []
    relations = j.get("value_dimension_relations")
    if not isinstance(relations, list):
        return
    scalar_dtype_by_parameter: dict[str, str] = {}
    scalar_coordinate_by_parameter: dict[str, str] = {}
    for argument in raw_arguments:
        if not isinstance(argument, Mapping) or not isinstance(argument.get("name"), str):
            continue
        if argument.get("kind") == "scalar" and isinstance(argument.get("dtype"), str):
            scalar_dtype_by_parameter[argument["name"]] = argument["dtype"]
        if isinstance(argument.get("shape_coordinate"), str):
            scalar_coordinate_by_parameter[argument["name"]] = argument[
                "shape_coordinate"
            ]
    for parameter_name, value_type in bound_types.items():
        if isinstance(parameter_name, str) and isinstance(value_type, str):
            scalar_dtype_by_parameter[parameter_name] = value_type
    for parameter_name, coordinate in bound_coordinates.items():
        if isinstance(parameter_name, str) and isinstance(coordinate, str):
            scalar_coordinate_by_parameter[parameter_name] = coordinate

    for parameter_name, (expected_logical, role) in logical_bindings.items():
        if role not in {"shape_scalar", "scalar"}:
            continue
        parameter = parameter_by_name.get(parameter_name)
        coordinate = scalar_coordinate_by_parameter.get(parameter_name)
        dimension = coordinate_to_dimension.get(coordinate)
        if (
            not isinstance(parameter_name, str)
            or parameter is None
        ):
            continue
        logical_ref = parameter.get("logical_ref")
        already_bound = (
            logical_ref == expected_logical
            and expected_logical in tensors
            and isinstance(tensors[expected_logical].get("shape"), list)
            and not tensors[expected_logical]["shape"]
        )
        if already_bound:
            continue
        if logical_ref is not None and re.fullmatch(r"t[0-9]+", str(logical_ref)):
            # A different existing t* binding is ambiguous: the constructor
            # cannot prove which authored scalar it should displace.
            if logical_ref != expected_logical:
                return
        elif logical_ref is not None and not (
            role in {"shape_scalar", "scalar"}
            and isinstance(dimension, str)
            and logical_ref == dimension
        ):
            return
        dtype = scalar_dtype_by_parameter.get(parameter_name)
        if not isinstance(dtype, str) or not dtype:
            return
        missing.append((parameter_name, parameter, dimension))

    if not missing:
        return

    input_names = [
        item["name"] for item in function["inputs"] if isinstance(item, dict)
    ]
    output_names = [
        item["name"] for item in function["outputs"] if isinstance(item, dict)
    ]
    if (
        len(input_names) != len(function["inputs"])
        or len(output_names) != len(function["outputs"])
        or any(
            not isinstance(name, str) or re.fullmatch(r"t[0-9]+", name) is None
            for name in (*input_names, *output_names)
        )
        or len(set(input_names)) != len(input_names)
        or len(set(output_names)) != len(output_names)
    ):
        return

    existing_inputs = list(dict.fromkeys(input_names))
    existing_outputs = [name for name in output_names if name not in existing_inputs]
    missing_by_parameter = {
        name: (parameter, dimension) for name, parameter, dimension in missing
    }
    ordered_missing = [
        (name, *missing_by_parameter[name])
        for name, (_logical, role) in logical_bindings.items()
        if role in {"shape_scalar", "scalar"} and name in missing_by_parameter
    ]
    ordered_tokens: list[tuple[str, str]] = [
        ("old", name) for name in existing_inputs
    ]
    ordered_tokens.extend(("new", name) for name, _parameter, _dimension in ordered_missing)
    ordered_tokens.extend(("old", name) for name in existing_outputs)
    if len({token for _kind, token in ordered_tokens}) != len(ordered_tokens):
        return
    target_tensors = allocator.declared("t")
    if len(target_tensors) != len(ordered_tokens):
        return
    old_to_new = {
        token: target_tensors[index]
        for index, (kind, token) in enumerate(ordered_tokens)
        if kind == "old"
    }
    new_to_new = {
        token: target_tensors[index]
        for index, (kind, token) in enumerate(ordered_tokens)
        if kind == "new"
    }

    def rewrite(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return re.sub(
                r"(?<![A-Za-z0-9_])t[0-9]+(?![A-Za-z0-9_])",
                lambda match: old_to_new.get(match.group(0), match.group(0)),
                value,
            )
        return value

    rewritten_j = rewrite(j)
    if not isinstance(rewritten_j, dict):
        return
    if "kernel_id" in j:
        rewritten_j["kernel_id"] = j["kernel_id"]
    rewritten_intent = rewritten_j.get("intent")
    rewritten_function = (
        rewritten_intent.get("intent_function")
        if isinstance(rewritten_intent, Mapping)
        else None
    )
    if isinstance(rewritten_function, dict) and "name" in function:
        rewritten_function["name"] = function["name"]

    rewritten_inputs = rewritten_function.get("inputs")
    if not isinstance(rewritten_inputs, list):
        return
    for parameter_name, _parameter, _dimension in ordered_missing:
        new_name = new_to_new[parameter_name]
        argument = next(
            (
                item
                for item in raw_arguments
                if isinstance(item, Mapping) and item.get("name") == parameter_name
            ),
            None,
        )
        dtype = scalar_dtype_by_parameter.get(parameter_name)
        if not isinstance(dtype, str):
            return
        rewritten_inputs.append(
            {
                "name": new_name,
                "dtype": dtype,
                "shape": [],
            }
        )

    rewritten_signature = rewritten_j.get("signature")
    if not isinstance(rewritten_signature, dict):
        return
    rewritten_parameters = rewritten_signature.get("parameters")
    if not isinstance(rewritten_parameters, list):
        return
    rewritten_by_name = {
        item.get("name"): item
        for item in rewritten_parameters
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    for parameter_name, _parameter, _dimension in ordered_missing:
        parameter = rewritten_by_name.get(parameter_name)
        if not isinstance(parameter, dict):
            return
        parameter["logical_ref"] = new_to_new[parameter_name]

    rewritten_relations = rewritten_j.get("value_dimension_relations")
    if not isinstance(rewritten_relations, list):
        return
    for parameter_name, _parameter, dimension in ordered_missing:
        value_ref = new_to_new[parameter_name]
        # A provider may have copied the physical constexpr carrier into the
        # relation before the logical rank-zero input was materialized.  The
        # Source binding proves this one replacement; retaining p* here would
        # make the later namespace check reject an otherwise closed relation.
        for relation in rewritten_relations:
            if (
                isinstance(relation, dict)
                and relation.get("value_ref") == parameter_name
            ):
                if dimension is None or relation.get("dimension_ref") != dimension:
                    return
                relation["value_ref"] = value_ref
        if dimension is None:
            continue
        if not any(
            isinstance(relation, Mapping)
            and relation.get("value_ref") == value_ref
            and relation.get("dimension_ref") == dimension
            and relation.get("relation") == "eq"
            for relation in rewritten_relations
        ):
            rewritten_relations.append(
                {
                    "value_ref": value_ref,
                    "dimension_ref": dimension,
                    "relation": "eq",
                }
            )

    payload["j"] = rewritten_j
    _rewrite_c_slot_references(payload, old_to_new)
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_source_shape_scalar_binding",
        path="j.intent.intent_function.inputs+outputs/value_dimension_relations",
        action=(
            "restore unique Harness scalar-to-shape J input and renumber later "
            "logical tensor handles around it"
        ),
    )


def _repair_logical_tensor_handles(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Canonicalize an unambiguous declaration-order permutation of t* handles."""

    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    j = payload.get("j")
    if not isinstance(j, Mapping):
        return
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping):
        return
    raw_inputs = function.get("inputs")
    raw_outputs = function.get("outputs")
    if not isinstance(raw_inputs, list) or not isinstance(raw_outputs, list):
        return
    input_names = [
        item.get("name")
        for item in raw_inputs
        if isinstance(item, Mapping)
    ]
    output_names = [
        item.get("name")
        for item in raw_outputs
        if isinstance(item, Mapping)
    ]
    if (
        len(input_names) != len(raw_inputs)
        or len(output_names) != len(raw_outputs)
        or any(
            not isinstance(name, str) or re.fullmatch(r"t[0-9]+", name) is None
            for name in (*input_names, *output_names)
        )
        or len(input_names) != len(set(input_names))
        or len(output_names) != len(set(output_names))
    ):
        return

    identity_names: list[tuple[tuple[str, int], str]] = []
    input_identity_by_name = {
        name: ("input", index) for index, name in enumerate(input_names)
    }
    for index, name in enumerate(input_names):
        identity_names.append((input_identity_by_name[name], name))
    seen_identities = set(input_identity_by_name.values())
    for index, name in enumerate(output_names):
        identity = input_identity_by_name.get(name, ("output", index))
        if identity not in seen_identities:
            identity_names.append((identity, name))
            seen_identities.add(identity)

    old_to_identity: dict[str, tuple[str, int]] = {}
    for identity, name in identity_names:
        previous = old_to_identity.get(name)
        if previous is not None and previous != identity:
            # A single authored spelling cannot be renamed to two logical
            # declarations without semantic evidence.
            return
        old_to_identity[name] = identity
    owned_tensors = allocator.declared("t")
    if len(owned_tensors) != len(identity_names):
        # SourceFacts owns the ABI tensor declaration count, including the
        # valid zero-tensor case.  A candidate with a different logical tensor
        # population needs semantic authoring, not a best-effort handle repair;
        # importantly, do not index an empty owner list below.
        return
    target_tensors = owned_tensors
    mapping = {
        name: target_tensors[index]
        for index, (_identity, name) in enumerate(identity_names)
    }
    if all(old == new for old, new in mapping.items()):
        return

    def rewrite(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return re.sub(
                r"(?<![A-Za-z0-9_])t[0-9]+(?![A-Za-z0-9_])",
                lambda match: mapping.get(match.group(0), match.group(0)),
                value,
            )
        return value

    rewritten_j = rewrite(j)
    # Kernel/function names are identifiers rather than logical tensor
    # references.  Keep them byte-for-byte even if a provider happened to use
    # a string such as "t0" as a name.
    if isinstance(rewritten_j, dict):
        if "kernel_id" in j:
            rewritten_j["kernel_id"] = j["kernel_id"]
        original_function = function.get("name")
        rewritten_intent = rewritten_j.get("intent")
        rewritten_function = (
            rewritten_intent.get("intent_function")
            if isinstance(rewritten_intent, Mapping)
            else None
        )
        if isinstance(rewritten_function, dict) and "name" in function:
            rewritten_function["name"] = original_function
    payload["j"] = rewritten_j
    if isinstance(payload.get("pi"), Mapping):
        payload["pi"] = rewrite(payload["pi"])
    _rewrite_c_slot_references(payload, mapping)
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_logical_handle_repair",
        path="j.intent.intent_function.inputs+outputs",
        action="canonicalize t* handles in inputs-then-output first-declaration order",
    )


def _repair_logical_parameter_references(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
    source_facts: SourceFacts,
) -> None:
    """Rewrite logical references from the Source-owned ABI binding map.

    ``p*`` names are physical ABI parameters.  The mapping to a logical ``t*``
    is a Source/Harness fact owned by the shared allocator, not a fact that a
    candidate may redefine in its own ``signature.logical_ref`` field.  The
    old implementation built ``p_to_logical`` from the candidate signature;
    a candidate that swapped or invented those fields could therefore make
    its own malformed ownership map look self-consistent.

    Only the allocator's closed subset is rewritten into J logical fields.
    A physical carrier absent from that subset is explicitly de-owned by
    clearing a stale candidate ``logical_ref``.  No tensor, operation,
    relation, or π/C binding is invented.  π carrier rows and C slots remain
    outside this transaction because they retain physical ``p*``/``c*``
    ownership.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    j = payload.get("j")
    if not isinstance(j, Mapping):
        return
    signature = j.get("signature")
    parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
    if not isinstance(parameters, list):
        return
    # This is the only authority for p* logical ownership.  In particular, do
    # not derive it from the candidate's signature, which is exactly the
    # field being repaired.  Ordinary logical carriers map to t*; a bound
    # constexpr tied to a Source shape coordinate maps directly to x*.
    p_to_logical = {
        abi_name: logical_name
        for abi_name, (logical_name, _role) in allocator.logical_binding_map().items()
        if re.fullmatch(r"p[0-9]+", abi_name)
        and re.fullmatch(r"t[0-9]+", logical_name)
    }
    compiler_facts = source_facts.compiler_facts
    bound_coordinates = compiler_facts.get("bound_constexpr_shape_coordinates", {})
    if not isinstance(bound_coordinates, Mapping):
        raise ValueError("Source bound constexpr shape coordinates are malformed")
    coordinates = tuple(name for name, _value in source_facts.shape_coordinates)
    dimensions = allocator.declared("x")
    if len(coordinates) != len(dimensions):
        raise ValueError("canonical x* allocation differs from Source coordinates")
    dimension_by_coordinate = dict(zip(coordinates, dimensions, strict=True))
    for abi_name, coordinate in bound_coordinates.items():
        if not isinstance(abi_name, str) or not isinstance(coordinate, str):
            raise ValueError("Source bound constexpr shape coordinate is malformed")
        dimension = dimension_by_coordinate.get(coordinate)
        if dimension is None:
            raise ValueError("Source bound constexpr shape coordinate is unknown")
        previous = p_to_logical.setdefault(abi_name, dimension)
        if previous != dimension:
            raise ValueError("Source ABI carrier has conflicting logical ownership")
    parameter_by_name: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return
        name = parameter.get("name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"p[0-9]+", name) is None
            or name in parameter_by_name
        ):
            return
        parameter_by_name[name] = parameter

    changed = False
    # Signature.logical_ref itself is part of the same Source -> J mapping.
    # Repair it before walking logical J references, and remove a stale link
    # from a carrier that the Source binder proved representation-only.
    for name, parameter in parameter_by_name.items():
        expected = p_to_logical.get(name)
        actual = parameter.get("logical_ref")
        if expected is not None:
            if actual != expected:
                parameter["logical_ref"] = expected
                changed = True
        elif actual is not None:
            parameter["logical_ref"] = None
            changed = True

    logical_reference_keys = frozenset(
        {
            "tensor",
            "target",
            "output",
            "operand",
            "operands",
            "value_ref",
            "logical_ref",
            # Structured operation and boundary records use distinct field
            # names for the same logical t*/v* value role.  Keeping these in
            # the same rewrite set prevents a p* draft from being repaired in
            # ordinary operands while remaining in gather/update/loop fields.
            "data",
            "indices",
            "base",
            "values",
            "compare",
            "observed",
            "left",
            "right",
            "source",
            "init_value",
            "body_argument",
            "next_value",
            "result_value",
            "yield_value",
            "yield",
            "members",
        }
    )
    def rewrite(value: Any, key: str | None = None) -> Any:
        nonlocal changed
        if isinstance(value, Mapping):
            rewritten: dict[Any, Any] = {}
            for item_key, item in value.items():
                if (
                    item_key in logical_reference_keys
                    and isinstance(item, str)
                ):
                    replacement = p_to_logical.get(item)
                    if replacement is not None:
                        rewritten[item_key] = replacement
                        changed = True
                        continue
                if item_key in logical_reference_keys and isinstance(item, list):
                    rewritten_items = []
                    for row in item:
                        replacement = (
                            p_to_logical.get(row)
                            if isinstance(row, str)
                            else None
                        )
                        if replacement is not None:
                            rewritten_items.append(replacement)
                            changed = True
                        else:
                            rewritten_items.append(rewrite(row, item_key))
                    rewritten[item_key] = rewritten_items
                    continue
                rewritten[item_key] = rewrite(item, str(item_key))
            return rewritten
        if isinstance(value, list):
            return [rewrite(item, key) for item in value]
        return value

    rewritten_j = rewrite(j)
    if not isinstance(rewritten_j, dict) or not changed:
        return
    payload["j"] = rewritten_j
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_logical_parameter_reference_repair",
        path="j.boundary + j.intent.intent_function",
        action=(
            "apply the unique Source-owned p* -> t* binding to signature and J "
            "logical fields while preserving physical ABI ownership in pi/C"
        ),
    )


def _repair_dimension_constraint_syntax(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Canonicalize closed operator spellings in dimension constraints.

    Providers commonly emit ``=``/``&&``/``||`` while the closed J wire uses
    Python-like ``==``/``and``/``or``.  This is a syntax-only rewrite; it does
    not add a dimension, alter a bound, or choose a relation.
    """

    j = payload.get("j")
    constraints = j.get("dimension_constraints") if isinstance(j, Mapping) else None
    if not isinstance(constraints, list):
        return
    rewritten: list[Any] = []
    changed = False
    for expression in constraints:
        if not isinstance(expression, str):
            rewritten.append(expression)
            continue
        canonical = expression.replace("&&", " and ").replace("||", " or ")
        canonical = re.sub(r"(?<![<>=!])=(?!=)", "==", canonical)
        rewritten.append(canonical)
        changed = changed or canonical != expression
    if not changed:
        return
    j["dimension_constraints"] = rewritten
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_dimension_constraint_syntax",
        path="j.dimension_constraints",
        action="normalize closed equality and boolean operator spellings",
    )


def _repair_canonical_index_bounds(
    payload: dict[str, Any],
    operation_records: Sequence[Mapping[str, Any]] | None,
    repairs: list[dict[str, Any]],
) -> None:
    """Construct the complete typed index-bound table from gather/scatter.

    An index-bound row has no independent semantic author: all four fields
    are an exact projection of a closed gather/scatter operation.  Retaining
    provider rows for other operations created a second boundary authority and
    let speculative ``iota/gather`` descriptions survive beside an ordinary
    pointwise graph.  The constructor therefore replaces this mechanical
    collection as one table, while an invalid gather/scatter operation itself
    remains reject-only.
    """

    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(boundary, dict) or not isinstance(boundary.get("index_bounds"), list):
        return
    if operation_records is None:
        return

    rows = boundary["index_bounds"]
    if any(not isinstance(row, Mapping) for row in rows):
        return
    expected_rows: list[dict[str, Any]] = []
    for operation in operation_records:
        if operation.get("op") not in {"gather", "scatter"}:
            continue
        node_ref = operation.get("id")
        # The canonical operation wire omits an empty optional ``operands``
        # field.  In particular, a legal ``iota`` declaration has no inputs
        # and MCP therefore need not spell ``"operands": []``.  Treat that
        # omission as the same zero-input wire while validating the flat SSA
        # graph; it does not invent an operand or weaken validation for a
        # non-list value.
        operands = operation.get("operands", [])
        attrs = operation.get("attrs")
        axis = attrs.get("axis") if isinstance(attrs, Mapping) else None
        if (
            not isinstance(node_ref, str)
            or not isinstance(operands, (tuple, list))
            or len(operands) < 2
            or not all(isinstance(item, str) for item in operands[:2])
            or not isinstance(axis, int)
            or isinstance(axis, bool)
        ):
            return
        expected_rows.append({
            "node_ref": node_ref,
            "data": operands[0],
            "indices": operands[1],
            "axis": axis,
        })
    if rows == expected_rows:
        return
    boundary["index_bounds"] = expected_rows
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_gather_index_bounds",
        path="j.boundary.index_bounds",
        action=(
            "derive the complete typed index-bounds table solely from closed "
            "gather/scatter operation records"
        ),
    )


def _canonical_operation_records(
    function: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Build one read-only operation table for the canonical construction.

    The table is intentionally derived from the authored operation object as a
    whole.  Output and scope construction must not independently rescan
    ``result`` fields while a third consumer reads ``type``/``operands`` from
    another view.  This helper does not invent an operation or repair an
    invalid operation contract; an incomplete/ambiguous table returns ``None``
    and leaves the candidate fail-closed.
    """

    records = canonical_operation_records(function)
    if records is None:
        return None
    checked: list[dict[str, Any]] = []
    for record in records:
        contract_valid = True
        try:
            # The operation object is validated as one closed contract.  The
            # table still records invalid rows for diagnostics/scope ordering,
            # but producer attachments never use an invalid operation as proof.
            validate_intent_op_structure(
                record["wire"], path=record["path"], recursive=True
            )
        except (TypeError, ValueError):
            contract_valid = False
        checked.append({**record, "contract_valid": contract_valid})
    return tuple(checked)


def _canonical_value_type_records(
    function: Mapping[str, Any],
    operation_records: tuple[Mapping[str, Any], ...],
) -> dict[str, dict[str, Any]] | None:
    """Read the explicit value-type record from the shared construction pass."""

    return build_canonical_value_type_records(function, operation_records)


def _canonical_output_records(
    payload: Mapping[str, Any],
    operation_records: tuple[Mapping[str, Any], ...],
    *,
    repairable_missing_returns: bool = False,
) -> tuple[dict[str, Any], ...] | None:
    """Read the shared output record from a raw candidate envelope."""
    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if (
        not isinstance(function, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(signature, Mapping)
    ):
        return None
    return build_canonical_output_records(
        function,
        boundary,
        operation_records,
        signature=signature,
        repairable_missing_returns=repairable_missing_returns,
    )


def _repair_canonical_output_records(
    payload: dict[str, Any],
    operation_records: tuple[Mapping[str, Any], ...],
    repairs: list[dict[str, Any]],
) -> None:
    """Construct returns and sparse output mappings from one output table.

    A return value is only materialized when the value already appears in a
    canonical mapping or canonical return object.  No producer, operation, or
    output is guessed.  Incomplete, string-valued, or conflicting return
    wires remain fail-closed and are never promoted into the canonical form.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(function, dict) or not isinstance(boundary, dict):
        return
    records = _canonical_output_records(
        payload,
        operation_records,
        repairable_missing_returns=True,
    )
    if records is None:
        return
    desired_returns = [
        {"output": record["output"], "value": record["return_value"]}
        for record in records
    ]
    returns = function.get("returns")
    if returns != desired_returns:
        function["returns"] = desired_returns
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_output_record_constructor",
            path="j.intent.intent_function.outputs + returns + j.boundary.output_mappings",
            action=(
                "derive one complete output ABI record per declared output and "
                "materialize returns in the declared output order"
            ),
        )

    # A mapping already names the output and value.  Its node_ref may be the
    # unique authored SSA producer or the unique boundary store/scatter anchor
    # proven by the shared output record; in either case it is never guessed.
    attached: list[str] = []
    mappings = boundary.get("output_mappings")
    if not isinstance(mappings, list):
        return
    for record in records:
        mapping_index = record.get("mapping_index")
        producer = record.get("producer_record")
        if (
            not isinstance(mapping_index, int)
            or mapping_index >= len(mappings)
            or not isinstance(producer, Mapping)
        ):
            continue
        item = mappings[mapping_index]
        if not isinstance(item, dict) or item.get("node_ref") is not None:
            continue
        item["node_ref"] = producer["id"]
        attached.append(str(record["output"]))
    if attached:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_output_producer_attachment",
            path="j.boundary.output_mappings[*].node_ref",
            action=(
                "attach each output mapping to its unique authored SSA producer "
                "without creating a producer or changing output values"
            ),
        )
    # Do not propagate the output producer into an owner-less store access or
    # effect.  The selected output value proves the ABI return producer, not a
    # provider-omitted physical store owner.  Access/effect node ownership is
    # preserved only when it was present on the provider-authored access wire.


def _materialize_root_effect_scope_from_source_ledger(
    payload: dict[str, Any],
    construction: CanonicalConstruction,
    repairs: list[dict[str, Any]],
) -> None:
    """Give direct Source transfers their sole possible lexical owner.

    A node-less load/store has no operation path from which the generic scope
    allocator can infer ownership.  After the two-stage ingress has discarded
    provider-written membership arrays, a flat function would otherwise have
    no proof that its direct effects belong to its only scope.  The observed
    Source transfer ledger supplies that proof, but only for the deliberately
    narrow case below: one function scope, no regions, and a one-to-one
    kind/tensor correspondence for every Source occurrence.  Its ordinal
    order becomes the scope's effect order.

    This is not a fallback for structured control flow.  Any nested scope,
    node-owned effect, duplicate candidate access, missing effect, or
    unavailable/ambiguous ledger leaves the artifact untouched and therefore
    fail-closed.
    """

    ledger = construction.transfer_ledger
    if ledger.status != "pass" or not ledger.occurrences:
        return
    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    if not isinstance(boundary, Mapping) or not isinstance(scopes, list):
        return
    if len(scopes) != 1 or not isinstance(scopes[0], dict):
        return
    root = scopes[0]
    if root.get("kind") != "function" or root.get("parent_scope") is not None:
        return
    accesses = boundary.get("accesses")
    effects = boundary.get("effects")
    if not isinstance(accesses, list) or not isinstance(effects, list):
        return
    if len(accesses) != len(ledger.occurrences) or len(effects) != len(accesses):
        return

    access_by_kind_tensor: dict[tuple[str, str], Mapping[str, Any]] = {}
    for access in accesses:
        if not isinstance(access, Mapping):
            return
        access_id = access.get("access_id")
        kind = access.get("kind")
        tensor = access.get("tensor")
        key = (kind, tensor)
        if (
            not isinstance(access_id, str)
            or not isinstance(kind, str)
            or not isinstance(tensor, str)
            or key in access_by_kind_tensor
        ):
            return
        access_by_kind_tensor[key] = access

    effect_by_access: dict[str, Mapping[str, Any]] = {}
    for effect in effects:
        if not isinstance(effect, Mapping):
            return
        effect_id = effect.get("effect_id")
        access_ref = effect.get("access_ref")
        if (
            not isinstance(effect_id, str)
            or not isinstance(access_ref, str)
            or access_ref in effect_by_access
            or effect.get("node_ref") is not None
        ):
            return
        effect_by_access[access_ref] = effect

    desired_effects: list[str] = []
    for occurrence in ledger.occurrences:
        logical = occurrence.logical_ref
        access = (
            access_by_kind_tensor.get((occurrence.kind, logical))
            if isinstance(logical, str)
            else None
        )
        if access is None:
            return
        access_id = access.get("access_id")
        effect = effect_by_access.get(access_id) if isinstance(access_id, str) else None
        expected_effect_kind = {
            "load": "read",
            "store": "write",
            "update": "update",
        }.get(occurrence.kind)
        if (
            effect is None
            or effect.get("kind") != expected_effect_kind
            or effect.get("target") != logical
            or not isinstance(effect.get("effect_id"), str)
        ):
            return
        desired_effects.append(effect["effect_id"])
    if len(set(desired_effects)) != len(desired_effects):
        return
    if root.get("effects") == desired_effects:
        return
    root["effects"] = desired_effects
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_root_effect_scope_from_source_ledger",
        path="j.scopes[*].effects",
        action=(
            "derive flat direct-effect ownership and Source occurrence order "
            "from the closed transfer ledger"
        ),
    )


def _materialize_flat_root_effect_membership(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Derive membership for direct effects when J has exactly one scope.

    With one root function scope and no region declaration, a direct boundary
    effect has exactly one lexical owner.  That includes both a node-less load
    and a store attached to a top-level operation: requiring every effect to
    be node-less made a normal load/load/store sequence impossible to
    materialize, because the loads then had no owner once provider-written
    scope rows were discarded.  The effect list's current order is retained;
    this helper makes no Source-order claim (the transfer-ledger helper above
    replaces it whenever that stronger proof is available).  It only removes
    the circular dependency between cleared scope rows and construction.
    """

    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    if not isinstance(boundary, Mapping) or not isinstance(scopes, list):
        return
    if len(scopes) != 1 or not isinstance(scopes[0], dict):
        return
    root = scopes[0]
    if root.get("kind") != "function" or root.get("parent_scope") is not None:
        return
    effects = boundary.get("effects")
    accesses = boundary.get("accesses")
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    body = function.get("body") if isinstance(function, Mapping) else None
    if (
        not isinstance(effects, list)
        or not isinstance(accesses, list)
        or not isinstance(node_ids, Mapping)
        or not isinstance(body, list)
        or any(not isinstance(operation, Mapping) for operation in body)
    ):
        return
    top_level_paths = {f"body[{index}]" for index in range(len(body))}
    access_ids = {
        row.get("access_id")
        for row in accesses
        if isinstance(row, Mapping) and isinstance(row.get("access_id"), str)
    }
    if len(access_ids) != len(accesses):
        return
    desired: list[str] = []
    for effect in effects:
        if not isinstance(effect, Mapping):
            return
        effect_id = effect.get("effect_id")
        access_ref = effect.get("access_ref")
        if (
            not isinstance(effect_id, str)
            or effect_id in desired
            or not isinstance(access_ref, str)
            or access_ref not in access_ids
        ):
            return
        node_ref = effect.get("node_ref")
        if node_ref is not None and (
            not isinstance(node_ref, str)
            or node_ids.get(node_ref) not in top_level_paths
        ):
            return
        desired.append(effect_id)
    if root.get("effects") == desired:
        return
    root["effects"] = desired
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_flat_root_effect_membership",
        path="j.scopes[*].effects",
        action=(
            "derive every direct boundary effect's only lexical owner in a "
            "single-scope function without changing boundary effect order"
        ),
    )


def _materialize_rectangular_domain_predicates(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Expand a closed rectangular domain extent into canonical predicate form.

    Source-owned input/output shapes declare the available ``x*`` extents.
    When a provider supplies the unambiguous shorthand ``dK: "xM"``, it has
    already selected both the coordinate and its Source-owned extent; the
    canonical J wire represents that same fact as ``"dK < xM"``.  Expanding
    this spelling is deterministic and preserves the domain exactly.  No
    compound predicate, literal, unknown extent, duplicate domain, or
    non-canonical handle is interpreted here.

    Access expressions and guards remain provider-authored semantic facts.
    In particular this helper never guesses that a symbolic dimension used in
    a predicate should be a coordinate axis.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(boundary, dict):
        return
    domains = boundary.get("domains")
    if not isinstance(domains, dict):
        return
    source_dimensions: set[str] = set()
    for collection in ("inputs", "outputs"):
        tensors = function.get(collection)
        if not isinstance(tensors, list):
            return
        for tensor in tensors:
            shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
            if not isinstance(shape, list):
                return
            source_dimensions.update(
                extent
                for extent in shape
                if isinstance(extent, str) and re.fullmatch(r"x[0-9]+", extent)
            )
    changed: list[str] = []
    for domain, extent in tuple(domains.items()):
        if (
            not isinstance(domain, str)
            or re.fullmatch(r"d[0-9]+", domain) is None
            or not isinstance(extent, str)
            or re.fullmatch(r"x[0-9]+", extent) is None
            or extent not in source_dimensions
        ):
            continue
        domains[domain] = f"{domain} < {extent}"
        changed.append(domain)
    if not changed:
        return
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_rectangular_domain_predicate",
        path="j.boundary.domains",
        action=(
            "expand each Source-owned rectangular d* -> x* extent shorthand "
            "into its canonical boolean domain predicate"
        ),
    )


def _repair_canonical_authoring_layout_impl(
    payload: dict[str, Any],
    trusted_view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Construct the one canonical node/access/signature/scope layout.

    This is deliberately a structural transaction.  It does not invent an
    operation, an operand, a predicate, or a Source semantic.  It only
    canonicalizes declaration-order handles when every affected declaration
    and reference is present and unambiguous.  Ambiguous graphs remain
    untouched and therefore fail closed in Decode/Gate1.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    j = payload.get("j")
    if not isinstance(j, dict):
        return
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, dict):
        return

    rows = semantic_operation_wire_rows(function)
    if rows is None:
        return
    if rows:
        old_operation_ids: list[str] = []
        for _path, operation in rows:
            old_id = operation.get("id")
            if not isinstance(old_id, str):
                continue
            if old_id in old_operation_ids:
                return
            old_operation_ids.append(old_id)

        old_node_map = j.get("node_ids")
        if old_node_map is not None and not isinstance(old_node_map, Mapping):
            return
        original_node_map = (
            dict(old_node_map) if isinstance(old_node_map, Mapping) else None
        )
        path_by_old_id = {
            operation.get("id"): path
            for path, operation in rows
            if isinstance(operation.get("id"), str)
        }
        path_by_unique_result: dict[str, str] = {}
        duplicate_results: set[str] = set()
        for path, operation in rows:
            result = operation.get("result")
            result_names = [result] if isinstance(result, str) else result
            if not isinstance(result_names, list):
                continue
            for result_name in result_names:
                if not isinstance(result_name, str):
                    continue
                if result_name in path_by_unique_result:
                    duplicate_results.add(result_name)
                else:
                    path_by_unique_result[result_name] = path
        for result_name in duplicate_results:
            path_by_unique_result.pop(result_name, None)
        path_set = {path for path, _operation in rows}
        def path_from_ordinal_handle(value: Any) -> str | None:
            if not isinstance(value, str):
                return None
            # Only an n* handle can be an ordinal node fallback. A v* handle
            # is an SSA result, and its numeric suffix says nothing about the
            # operation that produced it; treating v2 as body[2] can silently
            # attach an output or effect to the wrong node.
            match = re.fullmatch(r"n([0-9]+)", value)
            if match is None:
                return None
            ordinal = int(match.group(1))
            return rows[ordinal][0] if ordinal < len(rows) else None

        if isinstance(old_node_map, Mapping):
            for node_ref, raw_path in old_node_map.items():
                if not isinstance(node_ref, str) or not isinstance(raw_path, str):
                    return
                if raw_path not in path_set and raw_path not in path_by_old_id and raw_path not in path_by_unique_result:
                    # Older provider drafts used an operation/result spelling
                    # in this value.  If the node key or value is an ordinal
                    # n*/v* handle, its declaration-order path is closed by
                    # the authored body and can be recovered without choosing
                    # an operation or changing dataflow.
                    if path_from_ordinal_handle(raw_path) is None and path_from_ordinal_handle(node_ref) is None:
                        return

        canonical_node_names = allocator.declared("n")
        if len(canonical_node_names) != len(rows):
            return
        canonical_node_by_path = {
            path: canonical_node_names[index]
            for index, (path, _operation) in enumerate(rows)
        }
        path_by_canonical_node = {
            canonical: path for path, canonical in canonical_node_by_path.items()
        }
        node_ref_map: dict[str, str] = {}
        if isinstance(old_node_map, Mapping):
            for old_node, raw_path in old_node_map.items():
                path = path_by_old_id.get(
                    raw_path,
                    path_by_unique_result.get(raw_path, raw_path),
                )
                if path not in path_set:
                    path = path_from_ordinal_handle(raw_path) or path_from_ordinal_handle(old_node) or ""
                canonical = canonical_node_by_path.get(path)
                if canonical is None:
                    return
                node_ref_map[old_node] = canonical
                if isinstance(raw_path, str) and re.fullmatch(r"n[0-9]+", raw_path):
                    aliased_path = path_by_canonical_node.get(raw_path)
                    if aliased_path is not None:
                        node_ref_map[raw_path] = canonical_node_by_path[aliased_path]
        for old_id, path in path_by_old_id.items():
            canonical = canonical_node_by_path.get(path)
            if canonical is not None:
                node_ref_map[old_id] = canonical

        node_reference_fields = {
            "node_ref",
            "consumer_ref",
            "update_node_ref",
            "subject_ref",
        }

        def rewrite(value: Any) -> Any:
            if isinstance(value, Mapping):
                rewritten: dict[Any, Any] = {}
                for key, item in value.items():
                    if key in node_reference_fields and isinstance(item, str):
                        rewritten[key] = node_ref_map.get(item, item)
                    else:
                        rewritten[key] = rewrite(item)
                return rewritten
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value

        rewritten_j = rewrite(j)
        if not isinstance(rewritten_j, dict):
            return
        rewritten_intent = rewritten_j.get("intent")
        rewritten_function = (
            rewritten_intent.get("intent_function")
            if isinstance(rewritten_intent, Mapping)
            else None
        )
        if not isinstance(rewritten_function, dict):
            return
        rewritten_rows = semantic_operation_wire_rows(rewritten_function)
        if rewritten_rows is None:
            return
        if len(rewritten_rows) != len(rows):
            return
        for index, (_path, operation) in enumerate(rewritten_rows):
            operation["id"] = canonical_node_names[index]
        rewritten_j["node_ids"] = {
            canonical_node_names[index]: path
            for index, (path, _operation) in enumerate(rewritten_rows)
        }
        payload["j"] = rewritten_j
        j = rewritten_j
        intent = j.get("intent")
        function = (
            intent.get("intent_function") if isinstance(intent, Mapping) else None
        )
        if isinstance(payload.get("pi"), Mapping):
            payload["pi"] = rewrite(payload["pi"])
        _rewrite_c_slot_references(
            payload,
            {old: new for old, new in node_ref_map.items() if old != new},
        )
        rewritten_node_map = rewritten_j.get("node_ids")
        layout_changed = (
            tuple(
                operation.get("id")
                for _path, operation in rewritten_rows
            )
            != tuple(old_operation_ids)
            or original_node_map != rewritten_node_map
        )
        if layout_changed:
            _record_deterministic_repair(
                repairs,
                rule_code="deterministic_canonical_node_allocator",
                path="j.intent.intent_function.body + j.node_ids",
                action=(
                    "rebuild n* operation ids and node_ids from canonical lexical "
                    "body paths, rewriting only known node-reference fields"
                ),
            )

    boundary = j.get("boundary")
    if isinstance(boundary, dict):
        for field_name, prefix, reference_key in (
            ("accesses", "a", "access_ref"),
            ("effects", "e", "effect_ref"),
        ):
            rows = boundary.get(field_name)
            if not isinstance(rows, list):
                continue
            identifiers: list[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    return
                identifier = row.get("access_id" if prefix == "a" else "effect_id")
                if not isinstance(identifier, str) or identifier in identifiers:
                    return
                identifiers.append(identifier)
            target_names = allocator.declared(prefix)
            if len(target_names) != len(identifiers):
                return
            mapping = dict(zip(identifiers, target_names, strict=True))
            if all(old == new for old, new in mapping.items()):
                continue

            def rewrite_access_effect(value: Any) -> Any:
                if isinstance(value, Mapping):
                    rewritten: dict[Any, Any] = {}
                    for key, item in value.items():
                        if key == ("access_id" if prefix == "a" else "effect_id"):
                            rewritten[key] = mapping.get(item, item)
                        elif key == reference_key and isinstance(item, str):
                            rewritten[key] = mapping.get(item, item)
                        elif prefix == "a" and key == "access_ref" and isinstance(item, str):
                            rewritten[key] = mapping.get(item, item)
                        elif prefix == "e" and key == "effects" and isinstance(item, list):
                            rewritten[key] = [
                                mapping.get(effect_ref, effect_ref)
                                if isinstance(effect_ref, str)
                                else rewrite_access_effect(effect_ref)
                                for effect_ref in item
                            ]
                        else:
                            rewritten[key] = rewrite_access_effect(item)
                    return rewritten
                if isinstance(value, list):
                    return [rewrite_access_effect(item) for item in value]
                return value

            rewritten_j = rewrite_access_effect(payload["j"])
            if not isinstance(rewritten_j, dict):
                return
            payload["j"] = rewritten_j
            if isinstance(payload.get("pi"), Mapping):
                payload["pi"] = rewrite_access_effect(payload["pi"])
            _record_deterministic_repair(
                repairs,
                rule_code="deterministic_boundary_handle_allocator",
                path=f"j.boundary.{field_name}",
                action=f"canonicalize {prefix}* declarations and all {prefix}* references in declaration order",
            )

    view = trusted_view
    if isinstance(view, Mapping):
        expected_parameters = view.get("parameter_order")
        signature = j.get("signature")
        parameters = signature.get("parameters") if isinstance(signature, Mapping) else None
        if isinstance(expected_parameters, list) and isinstance(parameters, list):
            expected = [name for name in expected_parameters if isinstance(name, str) and name.startswith("p")]
            by_name = {
                row.get("name"): row
                for row in parameters
                if isinstance(row, Mapping) and isinstance(row.get("name"), str)
            }
            if (
                len(by_name) == len(parameters)
                and set(by_name) == set(expected)
                and [row.get("name") for row in parameters] != expected
            ):
                signature["parameters"] = [by_name[name] for name in expected]
                _record_deterministic_repair(
                    repairs,
                    rule_code="deterministic_source_parameter_allocator",
                    path="j.signature.parameters",
                    action="restore trusted p* ABI declaration order without renaming physical parameters",
                )


def _repair_canonical_authoring_layout(
    payload: dict[str, Any],
    trusted_view: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Commit the authoring-layout repair only as one closed transaction.

    The implementation above has deliberately conservative early returns for
    ambiguous rows.  It must therefore run on a private payload copy: an early
    return after rewriting node ids must not leave access/effect ids or their
    references on an older interpretation.  The structural construction bundle
    is the commit predicate; if it cannot be rebuilt, all edits and receipts
    are discarded and the strict CandidateCheck path remains authoritative.
    """

    original = copy.deepcopy(payload)
    repair_start = len(repairs)
    working = copy.deepcopy(payload)
    _repair_canonical_authoring_layout_impl(
        working,
        trusted_view,
        repairs,
        allocator=allocator,
    )

    j = working.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = (
        intent.get("intent_function")
        if isinstance(intent, Mapping)
        else None
    )
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    bundle = None
    if (
        isinstance(function, Mapping)
        and isinstance(boundary, Mapping)
        and isinstance(scopes, list)
        and isinstance(node_ids, Mapping)
    ):
        bundle = CanonicalConstructionRecords.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            repairable_scope_fields=True,
            dimension_relations=(
                j.get("dimension_relations", [])
                if isinstance(j, Mapping)
                else []
            ),
            value_dimension_relations=(
                j.get("value_dimension_relations", [])
                if isinstance(j, Mapping)
                else []
            ),
        )
    if bundle is None:
        payload.clear()
        payload.update(original)
        del repairs[repair_start:]
        return

    payload.clear()
    payload.update(working)


def _repair_canonical_record_links(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> CanonicalConstructionRecords | None:
    """Run the shared operation/output/scope link pass after handle repair.

    ``_repair_canonical_authoring_layout`` has intentionally conservative
    early exits for ambiguous node/access declarations.  Record linking is a
    separate transaction so one such diagnostic cannot leave output ABI and
    lexical scope construction on an older partial view.  It still edits only
    closed, already-authored relationships; unresolved ownership remains
    strict.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    if not isinstance(function, Mapping):
        return None
    operation_records = _canonical_operation_records(function)
    if operation_records is None:
        return None
    if not isinstance(scopes, list):
        return None
    if not isinstance(boundary, Mapping) or not isinstance(j.get("node_ids"), Mapping):
        return None

    # Record linking is the last transaction that can replace sibling
    # boundary rows before the strict construction bundle is committed.  A
    # flat direct load/store has no node path, so its only lexical owner is
    # the unique function scope.  Re-establish that already-proven ownership
    # against this final snapshot before asking ``canonical_scope_records``
    # to consume it.  The helper is deliberately a no-op for regions,
    # multiple scopes, malformed effects, or any non-unique ownership case;
    # it neither creates an effect nor alters provider access semantics.
    _materialize_flat_root_effect_membership(payload, repairs)
    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    if (
        not isinstance(function, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(scopes, list)
        or not isinstance(node_ids, Mapping)
    ):
        return None
    operation_records = _canonical_operation_records(function)
    if operation_records is None:
        return None
    # Lexical membership has a narrower proof boundary than the complete
    # structural bundle.  In particular, it is already uniquely determined
    # by the operation records, scope headers, canonical node map, and
    # boundary effect ownership.  Requiring unrelated typed-operation, SSA,
    # or output preconditions to pass before committing that host-owned
    # projection left an otherwise derivable ``definitions`` list empty.
    # Use the same canonical allocator in its repairable mode; strict bundle
    # construction below remains the sole commit gate.
    derived_scope_records = canonical_scope_records(
        function,
        scopes,
        boundary,
        node_ids,
        operation_records=operation_records,
        repairable_scope_fields=True,
    )
    if derived_scope_records is None:
        return None

    # Apply every derived scope record before any unrelated output transaction.
    # Scope definitions, captures, and effects have their own complete proof:
    # the allocator call above.  Coupling that host-owned lexical projection to
    # the later typed/output bundle made an unrelated incomplete J roll back a
    # proven scope repair, so CandidateCheck kept observing stale membership.
    # The final strict bundle build below is unchanged and still rejects that
    # incomplete J; this only preserves the independently proven scope fields.
    # definitions, captures, and effects are one construction output; updating
    # only the function scope would leave region scopes on a second, stale
    # interpretation.
    expected_scope_records = {
        record.get("scope_id"): record
        for record in derived_scope_records
        if isinstance(record.get("scope_id"), str)
    }
    changed_scope_ids: list[str] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        scope_id = scope.get("scope_id")
        expected = expected_scope_records.get(scope_id)
        if not isinstance(expected, Mapping):
            continue
        changed = False
        for field_name in ("definitions", "captures", "effects"):
            value = copy.deepcopy(expected[field_name])
            if scope.get(field_name) != value:
                scope[field_name] = value
                changed = True
        if changed and isinstance(scope_id, str):
            changed_scope_ids.append(scope_id)
    if changed_scope_ids:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_scope_allocator",
            path="j.scopes[*].definitions/captures/effects",
            action=(
                "derive every lexical scope's tensor/value definitions, captures, "
                "and Source-ordered effects from the shared construction table"
            ),
        )

    # Output/return/mapping repair is allowed to rewrite more than one sibling
    # field.  Keep only that edit transactional: if its post-edit wire cannot
    # rebuild one complete construction bundle, restore the scope-projected J
    # and discard only output-side provisional receipts.  A rejected candidate
    # remains rejected by the unchanged strict bundle gate; it just retains
    # lexical membership already derived from the canonical scope allocator.
    scope_projected_j = copy.deepcopy(j)
    output_repair_start = len(repairs)
    _repair_canonical_output_records(
        payload,
        operation_records,
        repairs,
    )
    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    if (
        not isinstance(function, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(scopes, list)
        or not isinstance(node_ids, Mapping)
    ):
        payload["j"] = scope_projected_j
        del repairs[output_repair_start:]
        return None

    post_operation_records = _canonical_operation_records(function)
    post_bundle = CanonicalConstructionRecords.build(
        function,
        boundary,
        scopes,
        node_ids,
        operation_records=post_operation_records,
        signature=j.get("signature") if isinstance(j, Mapping) else None,
        dimension_relations=j.get("dimension_relations", []),
        value_dimension_relations=j.get("value_dimension_relations", []),
    )
    if post_bundle is None:
        payload["j"] = scope_projected_j
        del repairs[output_repair_start:]
        return None
    return post_bundle
def _repair_canonical_non_tensor_handles(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    """Use the shared allocator for all candidate-authored namespaces.

    Only a complete, collision-free declaration permutation is normalized.  A
    missing declaration, duplicate producer, or incomplete loop is left
    untouched and therefore remains fail-closed.  Rewriting all dependent J/π
    references together is what prevents a local SSA rename from leaving
    stale scope, carry, effect, relation, normalization, or attached C-slot
    references behind.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        return
    raw_j = payload.get("j")
    if not isinstance(raw_j, Mapping):
        return
    declarations = canonical_authored_handle_declarations(raw_j)
    if declarations is None:
        return

    visible: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str):
                    match = re.fullmatch(
                        rf"({'|'.join(CANONICAL_AUTHORED_HANDLE_PREFIXES)})[0-9]+",
                        key,
                    )
                    if match:
                        visible.add(key)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            for match in re.finditer(
                rf"(?<![A-Za-z0-9_])({'|'.join(CANONICAL_AUTHORED_HANDLE_PREFIXES)})[0-9]+(?![A-Za-z0-9_])",
                value,
            ):
                visible.add(match.group(0))

    collect(raw_j)
    pi = payload.get("pi")
    if isinstance(pi, Mapping):
        collect(pi)

    # The transaction passes the one Source-owned allocator already extended
    # by ``for_candidate_j``.  Do not construct a local declaration table here:
    # an absent or conflicting namespace must remain untouched and fail closed.
    mappings: dict[str, dict[str, str]] = {}
    orphaned_prefixes: set[str] = set()
    for prefix in CANONICAL_AUTHORED_HANDLE_PREFIXES:
        declared = declarations.get(prefix, [])
        declared_set = set(declared)
        if any(
            value.startswith(prefix) and value not in declared_set
            for value in visible
        ):
            # An orphan handle is a semantic/visibility error, not a safe
            # numbering permutation. Abort the complete namespace transaction:
            # renaming only the other prefixes would leave a cross-namespace
            # candidate whose references no longer share one allocator view.
            orphaned_prefixes.add(prefix)
            continue
        owned = allocator.declared(prefix)
        if len(owned) != len(declared):
            # A candidate-owned namespace with incomplete declarations cannot
            # be safely normalized.  Source-owned x* is deliberately absent
            # from this loop and is checked against SourceFacts separately.
            orphaned_prefixes.add(prefix)
            continue
        mapping = dict(zip(declared, owned, strict=True))
        if any(old != new for old, new in mapping.items()):
            mappings[prefix] = mapping

    if orphaned_prefixes:
        return
    if not mappings:
        return

    def rewrite(value: Any, path: str = "") -> Any:
        if isinstance(value, Mapping):
            rewritten: dict[Any, Any] = {}
            for key, item in value.items():
                new_key = key
                if isinstance(key, str):
                    for mapping in mappings.values():
                        new_key = mapping.get(new_key, new_key)
                rewritten[new_key] = rewrite(
                    item,
                    f"{path}.{key}" if path else str(key),
                )
            return rewritten
        if isinstance(value, list):
            return [rewrite(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, str):
            if path.endswith("j.kernel_id") or path.endswith(
                "j.intent.intent_function.name"
            ):
                return value
            rewritten = value
            for mapping in mappings.values():
                rewritten = re.sub(
                    rf"(?<![A-Za-z0-9_])(?:{'|'.join(CANONICAL_AUTHORED_HANDLE_PREFIXES)})[0-9]+(?![A-Za-z0-9_])",
                    lambda match: mapping.get(match.group(0), match.group(0)),
                    rewritten,
                )
            return rewritten
        return value

    rewritten_j = rewrite(raw_j, "j")
    if not isinstance(rewritten_j, dict):
        return
    payload["j"] = rewritten_j
    if isinstance(pi, Mapping):
        rewritten_pi = rewrite(pi, "pi")
        if isinstance(rewritten_pi, dict):
            payload["pi"] = rewritten_pi
    _rewrite_c_slot_references(
        payload,
        {
            old: new
            for mapping in mappings.values()
            for old, new in mapping.items()
        },
    )
    for prefix in mappings:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_ssa_handle_allocator",
            path=f"j.{prefix}*",
            action=(
                f"canonicalize authored {prefix}* declarations in lexical order "
                "and rewrite all J/pi references together"
            ),
        )


def _repair_redundant_dimension_relations(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Drop only exact duplicate or identity self-relations.

    ``x = identity(x)`` contributes no semantic fact and is the common source
    of the false dependency cycle seen at CandidateCheck.  Any non-identity
    cycle or conflicting definition is retained for strict rejection; fixed
    code never guesses a replacement dimension relation.
    """

    j = payload.get("j")
    relations = j.get("dimension_relations") if isinstance(j, Mapping) else None
    if not isinstance(relations, list):
        return
    kept: list[Any] = []
    seen: set[str] = set()
    removed = 0
    for relation in relations:
        if not isinstance(relation, Mapping):
            kept.append(relation)
            continue
        try:
            identity = json.dumps(
                dict(relation),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            kept.append(relation)
            continue
        expression = relation.get("expr")
        is_self_identity = (
            isinstance(relation.get("result"), str)
            and isinstance(expression, Mapping)
            and expression.get("op") == "identity"
            and expression.get("arg") == relation.get("result")
            and set(expression) == {"op", "arg"}
        )
        if is_self_identity or identity in seen:
            removed += 1
            continue
        seen.add(identity)
        kept.append(relation)
    if removed:
        j["dimension_relations"] = kept
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_dimension_relation_cleanup",
            path="j.dimension_relations",
            action=(
                "remove exact duplicate and identity self-relations while "
                "preserving every non-trivial authored dimension definition"
            ),
        )


def _repair_result_relation_fields(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Remove update ownership from exact/tolerance result relations.

    ``update_node_ref`` is meaningful only for ``one_of_updates``.  Its
    absence from exact/tolerance rows is a closed relation contract, so
    deleting the misplaced field is safe; no output comparison class or
    update node is inferred here.
    """

    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    relations = (
        boundary.get("result_relations")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(relations, list):
        return
    changed = False
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if (
            relation.get("kind") in {"exact", "tolerance"}
            and "update_node_ref" in relation
        ):
            del relation["update_node_ref"]
            changed = True
    if changed:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_result_relation_ownership",
            path="j.boundary.result_relations",
            action=(
                "remove update_node_ref from exact/tolerance relations; only "
                "one_of_updates may own an update node"
            ),
        )


_NONEXACT_OUTPUT_UNARY_OPS = frozenset(
    {"sigmoid", "exp", "log", "rsqrt", "rsqrt_ftz", "sqrt", "sin", "cos", "tanh", "erf"}
)


def _derive_output_relation_kinds(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Construct typed output relations from the closed producer DAG.

    A non-update output has no provider-authored relation choice: its only
    possible row is ``exact`` or ``tolerance``, determined by the selected
    output value's typed producer chain.  Update semantics remain outside this
    constructor because their ``one_of_updates`` ownership is semantic and
    must stay explicitly proven.  This keeps normal output relations in the
    same construction record as returns and output mappings.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(boundary, Mapping):
        return
    mappings = boundary.get("output_mappings")
    relations = boundary.get("result_relations")
    updates = boundary.get("updates")
    if (
        not isinstance(mappings, list)
        or not isinstance(relations, list)
        or not isinstance(updates, list)
    ):
        return
    records = _canonical_operation_records(function)
    if records is None:
        return
    value_types = build_canonical_value_type_records(function, records)
    if value_types is None:
        return
    raw_authorizations = boundary.get("realization_authorizations")
    if not isinstance(raw_authorizations, list):
        return
    reassociation_nodes = {
        row.get("subject_ref")
        for row in raw_authorizations
        if isinstance(row, Mapping)
        and row.get("kind") == "reassociation"
        and isinstance(row.get("subject_ref"), str)
    }
    producer_by_value = {
        value: record
        for record in records
        for value in record.get("results", ())
        if isinstance(value, str)
    }

    def is_floating_value(value: Any) -> bool:
        value_type = value_types.get(value) if isinstance(value, str) else None
        dtype = value_type.get("dtype") if isinstance(value_type, Mapping) else None
        return isinstance(dtype, str) and dtype in {"f16", "bf16", "f32", "f64"}

    def nonexact(value: Any, seen: set[str]) -> bool:
        if not isinstance(value, str) or value in seen:
            return False
        record = producer_by_value.get(value)
        if record is None:
            return False
        seen.add(value)
        op = record.get("op")
        attrs = record.get("attrs")
        if record.get("id") in reassociation_nodes:
            return True
        if op in {"reduce", "contraction"} and is_floating_value(value):
            return True
        if (
            op == "map_unary"
            and isinstance(attrs, Mapping)
            and attrs.get("fn") in _NONEXACT_OUTPUT_UNARY_OPS
            and is_floating_value(value)
        ):
            return True
        operands = record.get("operands", ())
        return isinstance(operands, tuple) and any(
            nonexact(operand, seen) for operand in operands
        )

    mapping_by_output = {
        row.get("output"): row.get("value")
        for row in mappings
        if isinstance(row, Mapping)
        and isinstance(row.get("output"), str)
        and isinstance(row.get("value"), str)
    }
    outputs = function.get("outputs")
    if not isinstance(outputs, list):
        return
    output_names = [
        row.get("name")
        for row in outputs
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    ]
    if len(output_names) != len(outputs) or set(mapping_by_output) != set(output_names):
        return
    # A one-of relation is the single output-comparison class that cannot be
    # inferred from arithmetic: it makes a provider-authored logical-update
    # ownership claim.  Preserve it for strict cross-contract validation.  In
    # contrast, exact/tolerance is fully determined by the already-closed
    # typed producer DAG, just as it is for candidates without updates.
    records_by_id = {
        record.get("id"): record
        for record in records
        if isinstance(record.get("id"), str)
    }
    one_of_values: set[str] = set()
    for update in updates:
        if not isinstance(update, Mapping):
            return
        if update.get("duplicate_policy") != "one_of":
            continue
        node_ref = update.get("node_ref")
        record = records_by_id.get(node_ref)
        if not isinstance(record, Mapping) or record.get("op") != "scatter":
            # Leave an unknown or mismatched update untouched for the strict
            # LogicalUpdateSpec validator; never guess its output ownership.
            return
        results = record.get("results")
        if not isinstance(results, tuple) or len(results) != 1:
            return
        result = results[0]
        if not isinstance(result, str):
            return
        one_of_values.add(result)
    one_of_outputs = {
        output
        for output, value in mapping_by_output.items()
        if value in one_of_values
    }
    existing_one_of_by_output: dict[str, list[Mapping[str, Any]]] = {}
    unexpected_one_of_relations: list[Mapping[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, Mapping):
            return
        output = relation.get("output")
        kind = relation.get("kind")
        if not isinstance(output, str) or not isinstance(kind, str):
            return
        if kind == "one_of_updates":
            if output in output_names:
                existing_one_of_by_output.setdefault(output, []).append(relation)
            else:
                # Preserve an invalid ownership claim for the strict parser;
                # deriving unrelated exact/tolerance rows must not erase it.
                unexpected_one_of_relations.append(relation)
    expected: list[dict[str, Any]] = []
    for output in output_names:
        owned_relations = existing_one_of_by_output.get(output, ())
        if output in one_of_outputs or owned_relations:
            # Do not normalize, synthesize, or silently delete a provider's
            # update-ownership claim.  Missing, duplicate, or malformed rows
            # remain visible to the strict ResultRelationSpec/typed-boundary
            # checks below.
            expected.extend(
                copy.deepcopy(dict(relation))
                for relation in owned_relations
            )
            continue
        expected.append({
            "output": output,
            "kind": (
                "tolerance"
                if nonexact(mapping_by_output[output], set())
                else "exact"
            ),
        })
    expected.extend(
        copy.deepcopy(dict(relation))
        for relation in unexpected_one_of_relations
    )
    if relations != expected:
        boundary["result_relations"] = expected
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_output_relation_kind",
            path="j.boundary.result_relations",
            action=(
                "derive each non-one_of output relation from the closed typed "
                "producer graph, including tolerance for transcendental or "
                "reduction rounding, while preserving an explicitly authored "
                "one_of update ownership claim"
            ),
        )


def _repair_pi_normalization_scopes(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Apply the fixed scope belonging to each closed π class name."""

    pi = payload.get("pi")
    rows = pi.get("normalization_classes") if isinstance(pi, dict) else None
    if not isinstance(rows, list):
        return
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        expected_scope = NORMALIZATION_CLASS_SCOPES.get(name)
        if expected_scope is None or "scope" not in row:
            continue
        if row.get("scope") != expected_scope:
            row["scope"] = expected_scope
            changed = True
    if changed:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_pi_normalization_scope",
            path="pi.normalization_classes",
            action="synchronize each closed normalization class with its fixed scope",
        )


def _replace_logical_handle_spelling(
    value: Any,
    replacements: Mapping[str, str],
) -> Any:
    """Rewrite only whole logical ``t*`` handles in a private candidate view."""

    if isinstance(value, Mapping):
        return {
            key: _replace_logical_handle_spelling(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_logical_handle_spelling(item, replacements) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?<![A-Za-z0-9_])t[0-9]+(?![A-Za-z0-9_])",
            lambda match: replacements.get(match.group(0), match.group(0)),
            value,
        )
    return value


def _semantic_graph_allocator(
    allocator: CanonicalHandleAllocator,
    function: Mapping[str, Any],
) -> CanonicalHandleAllocator:
    """Reserve lexical ``n*``/``q*`` names from one operation graph.

    This is the first allocation step of canonical ingress.  It observes only
    the provider-authored operation tree: no provider handle spelling can
    influence the names it returns.  The same allocator instance is extended
    by the access/effect and SSA phases below; malformed regions do not get a
    partial namespace and remain fail-closed.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        raise ValueError("canonical semantic graph has no Source allocator")
    rows = semantic_operation_wire_rows(function)
    if rows is None:
        raise ValueError("candidate operation graph is malformed")

    region_count = 0

    def count_regions(items: object) -> None:
        nonlocal region_count
        if not isinstance(items, list):
            raise ValueError("candidate operation region body is malformed")
        for operation in items:
            if not isinstance(operation, Mapping):
                raise ValueError("candidate operation region contains a non-object")
            op_name = operation.get("op")
            if op_name == "loop":
                region_count += 1
                count_regions(operation.get("body"))
            elif op_name in {"effect_if", "value_if"}:
                for polarity in ("positive", "negative"):
                    arm = operation.get(polarity)
                    if not isinstance(arm, Mapping):
                        raise ValueError("candidate conditional arm is malformed")
                    region_count += 1
                    count_regions(arm.get("body"))

    inner = function.get("intent_function")
    if isinstance(inner, Mapping) and set(function) == {"intent_function"}:
        function = inner
    count_regions(function.get("body"))
    try:
        return allocator.with_declarations(
            {
                "n": CanonicalHandleAllocator.canonical_names("n", len(rows)),
                "q": CanonicalHandleAllocator.canonical_names(
                    "q", 1 + region_count
                ),
            }
        )
    except ValueError as exc:
        raise ValueError("canonical semantic graph namespace is inconsistent") from exc


def _materialize_scope_headers_from_operation_graph(
    body: list[Any],
    *,
    allocator: CanonicalHandleAllocator,
) -> list[dict[str, Any]]:
    """Derive the complete lexical scope tree from J region ownership.

    Scope handles, kinds, and parent links are mechanical consequences of the
    semantic operation tree.  In particular, a loop body and each conditional
    arm own exactly one child region.  The provider must not maintain a
    parallel ``j.scopes`` declaration whose headers can drift from those
    operation references.  This helper only establishes lexical headers;
    definitions/captures/effects remain the shared construction-record output
    after node/access/effect allocation.

    A region that is not present in the operation graph is deliberately not
    invented here.  In particular, ``program_instance`` has no operation-tree
    witness and remains reject-only unless a future Source-owned construction
    record can prove it.
    """

    scope_names = allocator.declared("q")
    if not scope_names or scope_names[0] != "q0":
        raise ValueError("canonical scope allocator lacks the function scope")
    scopes: list[dict[str, Any]] = [
        {
            "scope_id": scope_names[0],
            "kind": "function",
            "parent_scope": None,
            "definitions": [],
            "captures": [],
            "effects": [],
        }
    ]

    def allocate(kind: str, parent_scope: str) -> str:
        if len(scopes) >= len(scope_names):
            raise ValueError("canonical scope allocator is shorter than operation graph")
        scope_id = scope_names[len(scopes)]
        scopes.append(
            {
                "scope_id": scope_id,
                "kind": kind,
                "parent_scope": parent_scope,
                "definitions": [],
                "captures": [],
                "effects": [],
            }
        )
        return scope_id

    def visit(rows: object, parent_scope: str) -> None:
        if not isinstance(rows, list):
            raise ValueError("candidate operation region body is malformed")
        for operation in rows:
            if not isinstance(operation, dict):
                raise ValueError("candidate operation region contains a non-object")
            op_name = operation.get("op")
            if op_name == "loop":
                nested_body = operation.get("body")
                if not isinstance(nested_body, list):
                    raise ValueError("candidate loop lacks a body list")
                scope_id = allocate("loop_region", parent_scope)
                operation["scope_id"] = scope_id
                visit(nested_body, scope_id)
                continue
            if op_name not in {"effect_if", "value_if"}:
                continue
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if not isinstance(arm, dict):
                    raise ValueError("candidate conditional arm is malformed")
                nested_body = arm.get("body")
                if not isinstance(nested_body, list):
                    raise ValueError("candidate conditional arm lacks a body list")
                scope_id = allocate("branch_region", parent_scope)
                arm["scope_id"] = scope_id
                visit(nested_body, scope_id)

    visit(body, scope_names[0])
    if len(scopes) != len(scope_names):
        raise ValueError("canonical scope allocator is longer than operation graph")
    return scopes


def _seed_semantic_candidate_envelope(
    payload: Mapping[str, Any],
    construction: CanonicalConstruction,
) -> dict[str, Any]:
    """Project the bounded provider semantic surface onto one Source seed.

    This is the first phase of Lift ingress.  It is intentionally not a
    permissive artifact decoder: a candidate must still provide one J semantic
    function body and boundary object, and unknown root/J/boundary fields are
    retained so the final closed CandidateCheck can reject them.  What moves
    out of the provider's authority is only the duplicated mechanical surface:
    root C, ABI signature, Source tensor declarations, node map, scopes, and
    the empty/default pi rows.

    The result always has the normal canonical root envelope.  Consequently
    CandidateCheck, Decode, SourceReconcile, Replay, Gate1, and Gate2 retain
    exactly one complete artifact shape; no incomplete skeleton is ever
    accepted downstream or promoted to later authoring state.
    """

    if not isinstance(payload, Mapping):
        raise CanonicalIngressError(
            "semantic candidate is not an object",
            candidate_check_diagnostics=_semantic_candidate_surface_diagnostics(
                payload
            ),
        )
    surface_diagnostics = _semantic_candidate_surface_diagnostics(payload)
    if surface_diagnostics:
        raise CanonicalIngressError(
            "semantic candidate violates the canonical authoring surface",
            candidate_check_diagnostics=surface_diagnostics,
        )
    candidate_j = payload.get("j")
    if not isinstance(candidate_j, Mapping):
        raise CanonicalIngressError(
            "semantic candidate lacks j",
            candidate_check_diagnostics=(
                {
                    "rule_code": "semantic_j_required_path_missing",
                    "path": "j",
                    "observed_refs_or_types": ["expected:object"],
                },
            ),
        )
    candidate_intent = candidate_j.get("intent")
    candidate_function = (
        candidate_intent.get("intent_function")
        if isinstance(candidate_intent, Mapping)
        else None
    )
    candidate_boundary = candidate_j.get("boundary")
    if not isinstance(candidate_function, Mapping) or not isinstance(
        candidate_boundary, Mapping
    ):
        missing_path = (
            "j.intent.intent_function"
            if not isinstance(candidate_function, Mapping)
            else "j.boundary"
        )
        raise CanonicalIngressError(
            "semantic candidate lacks required J structure",
            candidate_check_diagnostics=(
                {
                    "rule_code": (
                        "semantic_intent_function_missing"
                        if missing_path == "j.intent.intent_function"
                        else "semantic_boundary_missing"
                    ),
                    "path": missing_path,
                    "observed_refs_or_types": ["expected:object"],
                },
            ),
        )
    if not isinstance(candidate_function.get("body"), list):
        raise CanonicalIngressError(
            "semantic candidate lacks a function body",
            candidate_check_diagnostics=(
                {
                    "rule_code": "semantic_intent_body_missing",
                    "path": "j.intent.intent_function.body",
                    "observed_refs_or_types": ["expected:array"],
                },
            ),
        )

    seed = canonical_source_wire_seed(construction)
    result = copy.deepcopy(seed)
    source_j = result["j"]
    source_function = source_j["intent"]["intent_function"]
    source_boundary = source_j["boundary"]

    # Function identity, Source logical input/output declarations, signature,
    # node map, and scope tree are all Source/operation-construction outputs.
    # The provider contributes only the operation graph here.  Keep unknown
    # function keys so the final schema is still the fail-closed authority.
    semantic_function = copy.deepcopy(dict(candidate_function))
    semantic_function["name"] = source_function["name"]
    semantic_function["inputs"] = copy.deepcopy(source_function["inputs"])
    semantic_function["outputs"] = copy.deepcopy(source_function["outputs"])
    semantic_function["returns"] = []
    source_j["intent"] = {"intent_function": semantic_function}

    # Boundary collection fields carry semantic choices, but their mechanical
    # ids/order are normalized later by the shared allocator/transfer ledger.
    # Do not whitelist-copy fields: retaining an unknown one makes the closed
    # final schema reject it instead of silently changing candidate meaning.
    source_boundary.update(copy.deepcopy(dict(candidate_boundary)))
    source_j["boundary"] = source_boundary

    # Source-owned J siblings must not be inherited from a provider draft.
    # Storage and non-Harness relation collections remain provider-owned until
    # their own construction record can prove a mechanical projection.  A
    # value/dimension equality is different: Source/Harness has already fixed
    # its rank-zero carrier and shape coordinate.  In particular, the empty
    # generic semantic skeleton must not erase that equality from the Source
    # seed before CandidateCheck observes it.
    for field_name in (
        "storage",
        "dimension_constraints",
        "dimension_relations",
    ):
        if field_name in candidate_j:
            source_j[field_name] = copy.deepcopy(candidate_j[field_name])
    # Node and lexical-scope identities are allocated only inside the second
    # construction transaction.  Keeping this seed empty prevents a preflight
    # scope walk from becoming a second q*/n* authority.
    source_j["node_ids"] = {}
    source_j["scopes"] = []

    candidate_pi = payload.get("pi")
    if candidate_pi is not None:
        if not isinstance(candidate_pi, Mapping):
            raise ValueError("candidate semantic pi must be an object")
        seeded_pi = result["pi"]
        seeded_pi.update(copy.deepcopy(dict(candidate_pi)))
        result["pi"] = seeded_pi

    # C is always the complete trusted Source/Harness ledger.  A submitted C
    # sibling is deliberately ignored rather than merged: it is mechanical
    # metadata and cannot acquire semantic ownership through this ingress.
    result["schema"] = Gate1Artifact.SCHEMA
    return result


def _materialize_source_owned_candidate_surface(
    payload: dict[str, Any],
    construction: CanonicalConstruction,
    repairs: list[dict[str, Any]],
) -> None:
    """Start a candidate from its sole Source/Harness-owned construction seed.

    Lift still supplies the semantic operation graph, boundary meaning and pi
    proof.  It does *not* own the duplicated ABI/C/tensor declarations, node
    table or lexical membership arrays.  Earlier ingress attempted to repair
    those fields only after a complete construction record already existed;
    consequently a provider spelling ``definitions: [\"n0\"]`` prevented the
    very scope allocator that could prove the correct list from running.

    This is the second, complete-artifact half of the two-stage ingress.  The
    preceding semantic projection has already replaced provider ABI spellings
    with the exact Source seed; this pass retains semantic graph/boundary/pi
    rows and rebuilds every remaining Source-owned/mechanical sibling from the
    same seed.  It never reintroduces a candidate-owned p*→t* translation.
    """

    candidate_j = payload.get("j")
    candidate_pi = payload.get("pi")
    if not isinstance(candidate_j, Mapping) or not isinstance(candidate_pi, Mapping):
        raise ValueError("candidate lacks a J/pi semantic surface")
    candidate_intent = candidate_j.get("intent")
    candidate_function = (
        candidate_intent.get("intent_function")
        if isinstance(candidate_intent, Mapping)
        else None
    )
    candidate_boundary = candidate_j.get("boundary")
    if not all(
        isinstance(item, Mapping)
        for item in (candidate_function, candidate_boundary)
    ):
        raise ValueError("candidate lacks a complete semantic/root surface")

    seed = canonical_source_wire_seed(construction)
    seed_j = seed["j"]
    seed_signature = seed_j["signature"]
    seed_parameters = seed_signature["parameters"]
    if not isinstance(seed_parameters, list):
        raise ValueError("canonical Source seed lacks an ABI signature")

    # The first-stage semantic ingress is already projected onto the exact
    # Source ABI/tensor seed.  There is no provider-authored p* table left to
    # translate, so J and pi share the seed's canonical logical spelling.
    semantic_j = copy.deepcopy(dict(candidate_j))
    semantic_pi = copy.deepcopy(dict(candidate_pi))
    semantic_intent = semantic_j.get("intent")
    semantic_function = (
        semantic_intent.get("intent_function")
        if isinstance(semantic_intent, Mapping)
        else None
    )
    if not isinstance(semantic_function, Mapping):
        raise ValueError("candidate semantic function is malformed")
    body = semantic_function.get("body")
    if not isinstance(body, list):
        raise ValueError("candidate semantic function has no body")

    # The Source seed owns function identity, logical inputs/outputs, ABI,
    # C ledger and Source-proven value/dimension relations.  Keep the
    # provider's operation graph and semantic boundary collections only.
    rebuilt = copy.deepcopy(seed)
    rebuilt_j = rebuilt["j"]
    rebuilt_function = rebuilt_j["intent"]["intent_function"]
    rebuilt_function["body"] = copy.deepcopy(body)

    semantic_boundary = semantic_j.get("boundary")
    if not isinstance(semantic_boundary, Mapping):
        raise ValueError("candidate semantic boundary is malformed")
    rebuilt_boundary = rebuilt_j["boundary"]
    for field_name in tuple(rebuilt_boundary):
        if field_name in semantic_boundary:
            rebuilt_boundary[field_name] = copy.deepcopy(semantic_boundary[field_name])

    # An output mapping has one semantic choice (the produced t*/v* value),
    # but its output name and node attachment are Source/operation-record
    # projections.  Do not let the provider independently maintain a third
    # spelling of output ownership beside returns and the lexical node table.
    candidate_mappings = semantic_boundary.get("output_mappings")
    seed_outputs = rebuilt_function.get("outputs")
    if not isinstance(candidate_mappings, list) or not isinstance(seed_outputs, list):
        raise ValueError("candidate lacks output mapping semantic selection")
    source_outputs = [
        row.get("name")
        for row in seed_outputs
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    ]
    if (
        len(source_outputs) != len(seed_outputs)
        or len(source_outputs) != len(set(source_outputs))
        or len(candidate_mappings) != len(source_outputs)
    ):
        raise ValueError("candidate output mapping selection has invalid arity")

    # Output names and their ABI order are fixed by SourceFacts.  The provider
    # supplies exactly one semantic choice per ordered output: the selected
    # logical/SSA value.  Its duplicate ``output`` spelling is deliberately
    # discarded, together with the independent return spelling below, so the
    # two mechanical projections cannot drift from the same selection.
    selected_values: dict[str, str] = {}

    def is_semantic_value_label(value: Any) -> bool:
        """Accept a bounded local value label, never a physical owner."""

        if not isinstance(value, str):
            return False
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None:
            return False
        # p*/c* are physical ABI/compiler namespaces.  They must never be
        # smuggled into a semantic output selection merely because the host
        # will later allocate v* handles.
        return re.fullmatch(r"[pc][0-9]+", value) is None

    for output, mapping in zip(source_outputs, candidate_mappings, strict=True):
        value = mapping.get("value") if isinstance(mapping, Mapping) else None
        if (
            not is_semantic_value_label(value)
            or output in selected_values
        ):
            raise ValueError("candidate output mapping selection is malformed")
        selected_values[output] = value
    rebuilt_boundary["output_mappings"] = [
        {"output": output, "value": selected_values[output], "node_ref": None}
        for output in source_outputs
    ]

    # Node ownership and the entire scope tree are derived from the operation
    # table.  A provider-written scope header is no more semantic than its
    # definitions/captures/effects arrays and is therefore never accepted.
    rebuilt_j["node_ids"] = {}
    rebuilt_j["scopes"] = []

    for field_name in (
        "storage",
        "dimension_constraints",
        "dimension_relations",
    ):
        if field_name in semantic_j:
            rebuilt_j[field_name] = copy.deepcopy(semantic_j[field_name])

    rebuilt_pi = rebuilt["pi"]
    for field_name in tuple(rebuilt_pi):
        if field_name != "schema" and field_name in semantic_pi:
            rebuilt_pi[field_name] = copy.deepcopy(semantic_pi[field_name])

    payload.clear()
    payload.update(rebuilt)
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_source_owned_surface",
        path="j.intent.inputs/outputs + j.signature + j.node_ids + j.scopes + c",
        action=(
            "rebuild Source-owned ABI/C/tensor rows and provider-independent "
            "node/scope fields from the canonical construction seed"
        ),
    )


def _prepare_canonical_access_reallocation(
    payload: dict[str, Any],
) -> None:
    """Turn an internal canonical access table into a transient semantic one.

    Source-ledger ordering operates on completed canonical rows.  After it
    proves a different occurrence order, the host must allocate fresh dense
    ``a*``/``e*`` identities from that order.  This adapter is private to that
    host-only transition: it never accepts provider input and it never leaves
    ``access_label`` in the final artifact.
    """

    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, dict) else None
    raw_effects = boundary.get("effects") if isinstance(boundary, dict) else None
    if not isinstance(raw_accesses, list) or not isinstance(raw_effects, list):
        raise ValueError("canonical access reallocation lacks access/effect tables")

    access_labels: dict[str, str] = {}
    for index, access in enumerate(raw_accesses):
        access_id = access.get("access_id") if isinstance(access, Mapping) else None
        if (
            not isinstance(access_id, str)
            or re.fullmatch(r"a[0-9]+", access_id) is None
            or access_id in access_labels
        ):
            raise ValueError("canonical access reallocation has invalid access identity")
        access_labels[access_id] = f"access{index}"

    effect_labels: dict[str, str] = {}
    effect_accesses: set[str] = set()
    for index, effect in enumerate(raw_effects):
        if not isinstance(effect, Mapping):
            raise ValueError("canonical access reallocation has malformed effect")
        effect_id = effect.get("effect_id")
        access_ref = effect.get("access_ref")
        if (
            not isinstance(effect_id, str)
            or re.fullmatch(r"e[0-9]+", effect_id) is None
            or effect_id in effect_labels
            or not isinstance(access_ref, str)
            or access_ref not in access_labels
            or access_ref in effect_accesses
        ):
            raise ValueError("canonical access reallocation has non-access effect")
        effect_labels[effect_id] = f"effect{index}"
        effect_accesses.add(access_ref)
    if effect_accesses != set(access_labels):
        raise ValueError("canonical access reallocation does not close access effects")

    def rewrite(value: Any) -> Any:
        if isinstance(value, Mapping):
            rewritten: dict[Any, Any] = {}
            for key, item in value.items():
                if key == "access_ref" and isinstance(item, str):
                    if item not in access_labels:
                        raise ValueError(
                            "canonical access reallocation has unknown access reference"
                        )
                    rewritten[key] = access_labels[item]
                elif key in {
                    "effect_id",
                    "effect_ref",
                    "subject_ref",
                    "update_effect_ref",
                } and isinstance(item, str) and item in effect_labels:
                    rewritten[key] = effect_labels[item]
                elif key in {"effect_id", "effect_ref", "update_effect_ref"} and isinstance(
                    item, str
                ):
                    if item not in effect_labels:
                        raise ValueError(
                            "canonical access reallocation has unknown effect reference"
                        )
                elif key == "effects" and isinstance(item, list):
                    # ``j.boundary.effects`` is the table of effect objects;
                    # only a lexical scope's ``effects`` field is a list of
                    # e* references.  Rewriting both as reference lists made
                    # the candidate-specific reorder path reject its own
                    # completed boundary table before it could reallocate
                    # access/effect identities.
                    if all(isinstance(effect, Mapping) for effect in item):
                        rewritten[key] = [rewrite(effect) for effect in item]
                    else:
                        rewritten[key] = [
                            effect_labels[effect_ref]
                            if isinstance(effect_ref, str) and effect_ref in effect_labels
                            else _raise_unknown_canonical_effect_reference(effect_ref)
                            for effect_ref in item
                        ]
                else:
                    rewritten[key] = rewrite(item)
            return rewritten
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    rewritten_boundary = rewrite(copy.deepcopy(boundary))
    if not isinstance(rewritten_boundary, dict):
        raise ValueError("canonical access reallocation lost boundary")
    rewritten_accesses = rewritten_boundary.get("accesses")
    if not isinstance(rewritten_accesses, list):
        raise ValueError("canonical access reallocation lost accesses")
    for access, local_label in zip(
        rewritten_accesses, access_labels.values(), strict=True
    ):
        if not isinstance(access, dict):
            raise ValueError("canonical access reallocation has malformed rewritten access")
        access.pop("access_id", None)
        access["access_label"] = local_label
    j["boundary"] = rewritten_boundary


def _raise_unknown_canonical_effect_reference(value: Any) -> Any:
    """Raise from a compact list rewrite without silently preserving an e*."""

    raise ValueError("canonical access reallocation has unknown effect reference")


def _materialize_boundary_effects_from_accesses(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> CanonicalHandleAllocator:
    """Allocate canonical identities for provider-authored access/effect pairs.

    The provider submits access meaning and the paired effect kind/target/node
    owner in one Source-occurrence row. The host validates that pair and only
    allocates dense ``a*``/``e*`` identities; it never infers a missing effect.
    A second invocation after Source-order reallocation may recover the same
    already-authored effect fields from the transient canonical effect table.
    The Source transfer ledger later establishes actual occurrence order.

    This is all-or-nothing.  It never fabricates a malformed access, converts
    an unknown kind, or leaves a partially rewritten reference table.
    """

    j = payload.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_accesses = boundary.get("accesses") if isinstance(boundary, dict) else None
    if not isinstance(raw_accesses, list):
        raise ValueError("canonical access/effect construction lacks an access array")
    try:
        allocator = allocator.with_declarations(
            {
                "a": CanonicalHandleAllocator.canonical_names("a", len(raw_accesses)),
                "e": CanonicalHandleAllocator.canonical_names("e", len(raw_accesses)),
            }
        )
    except ValueError as exc:
        raise ValueError(
            "canonical access/effect allocator declarations conflict"
        ) from exc
    access_names = allocator.declared("a")
    effect_names = allocator.declared("e")
    if len(access_names) != len(raw_accesses) or len(effect_names) != len(raw_accesses):
        raise ValueError("canonical access/effect allocator declarations are incomplete")

    old_to_new: dict[str, str] = {}
    old_effect_to_new: dict[str, str] = {}
    raw_effect_by_access: dict[str, Mapping[str, Any]] = {}
    raw_effects = boundary.get("effects") if isinstance(boundary, Mapping) else None
    if isinstance(raw_effects, list):
        for raw_effect in raw_effects:
            if not isinstance(raw_effect, Mapping):
                raise ValueError("candidate boundary effects are malformed")
            raw_access_ref = raw_effect.get("access_ref")
            raw_effect_id = raw_effect.get("effect_id")
            if (
                raw_access_ref is not None
                and (
                    not isinstance(raw_access_ref, str)
                    or not isinstance(raw_effect_id, str)
                    or raw_access_ref in raw_effect_by_access
                )
            ):
                raise ValueError("candidate boundary effects do not uniquely name accesses")
            if isinstance(raw_access_ref, str):
                raw_effect_by_access[raw_access_ref] = copy.deepcopy(
                    dict(raw_effect)
                )
    rewritten_accesses: list[dict[str, Any]] = []
    rewritten_effects: list[dict[str, Any]] = []
    expected_effect_kind = {"load": "read", "store": "write", "update": "update"}
    missing = object()
    for index, raw_access in enumerate(raw_accesses):
        if not isinstance(raw_access, Mapping):
            raise ValueError("candidate boundary access is malformed")
        access = copy.deepcopy(dict(raw_access))
        old_ref = access.pop("access_label", None)
        # Source ordinal is provider-wire identity evidence only.  It is
        # checked before construction and then removed together with the
        # local label, leaving exactly one canonical a* identity downstream.
        access.pop("ordinal", None)
        if (
            not isinstance(old_ref, str)
            or re.fullmatch(r"access[0-9]+", old_ref) is None
            or old_ref in old_to_new
        ):
            raise ValueError("candidate boundary access lacks a unique local access label")
        old_to_new[old_ref] = access_names[index]
        existing_effect = raw_effect_by_access.get(old_ref)
        old_effect_id = (
            existing_effect.get("effect_id")
            if isinstance(existing_effect, Mapping)
            else None
        )
        if isinstance(old_effect_id, str):
            old_effect_to_new[old_effect_id] = effect_names[index]
        kind = access.get("kind")
        tensor = access.get("tensor")
        node_ref = access.get("node_ref")
        if kind not in expected_effect_kind or not isinstance(tensor, str):
            raise ValueError("candidate boundary access lacks a supported kind or tensor")
        declared_effect_kind = access.pop("effect_kind", missing)
        declared_effect_target = access.pop("effect_target", missing)
        declared_effect_node_ref = access.pop("effect_node_ref", missing)
        if (
            declared_effect_kind is missing
            and declared_effect_target is missing
            and declared_effect_node_ref is missing
            and isinstance(existing_effect, Mapping)
        ):
            declared_effect_kind = existing_effect.get("kind", missing)
            declared_effect_target = existing_effect.get("target", missing)
            declared_effect_node_ref = existing_effect.get("node_ref", missing)
        if (
            declared_effect_kind is missing
            or declared_effect_target is missing
            or declared_effect_node_ref is missing
        ):
            raise ValueError(
                "candidate boundary access lacks provider-authored effect semantics"
            )
        if (
            declared_effect_kind != expected_effect_kind[kind]
            or declared_effect_target != tensor
            or declared_effect_node_ref != node_ref
        ):
            raise ValueError(
                "candidate boundary effect semantics differ from their access"
            )
        access_id = access_names[index]
        access["access_id"] = access_id
        rewritten_accesses.append(access)
        rewritten_effects.append(
            {
                "effect_id": effect_names[index],
                "kind": declared_effect_kind,
                "target": declared_effect_target,
                "node_ref": declared_effect_node_ref,
                "access_ref": access_id,
            }
        )

    def rewrite_access_ref(value: str) -> str:
        try:
            return old_to_new[value]
        except KeyError as exc:
            raise ValueError(
                "semantic access reference has no local access label"
            ) from exc

    def rewrite_access_references(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: (
                    rewrite_access_ref(item)
                    if key == "access_ref" and isinstance(item, str)
                    else old_effect_to_new.get(item, item)
                    if key in {"effect_ref", "subject_ref", "update_effect_ref"}
                    and isinstance(item, str)
                    else [
                        old_effect_to_new.get(effect_ref, effect_ref)
                        if isinstance(effect_ref, str)
                        else rewrite_access_references(effect_ref)
                        for effect_ref in item
                    ]
                    if key == "effects" and isinstance(item, list)
                    else rewrite_access_references(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite_access_references(item) for item in value]
        return value

    rewritten_boundary = rewrite_access_references(copy.deepcopy(boundary))
    if not isinstance(rewritten_boundary, dict):
        raise ValueError("canonical access/effect rewrite did not preserve boundary")
    rewritten_boundary["accesses"] = rewritten_accesses
    rewritten_boundary["effects"] = rewritten_effects
    j["boundary"] = rewritten_boundary
    # π may carry a uniquely proven access representation row.  It is not a
    # second access authority, so remap its typed access_ref together with the
    # boundary table rather than leaving it attached to the provider-order
    # spelling after Source-ledger ordering.
    pi = payload.get("pi")
    if isinstance(pi, Mapping):
        rewritten_pi = rewrite_access_references(copy.deepcopy(dict(pi)))
        if not isinstance(rewritten_pi, dict):
            raise ValueError("canonical access/effect rewrite did not preserve pi")
        payload["pi"] = rewritten_pi
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_access_effect_constructor",
        path="j.boundary.accesses + j.boundary.effects",
        action=(
            "allocate dense access/effect identities after validating the paired "
            "provider-authored effect kind/target/node semantics"
        ),
    )
    return allocator


def _materialize_computed_access_predicate_guards(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Project a submitted local ``i1`` access predicate into its guard row.

    The semantic provider surface deliberately gives a masked Source access
    one provider-local predicate label, while the canonical artifact gives a
    computed Layer-A ``i1`` value one legal boundary representation:
    ``access_guard_relations``.  Keeping the temporary ``predicate: v*`` after
    host SSA allocation is not an alternative representation -- functional
    boundary validation rejects it before it can consume the corresponding
    guard.  Conversely, requiring the provider to spell a canonical ``a*``
    relation is impossible by construction because a*/e* allocation happens
    after Source-order reconciliation.

    This adapter performs no semantic predicate inference.  It only converts
    the exact predicate value already submitted for an already allocated
    access, if its declared i1 shape has one *unique*, order-preserving mapping
    into that access tensor's declared shape.  Ambiguous/non-broadcastable
    cases stay as their authored predicate and consequently remain
    fail-closed in CandidateCheck.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if not all(isinstance(item, Mapping) for item in (function, boundary)):
        raise ValueError("canonical predicate-guard projection lacks J boundary")
    accesses = boundary.get("accesses")
    guards = boundary.get("access_guard_relations")
    if not isinstance(accesses, list) or not isinstance(guards, list):
        raise ValueError("canonical predicate-guard projection lacks access tables")

    value_types: dict[str, tuple[str, tuple[Any, ...]]] = {}

    def add_type(name: Any, value_type: Any) -> None:
        if not isinstance(name, str) or not isinstance(value_type, Mapping):
            return
        dtype = value_type.get("dtype")
        shape = value_type.get("shape")
        if isinstance(dtype, str) and isinstance(shape, list):
            value_types[name] = (dtype.casefold(), tuple(shape))

    for declaration in [
        *(function.get("inputs", ()) if isinstance(function.get("inputs"), list) else ()),
        *(function.get("outputs", ()) if isinstance(function.get("outputs"), list) else ()),
    ]:
        if isinstance(declaration, Mapping):
            add_type(declaration.get("name"), declaration)
    body = function.get("body")
    rows = semantic_operation_wire_rows({"body": body}) if isinstance(body, list) else None
    if rows is None:
        raise ValueError("canonical predicate-guard projection has malformed operation body")
    for _path, operation in rows:
        result = operation.get("result")
        if isinstance(result, str):
            add_type(result, operation.get("type"))

    guarded_accesses = {
        row.get("access_ref")
        for row in guards
        if isinstance(row, Mapping) and isinstance(row.get("access_ref"), str)
    }
    projected = 0
    for access in accesses:
        if not isinstance(access, dict):
            raise ValueError("canonical predicate-guard projection has malformed access")
        access_id = access.get("access_id")
        predicate = access.get("predicate")
        tensor = access.get("tensor")
        guard_authorization = access.get("semantic_guard_authorization")
        if (
            not isinstance(access_id, str)
            or access_id in guarded_accesses
            or not isinstance(predicate, str)
            or predicate not in value_types
            or not isinstance(tensor, str)
            or not isinstance(guard_authorization, str)
            or re.fullmatch(r"so_[0-9a-f]{32}", guard_authorization) is None
        ):
            continue
        predicate_type = value_types.get(predicate)
        target_type = value_types.get(tensor)
        if (
            predicate_type is None
            or target_type is None
            or not is_closed_boolean_dtype(predicate_type[0])
        ):
            continue
        # A guard relation's ``axes`` uses canonical broadcast-in-dim
        # terminology: it lists *inserted target axes*, not the source axes
        # which happen to match.  For [x0] -> [x0, x1], for example, the
        # unique relation is axes=[1], not axes=[0].
        if not computed_guard_projection_supported(access.get("kind")):
            # Updates have no closed guard representation. Preserve the exact
            # provider-authored predicate and reject rather than erasing the
            # physical mask with a host-authored ``true`` predicate.
            continue
        guard_axes = unique_access_guard_projection_axes(
            predicate_type[1], target_type[1]
        )
        if guard_axes is None:
            continue
        guards.append(
            {
                "access_ref": access_id,
                "value_ref": predicate,
                "axes": list(guard_axes),
                "polarity": "positive",
            }
        )
        # The guard relation is the canonical representation for a computed
        # predicate.  The access's logical predicate is therefore the neutral
        # domain predicate ``true``; the relation retains the provider-authored
        # i1 for target realization. For a store this is accepted only by the
        # closed output-only guarded-write profile downstream, which proves the
        # authored value producer is the complete logical output and that no
        # unmodelled pre-state/select merge was introduced. A candidate-specific
        # Source reorder may temporarily restore the provider-local spelling
        # while rebuilding a*/e* rows, so membership in the typed operation-
        # result table -- not a local-label spelling -- is the ownership test.
        access["predicate"] = "true"
        guarded_accesses.add(access_id)
        projected += 1
    if projected:
        _record_deterministic_repair(
            repairs,
            rule_code="deterministic_computed_access_guard_projection",
            path="j.boundary.accesses[*].predicate + j.boundary.access_guard_relations",
            action=(
                "project each uniquely typed provider-local i1 access predicate "
                "onto its allocated Source-order access guard relation"
            ),
        )


def _materialize_flat_semantic_ssa_handles(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> CanonicalHandleAllocator | None:
    """Allocate canonical n*/v* for one closed flat semantic operation DAG.

    A provider's local operation/result labels are not J semantics.  For a
    flat body with one result per operation, validate the complete def-use
    graph first, then allocate both namespaces in lexical transaction order.
    An operation id is optional at this semantic boundary; omitted ids receive
    a transaction-local label that never enters the artifact. This removes the
    need for the provider to maintain a dense n*/v* sequence or a parallel node
    map while preserving exact operation dependencies.

    Structured regions, multi-result operations, missing producers, ambiguous
    boundary node/value references, and forward uses involving an impure
    operation are deliberately left untouched. Those cases require the
    complete structured allocator path and remain fail-closed rather than
    being flattened or guessed.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    if (
        not isinstance(function, dict)
        or not isinstance(boundary, dict)
        or not isinstance(function.get("body"), list)
    ):
        return
    body = function["body"]
    if not body or any(
        not isinstance(operation, Mapping)
        or isinstance(operation.get("body"), list)
        or operation.get("positive") is not None
        or operation.get("negative") is not None
        or isinstance(operation.get("result"), list)
        for operation in body
    ):
        return
    try:
        allocator = allocator.with_declarations(
            {"v": CanonicalHandleAllocator.canonical_names("v", len(body))}
        )
    except ValueError:
        return None
    node_names = allocator.declared("n")
    value_names = allocator.declared("v")
    if len(node_names) != len(body) or len(value_names) != len(body):
        return None

    input_names = {
        item.get("name")
        for item in function.get("inputs", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(input_names) != len(function.get("inputs", [])):
        return
    producer_by_value: dict[str, int] = {}
    local_nodes: set[str] = set()
    synthetic_nodes: dict[int, str] = {}
    for index, operation in enumerate(body):
        authored_node = operation.get("id")
        # Node ids are host-owned mechanical names.  A semantic operation may
        # omit one entirely; keep an unforgeable transaction-local label until
        # the allocator assigns its canonical n*.  An explicitly supplied
        # local label remains useful only for a semantic boundary attachment
        # and is still checked for uniqueness below.
        old_node = (
            authored_node
            if isinstance(authored_node, str) and authored_node
            else f"@semantic-node-{index}"
        )
        old_value = operation.get("result")
        # Empty operands are optional in the canonical operation wire.  A
        # zero-input operation such as iota may therefore omit the field.
        operands = operation.get("operands", [])
        if (
            (authored_node is not None and not isinstance(authored_node, str))
            or (isinstance(authored_node, str) and not authored_node)
            or old_node in local_nodes
            or not isinstance(old_value, str)
            or not old_value
            or old_value in producer_by_value
            or old_value in input_names
            or not isinstance(operands, list)
            or any(not isinstance(operand, str) for operand in operands)
        ):
            return
        local_nodes.add(old_node)
        if authored_node is None:
            synthetic_nodes[index] = old_node
        producer_by_value[old_value] = index

    # Validate the complete semantic DAG before allocating either namespace.
    # A forward use is mechanically recoverable only through a stable reorder
    # of explicitly pure flat operations.  Effects, regions and unknown
    # values never move; they stay reject-only rather than being silently
    # reinterpreted by a local handle pass.
    predecessors: list[set[int]] = [set() for _ in body]
    successors: list[set[int]] = [set() for _ in body]
    needs_reorder = False
    for index, operation in enumerate(body):
        operands = operation.get("operands", [])
        for operand in operands:
            producer = producer_by_value.get(operand)
            if producer is None:
                if operand not in input_names:
                    return
                continue
            if producer == index:
                return
            predecessors[index].add(producer)
            successors[producer].add(index)
            needs_reorder = needs_reorder or producer > index
    if needs_reorder and any(
        operation.get("op") not in _TOPOLOGICALLY_REORDERABLE_OPS
        for operation in body
    ):
        return
    ready = [index for index, edges in enumerate(predecessors) if not edges]
    order: list[int] = []
    pending = [set(edges) for edges in predecessors]
    while ready:
        current = min(ready)
        ready.remove(current)
        order.append(current)
        for successor in sorted(successors[current]):
            pending[successor].discard(current)
            if not pending[successor]:
                ready.append(successor)
    if len(order) != len(body):
        return
    ordered_body = [body[index] for index in order]
    node_map = {
        (
            synthetic_nodes.get(original_index)
            or str(body[original_index]["id"])
        ): node_names[index]
        for index, original_index in enumerate(order)
    }
    value_map = {
        str(operation["result"]): value_names[index]
        for index, operation in enumerate(ordered_body)
    }

    def rewrite_node(value: Any) -> Any:
        if isinstance(value, Mapping):
            rewritten: dict[Any, Any] = {}
            for key, item in value.items():
                if key in {"node_ref", "consumer_ref", "update_node_ref", "subject_ref"}:
                    rewritten[key] = node_map.get(item, item) if isinstance(item, str) else item
                elif key in {"value_ref", "predicate_ref", "predicate", "result_ref"}:
                    rewritten[key] = value_map.get(item, item) if isinstance(item, str) else item
                else:
                    rewritten[key] = rewrite_node(item)
            return rewritten
        if isinstance(value, list):
            return [rewrite_node(item) for item in value]
        return value

    rewritten_body: list[dict[str, Any]] = []
    for original_index in order:
        operation = body[original_index]
        rewritten = copy.deepcopy(dict(operation))
        old_node = synthetic_nodes.get(original_index) or rewritten.get("id")
        old_value = rewritten["result"]
        rewritten["id"] = node_map[old_node]
        rewritten["result"] = value_map[old_value]
        if "operands" in rewritten:
            rewritten["operands"] = [
                value_map.get(operand, operand)
                for operand in rewritten["operands"]
            ]
        rewritten_body.append(rewritten)

    rewritten_boundary = rewrite_node(copy.deepcopy(boundary))
    if not isinstance(rewritten_boundary, dict):
        return
    mappings = rewritten_boundary.get("output_mappings")
    if not isinstance(mappings, list):
        return
    for mapping in mappings:
        if not isinstance(mapping, dict):
            return
        value = mapping.get("value")
        if isinstance(value, str):
            mapping["value"] = value_map.get(value, value)
    accesses = rewritten_boundary.get("accesses", [])
    if not isinstance(accesses, list):
        return
    for access in accesses:
        if not isinstance(access, dict):
            return
        if "index_expression" in access:
            access["index_expression"] = _rewrite_provider_local_value_references(
                access["index_expression"], value_map
            )
    updates = rewritten_boundary.get("updates", [])
    if not isinstance(updates, list):
        return
    for update in updates:
        if not isinstance(update, dict):
            return
        # ``node_ref`` crossed the same transaction in ``rewrite_node``.
        # The remaining LogicalUpdateSpec operands are in the provider-local
        # SSA namespace just like an output selection or broadcast operand,
        # so they must be rewritten from this exact completed graph as well.
        # This is an identity-only projection: an unknown label is retained
        # for strict validation rather than inferred or replaced.
        for field_name in ("base", "indices", "values"):
            value = update.get(field_name)
            if isinstance(value, str):
                update[field_name] = value_map.get(value, value)
    broadcasts = rewritten_boundary.get("broadcasts", [])
    if not isinstance(broadcasts, list):
        return
    for projection in broadcasts:
        if not isinstance(projection, dict):
            return
        operand = projection.get("operand")
        if isinstance(operand, str):
            # ``node_ref`` was rewritten by ``rewrite_node`` above, but a
            # broadcast's consumed value is deliberately named ``operand``
            # rather than ``value_ref``.  It is the same provider-local SSA
            # namespace and must cross this flat allocator transaction too;
            # otherwise a valid provider boundary row reaches canonical J
            # with an unallocated local spelling and fails schema decoding.
            projection["operand"] = value_map.get(operand, operand)
    function["body"] = rewritten_body
    j["boundary"] = rewritten_boundary
    j["node_ids"] = {
        node_names[index]: f"body[{index}]"
        for index in range(len(rewritten_body))
    }
    if isinstance(payload.get("pi"), Mapping):
        payload["pi"] = rewrite_node(payload["pi"])
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_flat_semantic_ssa_allocator",
        path="j.intent.intent_function.body + j.node_ids",
        action=(
            "validate and, only for a closed pure DAG, stably topologically order "
            "the semantic def-use graph before allocating canonical n*/v* ids "
            "with all typed node/value references"
        ),
    )
    return allocator


def _materialize_structured_node_handles(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> CanonicalHandleAllocator | None:
    """Allocate one lexical n*/node_ids projection for a structured J tree.

    Region syntax is semantic and remains untouched here: this helper neither
    reorders a loop/branch body nor invents its carry, yield, or SSA values.
    It owns only the duplicated node identity view.  A provider may omit every
    operation ``id``; supplied local ids are accepted solely as unambiguous
    references from semantic boundary rows and are rewritten with the same
    transaction.  This lets the later recursive SSA allocator retain a single
    remaining responsibility instead of competing with a parallel node map.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(function.get("body"), list):
        return

    body = function["body"]
    has_region = False
    rows: list[tuple[str, dict[str, Any]]] = []

    def visit(items: object, prefix: str) -> bool:
        nonlocal has_region
        if not isinstance(items, list):
            return False
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                return False
            path = f"{prefix}[{index}]"
            rows.append((path, item))
            op_name = item.get("op")
            if op_name == "loop":
                has_region = True
                if not visit(item.get("body"), f"{path}.body"):
                    return False
            elif op_name in {"effect_if", "value_if"}:
                has_region = True
                for arm_name in ("positive", "negative"):
                    arm = item.get(arm_name)
                    if not isinstance(arm, dict) or not visit(
                        arm.get("body"), f"{path}.{arm_name}.body"
                    ):
                        return False
        return True

    if not visit(body, "body") or not has_region:
        return

    node_names = allocator.declared("n")
    if len(node_names) != len(rows):
        return None
    old_to_new: dict[str, str] = {}
    for index, (_path, operation) in enumerate(rows):
        authored = operation.get("id")
        if authored is not None and (
            not isinstance(authored, str) or not authored or authored in old_to_new
        ):
            return
        if isinstance(authored, str):
            old_to_new[authored] = node_names[index]

    for index, (_path, operation) in enumerate(rows):
        operation["id"] = node_names[index]

    def rewrite(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: (
                    old_to_new.get(item, item)
                    if key in {"node_ref", "consumer_ref", "update_node_ref", "subject_ref"}
                    and isinstance(item, str)
                    else rewrite(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    rewritten = rewrite(j)
    if not isinstance(rewritten, dict):
        return
    rewritten["node_ids"] = {
        node: path
        for node, (path, _op) in zip(node_names, rows, strict=True)
    }
    payload["j"] = rewritten
    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_structured_node_allocator",
        path="j.intent.intent_function.body + j.node_ids",
        action=(
            "allocate lexical n* identities and node_ids for the complete "
            "structured operation tree without changing region semantics"
        ),
    )
    return allocator


def _materialize_structured_semantic_ssa_handles(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
    *,
    allocator: CanonicalHandleAllocator,
) -> CanonicalHandleAllocator | None:
    """Allocate every structured-region SSA binder in one transaction.

    A loop cannot be flattened into a pure topological DAG: its body arguments,
    scan names and induction variable are lexical binders, while ``next_value``
    is defined in the body and connected to a carry yield.  This pass therefore
    preserves region order and loop wiring, but removes the provider burden of
    maintaining canonical ``v*`` spellings.  It accepts only a complete tree
    whose result and region binders are unique; all unresolved references stay
    reject-only in the shared construction record.
    """

    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    if not isinstance(function, dict) or not isinstance(function.get("body"), list):
        return None
    rows = semantic_operation_wire_rows(function)
    if rows is None or not any(
        operation.get("op") in {"loop", "effect_if", "value_if"}
        for _path, operation in rows
    ):
        return None
    inputs = {
        row.get("name")
        for row in function.get("inputs", [])
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    if len(inputs) != len(function.get("inputs", [])):
        return None

    declared: list[str] = []
    seen: set[str] = set()

    def bind(value: Any) -> bool:
        if not isinstance(value, str) or not value or value in inputs:
            return False
        if value in seen:
            return False
        seen.add(value)
        declared.append(value)
        return True

    # The operation traversal is lexical and includes nested bodies/arms.  It
    # is the same traversal that supplies n*/node_ids, so v* and n* cannot use
    # competing body interpretations.
    for _path, operation in rows:
        result = operation.get("result")
        result_values = result if isinstance(result, list) else [result]
        if any(not bind(value) for value in result_values if value is not None):
            return None
        if operation.get("op") != "loop":
            continue
        carry = operation.get("carry")
        scan = operation.get("scan", [])
        trip = operation.get("trip")
        if not isinstance(carry, list) or not isinstance(scan, list):
            return None
        for state in carry:
            if not isinstance(state, Mapping) or not bind(state.get("body_argument")):
                return None
            # ``result_value`` is an already-declared loop result. It must not
            # become a second binder with a different allocator position.
            if state.get("result_value") not in seen:
                return None
        for item in scan:
            if not isinstance(item, Mapping) or not bind(item.get("name")):
                return None
        if isinstance(trip, Mapping) and "index" in trip and not bind(trip.get("index")):
            return None

    # Loop wiring is validated only after the complete lexical traversal has
    # collected every direct body producer.  Checking ``next_value`` while
    # visiting the loop header is order-sensitive and rejects a valid body
    # producer that appears later in the body.  The cross-field relationship
    # remains strict: loop results and yields must be positional projections of
    # carry roles, and every next value must be produced by that loop's direct
    # body (never by a nested region or a sibling operation).
    for path, operation in rows:
        if operation.get("op") != "loop":
            continue
        carry = operation.get("carry")
        result_values = operation.get("result")
        yielded = operation.get("yield")
        body = operation.get("body")
        if (
            not isinstance(carry, list)
            or not isinstance(result_values, list)
            or not isinstance(yielded, list)
            or not isinstance(body, list)
            or len(carry) != len(result_values)
            or len(carry) != len(yielded)
        ):
            return None
        expected_results: list[Any] = []
        expected_yields: list[Any] = []
        for state in carry:
            if not isinstance(state, Mapping):
                return None
            result_value = state.get("result_value")
            next_value = state.get("next_value")
            if not isinstance(result_value, str) or not isinstance(next_value, str):
                return None
            expected_results.append(result_value)
            expected_yields.append(next_value)
        if result_values != expected_results or yielded != expected_yields:
            return None

        direct_body_results: set[str] = set()
        for child in body:
            if not isinstance(child, Mapping):
                return None
            child_result = child.get("result")
            child_results = (
                child_result
                if isinstance(child_result, list)
                else [child_result]
            )
            if any(not isinstance(item, str) for item in child_results if item is not None):
                return None
            direct_body_results.update(
                item for item in child_results if isinstance(item, str)
            )
        if any(next_value not in direct_body_results for next_value in expected_yields):
            return None

    try:
        allocator = allocator.with_declarations(
            {"v": CanonicalHandleAllocator.canonical_names("v", len(declared))}
        )
    except ValueError:
        return None
    value_names = allocator.declared("v")
    if len(value_names) != len(declared):
        return None
    value_map = dict(zip(declared, value_names, strict=True))

    def value(value: Any) -> Any:
        return value_map.get(value, value) if isinstance(value, str) else value

    def rewrite_operation(operation: Any) -> bool:
        if not isinstance(operation, dict):
            return False
        operands = operation.get("operands")
        if isinstance(operands, list):
            operation["operands"] = [value(item) for item in operands]
        result = operation.get("result")
        if isinstance(result, list):
            operation["result"] = [value(item) for item in result]
        elif result is not None:
            operation["result"] = value(result)
        if operation.get("op") == "loop":
            carry = operation.get("carry")
            scan = operation.get("scan", [])
            trip = operation.get("trip")
            yielded = operation.get("yield")
            if not isinstance(carry, list) or not isinstance(scan, list) or not isinstance(yielded, list):
                return False
            for state in carry:
                if not isinstance(state, dict):
                    return False
                for field_name in (
                    "init_value",
                    "body_argument",
                    "next_value",
                    "result_value",
                ):
                    if field_name not in state:
                        return False
                    state[field_name] = value(state[field_name])
            for item in scan:
                if not isinstance(item, dict) or "name" not in item or "from" not in item:
                    return False
                item["name"] = value(item["name"])
                item["from"] = value(item["from"])
            if isinstance(trip, dict):
                for field_name in ("index", "upper_bound"):
                    if field_name in trip:
                        trip[field_name] = value(trip[field_name])
            operation["yield"] = [value(item) for item in yielded]
        if operation.get("op") in {"effect_if", "value_if"}:
            if "predicate" in operation:
                operation["predicate"] = value(operation["predicate"])
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if not isinstance(arm, dict) or not isinstance(arm.get("body"), list):
                    return False
                if "yield_value" in arm:
                    arm["yield_value"] = value(arm["yield_value"])
                if not all(rewrite_operation(child) for child in arm["body"]):
                    return False
        if operation.get("op") == "loop":
            body = operation.get("body")
            if not isinstance(body, list) or not all(rewrite_operation(child) for child in body):
                return False
        return True

    if not all(rewrite_operation(operation) for operation in function["body"]):
        return None

    boundary = j.get("boundary") if isinstance(j, dict) else None
    if isinstance(boundary, dict):
        for access in boundary.get("accesses", []):
            if not isinstance(access, dict):
                continue
            if "index_expression" in access:
                access["index_expression"] = _rewrite_provider_local_value_references(
                    access["index_expression"], value_map
                )
            # A provider-owned masked-access predicate is an exact local i1
            # result label, not a Source expression. Rewrite it in the same
            # one-way transaction as all other local SSA references so the
            # later canonical predicate->access_guard projection can find its
            # typed producer. ``value`` leaves logical literals, Source-owned
            # text, and unknown labels untouched; those remain fail-closed in
            # the existing parser/reference validation.
            if "predicate" in access:
                access["predicate"] = value(access["predicate"])
        for mapping in boundary.get("output_mappings", []):
            if isinstance(mapping, dict) and "value" in mapping:
                mapping["value"] = value(mapping["value"])
        for row in boundary.get("access_guard_relations", []):
            if isinstance(row, dict) and "value_ref" in row:
                row["value_ref"] = value(row["value_ref"])
        for row in boundary.get("index_bounds", []):
            if isinstance(row, dict):
                for field_name in ("data", "indices"):
                    if field_name in row:
                        row[field_name] = value(row[field_name])
        for row in boundary.get("updates", []):
            if isinstance(row, dict):
                for field_name in ("base", "indices", "values"):
                    if field_name in row:
                        row[field_name] = value(row[field_name])
        for row in boundary.get("broadcasts", []):
            if isinstance(row, dict) and "operand" in row:
                row["operand"] = value(row["operand"])
    for row in j.get("storage", []) if isinstance(j, dict) else []:
        if isinstance(row, dict) and isinstance(row.get("members"), list):
            row["members"] = [value(item) for item in row["members"]]
    for row in j.get("value_dimension_relations", []) if isinstance(j, dict) else []:
        if isinstance(row, dict) and "value_ref" in row:
            row["value_ref"] = value(row["value_ref"])
    pi = payload.get("pi")
    if isinstance(pi, Mapping):
        def rewrite_pi(item: Any) -> Any:
            if isinstance(item, Mapping):
                return {
                    key: (
                        value(child)
                        if key in {"value_ref", "predicate_ref", "result_ref"}
                        else rewrite_pi(child)
                    )
                    for key, child in item.items()
                }
            if isinstance(item, list):
                return [rewrite_pi(child) for child in item]
            return item
        rewritten_pi = rewrite_pi(copy.deepcopy(dict(pi)))
        if not isinstance(rewritten_pi, dict):
            return None
        payload["pi"] = rewritten_pi

    _record_deterministic_repair(
        repairs,
        rule_code="deterministic_structured_ssa_allocator",
        path="j.intent.intent_function.body",
        action=(
            "allocate all structured operation results and loop/branch lexical "
            "binders as one v* namespace and rewrite typed references together"
        ),
    )
    return allocator


_TOPOLOGICALLY_REORDERABLE_OPS = frozenset(
    {"const", "iota", "map_unary", "map_binary", "cast"}
)


def _require_pi_iteration_bindings_use_canonical_c_roles(
    payload: dict[str, Any],
) -> None:
    """Reject π tiled-domain rows lacking a Source-proven C tile role.

    π is evidence, not a cosmetic optional projection. Silently dropping an
    invalid provider row would turn an unproven claim into an empty witness and
    weaken the CandidateCheck boundary. The host may add a row only through a
    unique Source/J proof; provider-authored rows must already name an allowed
    trusted C role.
    """

    c = payload.get("c")
    pi = payload.get("pi")
    slots = c.get("slots") if isinstance(c, Mapping) else None
    bindings = pi.get("iteration_bindings") if isinstance(pi, Mapping) else None
    if not isinstance(slots, list) or not isinstance(bindings, list):
        return
    allowed = {
        row.get("name")
        for row in slots
        if isinstance(row, Mapping)
        and row.get("role") in {"tile_extent", "reduction_tile_extent"}
        and isinstance(row.get("name"), str)
    }
    if any(
        not isinstance(row, Mapping) or row.get("tile_slot") not in allowed
        for row in bindings
    ):
        raise ValueError(
            "pi iteration binding lacks a Source-proven C tile role"
        )


class _CanonicalConstructionBundleError(ValueError):
    """A complete bundle could not be formed from one candidate snapshot."""

    def __init__(self, diagnostics: Sequence[Mapping[str, Any]]) -> None:
        self.diagnostics = tuple(dict(item) for item in diagnostics)
        super().__init__("canonical_construction_bundle_incomplete")


_CANONICAL_CONSTRUCTION_BUNDLE_CATEGORY_ORDER = (
    "operation",
    "value_type",
    "scope",
    "boundary",
    "output",
    "dimension_relation",
    "value_dimension_relation",
    "typed_operation",
)


def _canonical_scope_bundle_diagnostic(
    function: Mapping[str, Any],
    scopes: Any,
    boundary: Any,
    node_ids: Any,
    operation_records: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Return one non-semantic scope-construction category for feedback.

    This is a deliberately small read-only projection of the existing scope
    constructor.  It identifies only mechanical ownership inputs that are
    already public in the canonical wire: the lexical scope tree, node map,
    and ownership witness for a node-less boundary effect.  It neither
    interprets provider operations nor exposes a candidate row.  When the
    existing allocator can derive a repairable scope bundle, its own derived
    membership is used only to classify the strict authored-field mismatch;
    it does not write the projection or alter admission.
    """

    if not isinstance(scopes, list) or any(
        not isinstance(scope, Mapping) for scope in scopes
    ):
        return "j.scopes", "one lexical scope header set"
    by_scope = {
        scope.get("scope_id"): scope
        for scope in scopes
        if isinstance(scope.get("scope_id"), str)
    }
    if len(by_scope) != len(scopes):
        return "j.scopes", "unique lexical scope identities"
    roots = [
        scope_id
        for scope_id, scope in by_scope.items()
        if scope.get("kind") == "function" and scope.get("parent_scope") is None
    ]
    if len(roots) != 1 or any(
        scope.get("parent_scope") is not None
        and scope.get("parent_scope") not in by_scope
        for scope in by_scope.values()
    ):
        return "j.scopes", "one rooted lexical scope tree"
    if not isinstance(node_ids, Mapping) or any(
        not isinstance(node, str) or not isinstance(path, str)
        for node, path in node_ids.items()
    ):
        return "j.node_ids", "one node identity for every lexical operation"
    expected_nodes = {
        record.get("id"): record.get("path")
        for record in operation_records
        if isinstance(record.get("id"), str) and isinstance(record.get("path"), str)
    }
    if expected_nodes != dict(node_ids):
        return "j.node_ids", "one node identity for every lexical operation"
    if not isinstance(boundary, Mapping) or not isinstance(
        boundary.get("effects"), list
    ):
        return "j.boundary.effects", "one boundary effect ownership table"

    # The strict allocator rejects an absent/stale lexical membership list,
    # while its repairable mode proves whether the same headers, operation
    # records, node map, and boundary rows can form one complete derived
    # scope view.  This is a diagnostic-only read of that exact authority:
    # report just the owning field name, never candidate values or a repaired
    # candidate body.
    derived_scopes = canonical_scope_records(
        function,
        scopes,
        boundary,
        node_ids,
        operation_records=operation_records,
        repairable_scope_fields=True,
    )
    if derived_scopes is not None:
        expected_by_scope = {
            record.get("scope_id"): record
            for record in derived_scopes
            if isinstance(record.get("scope_id"), str)
        }
        for scope in scopes:
            scope_id = scope.get("scope_id")
            expected = expected_by_scope.get(scope_id)
            if not isinstance(expected, Mapping):
                continue
            for field_name in ("definitions", "captures", "effects"):
                actual = scope.get(field_name)
                expected_values = expected.get(field_name)
                if not isinstance(actual, list):
                    return (
                        f"j.scopes[*].{field_name}",
                        f"complete derived lexical {field_name} membership",
                    )
                if field_name == "effects":
                    matched = actual == expected_values
                else:
                    matched = (
                        isinstance(expected_values, list)
                        and set(actual) == set(expected_values)
                    )
                if not matched:
                    return (
                        f"j.scopes[*].{field_name}",
                        f"complete derived lexical {field_name} membership",
                    )
    declared_effects = {
        effect_id
        for scope in by_scope.values()
        for effect_id in scope.get("effects", ())
        if isinstance(effect_id, str)
    }
    for effect in boundary["effects"]:
        if not isinstance(effect, Mapping) or not isinstance(
            effect.get("effect_id"), str
        ):
            return "j.boundary.effects", "well-formed boundary effect identities"
        node_ref = effect.get("node_ref")
        if node_ref is not None and node_ref not in node_ids:
            return "j.boundary.effects", "effect node references in the lexical node map"
        if node_ref is None and effect["effect_id"] not in declared_effects:
            return "j.scopes[*].effects", "one lexical owner for every node-less boundary effect"
    return "j.scopes", "lexically closed definitions, captures, and operation ownership"


def _canonical_construction_bundle_diagnostics(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Explain bundle absence by bounded semantic construction categories.

    This is deliberately a read-only projection of the same builder used for
    admission.  It does not inspect Source spellings, infer an operation, or
    offer a second raw-J grammar.  The ordered categories let a fresh Lift
    repair semantic fields without turning a private exception into prompt
    material.
    """

    j = payload.get("j") if isinstance(payload, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    rows: list[tuple[str, str, str]] = []

    def missing(category: str, path: str, expected: str) -> None:
        if category not in {item[0] for item in rows}:
            rows.append((category, path, expected))

    if not isinstance(function, Mapping) or not isinstance(function.get("body"), list):
        missing("operation", "j.intent.intent_function.body", "closed semantic operation rows")
    if not isinstance(boundary, Mapping):
        missing("boundary", "j.boundary", "closed boundary access and relation rows")
    if not isinstance(scopes, list):
        missing("scope", "j.scopes", "one lexical scope record set")
    if not isinstance(node_ids, Mapping):
        missing("operation", "j.node_ids", "one node identity for every operation")
    if not isinstance(signature, Mapping):
        missing("output", "j.signature", "Source-owned output ABI sequence")
    if isinstance(boundary, Mapping) and not isinstance(boundary.get("output_mappings"), list):
        missing("output", "j.boundary.output_mappings", "one ordered semantic output selection")
    if isinstance(j, Mapping):
        if not isinstance(j.get("dimension_relations", []), list):
            missing("dimension_relation", "j.dimension_relations", "closed dimension relation rows")
        if not isinstance(j.get("value_dimension_relations", []), list):
            missing("value_dimension_relation", "j.value_dimension_relations", "closed value-dimension relation rows")

    operation_records = (
        canonical_operation_records(function) if isinstance(function, Mapping) else None
    )
    if isinstance(function, Mapping) and operation_records is None:
        missing("operation", "j.intent.intent_function.body", "typed producer-before-consumer operations")
    if operation_records is not None:
        # ``build_structural`` deliberately returns one fail-closed ``None``
        # for every construction precondition.  Keep the commit predicate
        # unchanged, but project its existing independent inputs here so a
        # fresh semantic author receives the owning category rather than the
        # historically misleading catch-all ``typed_operation`` result.  No
        # candidate field is read back into a repair, inferred, or changed.
        if canonical_operation_objects(operation_records) is None:
            missing("typed_operation", "j.intent.intent_function.body", "typed operation record rows")
        if build_canonical_value_type_records(function, operation_records) is None:
            missing("value_type", "j.intent.intent_function.body", "closed input, output, and operation value types")
        if (
            isinstance(boundary, Mapping)
            and canonical_boundary_records(boundary) is None
        ):
            missing("boundary", "j.boundary", "closed boundary access and relation rows")
        if (
            isinstance(scopes, list)
            and isinstance(boundary, Mapping)
            and isinstance(node_ids, Mapping)
            and canonical_scope_records(
                function,
                scopes,
                boundary,
                node_ids,
                operation_records=operation_records,
            )
            is None
        ):
            scope_path, scope_expected = _canonical_scope_bundle_diagnostic(
                function,
                scopes,
                boundary,
                node_ids,
                operation_records,
            )
            missing("scope", scope_path, scope_expected)
    structural = None
    if (
        isinstance(function, Mapping)
        and isinstance(boundary, Mapping)
        and isinstance(scopes, list)
        and isinstance(node_ids, Mapping)
        and operation_records is not None
    ):
        structural = CanonicalConstructionRecords.build_structural(
            function,
            boundary,
            scopes,
            node_ids,
            operation_records=operation_records,
            dimension_relations=j.get("dimension_relations", []) if isinstance(j, Mapping) else [],
            value_dimension_relations=(
                j.get("value_dimension_relations", []) if isinstance(j, Mapping) else []
            ),
        )
        if structural is None and not rows:
            missing(
                "typed_operation",
                "j.intent.intent_function.body",
                "typed operations, boundary and lexical scope records",
            )
    if structural is not None and isinstance(signature, Mapping):
        complete = CanonicalConstructionRecords.build(
            function,
            boundary,
            scopes,
            node_ids,
            operation_records=operation_records,
            signature=signature,
            dimension_relations=j.get("dimension_relations", []) if isinstance(j, Mapping) else [],
            value_dimension_relations=(
                j.get("value_dimension_relations", []) if isinstance(j, Mapping) else []
            ),
        )
        if complete is None:
            missing("output", "j.boundary.output_mappings", "typed output, return and ABI-order records")
        elif complete.typed_operation_rows is None:
            missing("typed_operation", "j.intent.intent_function.body", "typed operation record rows")
    if not rows:
        missing("typed_operation", "j", "one complete typed construction bundle")
    rank = {
        category: index
        for index, category in enumerate(_CANONICAL_CONSTRUCTION_BUNDLE_CATEGORY_ORDER)
    }
    rows.sort(key=lambda item: (rank.get(item[0], len(rank)), item[1]))
    categories = ",".join(category for category, _path, _expected in rows)
    summary = {
        "rule_code": "canonical_construction_bundle_incomplete",
        "path": "j",
        "observed_refs_or_types": [f"missing_categories:{categories}"],
    }
    details = tuple(
        {
            "rule_code": f"canonical_construction_bundle_{category}_incomplete",
            "path": path,
            "observed_refs_or_types": [f"expected:{expected}"],
        }
        for category, path, expected in rows
    )
    return (summary, *details)


def _require_complete_canonical_construction_bundle(
    payload: Mapping[str, Any],
    *,
    bundle: CanonicalConstructionRecords | None = None,
) -> CanonicalConstructionRecords:
    """Require one strict operation/output/scope construction result.

    The provider-visible candidate surface has no independent operation-record
    wire.  The host must therefore prove that the final J snapshot can produce
    its shared construction bundle *before* CandidateCheck sees it.  This is
    a commit barrier, not a second validator and not a repair path: it writes
    nothing, infers nothing, and rejects a snapshot whose body, node map,
    scopes, output rows, boundary relations, or typed operation view disagree.
    Keeping this barrier at both transaction exits prevents a Source-ledger
    repair from leaving scopes or typed boundary rows attached to an older
    operation graph.
    """

    j = payload.get("j") if isinstance(payload, Mapping) else None
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    scopes = j.get("scopes") if isinstance(j, Mapping) else None
    node_ids = j.get("node_ids") if isinstance(j, Mapping) else None
    signature = j.get("signature") if isinstance(j, Mapping) else None
    if (
        not isinstance(function, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(scopes, list)
        or not isinstance(node_ids, Mapping)
        or not isinstance(signature, Mapping)
    ):
        raise _CanonicalConstructionBundleError(
            _canonical_construction_bundle_diagnostics(payload)
        )
    operation_records = canonical_operation_records(function)
    if operation_records is None:
        raise _CanonicalConstructionBundleError(
            _canonical_construction_bundle_diagnostics(payload)
        )
    if bundle is None:
        bundle = CanonicalConstructionRecords.build(
            function,
            boundary,
            scopes,
            node_ids,
            operation_records=operation_records,
            signature=signature,
            dimension_relations=j.get("dimension_relations", []),
            value_dimension_relations=j.get("value_dimension_relations", []),
        )
    if bundle is None or bundle.typed_operation_rows is None:
        raise _CanonicalConstructionBundleError(
            _canonical_construction_bundle_diagnostics(payload)
        )
    return bundle


def _finalize_canonical_construction_bundle(
    payload: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> CanonicalConstructionRecords:
    """Commit one candidate's derived output/scope/boundary bundle once."""

    # Source-order access/effect rebinding can replace the private working J
    # snapshot after the first ingress-time operation pass. IndexBoundsSpec is
    # host-derived from the *final* typed gather/scatter graph, so project it
    # immediately before the shared bundle is committed. This neither supplies
    # provider-owned semantics nor accepts an incomplete graph: the existing
    # helper remains a no-op unless every required operation field is already
    # closed, and the strict SemanticArchive validator remains final.
    j = payload.get("j")
    intent = j.get("intent") if isinstance(j, Mapping) else None
    function = (
        intent.get("intent_function") if isinstance(intent, Mapping) else None
    )
    operation_records = (
        _canonical_operation_records(function)
        if isinstance(function, Mapping)
        else None
    )
    _repair_canonical_index_bounds(payload, operation_records, repairs)
    bundle = _repair_canonical_record_links(payload, repairs)
    return _require_complete_canonical_construction_bundle(payload, bundle=bundle)


_CANONICAL_CONSTRUCTOR_FAILURE_RULES: tuple[
    tuple[tuple[str, ...], str, str, str], ...
] = (
    (
        ("output mapping",),
        "semantic_output_mapping_invalid",
        "j.boundary.output_mappings",
        "one ordered semantic value selection for every Source output",
    ),
    (
        ("semantic function", "function body", "operation body", "operation graph"),
        "semantic_operation_graph_invalid",
        "j.intent.intent_function.body",
        "a closed producer-before-consumer semantic operation graph",
    ),
    (
        ("scope", "region"),
        "canonical_scope_derivation_failed",
        "j.intent.intent_function.body",
        "lexically complete loop/branch regions with visible SSA references",
    ),
    (
        ("access", "effect", "transfer ledger"),
        "canonical_transfer_derivation_failed",
        "j.boundary.accesses",
        "one closed logical access row per provable Source occurrence",
    ),
    (
        ("pi iteration", "iteration binding"),
        "pi_iteration_binding_not_source_proven",
        "pi.iteration_bindings",
        "a tile_extent or reduction_tile_extent Source C slot",
    ),
    (
        ("access representation",),
        "pi_access_representation_not_source_proven",
        "pi.access_representation_bindings",
        "a unique Source/J access and carrier binding",
    ),
    (
        ("namespace", "handle", "allocator", "node_ids"),
        "canonical_handle_derivation_failed",
        "j.intent.intent_function.body",
        "unique local semantic labels without p*/c* logical references",
    ),
    (
        ("dimension", "index bounds", "gather", "scatter"),
        "semantic_dimension_or_index_contract_invalid",
        "j.boundary",
        "typed logical dimensions and complete gather/scatter boundary facts",
    ),
    (
        ("Source seed", "SourceFacts", "compiler", "ABI"),
        "canonical_source_construction_unavailable",
        "<canonical-constructor>",
        "the mounted Source/Harness canonical construction",
    ),
)


class _CanonicalConstructionPhaseError(ValueError):
    """Keep constructor phase ownership without exposing implementation text."""

    def __init__(self, rule_code: str, path: str, expected: str) -> None:
        super().__init__(rule_code)
        self.rule_code = rule_code
        self.path = path
        self.expected = expected


def _run_canonical_construction_phase(
    *,
    rule_code: str,
    path: str,
    expected: str,
    action: Callable[[], Any],
) -> Any:
    """Run one host phase and retain only its stable provider-facing class.

    A construction failure is not an additional J authoring contract.  The
    phase label identifies the one semantic responsibility a fresh relift may
    revisit while the causal exception remains host-private.
    """

    try:
        return action()
    except CanonicalIngressError:
        raise
    except _CanonicalConstructionBundleError:
        raise
    except _CanonicalConstructionPhaseError:
        raise
    except Exception as exc:
        raise _CanonicalConstructionPhaseError(
            rule_code, path, expected
        ) from exc


def _canonical_constructor_failure_diagnostic(
    exc: Exception,
) -> tuple[dict[str, Any], ...]:
    """Return one bounded, source-free constructor diagnostic for relift.

    Constructor exceptions are implementation details, not provider context.
    This table converts the stable failure families into one actionable path and
    expected semantic responsibility.  Unknown failures remain fail-closed but
    cannot leak a raw traceback, source spelling, or an accidental second
    authoring contract.
    """

    if isinstance(exc, _CanonicalConstructionPhaseError):
        return ({
            "rule_code": exc.rule_code,
            "path": exc.path,
            "observed_refs_or_types": [f"expected:{exc.expected}"],
        },)
    if isinstance(exc, _CanonicalConstructionBundleError):
        return tuple(exc.diagnostics)
    message = str(exc)
    folded = message.casefold()
    for needles, rule_code, path, expected in _CANONICAL_CONSTRUCTOR_FAILURE_RULES:
        if any(needle.casefold() in folded for needle in needles):
            return ({
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": [f"expected:{expected}"],
            },)
    return ({
        "rule_code": "canonical_constructor_unclassified",
        "path": "<canonical-constructor>",
        "observed_refs_or_types": [
            "expected:closed canonical semantic J surface",
        ],
    },)


def _canonical_construction_transaction(
    payload: dict[str, Any],
    request: SCRCLiftRequest,
    repairs: list[dict[str, Any]],
    *,
    construction: CanonicalConstruction,
) -> None:
    """Run the one deterministic Source/Harness-to-artifact ingress pass.

    The provider supplies semantic J and optional J-proven normalization
    classes. This
    transaction owns only facts that already exist in the anonymous
    Source/Harness view or in the candidate's closed declarations.  Keeping
    the order here explicit is important: C/ABI ownership is fixed before
    logical-reference repair, tensor/node/access handles are canonicalized
    before Source ledger matching, and the final namespace check sees the
    complete ``{j,c,pi}`` sibling layout.  A failed/ambiguous phase leaves the
    candidate for the strict checker; it never fabricates a partial artifact.
    """

    if construction.source_facts.digest != request.source_facts.digest:
        raise _CanonicalConstructionPhaseError(
            "canonical_source_construction_unavailable",
            "<canonical-constructor>",
            "the mounted Source/Harness canonical construction",
        )

    # First keep only the candidate's semantic surface and rebuild the
    # Source/Harness-owned sibling fields.  This must precede allocator
    # extension: otherwise a provider's stale signature/scope spelling can
    # determine the namespace that is supposed to correct it.
    _run_canonical_construction_phase(
        rule_code="semantic_operation_graph_invalid",
        path="j.intent.intent_function.body",
        expected="one closed semantic operation graph with Source-ordered output selections",
        action=lambda: _materialize_source_owned_candidate_surface(
            payload, construction, repairs
        ),
    )
    j_value = payload.get("j")
    intent_value = j_value.get("intent") if isinstance(j_value, Mapping) else None
    function_value = (
        intent_value.get("intent_function")
        if isinstance(intent_value, Mapping)
        else None
    )
    if not isinstance(function_value, Mapping):
        raise _CanonicalConstructionPhaseError(
            "semantic_operation_graph_invalid",
            "j.intent.intent_function.body",
            "a closed producer-before-consumer semantic operation graph",
        )
    # Reserve operation and lexical-scope namespaces from the graph before any
    # J mechanical row is emitted.  No helper below may spell n*/q*/a*/e*/v*
    # by ordinal outside this allocator transaction.
    allocator = _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j.intent.intent_function.body",
        expected="unique local operation labels with one canonical allocator",
        action=lambda: _semantic_graph_allocator(
            construction.allocator, function_value
        ),
    )
    if not isinstance(function_value.get("body"), list):
        raise _CanonicalConstructionPhaseError(
            "semantic_operation_graph_invalid",
            "j.intent.intent_function.body",
            "a list of lexically ordered semantic operations",
        )
    j_value["scopes"] = _run_canonical_construction_phase(
        rule_code="canonical_scope_derivation_failed",
        path="j.intent.intent_function.body",
        expected="lexically complete loop/branch regions with visible SSA references",
        action=lambda: _materialize_scope_headers_from_operation_graph(
            function_value["body"], allocator=allocator
        ),
    )
    allocator = _run_canonical_construction_phase(
        rule_code="canonical_transfer_derivation_failed",
        path="j.boundary.accesses",
        expected="one closed logical access row per provable Source occurrence",
        action=lambda: _materialize_boundary_effects_from_accesses(
            payload, repairs, allocator=allocator
        ),
    )
    flat_allocator = _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j.intent.intent_function.body",
        expected="unique local result labels in producer-before-consumer order",
        action=lambda: _materialize_flat_semantic_ssa_handles(
            payload, repairs, allocator=allocator
        ),
    )
    if flat_allocator is not None:
        allocator = flat_allocator
    structured_allocator = _run_canonical_construction_phase(
        rule_code="canonical_scope_derivation_failed",
        path="j.intent.intent_function.body",
        expected="lexically complete structured operation regions",
        action=lambda: _materialize_structured_node_handles(
            payload, repairs, allocator=allocator
        ),
    )
    if structured_allocator is not None:
        allocator = structured_allocator
    structured_ssa_allocator = _run_canonical_construction_phase(
        rule_code="canonical_scope_derivation_failed",
        path="j.intent.intent_function.body",
        expected="complete loop carry/body/yield and branch SSA wiring",
        action=lambda: _materialize_structured_semantic_ssa_handles(
            payload, repairs, allocator=allocator
        ),
    )
    if structured_ssa_allocator is not None:
        allocator = structured_ssa_allocator

    trusted_view = construction.source_facts.compiler_view()
    # C/p leakage is a semantic ingress error, not a late construction-record
    # failure.  Check before allocating node/SSA/scope projections so an
    # invalid physical capacity can never leave a partially rebuilt J view.
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="no p* or c* references in logical J semantics",
        action=lambda: _validate_canonical_namespace_ownership(payload, trusted_view),
    )

    # Extend the already-reserved graph allocator with any remaining typed J
    # namespaces.  Existing n*/q*/a*/e*/v* declarations must agree exactly;
    # this cannot create a second allocation view.
    # Every repair phase below receives this same object; no phase is allowed
    # to recount or independently allocate a namespace from the raw payload.
    candidate_allocator = _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="one canonical allocator for all authored J handles",
        action=lambda: allocator.for_candidate_j(payload.get("j", {})),
    )
    for prefix in ("n", "q", "a", "e", "v"):
        reserved = allocator.declared(prefix)
        if reserved and candidate_allocator.declared(prefix) != reserved:
            raise _CanonicalConstructionPhaseError(
                "canonical_handle_derivation_failed",
                "j.intent.intent_function.body",
                "one canonical allocation for every semantic handle",
            )
    allocator = candidate_allocator

    # Physical ABI and C are established first.  Later J repairs are not
    # allowed to use a c* slot as a logical placeholder.
    _run_canonical_construction_phase(
        rule_code="canonical_source_construction_unavailable",
        path="c",
        expected="the mounted Source/Harness C and ABI ledger",
        action=lambda: _repair_compiler_c(
            payload, trusted_view, repairs, allocator=allocator
        ),
    )
    _run_canonical_construction_phase(
        rule_code="pi_iteration_binding_not_source_proven",
        path="pi.iteration_bindings",
        expected="a tile_extent or reduction_tile_extent Source C slot",
        action=lambda: _require_pi_iteration_bindings_use_canonical_c_roles(payload),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_source_construction_unavailable",
        path="j.signature",
        expected="the mounted Source/Harness ABI declaration order",
        action=lambda: _repair_canonical_abi_layout(payload, trusted_view, repairs),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_source_construction_unavailable",
        path="j.signature",
        expected="the mounted Source/Harness constexpr parameter declarations",
        action=lambda: _repair_bound_constexpr_parameter_kinds(
            payload, trusted_view, repairs
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_source_construction_unavailable",
        path="j.signature",
        expected="the mounted Source/Harness scalar parameter ranks",
        action=lambda: _repair_bound_constexpr_parameter_rank(
            payload, trusted_view, repairs
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_source_construction_unavailable",
        path="j.signature",
        expected="Source-proven logical scalar bindings",
        action=lambda: _repair_source_shape_scalar_bindings(
            payload,
            trusted_view,
            repairs,
            allocator=allocator,
        ),
    )

    # The J declaration graph and all of its references are then canonicalized
    # together.  These helpers do not infer missing operations or dataflow.
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j.signature",
        expected="canonical t* declarations in Source ABI order",
        action=lambda: _repair_logical_tensor_handles(
            payload,
            repairs,
            allocator=allocator,
        ),
    )
    # Output selections remain semantic until the shared record-link transaction
    # has one closed operation/scope bundle.  That transaction is the sole
    # writer of output-mapping sibling links and ABI-order returns.
    _run_canonical_construction_phase(
        rule_code="semantic_dimension_or_index_contract_invalid",
        path="j.boundary.domains",
        expected="typed logical domain predicates",
        action=lambda: _materialize_rectangular_domain_predicates(payload, repairs),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_transfer_derivation_failed",
        path="j.boundary.effects",
        expected="Source-ordered boundary effect scope membership",
        action=lambda: _materialize_flat_root_effect_membership(payload, repairs),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_transfer_derivation_failed",
        path="j.boundary.effects",
        expected="one Source-ledger owner for every boundary effect",
        action=lambda: _materialize_root_effect_scope_from_source_ledger(
            payload,
            construction,
            repairs,
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="one synchronized J handle, scope, and declaration layout",
        action=lambda: _repair_canonical_authoring_layout(
            payload,
            trusted_view,
            repairs,
            allocator=allocator,
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="logical t*/v* references instead of physical p* carriers",
        action=lambda: _repair_logical_parameter_references(
            payload,
            repairs,
            allocator=allocator,
            source_facts=construction.source_facts,
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="one canonical non-tensor handle namespace",
        action=lambda: _repair_canonical_non_tensor_handles(
            payload,
            repairs,
            allocator=allocator,
        ),
    )
    _run_canonical_construction_phase(
        rule_code="pi_access_representation_not_source_proven",
        path="pi",
        expected="a J-proven normalization class with its fixed scope",
        action=lambda: _repair_pi_normalization_scopes(payload, repairs),
    )
    j_value = payload.get("j")
    intent_value = j_value.get("intent") if isinstance(j_value, Mapping) else None
    function_value = (
        intent_value.get("intent_function")
        if isinstance(intent_value, Mapping)
        else None
    )
    post_handle_records = _run_canonical_construction_phase(
        rule_code="semantic_operation_graph_invalid",
        path="j.intent.intent_function.body",
        expected="a closed operation record for every authored operation",
        action=lambda: _canonical_operation_records(function_value),
    )
    _run_canonical_construction_phase(
        rule_code="semantic_dimension_or_index_contract_invalid",
        path="j.boundary.index_bounds",
        expected="typed index bounds for every semantic gather/scatter",
        action=lambda: _repair_canonical_index_bounds(
            payload, post_handle_records, repairs
        ),
    )
    # Re-read the post-handle-repair function once and derive operation,
    # output, return, and flat-scope links from that single table.  This call
    # is deliberately outside the conservative layout helper so an ambiguous
    # node/access subphase cannot strand sibling ABI fields on a stale view.
    _run_canonical_construction_phase(
        rule_code="semantic_dimension_or_index_contract_invalid",
        path="j",
        expected="closed typed dimension and result relations",
        action=lambda: (
            _repair_dimension_constraint_syntax(payload, repairs),
            _repair_redundant_dimension_relations(payload, repairs),
            _repair_result_relation_fields(payload, repairs),
            _derive_output_relation_kinds(payload, repairs),
        ),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_handle_derivation_failed",
        path="j",
        expected="no p* or c* references in logical J semantics",
        action=lambda: _validate_canonical_namespace_ownership(payload, trusted_view),
    )
    _run_canonical_construction_phase(
        rule_code="canonical_construction_bundle_incomplete",
        path="j",
        expected="one typed operation, output, scope, and boundary construction bundle",
        action=lambda: _finalize_canonical_construction_bundle(payload, repairs),
    )


def _construct_source_seeded_candidate(
    original: dict[str, Any],
    request: SCRCLiftRequest,
    *,
    construction: CanonicalConstruction,
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate provider claims, then project the trusted seed and handles.

    The synchronizer supplies a private copy and its observed construction.
    Submission checks still precede all writes. This intermediate mapping is
    not accepted: the Source-transfer transaction and strict decoder remain
    mandatory. This function neither invents nor signs semantic receipts.
    """

    try:
        provider_namespace_diagnostics = _provider_semantic_namespace_diagnostics(
            original
        )
        if provider_namespace_diagnostics:
            raise CanonicalIngressError(
                "provider semantic candidate uses host-owned canonical handles",
                candidate_check_diagnostics=provider_namespace_diagnostics,
            )
        submission_diagnostics = _semantic_construction_submission_diagnostics(
            original,
            construction,
        )
        if submission_diagnostics:
            raise CanonicalIngressError(
                "semantic construction submission is incomplete",
                candidate_check_diagnostics=submission_diagnostics,
            )
        _rewrite_provider_dimension_aliases(original, construction)
        # Stage one accepts only the bounded semantic surface, then projects it
        # onto the sole Source/Harness seed.  Stage two still requires the
        # exact canonical root before any construction repair or CandidateCheck
        # can observe it.
        working = _run_canonical_construction_phase(
            rule_code="semantic_operation_graph_invalid",
            path="j",
            expected="the bounded semantic J and pi submission surface",
            action=lambda: _seed_semantic_candidate_envelope(
                original, construction
            ),
        )
        # Semantic submission has already passed exact owner verification.
        # Carry its existing private Source receipt only in the host artifact
        # so independent decoders/Boundary/fallback keep the same fail-closed
        # contract. No receipt is synthesized from provider text or a hash.
        for access in working["j"]["boundary"].get("accesses", ()):
            if isinstance(access, dict) and access.get("node_ref") is not None:
                access["semantic_owner_authorization"] = _private_source_owner_token(access, construction)
        _run_canonical_construction_phase(
            rule_code="canonical_source_construction_unavailable",
            path="<canonical-constructor>",
            expected="the Source-owned canonical artifact envelope",
            action=lambda: require_canonical_artifact_envelope(working),
        )
        _canonical_construction_transaction(
            working,
            request,
            repairs,
            construction=construction,
        )
        # This only proves that the private repaired object can be serialized;
        # CandidateCheck remains the sole wire and semantic authority.
        _canonical_candidate_text(working)
    except CanonicalIngressError:
        raise
    except Exception as exc:
        raise CanonicalIngressError(
            "semantic candidate cannot form the canonical Source-owned artifact",
            candidate_check_diagnostics=_canonical_constructor_failure_diagnostic(exc),
        ) from exc
    return working


def _close_candidate_source_transfers(
    working: dict[str, Any],
    *,
    construction: CanonicalConstruction,
    observation: Mapping[str, Any] | None,
    repairs: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Close existing Source facts against the private candidate transaction.

    Order/reallocation/rebinding, fills, domains, pi and guard projection all
    retain their existing checks. Return the final copy only after every phase
    succeeds; never recover from failure with the seed or a previous candidate.
    """

    # Source-ledger edits are part of canonical construction, rather than an
    # optional TTIR-only postscript.  The observer can refine the immutable
    # SourceFacts ledger (notably its lexical fill witness), but it must not
    # decide whether J access/effect order and host-derived pi ownership are
    # constructed.  Otherwise Responses, MCP, and non-TTIR frontends expose
    # the same provider surface yet produce different pi ownership layouts.
    # A Source preparation failure is therefore reported at ingress; it is
    # never converted into an invitation for the provider to author pi rows.
    fingerprint: Any = None
    source_runtime_parameter_uses: object = None
    if isinstance(observation, Mapping) and observation.get("status") == "pass":
        ttir = observation.get("ttir")
        if isinstance(ttir, str) and ttir:
            from intentir.verify.gate1.ttir import fingerprint_ttir

            python_interface = observation.get("python_interface")
            fingerprint = fingerprint_ttir(
                ttir,
                external_math=observation.get("_external_math"),
                python_interface=(
                    python_interface if isinstance(python_interface, Mapping) else None
                ),
            )
            observed_facts = observation.get("source_facts")
            if isinstance(observed_facts, Mapping):
                observed_source_facts = SourceFacts.from_json(observed_facts)
                if observed_source_facts.digest != construction.source_facts.digest:
                    raise CanonicalIngressError(
                        "Source observation facts differ from Lift request facts",
                        candidate_check_diagnostics=(
                            {
                                "rule_code": "canonical_source_construction_unavailable",
                                "path": "<source-transfer-ledger>",
                                "observed_refs_or_types": [
                                    "expected:Source observation lineage matching the mounted SourceFacts"
                                ],
                            },
                        ),
                    )
            source_runtime_parameter_uses = observation.get(
                "source_runtime_parameter_uses"
            )
            # ``construction`` was already joined with this exact observation
            # before semantic preflight.  Keep the fingerprint for the
            # candidate-specific Source/J reconciliation below, but do not
            # create a second occurrence-ledger transaction here.

    if construction.transfer_ledger.status != "pass":
        raise CanonicalIngressError(
            "Source transfer ledger is unavailable for canonical construction",
            candidate_check_diagnostics=_closed_projection_diagnostics(
                construction.transfer_ledger,
                fallback_rule_code="canonical_source_construction_unavailable",
                fallback_path="<source-transfer-ledger>",
                fallback_observation=(
                    "reason=source_transfer_occurrence_ledger_unavailable"
                ),
            ),
        )
    # A memory-free Source has no access/effect or carrier ownership to
    # materialize.  Its already complete seed remains the one authority;
    # forcing a synthetic empty occurrence binding would be another wire.
    if not construction.transfer_ledger.occurrences:
        return working, tuple(repairs)

    # This transaction is all-or-nothing.  An ambiguous occurrence match must
    # not leave only access/effect, fill, or pi synchronization visible to
    # CandidateCheck.
    source_working = copy.deepcopy(working)
    source_repairs: list[dict[str, Any]] = []
    try:
        from .compiler_observation import (
            bind_source_pi,
            materialize_source_program_domains,
            validate_source_access_fills_from_source_ledger,
            reorder_boundary_effects_from_source_ledger,
        )

        reorder = _run_canonical_construction_phase(
            rule_code="source_transfer_reorder_execution_failed",
            path="j.boundary.accesses+j.boundary.effects",
            expected="one closed occurrence/access/effect reorder transaction",
            action=lambda: reorder_boundary_effects_from_source_ledger(
                source_working,
                fingerprint,
                ledger=construction.transfer_ledger,
                source_facts=construction.source_facts,
                # `_semantic_construction_submission_diagnostics` has already
                # checked that the provider's only access identities are the
                # complete Source-order access0..accessN ordinal sequence. The
                # reconciler therefore consumes this sealed provenance rather
                # than re-identifying it from physical address term counts.
                source_order_sealed=True,
            ),
        )
        # Only the complete canonical ledger may flow to fill validation and the
        # Source->pi binder. ``source_access_matches`` is an ingress-local
        # projection returned for diagnostics; passing the whole reorder result
        # here would make that projection an accidental second authority.
        reorder_passed = (
            isinstance(reorder, Mapping) and reorder.get("status") == "pass"
        )
        reordered_ledger = (
            reorder.get("source_transfer_ledger")
            if reorder_passed
            else None
        )
        # The construction ledger is the input authority, not a fallback
        # after the candidate-specific reorder transaction.  If reorder is
        # ambiguous/unavailable, using the original ledger here would let
        # fill/π consumers bypass the failed effect-order proof.
        if not isinstance(reordered_ledger, Mapping):
            raise CanonicalIngressError(
                "candidate boundary cannot close against the Source transfer ledger",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    reorder,
                    fallback_rule_code="source_transfer_ledger_not_closed",
                    fallback_path="j.boundary.accesses",
                    fallback_observation=(
                        "reason=occurrence_access_effect_binding_unavailable"
                    ),
                ),
            )
        source_ledger_projection: SourceTransferLedger | Mapping[str, Any] = (
            reordered_ledger
        )
        reordered = reorder.get("artifact")
        if isinstance(reordered, Mapping) and reorder.get("applied"):
            source_working = copy.deepcopy(dict(reordered))
            source_repairs.extend(
                dict(item)
                for item in reorder["applied"]
                if isinstance(item, Mapping)
            )

        # The ledger has now proved one occurrence-to-access order.  Rebuild
        # both access and effect identities from that order through the same
        # candidate allocator used at ingress, then repeat the closed matcher
        # so fill/π see bindings for the post-allocation a*/e* names rather
        # than the pre-reorder provider spelling.
        source_j = source_working.get("j")
        if not isinstance(source_j, Mapping):
            raise CanonicalIngressError(
                "Source-ledger reorder lost the canonical J artifact",
                candidate_check_diagnostics=(
                    {
                        "rule_code": "source_transfer_rebuild_invalid",
                        "path": "j",
                        "observed_refs_or_types": ["expected:canonical J object"],
                    },
                ),
            )
        source_allocator = construction.allocator.for_candidate_j(source_j)
        # Reorder has returned a completed host artifact.  Convert its access
        # identities to a private transient semantic surface before the one
        # allocator rebuilds dense a*/e* names in Source occurrence order.
        # Provider ingress never sees this conversion.
        _run_canonical_construction_phase(
            rule_code="source_transfer_reallocation_failed",
            path="j.boundary.accesses+j.boundary.effects",
            expected="one closed canonical a*/e* allocation from the sealed ledger",
            action=lambda: (
                _prepare_canonical_access_reallocation(source_working),
                _materialize_boundary_effects_from_accesses(
                    source_working,
                    source_repairs,
                    allocator=source_allocator,
                ),
            ),
        )
        rebound = _run_canonical_construction_phase(
            rule_code="source_transfer_rebinding_execution_failed",
            path="j.boundary.accesses+j.boundary.effects",
            expected="one closed post-allocation occurrence/access/effect binding",
            action=lambda: reorder_boundary_effects_from_source_ledger(
                source_working,
                fingerprint,
                ledger=construction.transfer_ledger,
                source_facts=construction.source_facts,
                source_order_sealed=True,
            ),
        )
        if not isinstance(rebound, Mapping) or rebound.get("status") != "pass":
            raise CanonicalIngressError(
                "canonical access/effect allocation no longer closes Source order",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    rebound,
                    fallback_rule_code="source_transfer_rebinding_failed",
                    fallback_path="j.boundary.accesses",
                    fallback_observation=(
                        "reason=post_allocation_occurrence_binding_unavailable"
                    ),
                ),
            )
        rebound_ledger = rebound.get("source_transfer_ledger")
        if not isinstance(rebound_ledger, Mapping):
            raise CanonicalIngressError(
                "canonical access/effect allocation has no rebound Source ledger",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    rebound,
                    fallback_rule_code="source_transfer_rebinding_missing",
                    fallback_path="j.boundary.effects",
                    fallback_observation="reason=rebound_transfer_ledger_missing",
                ),
            )
        rebound_artifact = rebound.get("artifact")
        if isinstance(rebound_artifact, Mapping) and rebound.get("applied"):
            source_working = copy.deepcopy(dict(rebound_artifact))
            source_repairs.extend(
                dict(item)
                for item in rebound["applied"]
                if isinstance(item, Mapping)
            )
        source_ledger_projection = rebound_ledger

        fill_validation = _run_canonical_construction_phase(
            rule_code="source_transfer_fill_validation_execution_failed",
            path="j.boundary.accesses",
            expected="one closed occurrence-level masked-load fill validation",
            action=lambda: validate_source_access_fills_from_source_ledger(
                source_working,
                fingerprint,
                ledger=source_ledger_projection,
                source_facts=construction.source_facts,
            ),
        )
        if (
            not isinstance(fill_validation, Mapping)
            or fill_validation.get("status") != "pass"
        ):
            # A closed effect reorder is insufficient if occurrence-level
            # masked-load fill proof cannot close against the same rebound
            # ledger.  Continuing here used to leave a partly synchronized
            # J/C/pi candidate for a later checker to reinterpret.
            raise CanonicalIngressError(
                "canonical access/effect allocation cannot close Source fill facts",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    fill_validation,
                    fallback_rule_code="source_transfer_fill_binding_failed",
                    fallback_path="j.boundary.accesses",
                    fallback_observation=(
                        "reason=occurrence_fill_binding_unavailable"
                    ),
                ),
            )
        # Fresh MCP semantic entry intentionally has no provider write for a
        # host-owned program mapping. When the mounted compiler grid, Source
        # ABI shapes, and this exact rebound transfer ledger uniquely close a
        # launch coordinate, project that J domain before Source->pi carrier
        # binding. Ambiguity remains untouched for CandidateCheck to reject.
        program_domain_projection = _run_canonical_construction_phase(
            rule_code="source_program_domain_projection_execution_failed",
            path="j.boundary.domains",
            expected="one closed Source-proven program-domain projection",
            action=lambda: materialize_source_program_domains(
                source_working,
                source_facts=construction.source_facts,
                ledger=source_ledger_projection,
            ),
        )
        if (
            not isinstance(program_domain_projection, Mapping)
            or program_domain_projection.get("status") != "pass"
        ):
            raise CanonicalIngressError(
                "canonical access/effect allocation cannot project Source program domain",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    program_domain_projection,
                    fallback_rule_code="source_program_domain_projection_unavailable",
                    fallback_path="j.boundary.domains",
                    fallback_observation=(
                        "reason=projection_receipt_missing_or_malformed"
                    ),
                ),
            )
        projected = program_domain_projection.get("artifact")
        if isinstance(projected, Mapping) and program_domain_projection.get("applied"):
            source_working = copy.deepcopy(dict(projected))
            source_repairs.extend(
                dict(item)
                for item in program_domain_projection["applied"]
                if isinstance(item, Mapping)
            )

        binding = _run_canonical_construction_phase(
            rule_code="source_transfer_pi_binding_execution_failed",
            path="pi",
            expected="one closed Source address-carrier binding transaction",
            action=lambda: bind_source_pi(
                source_working,
                fingerprint,
                source_runtime_parameter_uses=source_runtime_parameter_uses,
                ledger=source_ledger_projection,
                source_facts=construction.source_facts,
            ),
        )
        if not isinstance(binding, Mapping) or binding.get("status") != "pass":
            # π carrier/address ownership is a consumer of the exact same
            # rebound occurrence ledger.  Do not allow a successful effect
            # reorder to mask a failed ownership proof.
            raise CanonicalIngressError(
                "canonical access/effect allocation cannot close Source pi ownership",
                candidate_check_diagnostics=_closed_projection_diagnostics(
                    binding,
                    fallback_rule_code="source_transfer_pi_binding_failed",
                    fallback_path="pi",
                    fallback_observation=(
                        "reason=tensor_address_carrier_binding_unavailable"
                    ),
                ),
            )
        bound = binding.get("artifact")
        if isinstance(bound, Mapping) and binding.get("applied"):
            source_working = copy.deepcopy(dict(bound))
            source_repairs.extend(
                dict(item)
                for item in binding["applied"]
                if isinstance(item, Mapping)
            )
        # A provider-local i1 predicate is submitted before the host has a
        # canonical a* access identity.  At this point the Source-order
        # reallocation and π ownership transaction have both closed, so the
        # host can perform the one mechanical predicate->guard projection
        # without asking the provider to author a host-owned a* relation.
        _run_canonical_construction_phase(
            rule_code="source_transfer_guard_projection_execution_failed",
            path="j.boundary.access_guard_relations",
            expected="a closed authorized computed-guard projection",
            action=lambda: _materialize_computed_access_predicate_guards(
                source_working,
                source_repairs,
            ),
        )
        # The transfer ledger can reorder boundary effects and attach a
        # uniquely proven fill/pi owner.  Scope membership, output links and
        # the shared construction bundle must be rebuilt from that same final
        # boundary snapshot; otherwise Source-order repair and lexical scope
        # consumers observe two different artifact states.
        _run_canonical_construction_phase(
            rule_code="source_transfer_final_bundle_execution_failed",
            path="j.boundary+j.intent+j.node_ids+j.scopes",
            expected="one final canonical bundle sharing the rebound ledger",
            action=lambda: _finalize_canonical_construction_bundle(
                source_working,
                source_repairs,
            ),
        )
        _canonical_candidate_text(source_working)
    except CanonicalIngressError:
        raise
    except _CanonicalConstructionBundleError as exc:
        raise CanonicalIngressError(
            "Source transfer construction cannot form a complete canonical bundle",
            candidate_check_diagnostics=_canonical_constructor_failure_diagnostic(exc),
        ) from exc
    except _CanonicalConstructionPhaseError as exc:
        raise CanonicalIngressError(
            "Source transfer construction phase failed",
            candidate_check_diagnostics=_canonical_constructor_failure_diagnostic(exc),
        ) from exc
    except Exception as exc:
        raise CanonicalIngressError(
            "Source transfer construction failed after semantic ingress",
            candidate_check_diagnostics=(
                {
                    "rule_code": "source_transfer_constructor_failed",
                    "path": "j.boundary",
                    "observed_refs_or_types": [
                        "expected:closed Source transfer ledger and canonical boundary"
                    ],
                },
            ),
        ) from exc

    return source_working, tuple([*repairs, *source_repairs])


def _synchronize_scrc_candidate(
    payload: Mapping[str, Any],
    request: SCRCLiftRequest,
    *,
    observation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    """Build one canonical ingress candidate transactionally.

    The provider authors semantic J and any provider-owned pi choices; MCP may
    omit every host-owned pi row.  This synchronizer applies only closed
    wire/ABI/Source facts already present in the candidate or trusted observer,
    including exact Source-proven program/tile projection.  All edits happen on
    private copies; callers receive a fully synchronized copy or an explicit
    ingress rejection.  It must never fall back to the untouched provider
    artifact after the constructor has failed, because that would create a
    second admission path around the one canonical construction authority.
    Every backend crosses this same construction boundary; an incomplete
    candidate is rejected rather than merged with prior candidate state.
    """

    original = copy.deepcopy(dict(payload))
    repairs: list[dict[str, Any]] = []
    # Authoring requirements are exposed from the observed Source occurrence
    # ledger before the provider writes accesses.  Apply that same authority
    # before semantic submission preflight and canonical construction; using
    # the pre-observation request here makes a complete MCP declaration appear
    # to have the wrong arity at finalize.
    observed_construction = _construction_with_source_observation(
        request,
        observation,
    )
    construction = (
        request.construction
        if observed_construction.authority_digest
        == request.construction.authority_digest
        else observed_construction
    )
    working = _construct_source_seeded_candidate(
        original,
        request,
        construction=construction,
        repairs=repairs,
    )
    return _close_candidate_source_transfers(
        working,
        construction=construction,
        observation=observation,
        repairs=repairs,
    )


def _construct_scrc_candidate(
    response_text: str,
    request: SCRCLiftRequest,
    *,
    observation: Mapping[str, Any] | None = None,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Construct one canonical Source-aware artifact before strict Decode.

    This private ingress transaction projects every semantic J surface onto
    the Source/Harness seed, allocates its mechanical canonical rows, and
    synchronizes the closed transfer ledger. It never creates missing
    provider-owned operation, loop, predicate, output-value, or normalization
    semantics, and it never changes the public decoder's fail-closed contract.
    Its deterministic Source projections are limited to exact host-owned
    program/tile rows. Masked-load fills remain provider-authored and are only
    validated against the unique Source transfer occurrence.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return response_text, ()
    synchronized, repairs = _synchronize_scrc_candidate(
        payload,
        request,
        observation=observation,
    )
    # This ingress function historically exposed its result as a "repair" and
    # returned the provider bytes unchanged when no receipt was emitted.  That
    # made a successful semantic submission bypass the very constructor which
    # had already built its complete canonical J/C/π envelope.  Construction
    # is not conditional on a repair receipt: every accepted semantic surface
    # must cross this exact transaction before CandidateCheck.
    return _canonical_candidate_text(synchronized), tuple(repairs)


def _candidate_delta_from_failure_receipt(
    receipt: _SCRCLiftFailureReceipt,
) -> dict[str, Any] | None:
    """Parse rejected bytes locally, return only their root-shape receipt."""

    if not receipt.response_text:
        return None
    try:
        payload = _json_object(receipt.response_text)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return None
    if len(encoded) > MAX_RELIFT_CANDIDATE_BYTES:
        return None
    return scrc_candidate_delta_receipt(payload)


def _source_ttir_observation_for_request(
    request: SCRCLiftRequest,
) -> Mapping[str, Any]:
    """Compile the exact host-only Source once for Source -> pi binding."""

    if request.source_facts.frontend.casefold() != "triton":
        return {
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    from .compiler_observation import observe_source_ttir

    return observe_source_ttir(
        request.source_text,
        request.source_compiler_view,
        source_function=request.harness_binding.source_function,
        import_roots=request.source_import_roots,
        source_facts=request.source_facts,
    )


def _construction_with_source_observation(
    request: SCRCLiftRequest,
    observation: Mapping[str, Any] | None,
) -> CanonicalConstruction:
    """Attach one already-observed Source ledger to the request authority.

    The Source occurrence ledger must be available to authoring before a
    provider chooses J accesses, effects, fills, owners, or pi rows.  It is
    still intentionally unbound at this point: occurrence -> J binding can
    only be proved against a candidate's authored boundary.  This helper never
    constructs that binding and never treats an unavailable observer as an
    empty ledger.
    """

    if not isinstance(observation, Mapping) or observation.get("status") != "pass":
        return request.construction
    ttir = observation.get("ttir")
    if not isinstance(ttir, str) or not ttir:
        return request.construction
    observed_facts = observation.get("source_facts")
    if isinstance(observed_facts, Mapping):
        facts = SourceFacts.from_json(observed_facts)
        if facts.digest != request.source_facts.digest:
            raise ValueError(
                "Source observation facts differ from the canonical Lift request"
            )
    from intentir.verify.gate1.ttir import fingerprint_ttir

    python_interface = observation.get("python_interface")
    return request.construction.with_observation(
        fingerprint_ttir(
            ttir,
            external_math=observation.get("_external_math"),
            python_interface=(
                python_interface if isinstance(python_interface, Mapping) else None
            ),
        )
    )


def _source_authoring_blockers(
    construction: CanonicalConstruction,
) -> tuple[dict[str, str], ...]:
    """Return closed Source-side blockers before any provider authoring turn.

    A masked boundary access has exactly two legal predicate representations:
    a fully closed logical ContractExpr supplied by the Source transfer wire,
    or one provider-local i1 consumed through an exact ledger-issued
    computed-guard receipt.  The host must not ask a provider to invent a
    third representation and later reject it in FunctionalBoundary.  This is
    an admission preflight only: it neither issues a receipt nor changes J,
    C, pi, a candidate, or the ledger.
    """

    if not isinstance(construction, CanonicalConstruction):
        raise TypeError("Source authoring blockers require canonical construction")
    if construction.transfer_ledger.status != "pass":
        return ()
    try:
        source_wire = canonical_source_wire_bundle(construction)
        transfer = source_wire.get("semantic_transfer_projection")
        rows = transfer.get("occurrences") if isinstance(transfer, Mapping) else None
    except Exception:
        return (
            {
                "rule_code": "source_authoring_contract_unavailable",
                "path": "semantic_transfer_projection",
            },
        )
    if not isinstance(rows, list):
        return (
            {
                "rule_code": "source_authoring_contract_unavailable",
                "path": "semantic_transfer_projection.occurrences",
            },
        )

    # Do not spend an authoring turn when the host cannot close the launch
    # domain that the provider would have to consume.  This is deliberately a
    # preflight over the Source-owned wire seed: it does not inspect a prior
    # candidate and it does not manufacture a J domain.  The detailed
    # projector reason stays private; the lifecycle receives one stable
    # Source-preparation class.
    if any(
        occurrence.kind in {"load", "store"}
        and any(
            re.fullmatch(r"program_axis:[xyz]", atom)
            for atoms, _coefficient in occurrence.address_terms
            for atom in atoms
        )
        for occurrence in construction.transfer_ledger.occurrences
    ):
        from .compiler_observation import materialize_source_program_domains

        seed = source_wire.get("artifact_seed")
        if not isinstance(seed, Mapping):
            return (
                {
                    "rule_code": "source_program_domain_projection_unavailable",
                    "path": "pi.program_bindings",
                },
            )
        projection = materialize_source_program_domains(
            seed,
            source_facts=construction.source_facts,
            ledger=construction.transfer_ledger,
        )
        if not isinstance(projection, Mapping) or projection.get("status") != "pass":
            return (
                {
                    "rule_code": "source_program_domain_projection_unavailable",
                    "path": "pi.program_bindings",
                },
            )

    bindings = {
        row["ordinal"]: row
        for row in construction.transfer_ledger.bindings
    }
    occurrences = {
        item.ordinal: item for item in construction.transfer_ledger.occurrences
    }
    blockers: list[dict[str, str]] = []

    def private_public_authorization_matches(
        private: object,
        public: object,
        *,
        owner: bool = False,
    ) -> bool:
        """Join one host receipt with its provider-facing projection.

        ``semantic_transfer_projection`` is intentionally sanitized.  In
        particular, it omits ``status`` and ``compiler_attestation`` and it
        never carries a Source owner token.  The old preflight read the
        sanitized row and consequently treated a valid computed-guard receipt
        as missing.  Keep the join host-local: validate the public contract
        against the authoritative ledger receipt without putting private
        fields back on the provider wire.
        """

        if not isinstance(private, Mapping):
            return public is None
        private_status = private.get("status")
        if private_status == "unavailable":
            return public is None and set(private) == {"status", "reason"}
        if private_status != "pass":
            return False
        token = private.get("token")
        family = private.get("operation_family")
        if (
            not isinstance(token, str)
            or not isinstance(family, str)
            or not isinstance(public, Mapping)
        ):
            return False
        if owner:
            # Owner tokens are host-private.  The public row may carry the
            # signed authoring outline, but never the token itself.
            expected_keys = {"operation_family"}
            outline = private.get("authoring_outline")
            if outline is not None:
                expected_keys.add("authoring_outline")
            if set(public) != expected_keys:
                return False
            if public.get("operation_family") != family or "token" in public:
                return False
            return outline is None or public.get("authoring_outline") == outline
        if set(public) != {"token", "operation_family"}:
            return False
        return (
            public.get("token") == token
            and public.get("operation_family") == family
        )

    def add_projection_blocker(index: int, reason: str) -> None:
        if not re.fullmatch(r"source_[a-z0-9_]+", reason):
            reason = "source_private_public_projection_mismatch"
        blockers.append(
            {
                "rule_code": reason,
                "path": (
                    "semantic_transfer_projection.occurrences["
                    f"{index}].semantic_authorizations"
                ),
            }
        )

    # The public occurrence array is a projection, not an authority.  Join it
    # to the private ledger by ordinal before checking any authoring blocker.
    # This catches stale/reordered/tampered public rows while allowing the
    # private receipt to remain private.
    if len(rows) != len(construction.transfer_ledger.occurrences):
        return (
            {
                "rule_code": "source_private_public_projection_mismatch",
                "path": "semantic_transfer_projection.occurrences",
            },
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            add_projection_blocker(index, "source_private_public_projection_mismatch")
            continue
        ordinal = row.get("ordinal")
        occurrence = occurrences.get(ordinal)
        private_binding = bindings.get(ordinal)
        if (
            not is_canonical_ordinal(ordinal)
            or ordinal != index
            or occurrence is None
            or not isinstance(private_binding, Mapping)
            or row.get("access_label") != f"access{ordinal}"
            or row.get("kind") != occurrence.kind
            or row.get("tensor") != occurrence.logical_ref
            or row.get("masked") is not occurrence.masked
        ):
            add_projection_blocker(index, "source_private_public_projection_mismatch")
            continue
        private_authorizations = private_binding.get("semantic_authorizations")
        authorizations = row.get("semantic_authorizations")
        if not isinstance(private_authorizations, Mapping) or not isinstance(
            authorizations, Mapping
        ):
            add_projection_blocker(index, "source_private_public_projection_mismatch")
            continue
        projection_mismatch = False
        for authorization_name, owner_authorization in (
            ("access_view", False),
            ("owner", True),
            ("computed_guard", False),
        ):
            if not private_public_authorization_matches(
                private_authorizations.get(authorization_name),
                authorizations.get(authorization_name),
                owner=owner_authorization,
            ):
                add_projection_blocker(index, "source_private_public_projection_mismatch")
                projection_mismatch = True
                break
        if projection_mismatch:
            continue
        # Every physical write is a final semantic obligation, not only a
        # masked or multiply-used store.  A simple unmasked store can still be
        # attached to the wrong J producer or output, so allowing it without
        # the exact Source/TTIR owner receipt would leave a gap in the
        # Source->J authorization chain.  Updates are included here and stay
        # fail-closed until their typed logical-update proof exists.
        needs_owner = bool(
            occurrence is not None
            and occurrence.kind in {"store", "update"}
        )
        owner = private_authorizations.get("owner")
        if needs_owner and not (
            isinstance(owner, Mapping) and owner.get("status") == "pass" and isinstance(owner.get("token"), str)
            and re.fullmatch(SOURCE_OWNER_TOKEN_PATTERN, owner["token"]) is not None
        ):
            private_owner = owner if isinstance(owner, Mapping) else {}
            reason = private_owner.get("reason", "source_exact_owner_binding_unavailable")
            if not isinstance(reason, str) or re.fullmatch(r"source_[a-z0-9_]+", reason) is None:
                reason = "source_exact_owner_binding_unavailable"
            blockers.append({
                "rule_code": reason,
                "path": f"semantic_transfer_projection.occurrences[{index}].semantic_authorizations.owner",
            })
        if row.get("masked") is not True or row.get("predicate") is not None:
            continue
        # The public guard projection intentionally has no status or
        # compiler_attestation.  Use the private ledger receipt for the
        # authorization decision; the public/private join above already
        # verified that any exposed token/family is the matching one.
        guard = private_authorizations.get("computed_guard")
        # A mapping-shaped ``unavailable`` receipt is not an authorization.
        # The previous check treated any Mapping as sufficient, allowing a
        # masked access with no Source predicate to reach the provider even
        # when the compiler-side i1 projection had failed.  Require the same
        # closed receipt shape that the ledger and semantic validator consume;
        # do not synthesize a guard or ask the provider to invent one.
        guard_valid = (
            isinstance(guard, Mapping)
            and guard.get("status") == "pass"
            and isinstance(guard.get("token"), str)
            and re.fullmatch(r"so_[0-9a-f]{32}", guard["token"]) is not None
            and isinstance(guard.get("operation_family"), str)
            and isinstance(guard.get("compiler_attestation"), str)
            and re.fullmatch(r"sc_[0-9a-f]{32}", guard["compiler_attestation"]) is not None
        )
        if not guard_valid:
            private_guard = guard
            reason = private_guard.get("reason") if isinstance(private_guard, Mapping) else None
            if reason not in {
                "source_predicate_access_anchor_unavailable",
                "source_compiler_predicate_access_projection_unavailable",
            }:
                reason = "source_computed_guard_authorization_unavailable"
            blockers.append(
                {
                    "rule_code": reason,
                    "path": f"semantic_transfer_projection.occurrences[{index}].predicate",
                }
            )
    return tuple(blockers)


def _reconcile_source_candidate(
    response_text: str,
    request: SCRCLiftRequest,
    *,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the shared Source-aware necessary-condition preflight.

    Both Responses and MCP evaluate the same Source-observer facts after
    strict decode. The result is reject-only diagnostic evidence; it never
    edits or completes J, C, or pi and never certifies a candidate.
    """

    try:
        payload = _json_object(request.user_input)
    except Exception:
        return {
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    frontend = str(payload.get("frontend", "triton")).casefold()
    if frontend != "triton":
        return {
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    anonymous_source = payload.get("anonymous_source")
    anonymous_view = request.source_facts.compiler_view()
    if not isinstance(anonymous_source, str) or not isinstance(
        anonymous_view, Mapping
    ):
        return {
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    if observation is None:
        observation = _source_ttir_observation_for_request(request)
    from .compiler_observation import reconcile_candidate_source_ttir

    return reconcile_candidate_source_ttir(
        response_text,
        observation,
        construction=request.construction,
    )


def decode_scrc_lift_artifact(
    response_text: str,
    request: SCRCLiftRequest,
) -> Gate1Artifact:
    """Apply the one canonical J/C/pi decoder to any Lift backend output."""

    payload = _json_object(response_text)
    candidate_j = payload.get("j")
    allocator = request.construction.allocator
    if isinstance(candidate_j, Mapping):
        allocator = allocator.for_candidate_j(candidate_j)
    expected_parameters = allocator.declared("p")
    artifact = _decode_scrc_lift_artifact_core(
        response_text,
        expected_parameters=expected_parameters,
        expected_tensors=allocator.declared("t"),
        expected_dimensions=allocator.declared("x"),
        trusted_compiler_facts=request.construction.source_facts.compiler_view(),
        required_schema=Gate1Artifact.SCHEMA,
        allocator=allocator,
    )
    _validate_source_identifier_privacy(artifact, request.harness_binding)
    return artifact


def _decode_scrc_lift_artifact_core(
    response_text: str,
    *,
    expected_parameters: tuple[str, ...],
    expected_tensors: tuple[str, ...],
    expected_dimensions: tuple[str, ...],
    trusted_compiler_facts: Mapping[str, Any] | None,
    required_schema: str,
    allocator: CanonicalHandleAllocator,
) -> Gate1Artifact:
    """Parse raw text and invoke the shared strict construction boundary.

    Parsing alone confers no candidate acceptance or Source authorization.
    Runner and CandidateCheck both retain every strict construction check.
    """

    payload = _json_object(response_text)
    return _construct_checked_scrc_artifact(
        payload,
        expected_parameters=expected_parameters,
        expected_tensors=expected_tensors,
        expected_dimensions=expected_dimensions,
        trusted_compiler_facts=trusted_compiler_facts,
        required_schema=required_schema,
        allocator=allocator,
    )


def _construct_checked_scrc_artifact(
    payload: Mapping[str, Any],
    *,
    expected_parameters: tuple[str, ...],
    expected_tensors: tuple[str, ...],
    expected_dimensions: tuple[str, ...],
    trusted_compiler_facts: Mapping[str, Any] | None,
    required_schema: str,
    allocator: CanonicalHandleAllocator,
) -> Gate1Artifact:
    """Construct the strict model, not an accepted lifecycle artifact.

    No permissive/pending mode: model, canonicalization, schema, anonymous
    contract and compiler contract remain mandatory and ordered. The caller
    still owes TypeCheck, Source authorization and the subsequent Gate chain.
    """
    try:
        parsed_artifact = Gate1Artifact.from_json(payload)
    except Exception as exc:
        raise _CandidateDecodeStageError("gate1_model", exc) from exc
    try:
        artifact = canonicalize_gate1_artifact(parsed_artifact)
    except Exception as exc:
        raise _CandidateDecodeStageError("j_canonicalization", exc) from exc
    if artifact.wire_schema != required_schema:
        raise _CandidateDecodeStageError(
            "wire_schema",
            ValueError("canonical artifact schema does not match this Lift round"),
        )
    try:
        _validate_anonymous_lift_core(
            artifact,
            expected_parameters,
            expected_tensors,
            expected_dimensions,
            allocator=allocator,
        )
    except Exception as exc:
        raise _CandidateDecodeStageError("anonymous_contract", exc) from exc
    try:
        _validate_compiler_view(
            artifact,
            trusted_compiler_facts,
        )
    except Exception as exc:
        raise _CandidateDecodeStageError("compiler_contract", exc) from exc
    return artifact


def _inspect_unaccepted_scrc_candidate(
    response_text: str,
    *,
    trusted_compiler_facts: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect raw candidate structure without constructing an artifact.

    An empty diagnostic list permits the existing allocator/decode path only;
    it is not CandidateCheck/TypeCheck success or an authorization receipt.
    Failed inputs return no usable payload and are never repaired here.
    """
    try:
        payload = _json_object(response_text)
    except Exception as exc:
        return {}, [
            {
                "rule_code": "candidate_json_parse_rejected",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"{type(exc).__name__}: complete one closed JSON "
                        "{schema,j,c,pi} root before submission"
                    )
                ],
            }
        ]
    # Preserve root -> schema -> namespace diagnostic precedence. The schema
    # check remains exactly the existing one, including its dependency policy;
    # strict construction cannot be skipped if it reports no diagnostics.
    diagnostics = _candidate_wire_envelope_diagnostics(response_text)
    if diagnostics:
        return {}, diagnostics
    diagnostics = _candidate_json_schema_diagnostics(response_text)
    if diagnostics:
        return {}, diagnostics
    diagnostics = _candidate_physical_namespace_diagnostics(
        payload, trusted_compiler_facts,
    )
    if diagnostics:
        return {}, diagnostics
    return payload, []


def check_scrc_lift_candidate(
    response_text: str,
    context: SCRCCandidateCheckContext,
) -> dict[str, Any]:
    """Run the authoritative decode/TypeCheck core and return diagnostics only."""

    if (
        context.schema_version != Gate1Artifact.SCHEMA
        or context.decode_protocol != SCRC_DECODE_PROTOCOL
        or context.typecheck_protocol != TYPECHECK_PROTOCOL
    ):
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_check_context_mismatch",
                    "path": "<context>",
                    "observed_refs_or_types": [
                        context.schema_version,
                        context.decode_protocol,
                        context.typecheck_protocol,
                    ],
                }
            ],
        }
    candidate_payload, surface_diagnostics = _inspect_unaccepted_scrc_candidate(
        response_text,
        trusted_compiler_facts=context.trusted_compiler_facts,
    )
    if surface_diagnostics:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": surface_diagnostics,
        }

    # Only a schema-valid canonical root may enter candidate-local allocator
    # derivation.  An arbitrary malformed nested row must become a structured
    # CandidateCheck rejection, not an exception in a declaration walker.
    candidate_allocator = context.allocator
    try:
        candidate_j = candidate_payload.get("j")
        if not isinstance(candidate_j, Mapping):
            raise TypeError("canonical candidate root.j must be an object")
        candidate_allocator = context.allocator.for_candidate_j(candidate_j)
    except Exception:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_allocator_surface_rejected",
                    "path": "j",
                    "observed_refs_or_types": [
                        "candidate J cannot establish the canonical handle "
                        "declaration table"
                    ],
                }
            ],
        }
    try:
        artifact = _decode_scrc_lift_artifact_core(
            response_text,
            expected_parameters=context.expected_parameters,
            expected_tensors=context.expected_tensors,
            expected_dimensions=context.expected_dimensions,
            trusted_compiler_facts=context.trusted_compiler_facts,
            required_schema=context.schema_version,
            allocator=candidate_allocator,
        )
    except Exception as exc:
        diagnostics = _candidate_decode_diagnostics(
            exc,
            response_text=response_text,
        )
        occupied_paths = {item["path"] for item in diagnostics}
        # A failed strict J model has already established the earliest safe
        # rejection category.  The supplemental surface walk assumes a
        # recordable J body and adds only a generic traversal-failure receipt
        # for this class of malformed input.  Do not obscure the owned J
        # authoring rule or attempt to traverse an unmodelled candidate.
        if not any(
            isinstance(item.get("rule_code"), str)
            and item["rule_code"].startswith("candidate_decode_j_")
            for item in diagnostics
        ):
            for item in _safe_candidate_surface_diagnostics(
                response_text,
                context,
                allocator=candidate_allocator,
            ):
                if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                    break
                if item["path"] in occupied_paths:
                    continue
                diagnostics.append(item)
                occupied_paths.add(item["path"])
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": diagnostics,
        }

    # This is the same callable used by the formal Runner.  CandidateCheck does
    # not retain or return the canonical artifact and therefore cannot become a
    # second acceptance path.
    try:
        report = type_check(
            artifact.j,
            artifact.c,
            artifact.pi,
            require_program_axes=artifact.wire_schema == artifact.SCHEMA,
        )
    except Exception:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_typed_surface_rejected",
                    "path": "<candidate>",
                    "observed_refs_or_types": [
                        "candidate passed decoding but its typed surface is "
                        "not a valid canonical mapping"
                    ],
                }
            ],
        }
    return {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass" if report.passed else "fail",
        "diagnostics": [
            {
                "rule_code": _candidate_typecheck_rule_code(
                    item.code.value,
                    item.detail,
                ),
                "path": _bounded_candidate_diagnostic_text(item.path),
                "observed_refs_or_types": _candidate_typecheck_observations(
                    item.ref,
                    item.detail,
                ),
            }
            for item in report.diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]
        ],
    }


def _candidate_wire_envelope_diagnostics(
    response_text: str,
) -> list[dict[str, Any]]:
    """Return the shared root-envelope diagnosis before dependent schema errors.

    A flattened J response causes JSON Schema to report many secondary errors
    (missing J fields, missing C/pi, and dependent operation failures). The
    typed decoder has the same ownership rule, so surface the one shared root
    diagnosis first and stop there. This is still reject-only: no field is moved
    and no malformed response becomes later authoring state.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return []
    shared_error = canonical_artifact_envelope_error(payload)
    if shared_error is not None:
        # The shared construction authority decides whether this transport may
        # enter repair/decode. Gate1's projection is used only to preserve the
        # more specific, bounded relift path (for example flattened J versus a
        # missing root sibling); it cannot make the shared ingress permissive.
        detailed = gate1_artifact_wire_diagnostics(payload)
        if detailed:
            diagnostics: list[dict[str, Any]] = []
            for item in detailed[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
                observed = [shared_error]
                message = item.get("message")
                if isinstance(message, str) and message != shared_error:
                    observed.append(message)
                diagnostics.append(
                    {
                        "rule_code": str(item.get("code", "artifact_wire_root")),
                        "path": str(item.get("path", "Gate1Artifact")),
                        "observed_refs_or_types": observed,
                    }
                )
            return diagnostics
        return [{
            "rule_code": "artifact_wire_root",
            "path": "Gate1Artifact",
            "observed_refs_or_types": [shared_error],
        }]
    diagnostics: list[dict[str, Any]] = []
    for item in gate1_artifact_wire_diagnostics(payload):
        observed: list[str] = []
        message = item.get("message")
        if isinstance(message, str):
            observed.append(message)
        for key in (
            "flattened_j_fields",
            "missing_root_fields",
            "extra_root_fields",
            "expected_root",
            "actual_root",
        ):
            value = item.get(key)
            if value:
                observed.append(
                    f"{key}="
                    + json.dumps(
                        value,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
        diagnostics.append(
            {
                "rule_code": str(item.get("code", "artifact_wire_root")),
                "path": _bounded_candidate_diagnostic_text(
                    str(item.get("path", "<candidate>"))
                ),
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(value)
                    for value in observed
                ],
            }
        )
    return diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]


def _candidate_physical_namespace_diagnostics(
    payload: Mapping[str, Any],
    trusted_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose physical p*/c* leakage before dependent typed decoding.

    This is a reject-only projection of the exact ownership check used by the
    canonical construction transaction.  It does not rewrite an ABI carrier
    into a logical value, and in particular never guesses that a C capacity is
    a J extent.  Reporting it before typed-boundary elaboration keeps relift
    guidance attached to the semantic cause rather than a stale downstream
    construction-record symptom.
    """

    try:
        _validate_canonical_namespace_ownership(payload, trusted_view)
    except ValueError as exc:
        message = str(exc)
        if "handle" in message or "J signature" in message:
            path_match = re.search(r" at (j(?:\.[^ ]+)?)$", message)
            return [
                {
                    "rule_code": "candidate_physical_handle_in_j",
                    "path": path_match.group(1) if path_match else "j",
                    "observed_refs_or_types": [
                        _bounded_candidate_diagnostic_text(message)
                    ],
                }
            ]
    return []


def _candidate_json_schema_diagnostics(
    response_text: str,
) -> list[dict[str, Any]]:
    """Aggregate independent structural errors before dependent Decode.

    This is the same closed wire schema already supplied to every Lift backend.
    It never fills a field or changes acceptance: an empty result merely allows
    the existing strict Decode/TypeCheck path to run.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return []
    try:
        import jsonschema
    except ImportError:
        return []

    schema = scrc_lift_core_json_schema()
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (
            tuple(str(item) for item in error.absolute_path),
            tuple(str(item) for item in error.absolute_schema_path),
            error.message,
        ),
    )
    diagnostics: list[dict[str, Any]] = []

    def path_value(value: Any, components: tuple[Any, ...]) -> Any:
        current = value
        for component in components:
            if isinstance(current, Mapping):
                current = current.get(component)
            elif isinstance(current, list) and isinstance(component, int):
                if component < 0 or component >= len(current):
                    return None
                current = current[component]
            else:
                return None
        return current

    for error in errors[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = ""
        for item in error.absolute_path:
            if isinstance(item, int):
                path += f"[{item}]"
            else:
                path += ("." if path else "") + str(item)
        rule_code = "schema_rejected"
        observed_message = error.message
        if path.endswith(".attrs") or path.endswith("]attrs"):
            operation = path_value(payload, tuple(error.absolute_path[:-1]))
            if isinstance(operation, Mapping) and operation.get("op") == "reduce":
                lowered = error.message.casefold()
                if (
                    "additional propert" in lowered
                    or "kind" in lowered
                    or "identity" in lowered
                ):
                    rule_code = "schema_rejected_reduce_attrs"
                    observed_message = f"op=reduce; {error.message}"
        diagnostics.append(
            {
                "rule_code": rule_code,
                "path": _bounded_candidate_diagnostic_text(
                    path or "<candidate>"
                ),
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(observed_message)
                ],
            }
        )
    return diagnostics


def _trusted_layer_c_declarations(
    view: Mapping[str, Any] | None,
    *,
    allocator: CanonicalHandleAllocator,
) -> dict[str, Any] | None:
    """Project the canonical SourceFacts C declarations for diagnostics only.

    Candidate diagnostics must not rebuild C order or dtype from a value map.
    ``schedule_values``, ``schedule_value_types``, ``parameter_order`` and
    ``compiler_slots`` are already closed by the Source construction pass; the
    projection below only verifies and reuses those records.
    """

    if not isinstance(view, Mapping) or not isinstance(
        allocator, CanonicalHandleAllocator
    ):
        return None
    values = view.get("schedule_values")
    value_types = view.get("schedule_value_types")
    parameter_order = view.get("parameter_order")
    compiler_slots = view.get("compiler_slots", [])
    compiler_roles = view.get("compiler_roles", {})
    if (
        not isinstance(values, Mapping)
        or not isinstance(value_types, Mapping)
        or not isinstance(parameter_order, list)
        or not isinstance(compiler_slots, list)
    ):
        return None
    if not isinstance(compiler_roles, Mapping):
        return None
    source_c_slots = [
        name
        for name in parameter_order
        if isinstance(name, str) and re.fullmatch(r"c[0-9]+", name)
    ]
    c_order = [*source_c_slots, *compiler_slots]
    allocator_c_order = allocator.declared("c")
    if (
        len(c_order) != len(set(c_order))
        or tuple(c_order)
        != allocator_c_order
        or set(values) != set(c_order)
        or set(value_types) != set(c_order)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"c[0-9]+", name) is None
            for name in compiler_slots
        )
        or len(compiler_slots) != len(set(compiler_slots))
        or set(compiler_slots) - set(values)
        or set(compiler_roles) != set(c_order)
    ):
        return None
    slots: list[dict[str, Any]] = []
    compiler_set = set(compiler_slots)
    for name in c_order:
        row: dict[str, Any] = {
            "slot": name,
            "kind": "compiler" if name in compiler_set else "constexpr",
            "value_type": value_types.get(name),
            "required": True,
        }
        if not isinstance(row["value_type"], str) or not row["value_type"]:
            return None
        role = compiler_roles.get(name)
        if isinstance(role, str):
            row["role"] = role
        slots.append(row)
    return {"slots": slots}


def _candidate_surface_diagnostics(
    response_text: str,
    context: SCRCCandidateCheckContext,
    *,
    allocator: CanonicalHandleAllocator,
) -> list[dict[str, Any]]:
    """Report independent authored-surface mistakes beside a Decode failure.

    The strict decoder remains the only acceptance boundary.  This pass reads
    raw JSON only after that decoder has already rejected the candidate and
    checks closed naming/ABI facts that do not depend on repairing or
    interpreting an invalid graph.  It never mutates, canonicalizes, or fills
    a field, and its diagnostics cannot turn rejection into acceptance.
    """

    if not isinstance(allocator, CanonicalHandleAllocator):
        return [
            {
                "rule_code": "construction_allocator_unavailable",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    "CandidateCheck cannot validate namespace ownership without "
                    "the canonical allocator"
                ],
            }
        ]
    try:
        payload = _json_object(response_text)
    except Exception:
        return []

    diagnostics: list[dict[str, Any]] = []

    def mismatch(
        rule_code: str,
        path: str,
        expected: Any,
        actual: Any,
    ) -> None:
        if len(diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
            return
        diagnostics.append(
            {
                "rule_code": rule_code,
                "path": path,
                "observed_refs_or_types": [
                    "closed structural contract mismatch; candidate values are withheld"
                ],
            }
        )

    j = payload.get("j")
    if not isinstance(j, Mapping):
        return diagnostics
    intent = j.get("intent")
    fn = intent.get("intent_function") if isinstance(intent, Mapping) else None
    operation_records = (
        _canonical_operation_records(fn)
        if isinstance(fn, Mapping)
        else None
    )
    canonical_output_records = (
        _canonical_output_records(payload, operation_records)
        if operation_records is not None
        else None
    )
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    if isinstance(parameters, list) and all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str)
        for item in parameters
    ):
        actual_parameters = [str(item["name"]) for item in parameters]
        for parameter_index, name in enumerate(actual_parameters):
            if re.fullmatch(r"p[0-9]+", name) is None:
                mismatch(
                    "construction_signature_parameter_namespace",
                    f"j.signature.parameters[{parameter_index}].name",
                    "p* physical Source ABI handle; c* belongs only to C.slots",
                    name,
                )
        expected_parameters = list(allocator.declared("p"))
        if actual_parameters != expected_parameters:
            mismatch(
                "construction_handle_sequence",
                "j.signature.parameters[*].name",
                expected_parameters,
                actual_parameters,
            )
        trusted_view = context.trusted_compiler_facts
        bound_constexprs = (
            trusted_view.get("bound_constexpr_values")
            if isinstance(trusted_view, Mapping)
            else None
        )
        if isinstance(bound_constexprs, Mapping):
            for name in bound_constexprs:
                parameter = next(
                    (
                        item
                        for item in parameters
                        if isinstance(item, Mapping)
                        and item.get("name") == name
                    ),
                    None,
                )
                if not isinstance(parameter, Mapping):
                    continue
                if parameter.get("kind") != "constexpr":
                    mismatch(
                        "construction_constexpr_parameter_kind",
                        f"j.signature.parameters[{name}].kind",
                        "constexpr",
                        parameter.get("kind"),
                    )

    if isinstance(fn, Mapping):
        raw_outputs = fn.get("outputs")
        raw_returns = fn.get("returns")
        if isinstance(raw_outputs, list) and isinstance(raw_returns, list):
            expected_outputs = [
                item.get("name")
                for item in raw_outputs
                if isinstance(item, Mapping)
                and isinstance(item.get("name"), str)
            ]
            actual_returns = [
                item.get("output")
                for item in raw_returns
                if isinstance(item, Mapping)
                and isinstance(item.get("output"), str)
            ]
            if actual_returns != expected_outputs:
                mismatch(
                    "construction_return_abi_order",
                    "j.intent.intent_function.returns[*].output",
                    expected_outputs,
                    actual_returns,
                )
            raw_output_mappings = (
                j.get("boundary", {}).get("output_mappings")
                if isinstance(j.get("boundary"), Mapping)
                else None
            )
            if isinstance(raw_output_mappings, list):
                for index, item in enumerate(raw_output_mappings):
                    if not isinstance(item, Mapping):
                        continue
                    output = item.get("output")
                    if isinstance(output, str) and output not in expected_outputs:
                        mismatch(
                            "construction_output_mapping_output_namespace",
                            f"j.boundary.output_mappings[{index}].output",
                            expected_outputs,
                            output,
                        )
            if canonical_output_records is not None:
                canonical_return_outputs = [
                    record["output"] for record in canonical_output_records
                ]
                if canonical_return_outputs != expected_outputs:
                    mismatch(
                        "construction_output_record_order",
                        "j.intent.intent_function.outputs + returns",
                        expected_outputs,
                        canonical_return_outputs,
                    )
                for record in canonical_output_records:
                    mapping_value = record.get("mapping_value")
                    if (
                        mapping_value is not None
                        and mapping_value != record["return_value"]
                    ):
                        mismatch(
                            "construction_return_output_agreement",
                            (
                                "j.boundary.output_mappings["
                                f"{record['mapping_index']}].value"
                            ),
                            record["return_value"],
                            mapping_value,
                        )

    tensors: list[Mapping[str, Any]] = []
    if isinstance(fn, Mapping):
        for collection in ("inputs", "outputs"):
            values = fn.get(collection)
            if isinstance(values, list):
                tensors.extend(
                    item for item in values if isinstance(item, Mapping)
                )
    tensor_names = [item.get("name") for item in tensors]
    if tensor_names and all(isinstance(item, str) for item in tensor_names):
        actual_tensors = list(dict.fromkeys(tensor_names))
        expected_tensors = list(allocator.declared("t"))
        if actual_tensors != expected_tensors:
            mismatch(
                "construction_handle_sequence",
                "j.intent.intent_function.inputs+outputs[*].name",
                expected_tensors,
                actual_tensors,
            )

    dimensions: list[str] = []
    for tensor in tensors:
        shape = tensor.get("shape")
        if not isinstance(shape, list):
            continue
        for extent in shape:
            if isinstance(extent, str) and extent not in dimensions:
                dimensions.append(extent)
    expected_dimensions = list(allocator.declared("x"))
    if dimensions != expected_dimensions:
        mismatch(
            "construction_handle_sequence",
            "j.intent.intent_function tensor shape dimensions",
            expected_dimensions,
            dimensions,
        )

    operations: list[tuple[str, Mapping[str, Any]]] = []
    if operation_records is None:
        mismatch(
            "construction_operation_records_unavailable",
            "j.intent.intent_function.body",
            "one complete canonical operation record table",
            "unavailable; raw-body fallback is forbidden",
        )
    else:
        operations = [
            (str(record["path"]), dict(record["wire"]))
            for record in operation_records
            if isinstance(record.get("path"), str)
            and isinstance(record.get("wire"), Mapping)
        ]
    for operation_path, operation in operations:
        if operation.get("op") == "cast":
            attrs = operation.get("attrs")
            if isinstance(attrs, Mapping) and "to" in attrs and "to_dtype" not in attrs:
                mismatch(
                    "construction_cast_attribute",
                    f"j.intent.intent_function.{operation_path}.attrs",
                    '{"to_dtype":"<closed dtype>"}',
                    dict(attrs),
                )
        if operation.get("op") == "transpose":
            attrs = operation.get("attrs")
            if not isinstance(attrs, Mapping) or "perm" not in attrs:
                mismatch(
                    "construction_transpose_attribute",
                    f"j.intent.intent_function.{operation_path}.attrs",
                    "closed transpose attribute {'perm': [input-axis ...]}",
                    attrs,
                )
        if operation.get("op") != "loop":
            continue
        trip = operation.get("trip")
        if isinstance(trip, Mapping) and "axis" not in trip:
            upper_bound = trip.get("upper_bound")
            trip_index = trip.get("index")
            if (
                isinstance(upper_bound, str)
                and re.fullmatch(r"x[0-9]+", upper_bound)
            ) or (
                isinstance(trip_index, str)
                and re.fullmatch(r"x[0-9]+", trip_index)
            ):
                mismatch(
                    "construction_loop_trip_namespace",
                    f"j.intent.intent_function.{operation_path}.trip",
                    "static {'axis':'x*'} or dynamic {'upper_bound':'t*/v*','index':'v*'}",
                    trip,
                )
        carries = operation.get("carry")
        if isinstance(carries, list):
            roles = [
                state.get(role)
                for state in carries
                if isinstance(state, Mapping)
                for role in ("body_argument", "next_value", "result_value")
            ]
            if len(roles) != len(set(roles)):
                mismatch(
                    "construction_loop_carry_role_namespace",
                    f"j.intent.intent_function.{operation_path}.carry",
                    "body_argument, next_value, and result_value are all distinct v* handles",
                    roles,
                )

    candidate_pi = payload.get("pi")
    if isinstance(candidate_pi, Mapping):
        program_axes = candidate_pi.get("program_axes")
        program_bindings = candidate_pi.get("program_bindings")
        if not isinstance(program_axes, list) or not isinstance(
            program_bindings, list
        ):
            mismatch(
                "construction_pi_program_mapping_xor",
                "pi.program_axes+pi.program_bindings",
                "both program mapping families are explicit arrays; one may be []",
                {
                    "program_axes": program_axes,
                    "program_bindings": program_bindings,
                },
            )
        if (
            isinstance(program_axes, list)
            and program_axes
            and isinstance(program_bindings, list)
            and program_bindings
        ):
            mismatch(
                "construction_pi_program_mapping_xor",
                "pi.program_axes+pi.program_bindings",
                "one populated complete program mapping family; the other array is []",
                {
                    "program_axes": program_axes,
                    "program_bindings": program_bindings,
                },
            )
    node_ids = [item.get("id") for _path, item in operations]
    # A candidate-owned namespace is authoritative only after the shared
    # construction table was recordable and the same allocator successfully
    # extended the Source-owned declarations.  When either prerequisite is
    # absent, ``[]`` means "no trusted declaration", not "the candidate must
    # have no nodes"; reporting every authored n*/q*/a*/e* row as a sequence
    # mismatch only obscures the primary construction failure.
    if (
        node_ids
        and allocator.declared("n")
        and all(isinstance(item, str) for item in node_ids)
    ):
        expected_nodes = list(allocator.declared("n"))
        if node_ids != expected_nodes:
            mismatch(
                "construction_handle_sequence",
                "j.intent.intent_function.body recursive node ids",
                expected_nodes,
                node_ids,
            )
    authored_node_map = j.get("node_ids")
    if (
        allocator.declared("n")
        and isinstance(authored_node_map, Mapping)
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in authored_node_map.items()
        )
    ):
        path_order = {
            path: index
            for index, (_operation_path, operation) in enumerate(operations)
            if isinstance((path := authored_node_map.get(operation.get("id"))), str)
        }
        ordered_keys = [
            key
            for key, _value in sorted(
                authored_node_map.items(),
                key=lambda item: path_order.get(item[1], len(path_order)),
            )
        ]
        expected_map_keys = list(allocator.declared("n"))
        if ordered_keys != expected_map_keys:
            mismatch(
                "construction_handle_sequence",
                "j.node_ids keys in authored body-path order",
                expected_map_keys,
                ordered_keys,
            )

    boundary = j.get("boundary")
    handle_specs = (
        ("d", "domains", None),
        ("a", "accesses", "access_id"),
        ("e", "effects", "effect_id"),
        ("q", "scopes", "scope_id"),
        ("b", "storage", "storage_id"),
    )
    for prefix, field_name, id_field in handle_specs:
        owner: Any = boundary if field_name in {"domains", "accesses", "effects"} else j
        values = owner.get(field_name) if isinstance(owner, Mapping) else None
        if id_field is None:
            actual = (
                sorted(
                    values,
                    key=lambda item: (
                        int(match.group(1))
                        if isinstance(item, str)
                        and (match := re.fullmatch(r"d([0-9]+)", item))
                        else math.inf,
                        str(item),
                    ),
                )
                if isinstance(values, Mapping)
                else None
            )
        elif isinstance(values, list) and all(
            isinstance(item, Mapping) and isinstance(item.get(id_field), str)
            for item in values
        ):
            actual = [str(item[id_field]) for item in values]
        else:
            actual = None
        if actual is None:
            continue
        expected = list(allocator.declared(prefix))
        if not expected and prefix in CANONICAL_CANDIDATE_OWNED_HANDLE_PREFIXES:
            continue
        if actual != expected:
            mismatch(
                "construction_handle_sequence",
                f"j.{field_name}" + (f"[*].{id_field}" if id_field else " keys"),
                expected,
                actual,
            )

    raw_dimension_constraints = j.get("dimension_constraints")
    if isinstance(raw_dimension_constraints, list):
        for constraint_index, constraint in enumerate(raw_dimension_constraints):
            if isinstance(constraint, Mapping):
                mismatch(
                    "construction_dimension_constraint_shape",
                    f"j.dimension_constraints[{constraint_index}]",
                    "one schema-valid ContractExpr string; value-to-dimension "
                    "equality belongs in j.value_dimension_relations",
                    constraint,
                )

    raw_value_dimension_relations = j.get("value_dimension_relations")
    if isinstance(raw_value_dimension_relations, list):
        for relation_index, relation in enumerate(raw_value_dimension_relations):
            if not isinstance(relation, Mapping):
                continue
            value_ref = relation.get("value_ref")
            dimension_ref = relation.get("dimension_ref")
            if not (
                isinstance(value_ref, str)
                and re.fullmatch(r"t[0-9]+", value_ref)
                and isinstance(dimension_ref, str)
                and re.fullmatch(r"x[0-9]+", dimension_ref)
            ):
                mismatch(
                    "construction_value_dimension_relation_namespace",
                    f"j.value_dimension_relations[{relation_index}]",
                    "{value_ref:t*, dimension_ref:x*, relation:eq}",
                    relation,
                )

    # A return is the ABI-level value binding for an output.  The same output
    # may also have a sparse Layer-B output mapping, but the two declarations
    # must name the identical Layer-A value.  This check is intentionally on
    # the raw candidate surface: a malformed C declaration must not hide an
    # otherwise independent return mismatch until a later relift round.
    if isinstance(fn, Mapping) and isinstance(boundary, Mapping):
        raw_returns = fn.get("returns")
        raw_output_mappings = boundary.get("output_mappings")
        if (
            canonical_output_records is None
            and isinstance(raw_returns, list)
            and isinstance(raw_output_mappings, list)
        ):
            returns_by_output: dict[str, tuple[int, Mapping[str, Any]]] = {}
            for index, item in enumerate(raw_returns):
                if (
                    isinstance(item, Mapping)
                    and isinstance(item.get("output"), str)
                    and item["output"] not in returns_by_output
                ):
                    returns_by_output[item["output"]] = (index, item)
            for mapping in raw_output_mappings:
                if not isinstance(mapping, Mapping):
                    continue
                output = mapping.get("output")
                expected_value = mapping.get("value")
                if not isinstance(output, str) or not isinstance(expected_value, str):
                    continue
                returned = returns_by_output.get(output)
                if returned is None:
                    continue
                return_index, return_item = returned
                actual_value = return_item.get("value")
                if actual_value != expected_value:
                    mismatch(
                        "construction_return_output_agreement",
                        f"j.intent.intent_function.returns[{return_index}].value",
                        expected_value,
                        actual_value,
                    )

        # A return/output pair may agree on one tile-shaped value while still
        # violating the full logical output type.  The raw type rows are
        # sufficient to report that independent error before Decode reaches a
        # separate malformed relation.
        if canonical_output_records is not None:
            output_types = {
                record["output"]: copy.deepcopy(record["output_type"])
                for record in canonical_output_records
            }
        else:
            output_types = {
                item.get("name"): {
                    "dtype": item.get("dtype"),
                    "shape": item.get("shape"),
                }
                for item in fn.get("outputs", [])
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
        value_types = (
            _canonical_value_type_records(fn, operation_records)
            if operation_records is not None
            else None
        )
        if value_types is None:
            value_types = {}
        if isinstance(raw_returns, list):
            for return_index, return_item in enumerate(raw_returns):
                if not isinstance(return_item, Mapping):
                    continue
                output = return_item.get("output")
                value = return_item.get("value")
                expected_type = output_types.get(output)
                actual_type = value_types.get(value)
                if (
                    isinstance(expected_type, Mapping)
                    and isinstance(actual_type, Mapping)
                    and dict(actual_type) != dict(expected_type)
                ):
                    mismatch(
                        "construction_return_type_agreement",
                        f"j.intent.intent_function.returns[{return_index}].value",
                        dict(expected_type),
                        dict(actual_type),
                    )

        # Each indexed operation has a closed, one-row typed precondition.
        # It is fully determined by the authored operation, so reporting the
        # exact missing row does not infer or repair a candidate.  Aggregate
        # these rows beside unrelated schema/decode failures so one gather at
        # a time cannot consume the bounded Lift budget.
        raw_index_bounds = boundary.get("index_bounds")
        index_bounds = (
            [item for item in raw_index_bounds if isinstance(item, Mapping)]
            if isinstance(raw_index_bounds, list)
            else []
        )
        for operation in operations:
            if operation.get("op") not in {"gather", "scatter"}:
                continue
            node_ref = operation.get("id")
            operands = operation.get("operands")
            attrs = operation.get("attrs")
            if (
                not isinstance(node_ref, str)
                or not isinstance(operands, list)
                or len(operands) < 2
                or not isinstance(attrs, Mapping)
                or isinstance(attrs.get("axis"), bool)
                or not isinstance(attrs.get("axis"), int)
            ):
                continue
            expected_bound = {
                "node_ref": node_ref,
                "data": operands[0],
                "indices": operands[1],
                "axis": attrs["axis"],
            }
            if not any(
                all(bound.get(key) == value for key, value in expected_bound.items())
                for bound in index_bounds
            ):
                authored_bounds = [
                    dict(bound)
                    for bound in index_bounds
                    if bound.get("node_ref") == node_ref
                ]
                mismatch(
                    "construction_index_bounds",
                    f"j.boundary.index_bounds[{node_ref}]",
                    expected_bound,
                    authored_bounds or None,
                )

        # Result-relation field presence is independent of graph decoding.
        # In particular, an ordinary exact result is not a logical update and
        # must not carry a dangling update node reference.
        raw_result_relations = boundary.get("result_relations")
        if isinstance(raw_result_relations, list):
            for relation_index, relation in enumerate(raw_result_relations):
                if not isinstance(relation, Mapping):
                    continue
                kind = relation.get("kind")
                has_update_ref = "update_node_ref" in relation
                if kind in {"exact", "tolerance"} and has_update_ref:
                    mismatch(
                        "construction_result_relation_update_reference",
                        f"j.boundary.result_relations[{relation_index}].update_node_ref",
                        "field omitted for exact/tolerance",
                        relation.get("update_node_ref"),
                    )
                elif kind == "one_of_updates" and not isinstance(
                    relation.get("update_node_ref"), str
                ):
                    mismatch(
                        "construction_result_relation_update_reference",
                        f"j.boundary.result_relations[{relation_index}].update_node_ref",
                        "non-empty n* reference for one_of_updates",
                        relation.get("update_node_ref"),
                    )

        # Scopes are a derived dominance archive, not an ABI inventory.  Use
        # the same operation table and lexical allocator as ingress repair;
        # otherwise CandidateCheck can report a different expected order from
        # the one used to construct the candidate.  All scopes are checked;
        # A function-only projection would silently hide loop/branch/program
        # ownership mismatches.
        raw_scopes = j.get("scopes")
        raw_body = fn.get("body")
        if (
            isinstance(raw_scopes, list)
            and isinstance(raw_body, list)
            and operation_records is not None
            and isinstance(boundary, Mapping)
            and isinstance(j.get("node_ids"), Mapping)
        ):
            expected_scopes = canonical_scope_records(
                fn,
                raw_scopes,
                boundary,
                j["node_ids"],
                operation_records=operation_records,
                repairable_scope_fields=True,
            )
            expected_scope_by_id = {
                record.get("scope_id"): record
                for record in expected_scopes or ()
                if isinstance(record.get("scope_id"), str)
            }
            for scope_index, scope in enumerate(raw_scopes):
                if not isinstance(scope, Mapping):
                    continue
                expected_scope = expected_scope_by_id.get(scope.get("scope_id"))
                if not isinstance(expected_scope, Mapping):
                    continue
                for field_name in ("definitions", "captures", "effects"):
                    actual = scope.get(field_name)
                    expected = expected_scope[field_name]
                    if actual == expected:
                        continue
                    if not isinstance(actual, list) or any(
                        not isinstance(item, str) for item in actual
                    ):
                        continue
                    mismatch(
                        "construction_scope_definitions",
                        f"j.scopes[{scope_index}].{field_name}",
                        expected,
                        actual,
                    )

    view = context.trusted_compiler_facts
    declarations = _trusted_layer_c_declarations(view, allocator=allocator)
    if isinstance(view, Mapping) and isinstance(parameters, list):
        by_name = {
            item.get("name"): item
            for item in parameters
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        arguments = view.get("arguments")
        if isinstance(arguments, list):
            for argument in arguments:
                if not isinstance(argument, Mapping):
                    continue
                name = argument.get("name")
                parameter = by_name.get(name)
                if not isinstance(name, str) or not isinstance(parameter, Mapping):
                    continue
                source_kind = argument.get("kind")
                allowed_kinds = {
                    "tensor": ["pointer", "tensor"],
                    "scalar": ["scalar"],
                    "opaque": ["opaque"],
                    "none": ["opaque"],
                }.get(source_kind)
                expected_fields: dict[str, Any] = {}
                if allowed_kinds is not None:
                    expected_fields["kind"] = allowed_kinds
                if "dtype" in argument:
                    expected_fields["dtype"] = _canonical_dtype(argument.get("dtype"))
                if source_kind == "tensor" and isinstance(argument.get("shape"), list):
                    expected_fields["rank"] = len(argument["shape"])
                if "mode" in argument:
                    expected_fields["mode"] = argument.get("mode")
                for field_name, expected in expected_fields.items():
                    actual = parameter.get(field_name)
                    if field_name == "dtype" and isinstance(actual, str):
                        actual = _canonical_dtype(actual)
                    matched = actual in expected if isinstance(expected, list) else actual == expected
                    if not matched:
                        mismatch(
                            "construction_abi_mismatch",
                            f"j.signature.parameters[{name}].{field_name}",
                            expected,
                            actual,
                        )
        # CandidateCheck keeps these ABI checks branch-local: the trusted
        # selected frontend view verifies physical parameters and C slots.
        # It is diagnostic evidence only and never replaces Triton Source TTIR
        # observation or the native compiler/Gate path of another frontend.
        c = payload.get("c")
        if isinstance(c, Mapping) and c.get("values") != view.get("schedule_values"):
            mismatch(
                "construction_abi_mismatch",
                "c.values",
                view.get("schedule_values"),
                c.get("values"),
            )
        if isinstance(c, Mapping):
            raw_slots = c.get("slots")
            slots_by_name = (
                {
                    item.get("name"): item
                    for item in raw_slots
                    if isinstance(item, Mapping)
                    and isinstance(item.get("name"), str)
                }
                if isinstance(raw_slots, list)
                else {}
            )
            bound_non_c = view.get("bound_constexpr_values")
            if isinstance(bound_non_c, Mapping):
                for name in bound_non_c:
                    if name in slots_by_name:
                        mismatch(
                            "construction_semantic_constexpr_in_c",
                            f"c.slots[{name}]",
                            "bound non-C constexpr carrier in J.signature only",
                            slots_by_name[name],
                        )

            # The candidate is required to spell the trusted C interface
            # exactly as the trusted declaration does.  Decode performs the
            # authoritative comparison, but report the concrete expected
            # field here as well so an unrelated decode failure cannot force
            # the owner to spend one relift merely discovering (for example)
            # whether c0 is constexpr or compiler.  Do not compare `required`
            # on raw JSON: ScheduleSlot supplies its default during Decode.
            if isinstance(declarations, Mapping) and isinstance(raw_slots, list):
                try:
                    expected_slots = [
                        {
                            "slot": item["slot"],
                            "kind": item["kind"],
                            "value_type": item["value_type"],
                            **(
                                {"role": item["role"]}
                                if "role" in item
                                else {}
                            ),
                        }
                        for item in declarations["slots"]
                    ]
                except (TypeError, ValueError) as exc:
                    mismatch(
                        "candidate_check_context_mismatch",
                        "harness_compiler_view.layer_c_declarations",
                        "closed trusted C declarations",
                        f"{type(exc).__name__}: {exc}",
                    )
                else:
                    if len(raw_slots) != len(expected_slots):
                        mismatch(
                            "construction_layer_c_declaration",
                            "c.slots",
                            f"{len(expected_slots)} trusted declaration(s)",
                            f"{len(raw_slots)} authored declaration(s)",
                        )
                    for index, expected_slot in enumerate(expected_slots):
                        if index >= len(raw_slots):
                            break
                        actual_slot = raw_slots[index]
                        if not isinstance(actual_slot, Mapping):
                            continue
                        expected_name = expected_slot["slot"]
                        if actual_slot.get("name") != expected_name:
                            mismatch(
                                "construction_layer_c_declaration",
                                f"c.slots[{index}].name",
                                expected_name,
                                actual_slot.get("name"),
                            )
                        for key in ("kind", "value_type"):
                            if actual_slot.get(key) != expected_slot.get(key):
                                mismatch(
                                    "construction_layer_c_declaration",
                                    f"c.slots[{index}].{key}",
                                    expected_slot.get(key),
                                    actual_slot.get(key),
                                )
                        if (
                            "role" in expected_slot
                            and actual_slot.get("role") != expected_slot["role"]
                        ):
                            mismatch(
                                "construction_layer_c_declaration",
                                f"c.slots[{index}].role",
                                expected_slot["role"],
                                actual_slot.get("role"),
                            )
        actual_aliases = boundary.get("aliases") if isinstance(boundary, Mapping) else None
        if actual_aliases != view.get("aliases"):
            mismatch(
                "construction_abi_mismatch",
                "j.boundary.aliases",
                view.get("aliases"),
                actual_aliases,
            )

    return diagnostics


def _safe_candidate_surface_diagnostics(
    response_text: str,
    context: SCRCCandidateCheckContext,
    *,
    allocator: CanonicalHandleAllocator,
) -> list[dict[str, Any]]:
    """Keep authored-surface diagnostics fail-closed and structurally typed.

    Surface diagnostics are reject-only supplemental reporting.  They must
    never turn a malformed provider object into an internal CandidateCheck
    exception (for example, a tuple reaching a mapping-only ``.get`` call).
    The strict decoder remains the acceptance authority; an exception here is
    therefore reported as one bounded diagnostic and cannot create a repair
    base or alter acceptance.
    """

    try:
        return _candidate_surface_diagnostics(
            response_text,
            context,
            allocator=allocator,
        )
    except Exception as exc:  # pragma: no cover - exercised by malformed real responses
        return [
            {
                "rule_code": "candidate_surface_diagnostics_failed",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    "exception:" + type(exc).__name__,
                ],
            }
        ]


def _candidate_typecheck_observations(
    ref: str | None,
    detail: str,
) -> list[str]:
    """Expose bounded artifact-only TypeCheck evidence without repairing it."""

    observations: list[str] = []
    if ref is not None:
        observations.append(_bounded_candidate_diagnostic_text(ref))
    if detail:
        observations.append(
            _bounded_candidate_diagnostic_text(f"detail: {detail}")
        )
    return observations


def _candidate_typecheck_rule_code(code: str, detail: str) -> str:
    """Refine functional-boundary failures without exposing their payload.

    The functional checker deliberately returns prose for local developer
    diagnostics. CandidateCheck may not relay that prose, a node/access name,
    or any candidate expression. Map only its finite, implementation-owned
    categories to fixed codes so fresh authoring can distinguish incompatible
    contracts without gaining a rejected-candidate repair base.
    """

    if code == "access_guard_relation_invalid":
        lowered = detail.casefold()
        if "one (access_ref, value_ref) pair may be declared only once" in lowered:
            return "access_guard_relation_duplicate"
        if "value_ref must name a real typed layer-a declaration" in lowered:
            return "access_guard_value_ref_unbound"
        if "access guard value_ref must have canonical dtype i1" in lowered:
            return "access_guard_value_type_invalid"
        if "relation-bound computed guard must not also appear" in lowered:
            return "access_guard_predicate_double_claim"
        if "access guard axes must be strictly increasing and unique" in lowered:
            return "access_guard_axes_invalid"
        if "guard shape cannot project to the access target" in lowered:
            return "access_guard_projection_invalid"
        if "access target must have one closed layer-a tensor type" in lowered:
            return "access_guard_target_type_unproved"
        if "relation-guarded load requires an explicit fill_value" in lowered:
            return "access_guard_load_fill_missing"
        if "access-guard relation v1 has no closed update merge contract" in lowered:
            return "access_guard_update_unsupported"
        return code
    if code != "functional_boundary_incomplete":
        return code
    if re.fullmatch(r"source_owner_[a-z0-9_]{1,80}", detail):
        return detail
    if detail in {
        "functional_logical_update_output_type_unclosed",
        "functional_logical_update_base_type_mismatch",
        "functional_logical_update_values_type_mismatch",
        "functional_logical_update_result_type_mismatch",
        "functional_logical_update_indices_shape_mismatch",
        "functional_logical_update_indices_dtype_invalid",
    }:
        return detail
    lowered = detail.casefold()
    if "multiple distinct node_ref=null access views" in lowered:
        return "functional_boundary_access_view_ambiguous"
    if (
        "node-less computed guarded load requires an explicit consumer extension"
        in lowered
    ):
        return "functional_boundary_guarded_load_consumer_extension_missing"
    if "computed layer-a predicate references" in lowered:
        return "functional_boundary_predicate_computed_reference"
    if "predicate upper bound is not an access axis" in lowered:
        return "functional_boundary_predicate_axis_unbound"
    if "predicate differs from the full logical domain" in lowered:
        return "functional_boundary_predicate_domain_mismatch"
    if "predicate lower bound" in lowered:
        return "functional_boundary_predicate_lower_bound_invalid"
    if "predicate" in lowered:
        return "functional_boundary_predicate_contract_invalid"
    return code


def _candidate_decode_classification(
    detail: str,
    path: str,
    *,
    fallback_rule_code: str = "decode_rejected",
) -> tuple[str, str]:
    """Map common decoder failures to the same actionable contract classes."""

    lowered = detail.casefold()
    if "returns" in lowered and "type does not match output" in lowered:
        return (
            "construction_return_type_agreement",
            path if path != "<candidate>" else "j.intent.intent_function.returns[*].value",
        )
    if "return" in lowered and "unknown layer-a value" in lowered:
        return (
            "construction_return_output_agreement",
            path if path != "<candidate>" else "j.intent.intent_function.returns[*].value",
        )
    if "logical tensor handles must be dense" in lowered:
        return (
            "construction_handle_sequence",
            path if path != "<candidate>" else "j.intent.intent_function.inputs+outputs[*].name",
        )
    if "broadcast operand is not consumed" in lowered:
        return (
            "construction_broadcast_reference",
            path if path != "<candidate>" else "j.boundary.broadcasts",
        )
    if "broadcast axes" in lowered and "operand shape" in lowered:
        return (
            "construction_broadcast_shape",
            path if path != "<candidate>" else "j.boundary.broadcasts",
        )
    if "gather mode" in lowered or "gather axis" in lowered:
        return (
            "construction_gather_attrs",
            path if path != "<candidate>" else "j.intent.intent_function.body[*].attrs",
        )
    if "invalid dtype" in lowered and "iota" in lowered:
        return (
            "construction_iota_attrs",
            path if path != "<candidate>" else "j.intent.intent_function.body[*].attrs",
        )
    if "constexpr parameter positions" in lowered or (
        "bound non-c" in lowered and "constexpr" in lowered
    ):
        return (
            "construction_constexpr_parameter_kind",
            path if path != "<candidate>" else "j.signature.parameters[*].kind",
        )
    return fallback_rule_code, path


def _candidate_j_model_classification(detail: str) -> tuple[str, str]:
    """Classify private J-model text into a closed semantic contract family."""

    lowered = detail.casefold()
    # Keep the decoder's exception private, but do not collapse structurally
    # distinct boundary failures into one prompt-opaque bucket.  These rules
    # expose only the canonical row family/path; they never publish an
    # authored handle, value, row, or exception fragment.
    if "layer b references unknown j nodes" in lowered or (
        "boundaryaccess.node_ref" in lowered
        or "effectspec.node_ref" in lowered
    ):
        return (
            "candidate_decode_j_boundary_node_owner_rejected",
            "j.boundary.accesses[*].node_ref+j.boundary.effects[*].node_ref",
        )
    if "boundaryaccess" in lowered or "boundary.accesses" in lowered:
        return (
            "candidate_decode_j_boundary_access_contract_rejected",
            "j.boundary.accesses",
        )
    if "effectspec" in lowered or "boundary.effects" in lowered:
        return (
            "candidate_decode_j_boundary_effect_contract_rejected",
            "j.boundary.effects",
        )
    if "outputmapping" in lowered or "boundary.output_mappings" in lowered:
        return (
            "candidate_decode_j_boundary_output_contract_rejected",
            "j.boundary.output_mappings",
        )
    if any(
        token in lowered
        for token in (
            "boundary",
            "access",
            "effect",
            "broadcast",
            "reduction",
            "output_mapping",
            "resultrelation",
            "indexbounds",
            "logicalupdate",
            "consumer_extension",
        )
    ):
        return "candidate_decode_j_boundary_contract_rejected", "j.boundary"
    if "signature" in lowered or "parameter" in lowered:
        return "candidate_decode_j_signature_contract_rejected", "j.signature"
    if "scope" in lowered or "storage" in lowered:
        return "candidate_decode_j_scope_storage_contract_rejected", "j.scopes+j.storage"
    if "dimension" in lowered or "contractexpr" in lowered:
        return "candidate_decode_j_dimension_contract_rejected", "j.dimension_relations"
    if any(
        token in lowered
        for token in (
            "intentfunction",
            "intentop",
            "operation",
            "body",
            "operand",
            "result",
            "attrs",
            "value type",
            "map_",
            "gather",
            "scatter",
            "iota",
            "reduce",
            "loop",
            "conditional",
            "node_ids",
        )
    ):
        return "candidate_decode_j_operation_contract_rejected", "j.intent.intent_function.body"
    return "candidate_decode_j_model_rejected", "j"


def _candidate_gate1_model_component_failure(
    response_text: str,
    *,
    fallback_detail: str,
) -> tuple[str, str]:
    """Locate a rejected Gate1 model component without exporting its content.

    This repeats only strict parsing already performed by ``Gate1Artifact``.
    It neither normalizes nor repairs the payload; its result is a closed
    component category and a private detail used only for finite rule
    selection.  If every independently parseable component succeeds, the
    original failure is necessarily one of Gate1's artifact-level cross-field
    invariants.
    """

    try:
        payload = _json_object(response_text)
    except Exception:
        return "gate1_model", fallback_detail
    # ``SemanticArchive`` parses the closed Boundary row grammar before it
    # checks J-wide node, scope, signature, and typed-boundary invariants.
    # Replaying that same strict parser first gives a safe structural stage
    # when a provider row cannot even form Boundary.  No parsed row or
    # exception detail leaves this function; the caller emits only the fixed
    # stage rule/path below.
    j_value = payload.get("j")
    if not isinstance(j_value, Mapping):
        return "j_model", fallback_detail
    boundary_value = j_value.get("boundary")
    if not isinstance(boundary_value, Mapping):
        return "j_boundary_wire_model", fallback_detail
    try:
        Boundary.from_json(dict(boundary_value))
    except Exception:
        return "j_boundary_wire_model", fallback_detail
    for stage, value, parser in (
        ("pi_model", payload.get("pi"), RealizationWitness.from_json),
        ("j_model", payload.get("j"), SemanticArchive.from_json),
        ("c_model", payload.get("c"), ImplementationConfig.from_json),
    ):
        if not isinstance(value, Mapping):
            return stage, fallback_detail
        try:
            parser(dict(value))
        except Exception as exc:
            if (
                stage == "j_model"
                and isinstance(exc, SemanticArchiveCrossContractError)
            ):
                # SemanticArchive has already parsed the strict Boundary wire
                # model above. Its constructor identifies only the fixed
                # validator entry that rejected cross-contract closure; never
                # derive a class from a private message or authored values.
                return f"j_boundary_cross_contract_{exc.contract}", ""
            detail = str(exc)
            if (
                stage == "j_model"
                and _candidate_j_model_classification(detail)[0]
                == "candidate_decode_j_boundary_contract_rejected"
            ):
                # Boundary's closed row grammar already parsed above. The
                # remaining strict J failure is therefore in the fixed
                # SemanticArchive cross-contract that joins Boundary to the
                # operation graph, node map, scope, or typed relations. Keep
                # the original detail process-local; this stage is the only
                # exported fact.
                return "j_boundary_cross_contract", detail
            return stage, detail
    return "artifact_cross_contract", fallback_detail


def _candidate_decode_stage_diagnostic(
    exc: Exception,
    *,
    response_text: str | None = None,
) -> dict[str, Any]:
    """Project a decoder exception onto a finite, content-free rule receipt."""

    stage_rules = {
        "gate1_model": ("candidate_decode_gate1_model_rejected", "<candidate>"),
        "j_model": ("candidate_decode_j_model_rejected", "j"),
        "j_boundary_wire_model": (
            "candidate_decode_j_boundary_wire_model_rejected",
            "j.boundary",
        ),
        "j_boundary_cross_contract": (
            "candidate_decode_j_boundary_cross_contract_rejected",
            "j.boundary+j.intent+j.node_ids+j.scopes",
        ),
        "j_boundary_cross_contract_node_path": (
            "candidate_decode_j_boundary_node_path_cross_contract_rejected",
            "j.node_ids+j.intent.intent_function.body",
        ),
        "j_boundary_cross_contract_dimension": (
            "candidate_decode_j_boundary_dimension_cross_contract_rejected",
            "j.dimension_relations+j.intent",
        ),
        "j_boundary_cross_contract_signature": (
            "candidate_decode_j_boundary_signature_cross_contract_rejected",
            "j.signature+j.intent+j.boundary.aliases",
        ),
        "j_boundary_cross_contract_typed_boundary": (
            "candidate_decode_j_boundary_typed_relation_cross_contract_rejected",
            "j.boundary+j.intent+j.node_ids",
        ),
        "j_boundary_cross_contract_typed_boundary_operation": (
            "candidate_decode_j_boundary_typed_operation_cross_contract_rejected",
            "j.intent.intent_function.body+j.node_ids",
        ),
        "j_boundary_cross_contract_typed_boundary_index_bounds": (
            "candidate_decode_j_boundary_index_bounds_cross_contract_rejected",
            "j.boundary.index_bounds+j.intent+j.node_ids",
        ),
        "j_boundary_cross_contract_typed_boundary_index_bounds_binding": (
            "candidate_decode_j_boundary_index_bounds_binding_cross_contract_rejected",
            "j.boundary.index_bounds+j.intent+j.node_ids",
        ),
        "j_boundary_cross_contract_typed_boundary_index_bounds_coverage": (
            "candidate_decode_j_boundary_index_bounds_coverage_cross_contract_rejected",
            "j.boundary.index_bounds+j.intent.intent_function.body",
        ),
        "j_boundary_cross_contract_typed_boundary_updates": (
            "candidate_decode_j_boundary_logical_update_cross_contract_rejected",
            "j.boundary.updates+j.boundary.index_bounds+j.intent+j.node_ids",
        ),
        "j_boundary_cross_contract_typed_boundary_result_relations": (
            "candidate_decode_j_boundary_result_relation_cross_contract_rejected",
            "j.boundary.result_relations+j.boundary.updates+j.intent+j.node_ids",
        ),
        "j_boundary_cross_contract_access_guard": (
            "candidate_decode_j_boundary_access_guard_cross_contract_rejected",
            "j.boundary.access_guard_relations+j.intent.intent_function.body",
        ),
        "j_boundary_cross_contract_scope_storage": (
            "candidate_decode_j_boundary_scope_storage_cross_contract_rejected",
            "j.boundary+j.intent+j.node_ids+j.scopes+j.storage",
        ),
        "c_model": ("candidate_decode_c_model_rejected", "c"),
        "pi_model": ("candidate_decode_pi_model_rejected", "pi"),
        "artifact_cross_contract": (
            "candidate_decode_artifact_cross_contract_rejected",
            "j+c+pi",
        ),
        "j_canonicalization": ("candidate_decode_j_canonicalization_rejected", "j"),
        "wire_schema": ("candidate_decode_wire_schema_rejected", "schema"),
        "anonymous_contract": ("candidate_decode_anonymous_contract_rejected", "j"),
        "compiler_contract": ("candidate_decode_compiler_contract_rejected", "c"),
    }
    if isinstance(exc, _CandidateDecodeStageError):
        stage = exc.stage
        detail = exc.detail
        if stage == "gate1_model" and isinstance(response_text, str):
            stage, detail = _candidate_gate1_model_component_failure(
                response_text,
                fallback_detail=detail,
            )
        fallback_rule_code, fallback_path = stage_rules.get(
            stage,
            ("candidate_decode_internal_rejected", "<candidate>"),
        )
        if stage == "j_boundary_wire_model" or stage.startswith(
            "j_boundary_cross_contract"
        ):
            # The strict Boundary parser itself failed. Its stage is already
            # the most specific safe structural fact; do not reclassify its
            # private message and accidentally broaden it back to generic J.
            rule_code, path = fallback_rule_code, fallback_path
        else:
            rule_code, path = _candidate_decode_classification(
                detail,
                fallback_path,
                fallback_rule_code=fallback_rule_code,
            )
            if (
                stage == "j_model"
                and rule_code == "candidate_decode_j_model_rejected"
            ):
                rule_code, path = _candidate_j_model_classification(detail)
        return {
            "rule_code": rule_code,
            "path": path,
            "observed_refs_or_types": [
                "strict decoder rejected this closed structural stage; "
                "candidate content is withheld"
            ],
        }
    return {
        "rule_code": "candidate_decode_internal_rejected",
        "path": "<candidate>",
        "observed_refs_or_types": [
            "strict decoder raised an unclassified internal rejection; "
            "candidate content is withheld"
        ],
    }


def _candidate_decode_diagnostics(
    exc: Exception,
    *,
    response_text: str | None = None,
) -> list[dict[str, Any]]:
    """Expose independent decoder errors with actionable contract classes."""

    if not isinstance(exc, SchemaError) or not exc.issues:
        return [
            _candidate_decode_stage_diagnostic(
                exc,
                response_text=response_text,
            )
        ]

    diagnostics: list[dict[str, Any]] = []
    for issue in exc.issues[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = issue.path
        if path == "LogicalSignature" or path.startswith("LogicalSignature."):
            path = "j.signature" + path.removeprefix("LogicalSignature")
        elif path == "SemanticArchive" or path.startswith("SemanticArchive."):
            path = "j" + path.removeprefix("SemanticArchive")
        rule_code, classified_path = _candidate_decode_classification(
            issue.detail,
            path,
            fallback_rule_code="candidate_decode_gate1_model_rejected",
        )
        diagnostics.append(
            {
                "rule_code": rule_code,
                "path": _bounded_candidate_diagnostic_text(classified_path),
                "observed_refs_or_types": [
                    "strict decoder rejected this closed structural field; "
                    "candidate content is withheld"
                ],
            }
        )
    return diagnostics


def _bounded_candidate_diagnostic_text(value: Any) -> str:
    """Keep source-free CandidateCheck diagnostics closed and compact."""

    text = str(value)
    marker = "...<truncated>"
    if len(text) <= _CANDIDATE_DIAGNOSTIC_TEXT_LIMIT:
        return text
    return text[: _CANDIDATE_DIAGNOSTIC_TEXT_LIMIT - len(marker)] + marker


def _response_usage(raw: Mapping[str, Any]) -> dict[str, int]:
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    return {
        key: value
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance((value := usage.get(key)), int)
        and not isinstance(value, bool)
        and value >= 0
    }


def _validate_compiler_view(
    artifact: Gate1Artifact,
    anonymous_view: Mapping[str, Any] | None,
) -> None:
    if anonymous_view is None:
        return
    bound_constexprs = set(anonymous_view["bound_constexpr_values"])
    slots = {item.name: item for item in artifact.c.slots}
    misplaced_bound_constexprs = sorted(bound_constexprs & set(slots))
    if misplaced_bound_constexprs:
        raise ValueError(
            "Lift moved bound non-C constexpr carrier(s) into pure C: "
            + ", ".join(misplaced_bound_constexprs)
        )
    if anonymous_view["parameter_order"]:
        expected_p = [
            name for name in anonymous_view["parameter_order"]
            if name.startswith("p")
        ]
        expected_c = [
            name for name in anonymous_view["parameter_order"]
            if name.startswith("c")
        ]
        actual_p = [item.name for item in artifact.j.signature.parameters]
        actual_c = [
            item.name for item in artifact.c.slots
            if item.kind is ScheduleSlotKind.CONSTEXPR
        ]
        if actual_p != expected_p or actual_c != expected_c:
            raise ValueError("Lift changed Harness-declared logical/C handles")
    if artifact.c.values != anonymous_view["schedule_values"]:
        raise ValueError("Lift changed trusted selected-frontend C values")
    actual_compiler_slots = [
        item.name for item in artifact.c.slots
        if item.kind is ScheduleSlotKind.COMPILER
    ]
    if actual_compiler_slots != anonymous_view["compiler_slots"]:
        raise ValueError("Lift changed trusted compiler slots")
    compiler_slot_set = set(anonymous_view["compiler_slots"])
    for name, role in anonymous_view["compiler_roles"].items():
        slot = slots.get(name)
        if (
            slot is None
            or slot.kind
            is not (
                ScheduleSlotKind.COMPILER
                if name in compiler_slot_set
                else ScheduleSlotKind.CONSTEXPR
            )
            or slot.role != role
        ):
            raise ValueError("Lift changed a trusted C schedule role")
    if [item.to_json() for item in artifact.j.boundary.aliases] != anonymous_view["aliases"]:
        raise ValueError("Lift changed trusted alias relation")
    expected_constexprs = set(anonymous_view["constexpr_parameters"])
    actual_constexprs = {
        item.name for item in artifact.j.signature.parameters
        if item.kind is ParameterKind.CONSTEXPR
    } | {
        item.name for item in artifact.c.slots
        if item.kind is ScheduleSlotKind.CONSTEXPR
    }
    if actual_constexprs != expected_constexprs:
        raise ValueError("Lift changed constexpr parameter positions")
    for name in expected_constexprs:
        slot = slots.get(name)
        if name in bound_constexprs:
            # A misplaced bound carrier was rejected above before the generic
            # handle-order mismatch so CandidateCheck can report the real cause.
            assert slot is None
        elif slot is None or slot.kind is not ScheduleSlotKind.CONSTEXPR:
            raise ValueError("Lift omitted a pure-C constexpr slot")
    signature = {item.name: item for item in artifact.j.signature.parameters}
    def carrier_dtype_matches(value: Any, dtype: str | None) -> bool:
        canonical = _canonical_dtype(dtype)
        if isinstance(value, bool):
            return canonical == "i1"
        if isinstance(value, int):
            return canonical in {"int", "index"} or bool(
                canonical and re.fullmatch(r"[iu][1-9][0-9]*", canonical)
            )
        if isinstance(value, float):
            return canonical in {"bf16", "f16", "f32", "f64", "float", "double"}
        return isinstance(value, str) and canonical in {"str", "string"}

    bound_constexpr_shape_coordinates = anonymous_view.get(
        "bound_constexpr_shape_coordinates", {}
    )
    for name, value in anonymous_view["bound_constexpr_values"].items():
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a bound semantic constexpr")
        if isinstance(value, str) and parameter.logical_ref is not None:
            raise ValueError(
                "A string specialization carrier cannot name a J logical value"
            )
        if parameter.logical_ref is None and not carrier_dtype_matches(
            value, parameter.dtype
        ):
            raise ValueError(
                "Harness specialization value type differs from its constexpr "
                "carrier dtype"
            )
    j_tensors = {
        item.name: item
        for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    dimensions: dict[str, int] = {}
    logical_values: dict[str, Any] = {}

    def bind_dimension(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("J symbolic dimension requires an integer valuation")
        previous = dimensions.setdefault(name, value)
        if previous != value:
            raise ValueError(f"J symbolic extent {name!r} is inconsistent")

    coordinate_to_dimension: dict[str, str] = {}
    dimension_to_coordinate: dict[str, str] = {}

    def bind_shape(
        declared: list[int | str],
        actual: list[Any],
        contract: list[int | str] | None,
        *,
        parameter_name: str,
    ) -> None:
        if len(declared) != len(actual):
            raise ValueError("J tensor rank differs from trusted compiler view")
        if contract is not None and len(contract) != len(actual):
            raise ValueError(
                "trusted shape contract rank differs from its concrete valuation"
            )
        contracts = contract if contract is not None else [None] * len(actual)
        for axis, (expected, extent, coordinate) in enumerate(
            zip(declared, actual, contracts, strict=True)
        ):
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise ValueError("trusted tensor extent must be an integer")
            if isinstance(coordinate, str):
                if not isinstance(expected, str):
                    raise ValueError(
                        "Lift specialized source-grounded shape coordinate "
                        f"{coordinate!r} at {parameter_name}[{axis}] to a literal"
                    )
                previous_dimension = coordinate_to_dimension.setdefault(
                    coordinate, expected
                )
                if previous_dimension != expected:
                    raise ValueError(
                        "Lift mapped one source-grounded shape coordinate to "
                        "multiple J dimensions"
                    )
                previous_coordinate = dimension_to_coordinate.setdefault(
                    expected, coordinate
                )
                if previous_coordinate != coordinate:
                    raise ValueError(
                        "Lift collapsed distinct source-grounded shape coordinates "
                        "into one J dimension"
                    )
            elif isinstance(coordinate, int) and expected != coordinate:
                raise ValueError(
                    "Lift parameterized a Harness-declared static shape extent "
                    f"at {parameter_name}[{axis}]"
                )
            if isinstance(expected, int):
                if expected != extent:
                    raise ValueError(
                        "J static tensor extent differs from trusted compiler view"
                    )
            else:
                bind_dimension(expected, extent)

    for argument in anonymous_view["arguments"]:
        if argument["name"].startswith("c"):
            if argument["name"] not in slots:
                raise ValueError("Lift omitted a trusted C argument")
            continue
        parameter = signature.get(argument["name"])
        if parameter is None:
            raise ValueError("Lift omitted a trusted compiler argument")
        expected_kind = {
            "tensor": ParameterKind.POINTER,
            "scalar": ParameterKind.SCALAR,
            "opaque": ParameterKind.OPAQUE,
            "none": ParameterKind.OPAQUE,
        }[argument["kind"]]
        allowed_kinds = {expected_kind}
        if expected_kind is ParameterKind.POINTER:
            allowed_kinds.add(ParameterKind.TENSOR)
        if parameter.kind not in allowed_kinds:
            raise ValueError("Lift changed a trusted compiler argument kind")
        if _canonical_dtype(parameter.dtype) != _canonical_dtype(argument["dtype"]):
            raise ValueError("Lift changed a trusted compiler argument dtype")
        if argument["kind"] == "tensor" and parameter.rank != len(argument["shape"]):
            raise ValueError("Lift changed a trusted compiler argument rank")
        if argument["kind"] == "tensor" and parameter.logical_ref in j_tensors:
            bind_shape(
                j_tensors[parameter.logical_ref].shape,
                argument["shape"],
                argument.get("shape_contract"),
                parameter_name=argument["name"],
            )
        if (
            argument["kind"] == "scalar"
            and isinstance(parameter.logical_ref, str)
            and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
        ):
            bind_dimension(parameter.logical_ref, argument.get("value"))
        if (
            argument["kind"] == "scalar"
            and isinstance(parameter.logical_ref, str)
            and parameter.logical_ref in j_tensors
            and not j_tensors[parameter.logical_ref].shape
        ):
            logical_values[parameter.logical_ref] = argument.get("value")
        expected_mode = argument.get("mode")
        if expected_mode is not None and parameter.mode.value != expected_mode:
            raise ValueError("Lift changed a trusted compiler argument mode")
    for name, value in anonymous_view["bound_constexpr_values"].items():
        parameter = signature.get(name)
        if (
            parameter is not None
            and isinstance(parameter.logical_ref, str)
            and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
        ):
            bind_dimension(parameter.logical_ref, value)
        if (
            parameter is not None
            and isinstance(parameter.logical_ref, str)
            and parameter.logical_ref in j_tensors
            and not j_tensors[parameter.logical_ref].shape
        ):
            logical_values[parameter.logical_ref] = value
        coordinate = bound_constexpr_shape_coordinates.get(name)
        if coordinate is None:
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a bound-constexpr source-grounded shape coordinate"
            )
        if parameter is not None and parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref if parameter is not None else None
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and any(
                relation.value_ref == logical_ref
                and relation.dimension_ref == dimension
                and relation.relation == "eq"
                for relation in artifact.j.value_dimension_relations
            )
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness bound-constexpr-to-shape-coordinate equality"
        )

    expected_shape_coordinates = set(
        anonymous_view.get("shape_coordinates", {})
    )
    if set(coordinate_to_dimension) != expected_shape_coordinates:
        missing = sorted(expected_shape_coordinates - set(coordinate_to_dimension))
        raise ValueError(
            "Lift omitted source-grounded shape coordinate(s) from J tensor shapes: "
            + ", ".join(missing)
        )

    for argument in anonymous_view["arguments"]:
        coordinate = argument.get("shape_coordinate")
        if not isinstance(coordinate, str):
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a scalar-bound source-grounded shape coordinate"
            )
        parameter = signature[argument["name"]]
        if parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and any(
                relation.value_ref == logical_ref
                and relation.dimension_ref == dimension
                and relation.relation == "eq"
                for relation in artifact.j.value_dimension_relations
            )
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness scalar-to-shape-coordinate equality"
        )

    try:
        validate_dimension_instantiation(artifact.j, dimensions)
        validate_value_dimension_instantiation(
            artifact.j, dimensions, logical_values
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    # Grid/source positions are Harness coordinates.  They are retained in the
    # extraction sidecar and deliberately absent from the proof artifact.


def _validate_anonymous_lift_core(
    artifact: Gate1Artifact,
    expected_parameters: tuple[str, ...],
    expected_tensors: tuple[str, ...],
    expected_dimensions: tuple[str, ...],
    *,
    allocator: CanonicalHandleAllocator,
) -> None:
    if not isinstance(allocator, CanonicalHandleAllocator):
        raise TypeError("Lift decode requires the canonical namespace allocator")
    if tuple(expected_parameters) != allocator.declared("p"):
        raise ValueError("Lift decode parameter expectation differs from allocator")
    if tuple(expected_tensors) != allocator.declared("t"):
        raise ValueError("Lift decode tensor expectation differs from allocator")
    if tuple(expected_dimensions) != allocator.declared("x"):
        raise ValueError("Lift decode dimension expectation differs from allocator")
    if artifact.j.kernel_id != "scrc_k0":
        raise ValueError("Lift kernel id must use scrc_k0")
    if artifact.j.intent.name != "scrc_j0":
        raise ValueError("Lift IntentFunction name must use scrc_j0")
    tensors = [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    # One logical inout tensor is deliberately declared in both collections.
    # Tensor ids therefore advance on first declaration, not on collection
    # entries.  SemanticArchive already rejects duplicate tensors within one
    # collection and non-identical input/output contracts.
    actual_tensors = list(dict.fromkeys(item.name for item in tensors))
    if actual_tensors != list(expected_tensors):
        raise ValueError(
            "J tensors must use the SourceFacts-owned first-declaration order "
            f"(expected {list(expected_tensors)}, got {actual_tensors})"
        )
    if list(artifact.j.node_ids) != list(allocator.declared("n")):
        raise ValueError("J node ids must use declaration-order handles n0, n1, ...")
    if [item.name for item in artifact.j.signature.parameters] != list(
        expected_parameters
    ):
        raise ValueError("J.signature parameters differ from anonymous source handles")
    dimensions: list[str] = []
    for tensor in tensors:
        for dim in tensor.shape:
            if isinstance(dim, str) and dim not in dimensions:
                dimensions.append(dim)
    if dimensions != list(expected_dimensions):
        raise ValueError(
            "J dimensions must use the SourceFacts-owned declaration order "
            f"(expected {list(expected_dimensions)}, got {dimensions})"
        )
    from intentir.ir.canonical_j import j_elaboration

    canonical_scope_records = j_elaboration(artifact.j).construction_records.scope_records
    for prefix, values in (
        ("d", list(artifact.j.boundary.domains)),
        ("a", [item.access_id for item in artifact.j.boundary.accesses]),
        ("e", [item.effect_id for item in artifact.j.boundary.effects]),
        ("q", [item["scope_id"] for item in canonical_scope_records]),
        ("b", [item.storage_id for item in artifact.j.storage]),
    ):
        if values != list(allocator.declared(prefix)):
            raise ValueError(f"J {prefix} handles are not declaration ordered")
    actual_c = [item.name for item in artifact.c.slots]
    if actual_c != list(allocator.declared("c")):
        raise ValueError("pure C slots must use c0, c1, ...")


def _validate_source_identifier_privacy(
    artifact: Gate1Artifact,
    binding: SCRCHarnessBinding,
) -> None:
    """Apply the privileged source-name leak check only in the formal decoder."""

    visible = {
        "j": artifact.j.to_json(),
        "c": artifact.c.to_json(),
        "pi": artifact.pi.to_json(),
    }
    visible_strings: list[tuple[str, str]] = []
    literal_fields = {
        "dtype", "fn", "frontend", "kind", "mechanism", "mode", "op",
        "placement", "producer", "role", "subscripts", "value_type",
    }

    def collect_strings(
        value: Any,
        *,
        path: str,
        field_name: str | None = None,
    ) -> None:
        if isinstance(value, str):
            if field_name not in literal_fields:
                visible_strings.append((path, value))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect_strings(
                    item,
                    path=f"{path}.{key}" if path else str(key),
                    field_name=key,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect_strings(
                    item,
                    path=f"{path}[{index}]",
                    field_name=field_name,
                )

    collect_strings(visible, path="")
    for source_name, opaque in binding.source_to_opaque:
        if source_name == opaque:
            continue
        # These are protocol-public, declaration-ordered handles/binders.  A
        # private Source local may coincidentally have the same spelling, but
        # seeing that spelling in canonical J/C/pi carries no Source identity.
        if re.fullmatch(
            r"(?:scrc_[jk][0-9]+|[abcdeiknpqrtuvx][0-9]+)", source_name
        ):
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])"
        for path, item in visible_strings:
            if re.search(pattern, item):
                raise SourceIdentifierPrivacyError(path)


def extract_scrc_artifact(
    source: str,
    name_hint: str | None = None,
    *,
    provider: ProviderConfig | None = None,
    compiler_view: SCRCCompilerView | None = None,
    effort: str | None = None,
    relift_context: SCRCReliftContext | None = None,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    source_import_roots: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
    frontend: str | None = "triton",
    source_facts: SourceFacts | None = None,
    construction: CanonicalConstruction | None = None,
) -> SCRCExtraction:
    """Perform exactly one source-aware provider call for one Lift attempt."""

    del name_hint
    if not isinstance(source_facts, SourceFacts):
        raise TypeError("Responses Lift requires the owning canonical SourceFacts")
    resolved_provider = provider or get_provider()
    resolved_effort = effort or resolved_provider.reasoning_effort
    resolved_timeout = resolve_source_aware_lift_timeout(
        resolved_effort,
        timeout_seconds,
    )
    request = prepare_scrc_lift(
        source,
        compiler_view=compiler_view,
        relift_context=relift_context,
        source_symbol=source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        source_import_roots=source_import_roots,
        frontend=frontend,
        source_facts=source_facts,
        construction=construction,
        authoring_transport="responses",
    )
    # Compile/observe anonymous Source before the one provider turn.  The
    # resulting occurrence ledger is then serialized in the direct Responses
    # canonical seed, rather than being discovered only after an already
    # authored candidate reaches ingress repair.
    from .compiler_observation import source_preparation_failure_reason

    source_observation = _source_ttir_observation_for_request(request)
    observed_construction = _construction_with_source_observation(
        request,
        source_observation,
    )
    if observed_construction.transfer_ledger.status != "pass":
        # The observer diagnostic may contain Source or host paths. Carry only
        # its closed rule code, or a deterministic status class, to the public
        # lifecycle receipt.
        observation_rule = source_preparation_failure_reason(
            source_observation, observed_construction.transfer_ledger,
        )
        attempt = _SCRCLiftFailureReceipt(
            diagnostic=(
                "Source preparation did not produce a closed transfer ledger: "
                f"{observed_construction.transfer_ledger.reason}"
            ),
            response_text="",
            attempts=0,
            termination_status="source_preparation_transfer_ledger_unavailable",
            source_failure_reason=observation_rule,
        )
        raise SCRCLiftError(attempt)
    authoring_blockers = _source_authoring_blockers(observed_construction)
    if authoring_blockers:
        attempt = _SCRCLiftFailureReceipt(
            diagnostic="Source preparation lacks a required exact owner or predicate authorization",
            response_text="",
            attempts=0,
            termination_status="source_preparation_authorization_unavailable",
            source_failure_reason=authoring_blockers[0]["rule_code"],
        )
        raise SCRCLiftError(attempt)
    if observed_construction is not request.construction:
        request = prepare_scrc_lift(
            source,
            compiler_view=compiler_view,
            relift_context=relift_context,
            source_symbol=source_symbol,
            source_kind=source_kind,
            source_parameter_order=source_parameter_order,
            source_to_opaque=source_to_opaque,
            source_import_roots=source_import_roots,
            frontend=frontend,
            source_facts=source_facts,
            construction=observed_construction,
            authoring_transport="responses",
        )
    instructions = request.instructions
    user = request.user_input
    prompt_sha = request.prompt_sha256
    try:
        reply = call_responses(
            instructions,
            user,
            provider=resolved_provider,
            effort=resolved_effort,
            # A real SCRC lifecycle treats overload/rate/transport/HTTP
            # failures as terminal infrastructure stops. Do not hide one
            # behind an in-place transport retry; another Lift call is legal
            # only after a completed candidate rejection.
            max_retries=1,
            # The full artifact schema is not a provider authoring schema: it
            # contains host-owned schema/C/ABI/handle/scope rows.  The compact
            # Source-sized schema is projected from that same canonical wire
            # and fixes the exact occurrence/output arity for this request.
            # CandidateCheck remains the local acceptance authority after the
            # host constructor has materialized the complete artifact.
            text_format=scrc_semantic_candidate_text_format(
                request.construction
            ),
            max_output_tokens=SCRC_LIFT_MAX_OUTPUT_TOKENS,
            timeout=resolved_timeout,
        )
    except ProviderError as exc:
        attempt = _SCRCLiftFailureReceipt(
            diagnostic=f"{type(exc).__name__}: {exc}",
            response_text="",
            attempts=getattr(exc, "attempts", 1),
            termination_status=getattr(exc, "failure_class", None),
            provider_failure_origin=getattr(exc, "failure_origin", None),
            provider_error=getattr(exc, "safe_error_envelope", None),
        )
        raise SCRCLiftError(
            attempt,
            candidate_delta=_candidate_delta_from_failure_receipt(attempt),
        ) from exc
    try:
        _json_object(reply.text)
    except Exception as exc:
        # A completed Responses request that does not contain one JSON object
        # has not reached semantic ingress at all.  It must terminate as a
        # provider-response preparation failure, rather than consuming a
        # relift turn with a misleading J/C/pi schema diagnostic.  Do not
        # retain the unparseable text as a repair draft.
        attempt = _SCRCLiftFailureReceipt(
            diagnostic="provider returned a malformed semantic candidate",
            response_text="",
            attempts=reply.attempts,
            termination_status="provider_malformed_candidate",
        )
        raise SCRCLiftError(attempt) from exc
    try:
        candidate_text, deterministic_repairs = _construct_scrc_candidate(
            reply.text,
            request,
            observation=source_observation,
        )
    except CanonicalIngressError as exc:
        diagnostics = list(exc.candidate_check_diagnostics)
        if not diagnostics:
            # An empty diagnostic is an internal classification defect, not a
            # reason to claim that the provider omitted host-owned c/schema.
            # Keep the lifecycle fail-closed, but expose the stable semantic
            # ingress class so the next fresh authoring turn can distinguish
            # it from CandidateCheck, TypeCheck, Replay, or Gate failures.
            diagnostics = [
                {
                    "rule_code": "canonical_constructor_unclassified",
                    "path": "<canonical-constructor>",
                    "observed_refs_or_types": [
                        "expected:closed_semantic_j_surface"
                    ],
                }
            ]
        candidate_check = {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": diagnostics,
        }
        attempt = _SCRCLiftFailureReceipt(
            diagnostic="canonical semantic ingress rejected candidate",
            response_text=reply.text,
            attempts=reply.attempts,
            candidate_check=candidate_check,
        )
        raise SCRCLiftError(
            attempt,
            candidate_delta=_candidate_delta_from_failure_receipt(attempt),
        ) from exc
    candidate_check_context = scrc_candidate_check_context(request)
    try:
        candidate_check = check_scrc_lift_candidate(
            candidate_text,
            candidate_check_context,
        )
    except Exception as exc:
        candidate_check = {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_check_failed",
                    "path": "<candidate>",
                    "observed_refs_or_types": [
                        _bounded_candidate_diagnostic_text(
                            f"{type(exc).__name__}: {exc}"
                        )
                    ],
                }
            ],
        }
    if candidate_check.get("status") != "pass":
        attempt = _SCRCLiftFailureReceipt(
            diagnostic="CandidateCheck rejected candidate",
            response_text=candidate_text,
            attempts=reply.attempts,
            candidate_check=candidate_check,
        )
        raise SCRCLiftError(
            attempt,
            candidate_delta=_candidate_delta_from_failure_receipt(attempt),
        )
    try:
        artifact = decode_scrc_lift_artifact(candidate_text, request)
    except Exception as exc:
        attempt = _SCRCLiftFailureReceipt(
            diagnostic=f"{type(exc).__name__}: {exc}",
            response_text=candidate_text,
            attempts=reply.attempts,
            candidate_check=candidate_check,
        )
        raise SCRCLiftError(
            attempt,
            candidate_delta=_candidate_delta_from_failure_receipt(attempt),
        ) from exc
    source_reconcile = _reconcile_source_candidate(
        candidate_text,
        request,
        observation=source_observation,
    )
    if source_reconcile.get("status") != "pass":
        attempt = _SCRCLiftFailureReceipt(
            diagnostic="source-aware SourceReconcile rejected candidate",
            response_text=candidate_text,
            attempts=reply.attempts,
            candidate_check=candidate_check,
            source_reconcile=source_reconcile,
            termination_status="source_reconcile_rejected",
        )
        raise SCRCLiftError(
            attempt,
            candidate_delta=_candidate_delta_from_failure_receipt(attempt),
        )
    provenance = SCRCProvenance(
        model=reply.model,
        prompt_sha256=prompt_sha,
        response_id=reply.raw.get("id"),
        response_model=(
            reply.raw.get("model")
            if isinstance(reply.raw.get("model"), str) else None
        ),
        usage=_response_usage(reply.raw),
        request_sha256=reply.request_sha256,
        response_sha256=hashlib.sha256(candidate_text.encode()).hexdigest(),
        raw_response_sha256=reply.raw_response_sha256,
        response_text=candidate_text,
        raw_response=reply.raw_response,
        attempts=reply.attempts,
        elapsed_ms=reply.elapsed_ms,
        deterministic_repairs=deterministic_repairs,
    )
    return SCRCExtraction(
        artifact,
        provenance,
        request.harness_binding,
        source,
        request.construction,
    )


@dataclass
class ResponsesSCRCRelifter:
    provider: ProviderConfig | None = None
    compiler_view: SCRCCompilerView | None = None
    effort: str | None = None
    source_symbol: str | None = None
    source_kind: str = "python"
    source_parameter_order: tuple[str, ...] = ()
    source_to_opaque: Mapping[str, str] | None = None
    source_import_roots: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    frontend: str | None = "triton"
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)
    _open_guidance: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _resolved_guidance: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple, init=False, repr=False
    )

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def relift(
        self,
        previous: SCRCExtraction,
        *,
        failure_kind: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> SCRCExtraction:
        round_index = len(self._records) + 1
        diagnostic_payload = (
            dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        )
        open_guidance = _merge_guidance_rows(
            self._open_guidance,
            diagnostic_payload.get("open_guidance", ()),
        )
        resolved_guidance = _merge_guidance_rows(
            self._resolved_guidance,
            diagnostic_payload.get("resolved_guidance", ()),
        )
        context = scrc_relift_context_from_feedback(
            {
                "kind": failure_kind,
                "diagnostics": diagnostic_payload,
                "open_guidance": open_guidance,
                "resolved_guidance": resolved_guidance,
                "candidate_delta": diagnostic_payload.get("candidate_delta"),
            },
            relift_round=round_index,
        )
        if context is None:
            raise TypeError("Responses relift requires explicit feedback")
        self._open_guidance = context.open_guidance
        self._resolved_guidance = context.resolved_guidance
        try:
            extraction = extract_scrc_artifact(
                previous.source,
                provider=self.provider,
                compiler_view=self.compiler_view,
                effort=self.effort,
                relift_context=context,
                source_symbol=self.source_symbol,
                source_kind=self.source_kind,
                source_parameter_order=self.source_parameter_order,
                source_to_opaque=self.source_to_opaque,
                source_import_roots=self.source_import_roots,
                timeout_seconds=self.timeout_seconds,
                frontend=self.frontend,
                source_facts=previous.construction.source_facts,
                construction=previous.construction,
            )
        except SCRCLiftError as exc:
            if exc.source_failure_class is not None:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    "Source preparation failed before provider Lift",
                    evidence={
                        "failure_class": "source_preparation",
                        "source_failure_class": exc.source_failure_class,
                        **(
                            {"source_failure_reason": exc.source_failure_reason}
                            if exc.source_failure_reason is not None
                            else {}
                        ),
                        "candidate_present": False,
                    },
                ) from exc
            if exc.provider_failure_class is not None:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    "Lift provider preparation failed before CandidateCheck",
                    evidence={
                        "failure_class": "lift_provider_preparation",
                        "provider_failure_class": exc.provider_failure_class,
                        **(
                            {"provider_failure_origin": exc.provider_failure_origin}
                            if exc.provider_failure_origin is not None
                            else {}
                        ),
                        **(
                            {"provider_error": dict(exc.provider_error)}
                            if isinstance(exc.provider_error, Mapping)
                            and exc.provider_error
                            else {}
                        ),
                        "candidate_present": False,
                    },
                ) from exc
            backend_evidence: dict[str, Any] = {}
            if exc.source_reconcile is not None:
                backend_evidence["source_reconcile"] = dict(exc.source_reconcile)
            if exc.candidate_check is not None:
                backend_evidence["candidate_check"] = dict(exc.candidate_check)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence=backend_evidence or None,
                )
            )
            raise
        except Exception as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        self._records.append(SCRCReliftRecord(round_index, extraction=extraction))
        return extraction


@dataclass
class ResponsesSCRCLiftAdapter:
    """Source-aware provider adapter for the unified SCRC lifecycle."""

    provider: ProviderConfig | None = None
    compiler_view: SCRCCompilerView | None = None
    effort: str | None = None
    source_symbol: str | None = None
    source_kind: str = "python"
    source_parameter_order: tuple[str, ...] = ()
    source_to_opaque: Mapping[str, str] | None = None
    source_import_roots: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    frontend: str | None = "triton"
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Responses SCRC Lift requires SourceProgram.text")
        source_kind = getattr(
            source_program, "lift_source_kind", self.source_kind
        )
        source_parameter_order = tuple(
            getattr(
                source_program,
                "lift_parameter_order",
                self.source_parameter_order,
            )
        )
        source_mapping = getattr(
            source_program,
            "lift_source_to_opaque",
            self.source_to_opaque,
        )
        source_to_opaque = (
            dict(source_mapping) if source_mapping is not None else None
        )
        source_symbol = getattr(
            source_program, "lift_source_symbol", self.source_symbol
        )
        gate_source = getattr(source_program, "gate_source", None)
        source_import_roots = tuple(self.source_import_roots)
        if not source_import_roots:
            source_import_roots = tuple(
                str(root)
                for root in getattr(gate_source, "import_roots", ())
            )
        round_index = len(self._records) + 1
        context = scrc_relift_context_from_feedback(
            feedback,
            relift_round=round_index,
        )
        provider = self.provider or get_provider()
        try:
            extraction = extract_scrc_artifact(
                source,
                provider=provider,
                compiler_view=self.compiler_view,
                effort=self.effort,
                relift_context=context,
                source_symbol=source_symbol,
                source_kind=source_kind,
                source_parameter_order=source_parameter_order,
                source_to_opaque=source_to_opaque,
                source_import_roots=source_import_roots,
                timeout_seconds=self.timeout_seconds,
                frontend=getattr(request, "frontend", self.frontend),
                source_facts=getattr(request, "source_facts", None),
                construction=getattr(request, "construction", None),
            )
        except SCRCLiftError as exc:
            if exc.source_failure_class is not None:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    "Source preparation failed before provider Lift",
                    evidence={
                        "failure_class": "source_preparation",
                        "source_failure_class": exc.source_failure_class,
                        **(
                            {"source_failure_reason": exc.source_failure_reason}
                            if exc.source_failure_reason is not None
                            else {}
                        ),
                        "candidate_present": False,
                    },
                ) from exc
            provider_failure = exc.provider_failure_class
            if provider_failure is not None:
                from intentir.verify.scrc_runner import SCRCInfrastructureError

                raise SCRCInfrastructureError(
                    "Lift provider preparation failed before CandidateCheck",
                    evidence={
                        "attempted_lift_call": round_index,
                        "lift_generation_failures": [
                            {
                                "rule_code": "lift_provider_generation_failed",
                                "path": "<lift-provider>",
                                "provider_failure_class": provider_failure,
                            **(
                                {
                                    "provider_failure_origin": (
                                        exc.provider_failure_origin
                                    )
                                }
                                if exc.provider_failure_origin is not None
                                else {}
                            ),
                                **(
                                    {"provider_error": dict(exc.provider_error)}
                                    if isinstance(exc.provider_error, Mapping)
                                    and exc.provider_error
                                    else {}
                                ),
                                "physical_provider_calls": exc.physical_calls,
                            }
                        ],
                        "candidate_present": False,
                    },
                ) from exc
            diagnostic = exc.diagnostic
            backend_evidence: dict[str, Any] = {}
            if exc.source_reconcile is not None:
                backend_evidence["source_reconcile"] = dict(exc.source_reconcile)
            if exc.candidate_check is not None:
                backend_evidence["candidate_check"] = dict(exc.candidate_check)
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    backend_evidence=backend_evidence or None,
                )
            )
            # Local import avoids making extraction depend on the verify package
            # during module initialization.
            from intentir.verify.scrc_runner import CandidateLiftError

            error_diagnostics: dict[str, Any] = {
                "diagnostic": diagnostic,
                "failure_class": "lift_response",
            }
            if exc.source_reconcile is not None:
                error_diagnostics["source_reconcile"] = dict(exc.source_reconcile)
            if exc.candidate_check is not None:
                error_diagnostics["candidate_check"] = dict(exc.candidate_check)
            raise CandidateLiftError(
                error_diagnostics,
                physical_calls=exc.physical_calls,
                candidate_delta=exc.candidate_delta,
            ) from exc
        self._records.append(
            SCRCReliftRecord(round_index, extraction=extraction)
        )
        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            extraction.artifact,
            physical_calls=extraction.provenance.attempts,
            metadata={
                "model": extraction.provenance.model,
                "response_model": extraction.provenance.response_model,
                "reasoning_effort": self.effort or provider.reasoning_effort,
                "response_id": extraction.provenance.response_id,
                "usage": dict(extraction.provenance.usage),
                "elapsed_ms": extraction.provenance.elapsed_ms,
                "artifact_schema": extraction.artifact.wire_schema,
                "decode_protocol": SCRC_DECODE_PROTOCOL,
                "typecheck_protocol": TYPECHECK_PROTOCOL,
                "deterministic_repairs": [
                    dict(item)
                    for item in extraction.provenance.deterministic_repairs
                    if isinstance(item, Mapping)
                ],
            },
        )


__all__ = [
    "ResponsesSCRCLiftAdapter", "ResponsesSCRCRelifter",
    "CANDIDATE_CHECK_PROTOCOL", "SCRC_DECODE_PROTOCOL", "SCRC_RELIFT_PROTOCOL",
    "SCRCCandidateCheckContext", "SCRCCompilerArgument",
    "SCRCCompilerView", "SCRCExtraction", "SCRCHarnessBinding",
    "SCRCLiftRequest", "SCRCProvenance", "SCRCReliftContext",
    "scrc_relift_context_from_feedback",
    "SourceIdentifierPrivacyError",
    "SCRCReliftRecord", "anonymize_native_source", "anonymize_source",
    "build_scrc_lift_instructions", "canonical_scalar_binding_dtype",
    "check_scrc_lift_candidate", "decode_scrc_lift_artifact", "extract_scrc_artifact", "prepare_scrc_lift", "scrc_candidate_check_context", "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
    "normalize_scrc_candidate_delta",
    "scrc_candidate_delta_receipt",
]
