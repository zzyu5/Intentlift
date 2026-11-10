"""Codex CLI Lift backend with external SCRC verification boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLISubmissionRejected,
    CodexCLIError,
    CodexCLISubmissionAccepted,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.provider import get_provider
from intentir.verify.scrc_runner import CandidateLiftError
from .miniswe import MiniSWELiftAdapter, _input_files, _trajectory_bytes
from .compiler_observation import (
    SOURCE_TTIR_RECONCILE_PROTOCOL,
    reconcile_candidate_source_ttir,
)
from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCProvenance,
    SCRCReliftRecord,
    SourceIdentifierPrivacyError,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
    _construct_scrc_candidate,
    _source_ttir_observation_for_request,
    prepare_scrc_lift,
    scrc_candidate_delta_receipt,
    scrc_candidate_check_context,
)
from .scrc_schema import (
    canonical_wire_bundle,
)
CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v41"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v101"

_FEEDBACK_CATEGORY_NAMES = (
    "interface",
    "predicate_effect",
    "op_multiset",
    "dataflow",
    "reduction",
)

_INITIAL_PROMPT = """Own one fresh isolated IntentIR Lift in /workspace. Input is
immutable; `output/candidate.json` is the only submission. Read the semantic
authoring entry and anonymous Source first; query one exact missing operation contract
only when the entry requires it.

Submit one JSON semantic candidate, never prose or a completed artifact. Author only
Source `t*`, local `op*/r*/dim*/access*`, semantic ops/attrs/types/SSA, boundary
access/predicate/fill meaning, structured control, and ordered output `{value}`.
`access_ref` names one local `accessN`. Schema, C, ABI, canonical handles, attachments,
returns, bounds, and pi are host-owned.

The transfer ledger is the occurrence authority: retain every proven load/store,
predicate, and fill; do not invent capacity iota or gather semantics. Never emit
canonical `n*/v*/x*/p*/c*/a*/e*`, C/ABI rows, returns, or pi bindings. CandidateCheck
and SourceReconcile fail closed. Every attempt starts from the semantic entry; prior
candidates and rejected drafts are never templates. Submit once, then never claim Replay
or Gate success."""


def _canonical_authoring_prompt() -> str:
    """Return the compact instruction; mounted schema entry is the only wire.

    The CLI adapter is an optional transport, not a second provider-facing
    contract.  Its workspace already mounts the schema-derived semantic entry,
    so pasting the skeleton and operation namespace into every provider prompt
    only duplicates bytes and lets the two render paths drift.
    """

    return _INITIAL_PROMPT


def _observer_diff_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    observer = diagnostics.get("observer_diff")
    if isinstance(observer, Mapping):
        return observer
    unresolved = diagnostics.get("unresolved_gate1")
    nested = (
        unresolved.get("diagnostics")
        if isinstance(unresolved, Mapping)
        else None
    )
    observer = nested.get("observer_diff") if isinstance(nested, Mapping) else None
    return observer if isinstance(observer, Mapping) else None


def _codex_feedback_files(feedback_text: str) -> dict[str, str]:
    """Partition exact closed feedback without exposing one bulk relift object.

    Category names already belong to the Gate1 diagnostic protocol.  This only
    separates bytes by that existing label and removes previous-artifact and
    repair-base compatibility fields. Fresh attempts never mount either one;
    this detail file must never become a candidate authority.
    """

    feedback = json.loads(feedback_text)
    categories: dict[str, Any] = {}
    if isinstance(feedback, Mapping):
        diagnostics = (
            feedback.get("diagnostics")
            if isinstance(feedback.get("diagnostics"), Mapping)
            else None
        )
        observer = _observer_diff_from_diagnostics(
            diagnostics
        )
        raw_categories = observer.get("categories") if isinstance(observer, Mapping) else None
        if isinstance(raw_categories, list):
            for category in raw_categories:
                if not isinstance(category, Mapping):
                    continue
                name = category.get("category")
                if isinstance(name, str) and name in _FEEDBACK_CATEGORY_NAMES:
                    categories[name] = dict(category)

    detail: Any = None
    if isinstance(feedback, Mapping):
        detail = {
            str(key): value
            for key, value in feedback.items()
            if key not in {
                "previous_artifact",
                "repair_base",
                "diagnostics",
            }
        }
        diagnostics = feedback.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            pruned = {
                str(key): value
                for key, value in diagnostics.items()
                if key not in {
                    "observer_diff",
                    "replay_counterexample",
                    "unresolved_gate1",
                }
            }
            unresolved = diagnostics.get("unresolved_gate1")
            if isinstance(unresolved, Mapping):
                unresolved_copy = dict(unresolved)
                nested = unresolved_copy.get("diagnostics")
                if isinstance(nested, Mapping):
                    unresolved_copy["diagnostics"] = {
                        str(key): value
                        for key, value in nested.items()
                        if key not in {
                            "observer_diff",
                            "replay_counterexample",
                        }
                    }
                pruned["unresolved_gate1"] = unresolved_copy
            if pruned:
                detail["diagnostics"] = pruned

    values: dict[str, Any] = {
        "feedback_detail.json": detail,
    }
    values.update(
        {
            f"feedback_category_{name}.json": categories.get(name)
            for name in _FEEDBACK_CATEGORY_NAMES
        }
    )
    rendered = {
        name: json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for name, value in values.items()
    }
    return rendered



def _codex_frontend_guidance(frontend: str | None, source_kind: str) -> str:
    """Describe the selected branch's own source/compiler observer."""

    selected = str(frontend or "triton").casefold()
    if selected == "triton":
        return (
            "This is the Triton/TTIR branch. When an exact source observer is "
            "mounted, use source-observer and focused source-ttir queries; do "
            "not load the complete observer into the conversation."
        )
    if selected == "cuda":
        return (
            "This is the native CUDA C++ branch. Use the selected CUDA source, "
            "harness-abi, source-outline, source-find, and CUDA compiler facts. "
            "source_ttir.json is not applicable and must not be queried."
        )
    if selected == "cute":
        return (
            "This is the independent CuTe branch. Use the CuTe source and its "
            "MLIR/PTX compiler evidence supplied by the CuTe path. Triton TTIR "
            "queries are not applicable."
        )
    if selected == "tilelang":
        return (
            "This is the independent TileLang branch. Use the TileLang Python/TIR "
            "source, harness-abi, source-outline, source-find, and TileLang "
            "compiler facts. Triton TTIR queries are not applicable."
        )
    return (
        f"This is the {selected} frontend branch with source_kind={source_kind}. "
        "Use only its selected source/compiler evidence; do not assume Triton TTIR."
    )


def _codex_frontend_prompt(
    base: str,
    *,
    frontend: str | None,
    source_kind: str,
) -> str:
    return (
        base
        + "\n\n<frontend-branch-guidance>\n"
        + _codex_frontend_guidance(frontend, source_kind)
        + "\n</frontend-branch-guidance>"
    )


def _codex_input_files(
    request: Any,
    *,
    source_ttir_observation: Mapping[str, Any] | None,
    frontend: str | None = None,
    source_kind: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build a small ordinary coding workspace for the source-aware owner."""

    complete = _input_files(
        request,
        source_ttir_observation=source_ttir_observation,
    )
    packet = json.loads(request.user_input)
    selected_frontend = str(
        frontend or packet.get("frontend") or "triton"
    ).casefold()
    selected_source_kind = str(
        source_kind or packet.get("source_kind") or "python"
    )
    branch_guidance = _codex_frontend_guidance(
        selected_frontend,
        selected_source_kind,
    )
    semantic_entry = packet.get("semantic_authoring_entry")
    if not isinstance(semantic_entry, Mapping):
        raise TypeError("Codex Lift request lacks canonical semantic authoring entry")
    observer_instructions = (
        "For an exact Triton compiler fact, use source-observer, "
        "source-ttir-outline, source-ttir-find, or source-ttir-def-use."
        if selected_frontend == "triton"
        else "For an exact branch compiler fact, use source-outline, source-find, "
        "harness, or harness-abi; do not use Triton source-ttir commands."
    )
    realization_instructions = (
        "Use Source/compiler observations only as read-only evidence for semantic "
        "operations and accesses; the host derives all C ownership and pi bindings."
    )
    files = {
        "source.py": complete["source.py"],
        "semantic_authoring_entry.json": (
            json.dumps(
                dict(semantic_entry),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ),
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "README.md": (
            "# IntentIR Lift workspace\n\n"
            "`semantic_authoring_entry.json` is the sole schema-derived semantic and "
            "Source-transfer navigation. Author unresolved J only; host rebuilds C/pi "
            "before fail-closed validation.\n\n"
            "For a `compiler.sensitive_numeric_index`, query exact selected-frontend "
            "evidence before typed arithmetic; never guess promotion or casts.\n\n"
            "Read `input/source.py` and `input/semantic_authoring_entry.json`. "
            + observer_instructions
            + " For a missing operation contract, use exactly:\n"
            "`python .agents/intentir_agent_tool.py lift_reference op NAME`\n"
            "The entry is the sole semantic wire; do not request final-artifact "
            "sections or ownership prose.\n\n"
            + realization_instructions
            + " These facts never choose a relation.\n\n"
            + branch_guidance
            + "\n\n"
            + "Write `output/candidate.json`, then run "
            "`python .agents/intentir_agent_tool.py submit_candidate`. The host exposes "
            "a public probe only after mechanical acceptance.\n"
        ),
    }
    files.update(_codex_feedback_files(complete["feedback.json"]))
    if "source_ttir.json" in complete:
        # The host observation carries canonical SourceFacts solely so the
        # post-submission reconciler can prove its own lineage.  That record
        # contains host allocator/ABI material and is not compiler evidence
        # the semantic provider may author against.  Mount only the selected
        # frontend facts used to understand Source operations.
        raw_observation = json.loads(complete["source_ttir.json"])
        if not isinstance(raw_observation, Mapping):
            raise TypeError("Source TTIR observation must be an object")
        provider_observation_fields = (
            "protocol",
            "authority",
            "function",
            "status",
            "diagnostic",
            "ttir",
            "source_runtime_parameter_uses",
            "python_interface",
            "compiler",
            "target",
            "elapsed_ms",
            "cache_hit",
        )
        provider_observation = {
            key: raw_observation[key]
            for key in provider_observation_fields
            if key in raw_observation
        }
        files["source_ttir.json"] = (
            json.dumps(
                provider_observation,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    return files, {}


def _lift_reference_query(
    references: Mapping[str, str],
    arguments: tuple[str, ...],
    *,
    frontend: str | None = "triton",
) -> Mapping[str, Any]:
    """Return one semantic operation leaf from the canonical wire authority."""

    del references, frontend
    if len(arguments) != 2 or arguments[0] != "op":
        raise ValueError("lift_reference requires exactly one semantic op NAME query")
    operations = canonical_wire_bundle()["profile"][
        "semantic_operation_contracts"
    ]
    name = arguments[1]
    result = operations.get(name) if isinstance(operations, Mapping) else None
    if not isinstance(result, Mapping):
        raise ValueError(f"unknown semantic operation {name!r}")
    return {
        "status": "pass",
        "query": ["op", name],
        "result": dict(result),
        "authority": "canonical semantic provider operation contract",
    }


def _source_aware_startup_context(
    workspace: Any,
    *,
    fresh_owner: bool,
    frontend: str | None = "triton",
    source_kind: str = "python",
) -> dict[str, Any]:
    """Return a path-only receipt; exact task state stays in ordinary files."""

    selected_frontend = str(frontend or "triton").casefold()
    return {
        "protocol": "intentir.codex-lift-startup-view.v2",
        "fresh_owner": fresh_owner,
        "frontend": selected_frontend,
        "source_kind": source_kind,
        "paths": {
            "source": "input/source.py",
            "semantic_authoring_entry": "input/semantic_authoring_entry.json",
            "source_observer": (
                "input/source_ttir.json"
                if selected_frontend == "triton"
                else None
            ),
            "bounded_reference_query": (
                "python .agents/intentir_agent_tool.py lift_reference"
            ),
            "feedback_frontier": "input/feedback_frontier.json",
            "feedback_detail": "input/feedback_detail.json",
            "feedback_categories": [
                f"input/feedback_category_{name}.json"
                for name in _FEEDBACK_CATEGORY_NAMES
            ],
            "candidate": "output/candidate.json",
        },
        "authority": (
            "path manifest only; exact bytes remain in files and no artifact "
            "field, semantic classification, or repair is generated. The "
            "semantic authoring entry and bounded operation query are the only "
            "provider-visible wire authorities."
        ),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return totals


@dataclass
class CodexCLILiftAdapter(MiniSWELiftAdapter):
    """Run one isolated source-aware Codex owner per Lift invocation."""

    session_factory: Callable[..., Any] = field(
        default=CodexCLISession,
        repr=False,
    )
    semantic_probe: Callable[[Any], Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
    )
    _codex_session: Any = field(default=None, init=False, repr=False)
    _invocation_active_elapsed_ms: float = field(default=0.0, init=False)
    _invocation_cli_turns: int = field(default=0, init=False)
    _invocation_model_segments: int = field(default=0, init=False)

    def close(self) -> None:
        session = self._codex_session
        self._codex_session = None
        if session is not None:
            session.close()
        self._invocation_active_elapsed_ms = 0.0
        self._invocation_cli_turns = 0
        self._invocation_model_segments = 0
        super().close()

    def _remaining_seconds(self) -> float | None:
        limit = self.session_wall_time_limit_seconds
        if limit is None:
            return None
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or limit <= 0
        ):
            raise ValueError(
                "Lift owner session wall-time limit must be None or positive"
            )
        return float(limit) - (self._invocation_active_elapsed_ms / 1000.0)

    def _terminal_time_error(self, round_index: int) -> Exception:
        diagnostic = {
            "diagnostic": "branch-local Codex Lift owner wall-time exhausted",
            "failure_class": "lift_owner_time_exhausted",
            "session_id": self._session_id,
            "active_elapsed_ms": self._invocation_active_elapsed_ms,
        }
        self._records.append(
            SCRCReliftRecord(
                round_index,
                diagnostic=diagnostic["diagnostic"],
                backend_evidence={
                    "backend": CODEX_CLI_LIFT_BACKEND,
                    "runtime": CODEX_CLI_AGENT_RUNTIME,
                    "owner_session_id": self._session_id,
                    "thread_id": getattr(self._codex_session, "thread_id", None),
                    "owner_active_elapsed_ms": self._invocation_active_elapsed_ms,
                },
            )
        )
        return CandidateLiftError(
            diagnostic,
            physical_calls=0,
            terminal=True,
        )

    @staticmethod
    def _candidate_missing_check(diagnostic: str) -> dict[str, Any]:
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_unavailable",
                    "path": "<candidate>",
                    "observed_refs_or_types": [diagnostic[:2048]],
                }
            ],
        }

    def lift(self, request: Any, feedback: Any) -> Any:
        if self.workflow != "single":
            raise ValueError("Codex CLI Lift supports only the single-owner workflow")
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Codex CLI Lift requires SourceProgram.text")
        # Close the previous CLI thread and disposable workspace before
        # constructing this invocation, just as the Responses/MiniSWE path
        # does. The fresh worker receives only compressed rule-level guidance.
        self._begin_fresh_agent_invocation()

        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        relift_context = self._relift_context(feedback)
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=relift_context,
            source_symbol=getattr(
                source_program, "lift_source_symbol", self.source_symbol
            ),
            source_kind=getattr(
                source_program, "lift_source_kind", self.source_kind
            ),
            source_parameter_order=tuple(
                getattr(
                    source_program,
                    "lift_parameter_order",
                    self.source_parameter_order,
                )
            ),
            source_to_opaque=(
                dict(
                    getattr(
                        source_program,
                        "lift_source_to_opaque",
                        self.source_to_opaque,
                    )
                )
                if getattr(
                    source_program,
                    "lift_source_to_opaque",
                    self.source_to_opaque,
                ) is not None
                else None
            ),
            frontend=getattr(request, "frontend", "triton"),
            source_facts=getattr(request, "source_facts", None),
            construction=getattr(request, "construction", None),
        )
        candidate_check_context = scrc_candidate_check_context(prepared)
        round_index = len(self._records) + 1
        prepared_packet = json.loads(prepared.user_input)
        selected_frontend = str(
            getattr(request, "frontend", None)
            or prepared_packet.get("frontend")
            or "triton"
        ).casefold()
        selected_source_kind = str(
            prepared_packet.get("source_kind", "python")
        )
        source_ttir_observation: dict[str, Any] | None = None
        if selected_frontend == "triton":
            source_ttir_observation = _source_ttir_observation_for_request(prepared)
        files, references = _codex_input_files(
            prepared,
            source_ttir_observation=source_ttir_observation,
            frontend=selected_frontend,
            source_kind=selected_source_kind,
        )
        try:
            fresh_workspace = self._create_session_workspace(files)
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "Codex CLI Lift requires the lift-agent dependencies",
                evidence={"backend": CODEX_CLI_LIFT_BACKEND},
            ) from exc
        workspace = self._session_workspace
        if workspace is None:
            raise RuntimeError("Codex Lift workspace was not created")
        candidate_path = workspace.candidate_path
        if candidate_path.exists():
            candidate_path.unlink()
        fresh_owner = self._codex_session is None
        call_turns: list[CodexCLITurn] = []
        check_results: list[dict[str, Any]] = []
        source_reconcile_results: list[dict[str, Any]] = []
        deterministic_repairs: list[dict[str, Any]] = []
        semantic_probe_results: list[dict[str, Any]] = []
        semantic_probe_by_candidate: dict[str, dict[str, Any]] = {}
        timeout_events: list[dict[str, Any]] = []
        accepted_submission_interrupts: list[dict[str, Any]] = []
        diagnostic_ready_interrupts: list[dict[str, Any]] = []
        live_checkpoint_events: list[dict[str, Any]] = []
        accepted_submission_event = threading.Event()
        diagnostic_ready_event = threading.Event()
        live_checkpoint_stop = threading.Event()
        evaluation_lock = threading.RLock()
        candidate_text: str | None = None
        last_checked_good: str | None = None
        timed_out_calls = 0
        timed_out_elapsed_ms = 0.0
        try:
            initial_candidate = workspace.read_candidate()
        except Exception:
            initial_candidate = None
        live_checkpoint_attempted = (
            {initial_candidate} if initial_candidate is not None else set()
        )

        def safe_check_receipt(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
            if not isinstance(value, Mapping):
                return None
            diagnostics: list[dict[str, str]] = []
            raw_diagnostics = value.get("diagnostics")
            if isinstance(raw_diagnostics, list):
                for raw in raw_diagnostics[:20]:
                    if not isinstance(raw, Mapping):
                        continue
                    rule_code = raw.get("rule_code")
                    path = raw.get("path")
                    if isinstance(rule_code, str) and isinstance(path, str):
                        diagnostics.append(
                            {
                                "rule_code": rule_code[:2048],
                                "path": path[:2048],
                            }
                        )
            receipt: dict[str, Any] = {
                "status": value.get("status"),
                "diagnostics": diagnostics,
            }
            protocol = value.get("protocol")
            if isinstance(protocol, str):
                receipt["protocol"] = protocol
            applicable = value.get("applicable")
            if isinstance(applicable, bool):
                receipt["applicable"] = applicable
            return receipt

        def current_candidate_delta() -> dict[str, Any] | None:
            if candidate_text is None:
                return None
            try:
                payload = json.loads(candidate_text)
            except (TypeError, ValueError, RecursionError):
                return None
            return scrc_candidate_delta_receipt(
                payload if isinstance(payload, Mapping) else None
            )

        def reject_current_candidate(
            *,
            failure_class: str,
            candidate_check: Mapping[str, Any] | None,
            source_reconcile: Mapping[str, Any] | None,
            semantic_probe_result: Mapping[str, Any] | None = None,
        ) -> None:
            diagnostics: dict[str, Any] = {
                "diagnostic": failure_class,
                "failure_class": failure_class,
                "candidate_present": candidate_text is not None,
            }
            safe_candidate_check = safe_check_receipt(candidate_check)
            safe_source_reconcile = safe_check_receipt(source_reconcile)
            if safe_candidate_check is not None:
                diagnostics["candidate_check"] = safe_candidate_check
            if safe_source_reconcile is not None:
                diagnostics["source_reconcile"] = safe_source_reconcile
            if isinstance(semantic_probe_result, Mapping):
                diagnostics["semantic_probe"] = {
                    key: semantic_probe_result[key]
                    for key in ("protocol", "status", "repairable")
                    if isinstance(semantic_probe_result.get(key), (str, bool))
                }
            evidence = {
                "backend": CODEX_CLI_LIFT_BACKEND,
                "runtime": CODEX_CLI_AGENT_RUNTIME,
                "owner_session_id": self._session_id,
                "fresh_session": True,
                "candidate_check": safe_candidate_check,
                "source_reconcile": safe_source_reconcile,
            }
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=failure_class,
                    backend_evidence=evidence,
                )
            )
            raise CandidateLiftError(
                diagnostics,
                physical_calls=self._invocation_model_segments,
                candidate_delta=current_candidate_delta(),
            )

        def check_current_candidate(*, trigger: str) -> dict[str, Any]:
            nonlocal candidate_text
            try:
                candidate_text = workspace.read_candidate()
            except Exception as exc:
                candidate_text = None
                check = self._candidate_missing_check(
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                try:
                    constructed_text, repairs = _construct_scrc_candidate(
                        candidate_text,
                        prepared,
                        observation=source_ttir_observation,
                        relift_context=relift_context,
                    )
                except Exception as exc:
                    constructed_text = candidate_text
                    repairs = ()
                    check = {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "diagnostics": [
                            {
                                "rule_code": "candidate_repair_failed",
                                "path": "<candidate>",
                                "observed_refs_or_types": [
                                    f"{type(exc).__name__}: {exc}"
                                ],
                            }
                        ],
                    }
                else:
                    if constructed_text != candidate_text:
                        _atomic_write_text(workspace.candidate_path, constructed_text)
                        candidate_text = workspace.read_candidate()
                        if candidate_text != constructed_text:
                            raise RuntimeError(
                                "candidate changed while deterministic repair was "
                                "being committed"
                            )
                    deterministic_repairs.extend(
                        dict(item) for item in repairs if isinstance(item, Mapping)
                    )
                    check = check_scrc_lift_candidate(
                        candidate_text,
                        candidate_check_context,
                    )
                    if check["status"] == "pass":
                        try:
                            # CandidateCheck remains source-free.  This adjacent
                            # decode only enforces the privacy boundary over the
                            # same exact bytes; it never rewrites the artifact.
                            decode_scrc_lift_artifact(candidate_text, prepared)
                        except SourceIdentifierPrivacyError as exc:
                            check = {
                                "protocol": CANDIDATE_CHECK_PROTOCOL,
                                "status": "fail",
                                "diagnostics": [
                                    {
                                        "rule_code": "source_identifier_privacy",
                                        "path": exc.path,
                                        "observed_refs_or_types": [
                                            "replace this free-form Source spelling "
                                            "with the correct anonymous handle or a "
                                            "source-neutral closed protocol value"
                                        ],
                                    }
                                ],
                            }
            check_results.append(
                {
                    "trigger": trigger,
                    "status": check["status"],
                    "diagnostics": [
                        dict(item) for item in check["diagnostics"]
                    ],
                    "candidate_sha256": (
                        hashlib.sha256(candidate_text.encode()).hexdigest()
                        if candidate_text is not None
                        else None
                    ),
                }
            )
            return check

        def reconcile_current_candidate(
            *, trigger: str, checkpoint_allowed: bool = True
        ) -> dict[str, Any]:
            nonlocal candidate_text, last_checked_good
            if candidate_text is None:
                try:
                    candidate_text = workspace.read_candidate()
                except Exception as exc:
                    result = {
                        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
                        "status": "fail",
                        "applicable": False,
                        "diagnostics": [
                            {
                                "rule_code": "candidate_unavailable",
                                "path": "<candidate>",
                                "anonymous_handles": [],
                                "observed_refs_or_types": [type(exc).__name__],
                                "line": None,
                            }
                        ],
                    }
                else:
                    result = (
                        reconcile_candidate_source_ttir(
                            candidate_text,
                            source_ttir_observation,
                            construction=prepared.construction,
                        )
                        if source_ttir_observation is not None
                        else {
                            "status": "pass",
                            "applicable": False,
                            "diagnostics": [],
                        }
                    )
            else:
                result = (
                    reconcile_candidate_source_ttir(
                        candidate_text,
                        source_ttir_observation,
                        construction=prepared.construction,
                    )
                    if source_ttir_observation is not None
                    else {
                        "status": "pass",
                        "applicable": False,
                        "diagnostics": [],
                    }
                )
            source_reconcile_results.append(
                {
                    "trigger": trigger,
                    "status": result["status"],
                    "applicable": result.get("applicable"),
                    "diagnostics": [
                        dict(item) for item in result.get("diagnostics", [])
                    ],
                    "candidate_sha256": (
                        hashlib.sha256(candidate_text.encode()).hexdigest()
                        if candidate_text is not None
                        else None
                    ),
                }
            )
            if (
                checkpoint_allowed
                and result["status"] == "pass"
                and candidate_text is not None
            ):
                # Exact bytes currently in the candidate file.  They may include
                # only the bounded deterministic ingress repairs recorded above;
                # both checks below consume this same byte sequence.
                last_checked_good = candidate_text
            return result

        def submit_candidate_tool() -> Mapping[str, Any]:
            """Run both reject-only preflights as one exact-byte transaction.

            CandidateCheck and SourceReconcile retain separate authority and
            diagnostics.  This wrapper only removes the model-facing sequencing
            overhead; the host still reruns both checks after the CLI turn.
            """

            with evaluation_lock:
                candidate_check = check_current_candidate(
                    trigger="submit_candidate"
                )
                source_reconcile = reconcile_current_candidate(
                    trigger="submit_candidate",
                    checkpoint_allowed=candidate_check["status"] == "pass",
                )
                passed = (
                    candidate_check["status"] == "pass"
                    and source_reconcile["status"] == "pass"
                )
                if passed:
                    accepted_submission_event.set()
                else:
                    # The first rejected submission terminates this fresh
                    # owner. It must not receive the rejected candidate back
                    # as an editable repair base in another CLI turn.
                    diagnostic_ready_event.set()
            return {
                "protocol": "intentir.codex-cli.lift-submit.v1",
                "status": "pass" if passed else "fail",
                "phase": (
                    "complete"
                    if passed
                    else (
                        "candidate_check"
                        if candidate_check["status"] != "pass"
                        else "source_reconcile"
                    )
                ),
                "candidate_check": candidate_check,
                "source_reconcile": source_reconcile,
                "authority": (
                    "diagnostic-only independent phase results over the same exact "
                    "candidate bytes; no artifact field is generated or repaired"
                ),
            }

        def host_preflight(
            *, trigger: str, checkpoint_allowed: bool = True
        ) -> tuple[dict[str, Any], dict[str, Any] | None]:
            with evaluation_lock:
                candidate_check = check_current_candidate(trigger=trigger)
                if candidate_check["status"] != "pass" or candidate_text is None:
                    return candidate_check, None
                source_reconcile = reconcile_current_candidate(
                    trigger=trigger,
                    checkpoint_allowed=checkpoint_allowed,
                )
                return candidate_check, source_reconcile

        def live_checkpoint_watch() -> None:
            """Deliver a stable LLM-authored draft once both preflights pass.

            This watcher never edits or completes a candidate.  It only avoids
            waiting for the remainder of a provider turn after exact bytes already
            satisfy the same transaction used by ``submit_candidate``.  A previous
            formal-round candidate is ignored until the owner changes its bytes.
            """

            nonlocal last_checked_good
            observed_candidate: str | None = None
            stable_since = 0.0
            while not live_checkpoint_stop.wait(0.25):
                try:
                    observed_text = workspace.read_candidate()
                except Exception:
                    observed_candidate = None
                    stable_since = 0.0
                    continue
                if observed_text in live_checkpoint_attempted:
                    continue
                if observed_text != observed_candidate:
                    observed_candidate = observed_text
                    stable_since = time.monotonic()
                    continue
                if time.monotonic() - stable_since < 3.0:
                    continue
                live_checkpoint_attempted.add(observed_text)
                candidate_check, source_reconcile = host_preflight(
                    trigger="live_checkpoint",
                    checkpoint_allowed=False,
                )
                with evaluation_lock:
                    checked_text = candidate_text
                    passed = (
                        candidate_check["status"] == "pass"
                        and source_reconcile is not None
                        and source_reconcile["status"] == "pass"
                        and checked_text is not None
                        and checked_text == observed_text
                    )
                    if passed:
                        last_checked_good = checked_text
                live_checkpoint_events.append(
                    {
                        "candidate_bytes": (
                            len(checked_text.encode())
                            if checked_text is not None
                            else 0
                        ),
                        "candidate_check": candidate_check["status"],
                        "source_reconcile": (
                            source_reconcile["status"]
                            if source_reconcile is not None
                            else "not_run"
                        ),
                        "accepted": passed,
                    }
                )
                if checked_text != observed_text:
                    live_checkpoint_attempted.discard(observed_text)
                if passed:
                    accepted_submission_event.set()
                    return

        def semantic_probe_tool() -> Mapping[str, Any]:
            """Execute one public Source-vs-J diagnostic over exact candidate bytes."""

            with evaluation_lock:
                candidate_check, source_reconcile = host_preflight(
                    trigger="semantic_probe"
                )
                probe_candidate_text = candidate_text
            if candidate_check["status"] != "pass":
                result: dict[str, Any] = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "candidate_check",
                    "diagnostic": "current candidate has not passed CandidateCheck",
                }
            elif source_reconcile is None or source_reconcile["status"] != "pass":
                result = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "source_reconcile",
                    "diagnostic": "current candidate has not passed SourceReconcile",
                }
            elif probe_candidate_text is None or self.semantic_probe is None:
                result = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "execution",
                    "diagnostic": "this Lift invocation has no public probe runner",
                }
            else:
                artifact = decode_scrc_lift_artifact(
                    probe_candidate_text, prepared
                )
                result = dict(self.semantic_probe(artifact))
            result.setdefault(
                "authority",
                "public diagnostic only; not CandidateCheck or certification",
            )
            candidate_sha256 = (
                hashlib.sha256(probe_candidate_text.encode()).hexdigest()
                if probe_candidate_text is not None
                else None
            )
            result["candidate_sha256"] = candidate_sha256
            semantic_probe_results.append(dict(result))
            if candidate_sha256 is not None:
                semantic_probe_by_candidate[candidate_sha256] = dict(result)
            if (
                result.get("status") == "diverged"
                or result.get("repairable") is True
            ):
                diagnostic_ready_event.set()
            return result

        def actionable_semantic_probe() -> Mapping[str, Any] | None:
            """Return one Source/J rejection without reopening this owner."""

            if candidate_text is None:
                return None
            candidate_sha256 = hashlib.sha256(candidate_text.encode()).hexdigest()
            result = semantic_probe_by_candidate.get(candidate_sha256)
            if result is None:
                result = dict(semantic_probe_tool())
            # An executed Source-vs-J mismatch is a counterexample. Most RefJ
            # construction ``unavailable`` results remain a consumer capability
            # gap, but the probe may narrowly mark one directly expressible J
            # omission as repairable. Do not feed back any other unsupported form.
            if (
                result.get("status") == "diverged"
                or result.get("repairable") is True
            ):
                return result
            return None

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "submit_candidate": submit_candidate_tool,
                "semantic_probe": semantic_probe_tool,
            },
            argument_handlers={
                "lift_reference": lambda arguments: _lift_reference_query(
                    references,
                    arguments,
                    frontend=selected_frontend,
                )
            },
        )
        live_checkpoint_thread: threading.Thread | None = None
        try:
            bridge.start()
            live_checkpoint_thread = threading.Thread(
                target=live_checkpoint_watch,
                name="intentir-codex-lift-live-checkpoint",
                daemon=True,
            )
            live_checkpoint_thread.start()
            if self._codex_session is None:
                self._codex_session = self.session_factory(
                    provider=provider,
                    input_dir=workspace.input_dir,
                    output_dir=workspace.output_dir,
                    scratch_dir=workspace.scratch_dir,
                    tmp_dir=workspace.tmp_dir,
                    model=provider.model,
                    effort=effective_effort,
                )
            startup_context = _source_aware_startup_context(
                workspace,
                fresh_owner=fresh_owner,
                frontend=selected_frontend,
                source_kind=selected_source_kind,
            )
            prompt = (
                _codex_frontend_prompt(
                    _canonical_authoring_prompt(),
                    frontend=selected_frontend,
                    source_kind=selected_source_kind,
                )
                + "\n\n<source-aware-startup>\n"
                + json.dumps(
                    startup_context,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n</source-aware-startup>\n"
            )
            while True:
                remaining = self._remaining_seconds()
                if remaining is not None and remaining <= 0:
                    raise self._terminal_time_error(round_index)
                turn_timeout = self.model_timeout_seconds
                if remaining is not None:
                    turn_timeout = (
                        remaining
                        if turn_timeout is None
                        else min(remaining, turn_timeout)
                    )
                try:
                    turn = self._codex_session.run(
                        prompt,
                        timeout_seconds=turn_timeout,
                        effort=effective_effort,
                        stop_requested=self.stop_requested,
                        submission_accepted=accepted_submission_event.is_set,
                        submission_rejected=diagnostic_ready_event.is_set,
                    )
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                except CodexCLICancelled as exc:
                    self._invocation_active_elapsed_ms += exc.elapsed_ms
                    self._invocation_cli_turns += 1
                    self._invocation_model_segments += exc.physical_calls
                    cancellation_evidence = {
                        "backend": CODEX_CLI_LIFT_BACKEND,
                        "runtime": CODEX_CLI_AGENT_RUNTIME,
                        "owner_session_id": self._session_id,
                        "thread_id": getattr(
                            self._codex_session, "thread_id", None
                        ),
                        "winner_branch_id": exc.winner_branch_id,
                        "cancelled_elapsed_ms": exc.elapsed_ms,
                        "cancelled_stdout_sha256": exc.stdout_sha256,
                        "cancelled_stderr_sha256": exc.stderr_sha256,
                        "turns": [item.to_evidence() for item in call_turns],
                        "candidate_check": list(check_results),
                        "source_reconcile": list(source_reconcile_results),
                        "deterministic_repairs": list(deterministic_repairs),
                        "semantic_probe": list(semantic_probe_results),
                        "live_checkpoint": list(live_checkpoint_events),
                        "tool_bridge": bridge.evidence,
                    }
                    self._records.append(
                        SCRCReliftRecord(
                            round_index,
                            diagnostic=str(exc),
                            backend_evidence=cancellation_evidence,
                        )
                    )
                    raise CandidateLiftError(
                        {
                            "diagnostic": str(exc),
                            "failure_class": "candidate_superseded",
                            "winner_branch_id": exc.winner_branch_id,
                        },
                        physical_calls=exc.physical_calls,
                    ) from exc
                except CodexCLISubmissionAccepted as exc:
                    self._invocation_active_elapsed_ms += exc.elapsed_ms
                    self._invocation_cli_turns += 1
                    self._invocation_model_segments += exc.physical_calls
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                    accepted_submission_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "pending_exact_checkpoint_recheck",
                        }
                    )
                    if last_checked_good is None:
                        raise CodexCLIError(
                            "accepted submission has no exact checked checkpoint"
                        ) from exc
                    _atomic_write_text(workspace.candidate_path, last_checked_good)
                    check, source_reconcile = host_preflight(
                        trigger="host_accepted_submission_restore"
                    )
                    if (
                        check["status"] != "pass"
                        or source_reconcile is None
                        or source_reconcile["status"] != "pass"
                        or candidate_text is None
                    ):
                        raise CodexCLIError(
                            "accepted submission failed independent host recheck"
                        ) from exc
                    accepted_submission_interrupts[-1]["recovery"] = (
                        "exact_checkpoint_restored"
                    )
                    semantic_rejection = actionable_semantic_probe()
                    if semantic_rejection is not None:
                        reject_current_candidate(
                            failure_class="semantic_probe_rejected",
                            candidate_check=check,
                            source_reconcile=source_reconcile,
                            semantic_probe_result=semantic_rejection,
                        )
                    break
                except CodexCLISubmissionRejected as exc:
                    self._invocation_active_elapsed_ms += exc.elapsed_ms
                    self._invocation_cli_turns += 1
                    self._invocation_model_segments += exc.physical_calls
                    diagnostic_ready_event.clear()
                    accepted_submission_event.clear()
                    diagnostic_ready_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "fresh_attempt_required",
                        }
                    )
                    check, source_reconcile = host_preflight(
                        trigger="host_live_diagnostic_recheck"
                    )
                    if (
                        check["status"] == "pass"
                        and source_reconcile is not None
                        and source_reconcile["status"] == "pass"
                        and candidate_text is not None
                    ):
                        semantic_rejection = actionable_semantic_probe()
                        if semantic_rejection is not None:
                            reject_current_candidate(
                                failure_class="semantic_probe_rejected",
                                candidate_check=check,
                                source_reconcile=source_reconcile,
                                semantic_probe_result=semantic_rejection,
                            )
                        break
                    reject_current_candidate(
                        failure_class=(
                            "candidate_rejected"
                            if check["status"] != "pass"
                            else "source_reconcile_rejected"
                        ),
                        candidate_check=check,
                        source_reconcile=source_reconcile,
                    )
                except CodexCLIError as exc:
                    if "wall-time limit" in str(exc):
                        elapsed = (turn_timeout or 0.0) * 1000.0
                        self._invocation_active_elapsed_ms += elapsed
                        self._invocation_cli_turns += 1
                        self._invocation_model_segments += 1
                        timed_out_calls += 1
                        timed_out_elapsed_ms += elapsed
                        timeout_events.append(
                            {
                                "turn_timeout_seconds": turn_timeout,
                                "thread_id": getattr(
                                    self._codex_session, "thread_id", None
                                ),
                                "recovery": "fresh_attempt_required",
                            }
                        )
                    raise
                call_turns.append(turn)
                self._invocation_active_elapsed_ms += turn.elapsed_ms
                self._invocation_cli_turns += 1
                self._invocation_model_segments += turn.model_segments

                check, source_reconcile = host_preflight(trigger="host_post_turn")
                if (
                    check["status"] == "pass"
                    and source_reconcile is not None
                    and source_reconcile["status"] == "pass"
                    and candidate_text is not None
                ):
                    semantic_rejection = actionable_semantic_probe()
                    if semantic_rejection is not None:
                        reject_current_candidate(
                            failure_class="semantic_probe_rejected",
                            candidate_check=check,
                            source_reconcile=source_reconcile,
                            semantic_probe_result=semantic_rejection,
                        )
                    break
                reject_current_candidate(
                    failure_class=(
                        "candidate_rejected"
                        if check["status"] != "pass"
                        else "source_reconcile_rejected"
                    ),
                    candidate_check=check,
                    source_reconcile=source_reconcile,
                )
        except Exception as exc:
            if isinstance(exc, CandidateLiftError):
                raise
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            evidence = {
                "backend": CODEX_CLI_LIFT_BACKEND,
                "runtime": CODEX_CLI_AGENT_RUNTIME,
                "owner_session_id": self._session_id,
                "thread_id": getattr(self._codex_session, "thread_id", None),
                "fresh_session": fresh_owner,
                "turns": [turn.to_evidence() for turn in call_turns],
                "candidate_check": list(check_results),
                "source_reconcile": list(source_reconcile_results),
                "deterministic_repairs": list(deterministic_repairs),
                "semantic_probe": list(semantic_probe_results),
                "live_checkpoint": list(live_checkpoint_events),
                "timeout_events": list(timeout_events),
                "accepted_submission_interrupts": list(
                    accepted_submission_interrupts
                ),
                "diagnostic_ready_interrupts": list(
                    diagnostic_ready_interrupts
                ),
                "tool_bridge": bridge.evidence,
            }
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    backend_evidence=evidence,
                )
            )
            raise SCRCInfrastructureError(
                f"Codex CLI Lift infrastructure failed: {type(exc).__name__}: {exc}",
                evidence={
                    key: value
                    for key, value in evidence.items()
                    if key != "turns"
                },
            ) from exc
        finally:
            live_checkpoint_stop.set()
            if live_checkpoint_thread is not None:
                live_checkpoint_thread.join()
            bridge.close()

        assert candidate_text is not None
        artifact = decode_scrc_lift_artifact(candidate_text, prepared)
        usage = _aggregate_usage(call_turns)
        trajectory = {
            "protocol": CODEX_CLI_AGENT_RUNTIME,
            "backend": CODEX_CLI_LIFT_BACKEND,
            "reasoning_effort": effective_effort,
            "thread_id": self._codex_session.thread_id,
            "turns": [turn.to_evidence() for turn in call_turns],
            "candidate_check": list(check_results),
            "source_reconcile": list(source_reconcile_results),
            "deterministic_repairs": list(deterministic_repairs),
            "semantic_probe": list(semantic_probe_results),
            "live_checkpoint": list(live_checkpoint_events),
            "timeout_events": list(timeout_events),
            "accepted_submission_interrupts": list(
                accepted_submission_interrupts
            ),
            "diagnostic_ready_interrupts": list(
                diagnostic_ready_interrupts
            ),
            "tool_bridge": bridge.evidence,
            "source_ttir_observation": (
                {
                    key: value
                    for key, value in source_ttir_observation.items()
                    if key not in {"ttir", "fingerprint", "source_facts"}
                }
                if source_ttir_observation is not None
                else None
            ),
            "isolation": {
                "outer": "bubblewrap-host-fs-v1",
                "input": "read-only",
                "output_work_tmp": "write-only-scope",
                "tool_network": "disabled",
                "provider_secret_in_tool_env": False,
            },
        }
        raw_trajectory = _trajectory_bytes(trajectory)
        elapsed_ms = (
            sum(turn.elapsed_ms for turn in call_turns)
            + timed_out_elapsed_ms
            + sum(
                float(item["elapsed_ms"])
                for item in accepted_submission_interrupts
            )
            + sum(
                float(item["elapsed_ms"])
                for item in diagnostic_ready_interrupts
            )
        )
        physical_calls = (
            sum(turn.model_segments for turn in call_turns)
            + timed_out_calls
            + len(accepted_submission_interrupts)
            + len(diagnostic_ready_interrupts)
        )
        provenance = SCRCProvenance(
            model=provider.model,
            prompt_sha256=prepared.prompt_sha256,
            response_id=self._codex_session.thread_id,
            response_model=provider.model,
            usage=usage,
            response_sha256=hashlib.sha256(candidate_text.encode()).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=candidate_text,
            raw_response=raw_trajectory,
            attempts=(
                len(call_turns)
                + timed_out_calls
                + len(accepted_submission_interrupts)
                + len(diagnostic_ready_interrupts)
            ),
            elapsed_ms=elapsed_ms,
            backend=CODEX_CLI_LIFT_BACKEND,
            termination_status="CandidateCheckAndSourceReconcilePass",
            trajectory=trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
            prepared.construction,
        )
        backend_evidence = {
            "backend": CODEX_CLI_LIFT_BACKEND,
            "knowledge_profile": CODEX_CLI_LIFT_KNOWLEDGE,
            "runtime": CODEX_CLI_AGENT_RUNTIME,
            "fresh_session": fresh_owner,
            "fresh_workspace": fresh_workspace,
            "persistent_owner": False,
            "persistent_workspace": False,
            "owner_session_id": self._session_id,
            "thread_id": self._codex_session.thread_id,
            "cli_turns": (
                len(call_turns)
                + timed_out_calls
                + len(accepted_submission_interrupts)
                + len(diagnostic_ready_interrupts)
            ),
            "model_segments_lower_bound": physical_calls,
            "tool_actions": sum(turn.tool_actions for turn in call_turns),
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "candidate_check": list(check_results),
            "source_reconcile": list(source_reconcile_results),
            "deterministic_repairs": list(deterministic_repairs),
            "semantic_probe": list(semantic_probe_results),
            "live_checkpoint": list(live_checkpoint_events),
            "timeout_events": list(timeout_events),
            "accepted_submission_interrupts": list(
                accepted_submission_interrupts
            ),
            "diagnostic_ready_interrupts": list(
                diagnostic_ready_interrupts
            ),
            "tool_bridge": bridge.evidence,
            "trajectory": trajectory,
        }
        self._records.append(
            SCRCReliftRecord(
                round_index,
                extraction=extraction,
                backend_evidence=backend_evidence,
            )
        )

        from intentir.verify.scrc_runner import LiftCallResult

        return LiftCallResult(
            artifact,
            physical_calls=physical_calls,
            metadata={
                key: value
                for key, value in backend_evidence.items()
                if key != "trajectory"
            }
            | {
                "model": provider.model,
                "reasoning_effort": effective_effort,
                "artifact_schema": artifact.wire_schema,
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "owner_active_elapsed_ms": self._invocation_active_elapsed_ms,
                "owner_cli_turns": self._invocation_cli_turns,
                "owner_model_segments_lower_bound": (
                    self._invocation_model_segments
                ),
                "physical_provider_call_observability": (
                    "Codex CLI exposes model segments, not raw HTTP request count"
                ),
                "workflow": "single",
            },
        )


__all__ = [
    "CODEX_CLI_LIFT_BACKEND",
    "CODEX_CLI_LIFT_KNOWLEDGE",
    "CodexCLILiftAdapter",
]
