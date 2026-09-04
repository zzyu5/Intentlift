"""Codex CLI Lift backend with external SCRC verification boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIDiagnosticReady,
    CodexCLIError,
    CodexCLISubmissionAccepted,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.verify.scrc_runner import CandidateLiftError
from minisweagent.config.intentir_schema_query import schema_sections

from .miniswe import (
    MiniSWELiftAdapter,
    _input_files,
    _lift_invocation_limits,
    _lift_turn_timeout,
    _trajectory_bytes,
)
from .scrc import (
    CANDIDATE_CHECK_PROTOCOL,
    SCRCExtraction,
    SCRCProvenance,
    SCRCReliftRecord,
    SourceIdentifierPrivacyError,
    _decode_value_free_lift_candidate,
    _unbound_lift_artifact_payload,
    _value_free_lift_candidate_payload,
    _validate_source_identifier_privacy,
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
CODEX_CLI_LIFT_BACKEND = "openai.codex-cli.intentir-lift.v41"
CODEX_CLI_LIFT_KNOWLEDGE = "intentir.lift-knowledge.v101"
_INACTIVE_SOURCE_RECONCILE_PROTOCOL = "intentir.lift-source-reconcile.v1"

_FEEDBACK_CATEGORY_NAMES = (
    "interface",
    "predicate_effect",
    "op_multiset",
    "dataflow",
    "reduction",
)

_INITIAL_PROMPT = """You are the sole source-aware IntentIR Lift owner for this branch.
Work only under the isolated /workspace; input/ is immutable and output/, work/,
and tmp/ are writable. There is no repository or network.

Act like a coding agent, not a staged JSON generator. Start with the one compact
`source-brief` command in input/README.md, then read input/source.py. The brief includes
the complete value-free Harness ABI/C-declaration ledger; you decide all meaning.
Trusted Source compilation and TTIR remain host-private and are never model input.
When Source exposes a fused operation or ambiguous implicit numeric conversion,
preserve only what Source establishes and never guess a hardware-specialized width
or structure. Compiler realization is not a J authorship or Lift-acceptance
authority. Query schema leaves, operation contracts and ownership
terms through `python .agents/intentir_agent_tool.py lift_reference ...`; complete
references are deliberately not mounted as readable workspace files.

You alone author the complete output/candidate.json. J owns all logical semantics;
C owns only implementation choices; pi is only the typed Gate1 normalization
witness in the supplied schema. It has no realization graph. Author only closed direct
program/iteration/control/materialization/access-representation relations bound to
existing J axes, values, accesses and C slots. Query the exact pi schema leaves when
needed; never invent a physical Source SSA id or target recipe in
pi. A relation whose fixed unary recognizer cannot prove remains hard.
Fixed code must not choose or fill any field, handle, relation, operation, or shape.
Preserve anonymous ABI order, parametric shape identities, logical control/time and
effects in J. Derive logical program coordinates only from Source semantics; never
copy launch axes or a backend execution topology into J. Use dense
declaration-ordered opaque handles and never copy Source identifiers into the artifact.

Every inout tensor requires explicit pre-state access/effect, post-state access/effect,
and output_mapping. A static affine inout store is owned by that boundary access/effect
and an identity output mapping; do not invent semantic scatter/update nodes merely to
rebuild the full tensor. Use scatter only for a data-dependent J index whose closed
index/source shapes agree at the data rank. Preserve J access multiplicity. An access
representation authorization may change pointer construction only at Lower time and
never changes tensor, logical index, mask, value, or effect. `candidate-attachments`
projects anchors already authored in your draft; it provides no Source or hardware
realization evidence.

If output/candidate.json already exists, it is the exact preceding model-authored
candidate: read input/feedback_frontier.json and input/feedback_detail.json first, then
inspect and patch that candidate rather than recreating it. Otherwise write one meaningful
complete candidate early. Do not postpone authoring until after exhaustive reading. As
soon as the first complete JSON exists, run
`python .agents/intentir_agent_tool.py submit_candidate` before manual rereading or
broad schema inspection. Let its closed diagnostics drive focused patches.
Organize scratch under work/ however useful. The final J/C/pi and every semantic
field remain entirely your work. This transaction runs the source-free CandidateCheck
on the exact bytes. The host then runs the public Source-vs-J semantic probe on those
exact passing bytes and returns a same-thread diagnostic if they diverge.
That probe never certifies or edits the candidate: repair only when the evidence
shows your J is wrong, rerun submit_candidate, and leave small floating deltas or
unsupported execution as diagnostics. The host independently reruns Decode and
TypeCheck, then starts fresh source-blind Replay and the formal gates. Do not claim
certification.
"""


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
    separates bytes by that existing label and removes the duplicate previous
    artifact, which the persistent owner already has in output/candidate.json.
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
            if key not in {"previous_artifact", "previous_candidate", "diagnostics"}
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



def _codex_input_files(
    request: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build a small ordinary coding workspace for the source-aware owner."""

    complete = _input_files(request)
    files = {
        "source.py": complete["source.py"],
        "harness_view.json": complete["harness_view.json"],
        "feedback_frontier.json": complete["feedback_frontier.json"],
        "README.md": (
            "# IntentIR Lift workspace\n\n"
            "J owns logical semantics, C only implementation choices, and pi only "
            "contains closed direct normalization relations bound to J/C. "
            "You author every "
            "semantic field; fixed code only projects evidence, checks, and rejects.\n\n"
            "Start once with:\n"
            "`python input/tools/context_query.py --input-root input --output-root output source-brief`\n\n"
            "Then read `input/source.py`. Trusted Source TTIR, compiler schedule, "
            "launch/grid, target metadata and actual C values are deliberately "
            "host-private. Never infer them from Source surface syntax. For wire syntax, use:\n"
            "`python .agents/intentir_agent_tool.py lift_reference authoring-brief`\n"
            "Then request only `section NAME`, `op NAME`, or `contract TERM` needed by "
            "the current draft. Complete references are not readable files.\n\n"
            "After J/C exists, run the read-only `candidate-attachments` view before "
            "authoring pi; it projects only your own draft.\n\n"
            "Use `lift_reference contract TERM` only when ownership is unclear. Write "
            "`output/candidate.json` early and patch it in "
            "place. Run `python .agents/intentir_agent_tool.py submit_candidate`; the "
            "host supplies the public semantic probe after a mechanically passing draft.\n"
        ),
        "tools/context_query.py": complete["tools/context_query.py"],
        "candidate_check.json": "null\n",
        "semantic_probe.json": "null\n",
    }
    files.update(_codex_feedback_files(complete["feedback.json"]))
    references = {
        "schema": complete["schema/artifact.schema.json"],
        "contract": complete["knowledge/lift_contract.md"],
        "operations": complete["knowledge/intent_ops.md"],
        "preflight": complete["knowledge/preflight.md"],
    }
    return files, references


_REFERENCE_OP_HEADING = re.compile(r"^  ([a-z][a-z0-9_]*)\s*:")


def _reference_operation_sections(reference: str) -> dict[str, str]:
    lines = reference.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _REFERENCE_OP_HEADING.match(line)
        if match:
            starts.append((index, match.group(1)))
    return {
        name: "\n".join(
            lines[
                start : (
                    starts[position + 1][0]
                    if position + 1 < len(starts)
                    else len(lines)
                )
            ]
        ).rstrip()
        for position, (start, name) in enumerate(starts)
    }


def _lift_reference_query(
    references: Mapping[str, str], arguments: tuple[str, ...]
) -> Mapping[str, Any]:
    """Return one bounded exact reference fragment without mounting bulk files."""

    if not arguments:
        raise ValueError(
            "lift_reference requires authoring-brief, section NAME, op NAME, or contract TERM"
        )
    schema = json.loads(references["schema"])
    if not isinstance(schema, Mapping):
        raise TypeError("Lift schema root must be an object")
    sections = schema_sections(dict(schema))
    root_properties = schema["properties"]
    j_schema = root_properties["j"]
    j_properties = j_schema["properties"]
    intent_schema = j_properties["intent"]
    intent_properties = intent_schema["properties"]
    intent_function_schema = intent_properties["intent_function"]
    boundary_schema = j_properties["boundary"]
    signature_schema = j_properties["signature"]
    c_schema = root_properties["c"]
    pi_schema = root_properties["pi"]
    sections.update(
        {
            "node_ids": j_properties["node_ids"],
            "dimension_constraint": j_properties["dimension_constraints"]["items"],
        }
    )
    operations = _reference_operation_sections(references["operations"])
    command = arguments[0]
    result: Any
    if command == "authoring-brief" and len(arguments) == 1:
        def object_shape(value: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "required": value.get("required", []),
                "fields": list(value.get("properties", {})),
            }

        result = {
            "object_shapes": {
                "root": object_shape(schema),
                "j": object_shape(j_schema),
                "j.intent": object_shape(intent_schema),
                "j.intent.intent_function": object_shape(intent_function_schema),
                "j.boundary": object_shape(boundary_schema),
                "j.signature": object_shape(signature_schema),
                "c": object_shape(c_schema),
                "pi": object_shape(pi_schema),
            },
            "closed_field_schemas": {
                "root.schema": root_properties["schema"],
                "j.kernel_id": j_properties["kernel_id"],
                "j.node_ids": j_properties["node_ids"],
                "pi.schema": pi_schema["properties"]["schema"],
            },
            "sections": sorted(sections),
            "ordinary_operations": sorted(operations),
            "structured_operations": ["loop", "effect_if", "value_if"],
            "mandatory_numeric_authoring": [
                "Preserve a fused numeric event only when anonymous Source syntax or "
                "its public language contract establishes that semantic operation.",
                "Preserve Source-explicit cast order and logical operand/result dtypes; "
                "never import a compiler-selected arithmetic width into J.",
                "Keep logical index arithmetic independent of ABI pointer widths, "
                "address representation, vectorization, and backend layout choices.",
            ],
        }
    elif command == "section" and len(arguments) == 2:
        name = arguments[1]
        if name not in sections:
            raise ValueError(f"unknown schema section {name!r}; choose {sorted(sections)}")
        result = sections[name]
    elif command == "op" and len(arguments) == 2:
        name = arguments[1]
        if name in {"loop", "effect_if", "value_if"}:
            variants = schema["$defs"]["intent_op"]["anyOf"]
            result = next(
                variant
                for variant in variants
                if name in variant.get("properties", {}).get("op", {}).get("enum", [])
            )
        elif name in operations:
            result = operations[name]
        else:
            raise ValueError(f"unknown Intent operation {name!r}")
    elif command == "contract" and len(arguments) >= 2:
        needle = " ".join(arguments[1:]).casefold()
        matches: list[dict[str, Any]] = []
        for label in ("contract", "preflight"):
            lines = references[label].splitlines()
            for index, line in enumerate(lines):
                if needle not in line.casefold():
                    continue
                start = max(0, index - 3)
                end = min(len(lines), index + 4)
                matches.append(
                    {
                        "source": label,
                        "line_start": start + 1,
                        "text": "\n".join(lines[start:end]),
                    }
                )
                if len(matches) >= 8:
                    break
            if len(matches) >= 8:
                break
        result = matches
    else:
        raise ValueError(
            "lift_reference accepts authoring-brief, section NAME, op NAME, or contract TERM"
        )
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if len(encoded) > 12_000:
        raise ValueError("reference result exceeds 12000 characters; request a narrower item")
    return {
        "status": "pass",
        "query": list(arguments),
        "result": result,
        "authority": "exact source-neutral reference fragment; no J/C/pi field is generated",
    }


_EXTERNAL_CONTINUATION_PROMPT = """The same branch-local source-aware Lift task continues in the same workspace and thread.
Read input/feedback_frontier.json first. Open only a named
input/feedback_category_NAME.json when its bounded frontier is insufficient; that
file contains the exact differences for the named category.
input/feedback_detail.json contains exact non-observer diagnostics. The previous
artifact already exists at output/candidate.json and is never duplicated in input.
After a formal Gate1 failure, only the bounded typed observer diff is returned.
The preceding Replay source, Replay TTIR, agent context, and implementation choices
remain hidden; repair J/C only when Source plus the typed frontier proves that the
current artifact is incomplete. Otherwise resubmit the unchanged artifact for a fresh
source-blind Replay.
Your own prior formal candidates are immutable under work/candidate_history/;
consult or restore them when a repair regresses, but decide every edit yourself.
Treat the first exact missing/extra operation only as an entry point: inspect its
anonymous Source neighborhood and the bounded local frontier against the current J
node and boundary. The frontier describes a mismatch but does not expose or authorize
the Replay implementation that produced it. Preserve cast order and operation spelling when they are semantic. If J
already states the complete logical fact but Replay failed to realize it, do not add a
pi escape hatch; resubmit the unchanged artifact for a fresh source-blind Replay.
Never copy block/thread/warp mapping, launch geometry, address topology, layout,
pipeline, vector width, physical loop structure, or another Replay/compiler
realization into J or pi. Such choices belong to a target Layer-C plugin.
Before editing local operations, check whether one omitted Source-visible loop,
effect_if, early terminal branch, continuation, or effect region explains the
downstream op/dataflow/predicate differences; if so, repair that complete J scope
rather than its symptoms. Change pi only when Source evidence and an existing J/C
anchor select one closed normalization relation; a row never licenses an arbitrary
diff. Conversely, when
the current J/B/C already authors the exact logical fact and Source-local boundary fact
but the preceding Replay alone failed to consume it, do not mutate the artifact to
imitate that Replay: resubmit the unchanged artifact for a fresh source-blind Replay.
Use the compact source query only where the named feedback needs exact Source syntax,
then revise the complete model-authored J/C/pi in place. If a
named relation's ownership is unclear, query only that term through
`python .agents/intentir_agent_tool.py lift_reference contract TERM`. Treat feedback as
a semantic diagnostic, not as a patch or a lowering recipe. Run
`python .agents/intentir_agent_tool.py submit_candidate`;
the host automatically probes the exact passing J and returns any public divergence
to this same thread. The host alone runs formal gates.
"""

_CANDIDATE_CHECK_CONTINUATION_PROMPT = """The host submission preflight rejected the current exact candidate bytes.
Read input/candidate_check.json and repair output/candidate.json in this same
thread. Make coordinated J/C/pi changes when required. Rerun
`python .agents/intentir_agent_tool.py submit_candidate` and finish immediately
after the complete candidate passes. Query a rejected relation's exact field name with
`lift_reference contract TERM` when its ownership is unclear. The transaction is diagnostic-only.
"""

_SEMANTIC_PROBE_CONTINUATION_PROMPT = """The current exact candidate passed CandidateCheck, and the host ran the public Source-vs-J semantic probe.
Read input/semantic_probe.json in this same thread. It is diagnostic-only and does
not accept, reject, certify, or edit J. If the observed divergence exposes an error
in your authored J, patch output/candidate.json and rerun submit_candidate. An unavailable
J reference construction/execution is a fixed-consumer capability diagnostic, not a J
counterexample; leave the already checked candidate unchanged. If the remaining delta is
only an expected floating reduction difference, also leave it unchanged and finish. Do
not claim Gate membership.
"""


def _source_aware_startup_context(workspace: Any, *, fresh_owner: bool) -> dict[str, Any]:
    """Return a path-only receipt; exact task state stays in ordinary files."""

    return {
        "protocol": "intentir.codex-lift-startup-view.v2",
        "fresh_owner": fresh_owner,
        "paths": {
            "source": "input/source.py",
            "harness": "input/harness_view.json",
            "source_query": "input/tools/context_query.py",
            "bounded_reference_query": (
                "python .agents/intentir_agent_tool.py lift_reference"
            ),
            "feedback_frontier": "input/feedback_frontier.json",
            "feedback_detail": "input/feedback_detail.json",
            "feedback_categories": [
                f"input/feedback_category_{name}.json"
                for name in _FEEDBACK_CATEGORY_NAMES
            ],
            "candidate_check": "input/candidate_check.json",
            "semantic_probe": "input/semantic_probe.json",
            "candidate": "output/candidate.json",
            "candidate_history": "work/candidate_history/",
        },
        "authority": (
            "path manifest only; exact bytes remain in files and no artifact "
            "field, semantic classification, or repair is generated"
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


def _atomic_write_json(path: Path, value: Mapping[str, Any] | None) -> None:
    text = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        if value is not None
        else "null\n"
    )
    _atomic_write_text(path, text)


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return totals


@dataclass
class CodexCLILiftAdapter(MiniSWELiftAdapter):
    """Persistent source-aware Codex owner across formal Gate1 continuations."""

    session_factory: Callable[..., Any] = field(
        default=CodexCLISession,
        repr=False,
    )
    repair_effort: str | None = None
    semantic_probe: Callable[[Any], Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
    )
    _codex_session: Any = field(default=None, init=False, repr=False)

    def close(self) -> None:
        session = self._codex_session
        self._codex_session = None
        if session is not None:
            session.close()
        super().close()

    def _remaining_seconds(self) -> float | None:
        if self.session_wall_time_limit_seconds in (None, 0):
            return None
        if (
            isinstance(self.session_wall_time_limit_seconds, bool)
            or not isinstance(self.session_wall_time_limit_seconds, (int, float))
            or not math.isfinite(float(self.session_wall_time_limit_seconds))
            or self.session_wall_time_limit_seconds < 0
        ):
            raise ValueError(
                "Lift owner session wall-time limit must be null, zero, or a "
                "finite non-negative number"
            )
        return self.session_wall_time_limit_seconds - (
            self._session_active_elapsed_ms / 1000.0
        )

    def _terminal_time_error(self, round_index: int) -> Exception:
        diagnostic = {
            "diagnostic": "branch-local Codex Lift owner wall-time exhausted",
            "failure_class": "lift_owner_time_exhausted",
            "session_id": self._session_id,
            "active_elapsed_ms": self._session_active_elapsed_ms,
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
                    "session_active_elapsed_ms": self._session_active_elapsed_ms,
                },
                continuation_candidate=(
                    self._read_continuation_candidate(self._session_workspace)
                    if self._session_workspace is not None
                    else None
                ),
            )
        )
        return CandidateLiftError(
            diagnostic,
            physical_calls=0,
            terminal=True,
        )

    def _invocation_limit_error(
        self,
        round_index: int,
        *,
        reason: str,
        invocation_turns: int,
        invocation_elapsed_ms: float,
        physical_calls: int,
        semantic_probe: Mapping[str, Any] | None = None,
    ) -> CandidateLiftError:
        """Stop one Codex invocation while retaining its exact current draft."""

        workspace = self._session_workspace
        candidate = (
            self._read_continuation_candidate(workspace)
            if workspace is not None
            else None
        )
        if candidate is not None:
            try:
                candidate = _value_free_lift_candidate_payload(candidate)
            except (TypeError, ValueError, RecursionError, json.JSONDecodeError):
                candidate = None
        diagnostic = (
            "Codex Lift invocation reached its bounded "
            f"{reason} limit before a complete candidate was returned"
        )
        diagnostics: dict[str, Any] = {
            "diagnostic": diagnostic,
            "failure_class": "lift_worker_no_candidate",
            "termination_status": "LiftInvocationLimit",
            "resource_limited": True,
            "retryable": True,
            "limit_kind": reason,
            "invocation_model_turns": invocation_turns,
            "invocation_elapsed_ms": invocation_elapsed_ms,
            "continuation": (
                "the persistent Lift owner workspace and Codex thread remain "
                "available to the next logical Lift invocation"
            ),
        }
        if semantic_probe is not None:
            diagnostics["semantic_probe"] = dict(semantic_probe)
        if candidate is not None:
            encoded = json.dumps(
                candidate,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
            diagnostics["continuation_draft"] = {
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
            }
        self._records.append(
            SCRCReliftRecord(
                round_index,
                diagnostic=diagnostic,
                backend_evidence={
                    "backend": CODEX_CLI_LIFT_BACKEND,
                    "runtime": CODEX_CLI_AGENT_RUNTIME,
                    "owner_session_id": self._session_id,
                    "thread_id": getattr(self._codex_session, "thread_id", None),
                    "resource_limited": True,
                    "limit_kind": reason,
                    "invocation_model_turns": invocation_turns,
                    "invocation_elapsed_ms": invocation_elapsed_ms,
                },
                continuation_candidate=candidate,
            )
        )
        return CandidateLiftError(
            diagnostics,
            physical_calls=max(0, physical_calls),
            terminal=False,
            candidate_draft=candidate,
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

    @staticmethod
    def _archive_formal_candidate(workspace: Any, feedback: Any) -> None:
        """Preserve this owner's exact formal candidates without selecting one."""

        formal_round = getattr(feedback, "formal_round", None)
        artifact = getattr(feedback, "previous_artifact", None)
        kind = getattr(getattr(feedback, "kind", None), "value", None)
        if (
            kind != "gate1"
            or not isinstance(formal_round, int)
            or isinstance(formal_round, bool)
            or formal_round < 1
            or artifact is None
        ):
            return
        payload = getattr(artifact, "to_json", None)
        if not callable(payload):
            return
        encoded = (
            json.dumps(
                _unbound_lift_artifact_payload(payload()),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        history_dir = workspace.scratch_dir / "candidate_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        destination = history_dir / f"gate1_round_{formal_round:03d}.json"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    "branch-local formal candidate history changed for one round"
                )
            return
        _atomic_write_text(destination, encoded)

    def lift(self, request: Any, feedback: Any) -> Any:
        if self.workflow != "single":
            raise ValueError("Codex CLI Lift supports only the single-owner workflow")
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Codex CLI Lift requires SourceProgram.text")
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        if self._session_source_sha256 is None:
            self._session_source_sha256 = source_sha256
        elif self._session_source_sha256 != source_sha256:
            raise ValueError("one Codex Lift owner cannot cross Source programs")

        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        prepared = prepare_scrc_lift(
            source,
            compiler_view=self.compiler_view,
            relift_context=self._relift_context(feedback),
            source_symbol=(
                getattr(source_program, "lift_source_symbol", None)
                or self.source_symbol
            ),
            source_kind=getattr(source_program, "lift_source_kind", "python"),
            source_parameter_order=tuple(
                getattr(source_program, "lift_parameter_order", ())
            ),
            source_to_opaque=(
                dict(getattr(source_program, "lift_source_to_opaque", ()))
                or None
            ),
        )
        candidate_check_context = scrc_candidate_check_context(prepared)
        round_index = len(self._records) + 1
        invocation_turn_limit, invocation_wall_time_limit = _lift_invocation_limits(
            self.worker_config
        )
        files, references = _codex_input_files(prepared)
        try:
            fresh_workspace = self._create_or_refresh_session_workspace(
                files,
                mutable_paths=frozenset(
                    {
                        "feedback_frontier.json",
                        "feedback_detail.json",
                        *{
                            f"feedback_category_{name}.json"
                            for name in _FEEDBACK_CATEGORY_NAMES
                        },
                        "candidate_check.json",
                        "semantic_probe.json",
                    }
                ),
            )
        except ImportError as exc:
            from intentir.verify.scrc_runner import SCRCInfrastructureError

            raise SCRCInfrastructureError(
                "Codex CLI Lift requires the lift-agent dependencies",
                evidence={"backend": CODEX_CLI_LIFT_BACKEND},
            ) from exc
        workspace = self._session_workspace
        if workspace is None:
            raise RuntimeError("Codex Lift workspace was not created")
        self._archive_formal_candidate(workspace, feedback)
        fresh_owner = self._codex_session is None
        previous_artifact = getattr(feedback, "previous_artifact", None)
        previous_candidate = getattr(feedback, "previous_candidate", None)
        candidate_path = workspace.output_dir / "candidate.json"
        if fresh_owner and not candidate_path.exists() and (
            previous_artifact is not None or previous_candidate is not None
        ):
            # Cross-process continuation still starts a fresh isolated coding
            # session, but it must not force the model to reserialize the last
            # complete artifact from prose.  This is an exact copy of the prior
            # model-authored candidate; fixed code neither edits nor completes it.
            payload = (
                _value_free_lift_candidate_payload(previous_candidate)
                if isinstance(previous_candidate, Mapping)
                else _unbound_lift_artifact_payload(previous_artifact.to_json())
            )
            _atomic_write_text(
                candidate_path,
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
            )
        call_turns: list[CodexCLITurn] = []
        check_results: list[dict[str, Any]] = []
        source_reconcile_results: list[dict[str, Any]] = []
        semantic_probe_results: list[dict[str, Any]] = []
        semantic_probe_by_candidate: dict[str, dict[str, Any]] = {}
        semantic_probe_feedback_delivered: set[str] = set()
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
        invocation_started = time.monotonic()
        invocation_turns = 0

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
                check = check_scrc_lift_candidate(
                    candidate_text,
                    candidate_check_context,
                )
                if check["status"] == "pass":
                    try:
                        # CandidateCheck remains value-free. The adjacent
                        # submission boundary enforces Source-name privacy using
                        # only the already-authored candidate; BindSource and all
                        # realization checks run after owner acceptance.
                        provisional = _decode_value_free_lift_candidate(
                            candidate_text,
                            expected_parameters=(
                                candidate_check_context.expected_parameters
                            ),
                            semantic_compiler_view=(
                                candidate_check_context.semantic_compiler_view
                            ),
                            required_schema=(
                                candidate_check_context.schema_version
                            ),
                        )
                        _validate_source_identifier_privacy(
                            provisional,
                            prepared.harness_binding,
                        )
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
            """Emit an inert legacy receipt without consulting realization state.

            The field remains in telemetry and the submit response for consumers
            of the earlier wire shape. It cannot accept, reject, diagnose, or
            modify J/C/pi and deliberately reads no Source TTIR, compiler result,
            backend metadata, candidate structure, or C value.
            """

            nonlocal candidate_text, last_checked_good
            result = {
                "protocol": _INACTIVE_SOURCE_RECONCILE_PROTOCOL,
                "status": "pass",
                "applicable": False,
                "diagnostics": [],
                "authority": (
                    "compatibility receipt only; compiler realization is not "
                    "a J authorship or Lift-acceptance authority"
                ),
            }
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
                and candidate_text is not None
            ):
                # Exact LLM-authored bytes only. CandidateCheck is enforced by
                # callers; this compatibility receipt adds no acceptance rule.
                last_checked_good = candidate_text
            return result

        def submit_candidate_tool() -> Mapping[str, Any]:
            """Run CandidateCheck over one exact model-authored byte sequence."""

            with evaluation_lock:
                candidate_check = check_current_candidate(
                    trigger="submit_candidate"
                )
                source_reconcile = reconcile_current_candidate(
                    trigger="submit_candidate",
                    checkpoint_allowed=candidate_check["status"] == "pass",
                )
                passed = candidate_check["status"] == "pass"
                if passed:
                    accepted_submission_event.set()
            return {
                "protocol": "intentir.codex-cli.lift-submit.v1",
                "status": "pass" if passed else "fail",
                "phase": (
                    "complete"
                    if passed
                    else "candidate_check"
                ),
                "candidate_check": candidate_check,
                "source_reconcile": source_reconcile,
                "authority": (
                    "CandidateCheck is the only Lift acceptance preflight; the "
                    "source_reconcile field is an inert compatibility receipt and "
                    "no artifact field is generated or repaired"
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

            nonlocal candidate_text, last_checked_good
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
                if checked_text == observed_text:
                    if candidate_check["status"] != "pass":
                        _atomic_write_json(
                            workspace.input_dir / "candidate_check.json",
                            candidate_check,
                        )
                    # A failing passive checkpoint must not split one coding-agent
                    # tool loop into a new provider turn.  The owner receives the
                    # same closed diagnostics synchronously through its explicit
                    # ``submit_candidate`` tool and can patch/recheck in the current
                    # turn.  Only a passing checkpoint may interrupt early.

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
            elif probe_candidate_text is None or self.semantic_probe is None:
                result = {
                    "protocol": "intentir.lift.public-semantic-probe.v1",
                    "status": "unavailable",
                    "phase": "execution",
                    "diagnostic": "this Lift invocation has no public probe runner",
                }
            else:
                try:
                    artifact = decode_scrc_lift_artifact(
                        probe_candidate_text, prepared
                    )
                except Exception:
                    # CandidateCheck has already accepted the value-free draft.
                    # Any remaining Decode failure belongs to private
                    # BindSource/Source realization and must not become an
                    # owner-visible repair hint through this diagnostic tool.
                    result = {
                        "protocol": "intentir.lift.public-semantic-probe.v1",
                        "status": "unavailable",
                        "phase": "host_private_bind_source",
                        "diagnostic": (
                            "host-private BindSource rejected the accepted "
                            "Lift draft"
                        ),
                    }
                else:
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
            _atomic_write_json(
                workspace.input_dir / "semantic_probe.json",
                result,
            )
            return result

        def semantic_probe_needs_owner_turn() -> bool:
            """Run one public probe per exact passing draft and surface divergence."""

            if candidate_text is None:
                return False
            candidate_sha256 = hashlib.sha256(candidate_text.encode()).hexdigest()
            result = semantic_probe_by_candidate.get(candidate_sha256)
            if result is None:
                result = dict(semantic_probe_tool())
            # Only an executed Source-vs-J mismatch is a counterexample for the
            # Lift owner.  RefJ construction ``unavailable`` reports a consumer
            # capability gap; feeding it back makes J conform to an incomplete
            # evaluator and can overwrite an already well-typed semantic artifact.
            actionable = result.get("status") == "diverged"
            if (
                actionable
                and candidate_sha256 not in semantic_probe_feedback_delivered
            ):
                semantic_probe_feedback_delivered.add(candidate_sha256)
                return True
            return False

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "submit_candidate": submit_candidate_tool,
                "semantic_probe": semantic_probe_tool,
            },
            argument_handlers={
                "lift_reference": lambda arguments: _lift_reference_query(
                    references, arguments
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
                workspace, fresh_owner=fresh_owner
            )
            prompt = (
                (_INITIAL_PROMPT if fresh_owner else _EXTERNAL_CONTINUATION_PROMPT)
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
            selected_effort = effective_effort
            while True:
                remaining_session = self._remaining_seconds()
                remaining_invocation = (
                    invocation_wall_time_limit
                    - (time.monotonic() - invocation_started)
                    if invocation_wall_time_limit is not None
                    else None
                )
                current_physical_calls = (
                    sum(turn.model_segments for turn in call_turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                )
                if (
                    remaining_invocation is not None
                    and remaining_invocation <= 0
                ):
                    raise self._invocation_limit_error(
                        round_index,
                        reason="wall time",
                        invocation_turns=invocation_turns,
                        invocation_elapsed_ms=(
                            time.monotonic() - invocation_started
                        )
                        * 1000.0,
                        physical_calls=current_physical_calls,
                        semantic_probe=(
                            semantic_probe_results[-1]
                            if semantic_probe_results
                            else None
                        ),
                    )
                if invocation_turns >= invocation_turn_limit:
                    raise self._invocation_limit_error(
                        round_index,
                        reason="model-turn",
                        invocation_turns=invocation_turns,
                        invocation_elapsed_ms=(
                            time.monotonic() - invocation_started
                        )
                        * 1000.0,
                        physical_calls=current_physical_calls,
                        semantic_probe=(
                            semantic_probe_results[-1]
                            if semantic_probe_results
                            else None
                        ),
                    )
                if remaining_session is not None and remaining_session <= 0:
                    raise self._terminal_time_error(round_index)
                turn_limits = [
                    value
                    for value in (
                        remaining_session,
                        remaining_invocation,
                        _lift_turn_timeout(
                            selected_effort,
                            self.model_timeout_seconds,
                        ),
                    )
                    if value is not None
                ]
                turn_timeout = min(turn_limits) if turn_limits else None
                invocation_turns += 1
                try:
                    turn = self._codex_session.run(
                        prompt,
                        timeout_seconds=turn_timeout,
                        effort=selected_effort,
                        stop_requested=self.stop_requested,
                        submission_accepted=accepted_submission_event.is_set,
                        diagnostic_ready=diagnostic_ready_event.is_set,
                    )
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                except CodexCLICancelled as exc:
                    self._session_active_elapsed_ms += exc.elapsed_ms
                    self._session_model_turns += 1
                    self._session_physical_provider_calls += exc.physical_calls
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
                        "semantic_probe": list(semantic_probe_results),
                        "live_checkpoint": list(live_checkpoint_events),
                        "tool_bridge": bridge.evidence,
                    }
                    self._records.append(
                        SCRCReliftRecord(
                            round_index,
                            diagnostic=str(exc),
                            backend_evidence=cancellation_evidence,
                            continuation_candidate=(
                                self._read_continuation_candidate(workspace)
                            ),
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
                    self._session_active_elapsed_ms += exc.elapsed_ms
                    self._session_model_turns += 1
                    self._session_physical_provider_calls += exc.physical_calls
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
                        or candidate_text is None
                    ):
                        raise CodexCLIError(
                            "accepted submission failed independent host recheck"
                        ) from exc
                    accepted_submission_interrupts[-1]["recovery"] = (
                        "exact_checkpoint_restored"
                    )
                    if semantic_probe_needs_owner_turn():
                        prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                        selected_effort = self.repair_effort or effective_effort
                        continue
                    break
                except CodexCLIDiagnosticReady as exc:
                    self._session_active_elapsed_ms += exc.elapsed_ms
                    self._session_model_turns += 1
                    self._session_physical_provider_calls += exc.physical_calls
                    diagnostic_ready_event.clear()
                    accepted_submission_event.clear()
                    diagnostic_ready_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "same_thread_closed_diagnostic",
                        }
                    )
                    check, source_reconcile = host_preflight(
                        trigger="host_live_diagnostic_recheck"
                    )
                    if (
                        check["status"] == "pass"
                        and candidate_text is not None
                    ):
                        if semantic_probe_needs_owner_turn():
                            prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                            selected_effort = self.repair_effort or effective_effort
                            continue
                        break
                    if check["status"] != "pass":
                        _atomic_write_json(
                            workspace.input_dir / "candidate_check.json",
                            check,
                        )
                        prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                    else:
                        raise CodexCLIError(
                            "passing CandidateCheck produced no candidate bytes"
                        )
                    selected_effort = self.repair_effort or effective_effort
                    continue
                except CodexCLIError as exc:
                    if "wall-time limit" in str(exc):
                        elapsed = turn_timeout * 1000.0
                        self._session_active_elapsed_ms += elapsed
                        self._session_model_turns += 1
                        self._session_physical_provider_calls += 1
                        timed_out_calls += 1
                        timed_out_elapsed_ms += elapsed
                        timeout_event = {
                            "turn_timeout_seconds": turn_timeout,
                            "thread_id": getattr(
                                self._codex_session, "thread_id", None
                            ),
                            "recovery": None,
                        }
                        timeout_events.append(timeout_event)

                        check, source_reconcile = host_preflight(
                            trigger="host_timeout_recheck"
                        )
                        if (
                            check["status"] == "pass"
                            and candidate_text is not None
                        ):
                            if semantic_probe_needs_owner_turn():
                                timeout_event["recovery"] = (
                                    "current_exact_candidate_semantic_probe"
                                )
                                prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                                selected_effort = self.repair_effort or effective_effort
                                continue
                            timeout_event["recovery"] = "current_exact_candidate"
                            break
                        if last_checked_good is not None:
                            _atomic_write_text(
                                workspace.candidate_path, last_checked_good
                            )
                            check, source_reconcile = host_preflight(
                                trigger="host_timeout_checkpoint_restore"
                            )
                            if (
                                check["status"] == "pass"
                                and candidate_text is not None
                            ):
                                if semantic_probe_needs_owner_turn():
                                    timeout_event["recovery"] = (
                                        "checkpoint_semantic_probe"
                                    )
                                    prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                                    selected_effort = self.repair_effort or effective_effort
                                    continue
                                timeout_event["recovery"] = (
                                    "same_invocation_checked_checkpoint"
                                )
                                break

                        remaining_after_timeout = self._remaining_seconds()
                        remaining_invocation_after_timeout = (
                            invocation_wall_time_limit
                            - (time.monotonic() - invocation_started)
                            if invocation_wall_time_limit is not None
                            else None
                        )
                        if (
                            remaining_invocation_after_timeout is not None
                            and remaining_invocation_after_timeout <= 0
                        ):
                            timeout_event["recovery"] = "invocation_time_exhausted"
                            raise self._invocation_limit_error(
                                round_index,
                                reason="wall time",
                                invocation_turns=invocation_turns,
                                invocation_elapsed_ms=(
                                    time.monotonic() - invocation_started
                                )
                                * 1000.0,
                                physical_calls=(
                                    sum(turn.model_segments for turn in call_turns)
                                    + timed_out_calls
                                    + len(accepted_submission_interrupts)
                                    + len(diagnostic_ready_interrupts)
                                ),
                                semantic_probe=(
                                    semantic_probe_results[-1]
                                    if semantic_probe_results
                                    else None
                                ),
                            ) from exc
                        if (
                            remaining_after_timeout is not None
                            and remaining_after_timeout <= 0
                        ):
                            timeout_event["recovery"] = "owner_time_exhausted"
                            raise self._terminal_time_error(round_index) from exc
                        if getattr(self._codex_session, "thread_id", None) is None:
                            timeout_event["recovery"] = "no_resumable_thread"
                            raise CodexCLIError(
                                "initial Codex Lift turn timed out without a "
                                "resumable thread or checked candidate"
                            ) from exc
                        if check["status"] != "pass":
                            _atomic_write_json(
                                workspace.input_dir / "candidate_check.json",
                                check,
                            )
                            prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                        else:
                            raise CodexCLIError(
                                "passing CandidateCheck produced no candidate bytes"
                            )
                        timeout_event["recovery"] = "same_thread_continuation"
                        selected_effort = self.repair_effort or effective_effort
                        continue
                    raise
                call_turns.append(turn)
                self._session_active_elapsed_ms += turn.elapsed_ms
                self._session_model_turns += 1
                self._session_physical_provider_calls += turn.model_segments

                check, source_reconcile = host_preflight(trigger="host_post_turn")
                if (
                    check["status"] == "pass"
                    and candidate_text is not None
                ):
                    if semantic_probe_needs_owner_turn():
                        prompt = _SEMANTIC_PROBE_CONTINUATION_PROMPT
                        selected_effort = self.repair_effort or effective_effort
                        continue
                    break
                if check["status"] != "pass":
                    _atomic_write_json(
                        workspace.input_dir / "candidate_check.json",
                        check,
                    )
                    prompt = _CANDIDATE_CHECK_CONTINUATION_PROMPT
                else:
                    raise CodexCLIError(
                        "passing CandidateCheck produced no candidate bytes"
                    )
                selected_effort = self.repair_effort or effective_effort
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
                    continuation_candidate=self._read_continuation_candidate(
                        workspace
                    ),
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
        try:
            artifact = decode_scrc_lift_artifact(candidate_text, prepared)
        except Exception as exc:
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                {
                    "diagnostic": (
                        "host-private BindSource rejected the accepted Lift draft"
                    ),
                    "failure_class": "bind_source_rejected",
                    "authority": (
                        "host-private BindSource/Gate boundary; no Source "
                        "realization diagnostic is returned to the Lift owner"
                    ),
                },
                physical_calls=max(1, sum(turn.model_segments for turn in call_turns)),
                terminal=True,
            ) from exc
        usage = _aggregate_usage(call_turns)
        trajectory = {
            "protocol": CODEX_CLI_AGENT_RUNTIME,
            "backend": CODEX_CLI_LIFT_BACKEND,
            "thread_id": self._codex_session.thread_id,
            "turns": [turn.to_evidence() for turn in call_turns],
            "candidate_check": list(check_results),
            "source_reconcile": list(source_reconcile_results),
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
            termination_status="CandidateCheckPass",
            trajectory=trajectory,
        )
        extraction = SCRCExtraction(
            artifact,
            provenance,
            prepared.harness_binding,
            source,
        )
        backend_evidence = {
            "backend": CODEX_CLI_LIFT_BACKEND,
            "knowledge_profile": CODEX_CLI_LIFT_KNOWLEDGE,
            "runtime": CODEX_CLI_AGENT_RUNTIME,
            "fresh_session": fresh_owner,
            "fresh_workspace": fresh_workspace,
            "persistent_owner": True,
            "persistent_workspace": True,
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
            "invocation_model_turn_limit": invocation_turn_limit,
            "invocation_wall_time_limit_seconds": invocation_wall_time_limit,
            "invocation_model_turns": (
                len(call_turns)
                + timed_out_calls
                + len(accepted_submission_interrupts)
                + len(diagnostic_ready_interrupts)
            ),
            "invocation_elapsed_ms": elapsed_ms,
            "usage": usage,
            "candidate_check": list(check_results),
            "source_reconcile": list(source_reconcile_results),
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
                "artifact_schema": artifact.wire_schema,
                "layer_c_interface": extraction.layer_c_interface_payload,
                "layer_c_interface_digest": (
                    extraction.layer_c_interface_digest
                ),
                "decode_protocol": candidate_check_context.decode_protocol,
                "typecheck_protocol": candidate_check_context.typecheck_protocol,
                "session_active_elapsed_ms": self._session_active_elapsed_ms,
                "session_cli_turns": self._session_model_turns,
                "session_model_segments_lower_bound": (
                    self._session_physical_provider_calls
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
