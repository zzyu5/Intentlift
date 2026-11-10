"""SCRC Gate2 three-way execution and external-Q pairwise closure."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import SemanticArchive

from .compare import compare_outputs, validate_j_inputs
from .model import (
    ExecutionBinding,
    ExecutionResult,
    Gate2Result,
    Gate2Status,
    Gate2SuiteResult,
    execution_spec_contract,
    json_sha256,
    mapping_digest,
    output_binding_digest,
    output_digest,
    program_identity_digests,
)
from .policy import (
    ComparisonPolicy,
    ComparisonPolicyError,
    DEFAULT_GATE2_POLICY,
)
from .scalar import (
    ScalarLoweringRequest,
    ScalarLoweringRunner,
    ScalarLoweringSpec,
    ScalarLoweringUnavailable,
)
from .triton import TritonExecutionSpec, TritonRunner


GATE2_EXECUTION_PROTOCOL = "source_replay_j_three_edges_q_v5"


def _input_digest(values: Mapping[str, Any]) -> str:
    """Hash exact trial inputs without lossy JSON float serialization."""

    digest = hashlib.sha256()
    digest.update(b"intentir-gate2-inputs-v1\0")
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise ValueError("Gate2 input names must be non-empty strings")
        value = np.asarray(values[name])
        if value.dtype.hasobject:
            raise ValueError(f"Gate2 input {name!r} has object dtype")
        dense = np.ascontiguousarray(value)
        header = json.dumps(
            {"name": name, "dtype": dense.dtype.str, "shape": list(dense.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = dense.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def clone_gate2_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Give every Gate2 leg an isolated view of the committed pre-state."""

    result: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            result[name] = value.copy()
        else:
            result[name] = copy.deepcopy(value)
    return result


_clone_values = clone_gate2_values


@dataclass(frozen=True)
class Gate2Case:
    case_id: str
    j: SemanticArchive
    values: Mapping[str, Any]
    triton: TritonExecutionSpec
    refeval: ScalarLoweringRequest
    output_keys: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Gate2 case_id must not be empty")
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("Gate2 case requires the canonical J object")
        if not self.output_keys:
            raise ValueError("Gate2 requires at least one output key")
        if self.refeval.j is not self.j:
            raise ValueError(
                "Gate2 RefEval request and comparator must receive the same J object"
            )
        if isinstance(self.refeval, ScalarLoweringSpec):
            reference_outputs = set(self.refeval.output_binding_map.values())
            if set(self.output_keys) != reference_outputs:
                raise ValueError(
                    "Gate2 compared outputs differ from RefEval output bindings"
                )
        if len(self.output_keys) != len(set(self.output_keys)):
            raise ValueError("Gate2 output keys must be unique")


@dataclass(frozen=True)
class Gate2Suite:
    suite_id: str
    cases: tuple[Gate2Case, ...]
    c_values: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case")
        names = [name for name, _ in self.c_values]
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(
                "Gate2 suite c0 bindings must have unique non-empty slot names"
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite case_id values must be unique")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class _LineageValidationError(ValueError):
    pass


def _manifest_execution_identity(case: Gate2Case) -> dict[str, Any]:
    """Validate and return the execution snapshot carried by a Gate2 case."""

    payload = case.metadata.get("generated_target_source_manifest")
    digest = case.metadata.get("generated_target_source_manifest_digest")
    if not isinstance(payload, Mapping) or not _is_sha256(digest):
        raise _LineageValidationError(
            "Gate2 case lacks a complete generated-source manifest binding"
        )
    if json_sha256(payload) != digest:
        raise _LineageValidationError(
            "Gate2 case manifest payload does not match its digest"
        )
    required = (
        "source_sha256",
        "source_slice_sha256",
        "source_location",
        "device_entry_symbol",
    )
    if any(key not in payload for key in required):
        raise _LineageValidationError(
            "Gate2 case manifest is missing source identity fields"
        )
    if any(
        not isinstance(payload[key], str) or not payload[key]
        for key in required
    ):
        raise _LineageValidationError(
            "Gate2 case manifest source identity fields are malformed"
        )
    for digest_key in ("source_sha256", "source_slice_sha256"):
        if not _is_sha256(payload[digest_key]):
            raise _LineageValidationError(
                f"Gate2 case manifest field {digest_key!r} is not a SHA-256"
            )
    output_binding = case.metadata.get("manifest_output_binding_digest")
    if not _is_sha256(output_binding):
        raise _LineageValidationError(
            "Gate2 case lacks the manifest output-binding digest"
        )
    if case.metadata.get("source_sha256") != payload["source_sha256"]:
        raise _LineageValidationError("Gate2 source SHA lineage differs from manifest")
    if case.metadata.get("source_slice_sha256") != payload["source_slice_sha256"]:
        raise _LineageValidationError(
            "Gate2 source-slice lineage differs from manifest"
        )
    if case.metadata.get("source_location") != payload["source_location"]:
        raise _LineageValidationError(
            "Gate2 source location lineage differs from manifest"
        )
    if case.metadata.get("device_entry_symbol") != payload["device_entry_symbol"]:
        raise _LineageValidationError(
            "Gate2 device-entry lineage differs from manifest"
        )
    return dict(payload)


def _make_execution_binding(
    program: Any,
    spec: Any,
    case: Gate2Case,
    *,
    leg: str,
    runner: Any,
) -> ExecutionBinding:
    """Create the per-leg binding before the runner reaches compile/warmup."""

    manifest = _manifest_execution_identity(case)
    frontend = str(case.metadata.get("frontend", "triton"))
    raw_program_digest, executed_program_digest = program_identity_digests(
        program,
        frontend=frontend,
    )
    if leg == "source":
        expected_raw = case.metadata.get("source_program_sha256")
        expected_executed = case.metadata.get("source_execution_program_sha256")
        if not _is_sha256(expected_raw) or not _is_sha256(expected_executed):
            raise _LineageValidationError(
                "Gate2 case lacks the prepared Source program identity"
            )
    else:
        # The final accepted Replay is already available at this boundary.  Its
        # exact bytes are bound now, and the runner re-reads them immediately
        # before compilation so a changed module/file cannot pass by metadata.
        expected_raw = raw_program_digest
        expected_executed = executed_program_digest
    contract = execution_spec_contract(spec, leg=leg)
    compiler_identity = contract["compiler_identity"]
    del runner
    return ExecutionBinding(
        leg=leg,
        manifest_digest=str(case.metadata["generated_target_source_manifest_digest"]),
        source_sha256=manifest["source_sha256"],
        source_slice_sha256=manifest["source_slice_sha256"],
        source_location=manifest["source_location"],
        selected_device_entry=manifest["device_entry_symbol"],
        program_source_sha256=expected_raw,
        program_execution_sha256=expected_executed,
        program_entry=str(getattr(program, "function")),
        abi_order=contract["abi_order"],
        abi_digest=contract["abi_digest"],
        schedule_names=contract["schedule_names"],
        schedule_digest=contract["schedule_digest"],
        compiler_identity=compiler_identity,
        compiler_digest=contract["compiler_digest"],
        grid=contract["grid"],
        block=contract["block"],
        shared_mem=contract["shared_mem"],
        output_keys=tuple(case.output_keys),
        manifest_output_binding_digest=str(
            case.metadata["manifest_output_binding_digest"]
        ),
    )


def _bind_execution_spec(spec: Any, binding: ExecutionBinding) -> Any:
    if not hasattr(spec, "execution_binding"):
        raise _LineageValidationError(
            "Gate2 execution spec has no execution_binding boundary"
        )
    return replace(spec, execution_binding=binding)


def _validate_execution_lineage(
    result: ExecutionResult,
    *,
    expected_source_slice: str,
    expected_manifest_digest: str,
    require_artifact: bool,
    require_source_lineage: bool = True,
    expected_binding: ExecutionBinding | None = None,
) -> dict[str, Any]:
    """Validate runner-owned digests against the returned real observations."""

    if not result.ok:
        raise _LineageValidationError(
            f"{result.leg} execution did not produce lineage: {result.error}"
        )
    metadata = dict(result.metadata)
    if expected_binding is not None:
        if metadata.get("execution_binding_digest") != expected_binding.digest:
            raise _LineageValidationError(
                f"{result.leg} execution binding digest is missing or differs"
            )
        if metadata.get("execution_binding") != expected_binding.to_json():
            raise _LineageValidationError(
                f"{result.leg} execution binding payload differs"
            )
    if require_source_lineage and metadata.get("source_slice_sha256") != expected_source_slice:
        raise _LineageValidationError(
            f"{result.leg} source-slice lineage differs from the prepared manifest"
        )
    if require_source_lineage and metadata.get("generated_target_source_manifest_digest") != expected_manifest_digest:
        raise _LineageValidationError(
            f"{result.leg} generated-source manifest lineage differs from the case"
        )
    actual_output_digest = output_digest(result.outputs)
    actual_binding_digest = output_binding_digest(result.outputs)
    if metadata.get("output_digest") != actual_output_digest:
        raise _LineageValidationError(
            f"{result.leg} output digest is missing or does not match returned bytes"
        )
    if metadata.get("output_binding_digest") != actual_binding_digest:
        raise _LineageValidationError(
            f"{result.leg} output binding digest is missing or does not match returned outputs"
        )
    if expected_binding is not None and tuple(result.outputs) != expected_binding.output_keys:
        raise _LineageValidationError(
            f"{result.leg} output keys differ from the bound manifest ABI"
        )
    if require_artifact:
        if not _is_sha256(metadata.get("artifact_digest")):
            raise _LineageValidationError(
                f"{result.leg} did not expose a real compiled artifact digest"
            )
        artifact_kind = metadata.get("artifact_kind")
        if not isinstance(artifact_kind, str) or not artifact_kind:
            raise _LineageValidationError(
                f"{result.leg} did not expose a compiled artifact kind"
            )
    return metadata


def _collect_case_lineage(
    case: Gate2Case,
    source_result: ExecutionResult,
    replay_result: ExecutionResult,
    reference_result: ExecutionResult,
    source_binding: ExecutionBinding | None = None,
    replay_binding: ExecutionBinding | None = None,
) -> dict[str, Any]:
    """Build the ordered manifest lifecycle from actual Gate2 leg evidence."""

    if not case.metadata.get("lineage_required"):
        return {}
    expected_slice = case.metadata.get("source_slice_sha256")
    expected_manifest = case.metadata.get("generated_target_source_manifest_digest")
    if not _is_sha256(expected_slice) or not _is_sha256(expected_manifest):
        raise _LineageValidationError(
            "Gate2 case requested lineage but lacks a valid manifest/slice digest"
        )
    source_metadata = _validate_execution_lineage(
        source_result,
        expected_source_slice=expected_slice,
        expected_manifest_digest=expected_manifest,
        require_artifact=True,
        expected_binding=source_binding,
    )
    replay_metadata = _validate_execution_lineage(
        replay_result,
        expected_source_slice=expected_slice,
        expected_manifest_digest=expected_manifest,
        require_artifact=True,
        expected_binding=replay_binding,
    )
    frontend = str(
        case.metadata.get(
            "frontend",
            source_metadata.get("frontend", "triton"),
        )
    )
    compile_digest = mapping_digest(
        {
            "source": source_metadata["artifact_digest"],
            "replay": replay_metadata["artifact_digest"],
        },
        purpose=f"gate2-{frontend}-compiled-pair",
    )
    target_binding_digest = mapping_digest(
        {
            "source": source_metadata["output_binding_digest"],
            "replay": replay_metadata["output_binding_digest"],
        },
        purpose=f"gate2-{frontend}-target-output-bindings",
    )
    consumer = f"gate2.{frontend}.source+replay"
    records: list[dict[str, Any]] = [
        {
            "phase": "source_compile",
            "consumer": consumer,
            "artifact_kind": f"{frontend}.compiled_pair",
            "artifact_digest": compile_digest,
            "source_slice_sha256": expected_slice,
            "output_digest": None,
            "output_binding_digest": None,
        },
        {
            "phase": "source_execute",
            "consumer": consumer,
            "artifact_kind": f"{frontend}.compiled_pair",
            "artifact_digest": compile_digest,
            "source_slice_sha256": expected_slice,
            "output_digest": None,
            "output_binding_digest": target_binding_digest,
        },
    ]
    if reference_result.unsupported:
        return {
            "lineage_status": "target_execution_complete_reference_unsupported",
            "lineage_complete": False,
            "lifecycle_records": records,
            "source_artifact_digest": source_metadata["artifact_digest"],
            "replay_artifact_digest": replay_metadata["artifact_digest"],
        }
    if not reference_result.ok:
        return {
            "lineage_status": "target_execution_complete_reference_failed",
            "lineage_complete": False,
            "lifecycle_records": records,
            "source_artifact_digest": source_metadata["artifact_digest"],
            "replay_artifact_digest": replay_metadata["artifact_digest"],
        }
    reference_metadata = _validate_execution_lineage(
        reference_result,
        expected_source_slice=expected_slice,
        expected_manifest_digest=expected_manifest,
        require_artifact=False,
        require_source_lineage=False,
    )
    combined_outputs: dict[str, np.ndarray] = {}
    for leg, result in (
        ("source", source_result),
        ("replay", replay_result),
        ("j", reference_result),
    ):
        combined_outputs.update(
            {f"{leg}:{name}": np.asarray(value) for name, value in result.outputs.items()}
        )
    all_output_digest = output_digest(combined_outputs)
    all_output_bindings = output_binding_digest(combined_outputs)
    records.append(
        {
            "phase": "output_digest",
            "consumer": f"gate2.{frontend}.source+replay+j",
            "artifact_kind": "outputs",
            "artifact_digest": all_output_digest,
            "source_slice_sha256": expected_slice,
            "output_digest": all_output_digest,
            "output_binding_digest": all_output_bindings,
        }
    )
    return {
        "lineage_status": "execution_complete_reingress_pending",
        "lineage_complete": False,
        "lifecycle_records": records,
        "source_artifact_digest": source_metadata["artifact_digest"],
        "replay_artifact_digest": replay_metadata["artifact_digest"],
        "reference_output_digest": reference_metadata["output_digest"],
    }


class Gate2Runner:
    def __init__(
        self,
        *,
        source_runner: TritonRunner | None = None,
        replay_runner: TritonRunner | None = None,
        refeval_runner: ScalarLoweringRunner | None = None,
        comparison_policy: ComparisonPolicy | None = None,
    ) -> None:
        self.source_runner = source_runner or TritonRunner()
        self.replay_runner = replay_runner or TritonRunner()
        self.refeval_runner = refeval_runner or ScalarLoweringRunner()
        self.comparison_policy = comparison_policy or DEFAULT_GATE2_POLICY

    def run(
        self,
        source: KernelSource,
        replay: KernelSource,
        case: Gate2Case,
    ) -> Gate2Result:
        case_started = time.monotonic()
        stage_metrics: dict[str, dict[str, Any]] = {}

        def record_stage(
            name: str,
            started: float,
            status: str,
            **details: Any,
        ) -> None:
            stage_metrics[name] = {
                "elapsed_ms": max(
                    0.0,
                    (time.monotonic() - started) * 1000.0,
                ),
                "runs": 1,
                "status": status,
                **details,
            }

        def run_execution(
            name: str,
            callback: Any,
        ) -> ExecutionResult:
            started = time.monotonic()
            try:
                result = callback()
            except Exception as exc:
                result = ExecutionResult(
                    name,
                    False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            elapsed_ms = max(0.0, (time.monotonic() - started) * 1000.0)
            result.metadata = {
                **dict(result.metadata),
                "elapsed_ms": elapsed_ms,
                "execution_stage": name,
            }
            record_stage(
                name,
                started,
                "passed" if result.ok else (
                    "unsupported" if result.unsupported else "failed"
                ),
            )
            return result

        def finish(result: Gate2Result) -> Gate2Result:
            result.metadata = {
                **dict(result.metadata),
                "stage_metrics": {
                    **stage_metrics,
                    "gate2_case_total": {
                        "elapsed_ms": max(
                            0.0,
                            (time.monotonic() - case_started) * 1000.0,
                        ),
                        "runs": 1,
                        "status": result.status.value,
                    },
                },
            }
            return result

        input_digest = _input_digest(case.values)
        q_metadata = {
            "protocol": GATE2_EXECUTION_PROTOCOL,
            "comparison_policy": self.comparison_policy.to_json(),
        }
        try:
            self.comparison_policy.validate_j(case.j)
        except ComparisonPolicyError as exc:
            error = f"Gate2 comparison policy rejected J: {exc}"
            failed = ExecutionResult(
                "source", False, error=error, metadata=q_metadata
            )
            return finish(Gate2Result(
                case_id=case.case_id,
                status=Gate2Status.COMPARISON_UNSUPPORTED,
                source=failed,
                replay=ExecutionResult(
                    "replay", False, error=error, metadata=q_metadata
                ),
                reference=ExecutionResult(
                    "j", False, error=error, metadata=q_metadata
                ),
                input_digest=input_digest,
                metadata={**dict(case.metadata), **q_metadata},
            ))
        if isinstance(case.refeval, ScalarLoweringSpec):
            try:
                validate_j_inputs(
                    case.j,
                    case.refeval.input_binding_map,
                    case.values,
                )
            except (NotImplementedError, TypeError, ValueError) as exc:
                error = f"J input precondition failed: {type(exc).__name__}: {exc}"
                metadata = {
                    "protocol": GATE2_EXECUTION_PROTOCOL,
                    "failure_class": "j_input_precondition",
                    "phase": "input_binding_validation",
                }
                return finish(Gate2Result(
                    case_id=case.case_id,
                    status=Gate2Status.EXECUTION_FAILED,
                    source=ExecutionResult("source", False, error=error, metadata=metadata),
                    replay=ExecutionResult("replay", False, error=error, metadata=metadata),
                    reference=ExecutionResult("j", False, error=error, metadata=metadata),
                    input_digest=input_digest,
                    metadata={**dict(case.metadata), **q_metadata},
                ))

        source_spec = case.triton
        replay_spec = case.triton
        source_binding: ExecutionBinding | None = None
        replay_binding: ExecutionBinding | None = None
        if case.metadata.get("lineage_required"):
            try:
                source_binding = _make_execution_binding(
                    source,
                    source_spec,
                    case,
                    leg="source",
                    runner=self.source_runner,
                )
                replay_binding = _make_execution_binding(
                    replay,
                    replay_spec,
                    case,
                    leg="replay",
                    runner=self.replay_runner,
                )
                source_spec = _bind_execution_spec(source_spec, source_binding)
                replay_spec = _bind_execution_spec(replay_spec, replay_binding)
            except Exception as exc:
                error = (
                    "Gate2 execution binding could not be established: "
                    f"{type(exc).__name__}: {exc}"
                )
                metadata = {
                    **dict(case.metadata),
                    **q_metadata,
                    "failure_class": "lineage_invalid",
                    "phase": "execution_binding",
                    "lineage_status": "failed_closed",
                    "lineage_failure": error,
                }
                return finish(Gate2Result(
                    case_id=case.case_id,
                    status=Gate2Status.EXECUTION_FAILED,
                    source=ExecutionResult("source", False, error=error, metadata=metadata),
                    replay=ExecutionResult("replay", False, error=error, metadata=metadata),
                    reference=ExecutionResult("j", False, error=error, metadata=metadata),
                    input_digest=input_digest,
                    metadata=metadata,
                ))
        source_result = run_execution(
            "source_execution",
            lambda: self.source_runner.run(
                "source", source, source_spec, _clone_values(case.values)
            ),
        )
        replay_result = run_execution(
            "replay_execution",
            lambda: self.replay_runner.run(
                "replay", replay, replay_spec, _clone_values(case.values)
            ),
        )
        reference_result = run_execution(
            "refeval_execution",
            (
                case.refeval.execution_result
                if isinstance(case.refeval, ScalarLoweringUnavailable)
                else lambda: self.refeval_runner.run(
                    case.refeval, _clone_values(case.values)
                )
            ),
        )
        try:
            lineage_metadata = _collect_case_lineage(
                case,
                source_result,
                replay_result,
                reference_result,
                source_binding,
                replay_binding,
            )
        except _LineageValidationError as exc:
            metadata = {
                **dict(case.metadata),
                **q_metadata,
                "failure_class": "lineage_invalid",
                "phase": "execution_lineage",
                "lineage_status": "failed_closed",
                "lineage_failure": str(exc),
            }
            return finish(Gate2Result(
                case_id=case.case_id,
                status=Gate2Status.EXECUTION_FAILED,
                source=source_result,
                replay=replay_result,
                reference=reference_result,
                input_digest=input_digest,
                metadata=metadata,
            ))
        comparison_kwargs = (
            {
                "j": case.j,
                "input_bindings": case.refeval.input_binding_map,
                "output_bindings": case.refeval.output_binding_map,
                "inputs": case.values,
                "policy": self.comparison_policy,
            }
            if isinstance(case.refeval, ScalarLoweringSpec)
            else None
        )
        source_replay = None
        source_j = None
        replay_j = None
        if (
            source_result.ok
            and replay_result.ok
            and reference_result.ok
            and comparison_kwargs is not None
        ):
            comparison_started = time.monotonic()
            try:
                source_replay = compare_outputs(
                    "source",
                    source_result.outputs,
                    "replay",
                    replay_result.outputs,
                    **comparison_kwargs,
                )
                source_j = compare_outputs(
                    "source",
                    source_result.outputs,
                    "j",
                    reference_result.outputs,
                    **comparison_kwargs,
                )
                replay_j = compare_outputs(
                    "replay",
                    replay_result.outputs,
                    "j",
                    reference_result.outputs,
                    **comparison_kwargs,
                )
                record_stage(
                    "pairwise_comparison",
                    comparison_started,
                    "passed" if (
                        source_replay.passed
                        and source_j.passed
                        and replay_j.passed
                    ) else "failed",
                )
            except Exception as exc:
                record_stage(
                    "pairwise_comparison",
                    comparison_started,
                    "failed",
                )
                # A shape/binding mismatch at this point is a broken Gate2
                # case or Harness ABI, not a reference capability result.  Keep
                # the three execution records and classify the closure as an
                # execution failure with an explicit harness phase.
                error = (
                    "Gate2 comparison input/output binding is invalid: "
                    f"{type(exc).__name__}: {exc}"
                )
                metadata = {
                    **dict(case.metadata),
                    **q_metadata,
                    "failure_class": "harness_invalid",
                    "phase": "pairwise_comparison",
                    "error": error,
                }
                return finish(Gate2Result(
                    case_id=case.case_id,
                    status=Gate2Status.EXECUTION_FAILED,
                    source=source_result,
                    replay=replay_result,
                    reference=reference_result,
                    input_digest=input_digest,
                    metadata=metadata,
                ))

        if not source_result.ok or not replay_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        elif reference_result.unsupported:
            status = Gate2Status.REFERENCE_UNSUPPORTED
        elif not reference_result.ok:
            status = Gate2Status.EXECUTION_FAILED
        else:
            assert source_replay is not None and source_j is not None
            assert replay_j is not None
            if source_replay.passed and source_j.passed and replay_j.passed:
                status = Gate2Status.PASS
            else:
                status = Gate2Status.PAIRWISE_DIVERGED
        return finish(Gate2Result(
            case_id=case.case_id,
            status=status,
            source=source_result,
            replay=replay_result,
            reference=reference_result,
            source_replay=source_replay,
            source_j=source_j,
            replay_j=replay_j,
            input_digest=input_digest,
            metadata={
                **dict(case.metadata),
                **q_metadata,
                **lineage_metadata,
            },
        ))

    def run_suite(
        self,
        source: KernelSource,
        replay: KernelSource,
        suite: Gate2Suite,
    ) -> Gate2SuiteResult:
        """Run every matrix member; one failure keeps the suite from passing."""

        return Gate2SuiteResult(
            suite.suite_id,
            tuple(self.run(source, replay, case) for case in suite.cases),
            suite.c_values,
        )


__all__ = [
    "GATE2_EXECUTION_PROTOCOL",
    "Gate2Case",
    "Gate2Runner",
    "Gate2Suite",
    "clone_gate2_values",
]
