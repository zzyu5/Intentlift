"""CuTe frontend arm for the shared LLM-direct Gate1 protocol."""

from __future__ import annotations

from typing import Any, Mapping

from .cute import CuteComparator, CuteFingerprint, fingerprint_cute
from .cute_compiler import (
    CuteCompileCase,
    CuteCompileResult,
    CuteCompiler,
    CuteKernelSource,
)
from .model import Gate1Artifact, Gate1Status, ParameterKind
from .orchestrate import (
    Gate1ArmEvidence,
    Gate1Config,
    Gate1ExecutionResult,
    canonical_artifact_digest,
)
from .profile import CUTE_OBSERVER_PROFILE
from .replay import ReplayWriter


def _annotation_kind(annotation: Any) -> str:
    if annotation is None:
        return "unspecified"
    spelling = str(annotation).replace(" ", "").lower()
    if spelling.endswith("tensor"):
        return "tensor"
    if spelling.endswith("copyatom"):
        return "opaque"
    if any(
        spelling.endswith(name)
        for name in (
            "boolean",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "bfloat16",
        )
    ):
        return "scalar"
    return "unknown"


def _opaque_cute_interface(
    artifact: Gate1Artifact,
    metadata: Mapping[str, Any],
    case: CuteCompileCase,
) -> dict[str, Any]:
    device = metadata.get("device")
    entry = metadata.get("entry")
    if not isinstance(device, Mapping) or not isinstance(entry, Mapping):
        raise ValueError("CuTe compiler metadata lacks device/entry interfaces")
    device_names = device.get("arg_names")
    annotations = device.get("annotations")
    if not isinstance(device_names, list) or not all(
        isinstance(name, str) for name in device_names
    ):
        raise ValueError("CuTe device argument names are malformed")
    if not isinstance(annotations, list) or len(annotations) != len(device_names):
        raise ValueError("CuTe device annotations are malformed")
    parameters = artifact.j.signature.parameters
    if len(device_names) != len(parameters):
        raise ValueError("CuTe device arity differs from the artifact harness")
    expected_categories = {
        ParameterKind.POINTER: "tensor",
        ParameterKind.TENSOR: "tensor",
        ParameterKind.SCALAR: "scalar",
        ParameterKind.OPAQUE: "opaque",
    }
    normalized_device: list[dict[str, Any]] = []
    for index, (parameter, annotation) in enumerate(
        zip(parameters, annotations, strict=True)
    ):
        if parameter.kind is ParameterKind.CONSTEXPR:
            raise ValueError("CuTe device interface may not expose constexpr arguments")
        expected = expected_categories[parameter.kind]
        observed = _annotation_kind(annotation)
        if observed not in {expected, "unspecified"}:
            raise ValueError(
                f"CuTe device argument {index} annotation differs from harness kind"
            )
        if observed == "unspecified" and expected in {"tensor", "opaque"}:
            raise ValueError(
                f"CuTe device argument {index} requires an explicit public DSL type"
            )
        normalized_device.append(
            {
                "position": index,
                "kind": parameter.kind.value,
                "dtype": parameter.dtype,
                "rank": parameter.rank,
            }
        )
    entry_names = entry.get("arg_names")
    if not isinstance(entry_names, list) or not all(
        isinstance(name, str) for name in entry_names
    ):
        raise ValueError("CuTe entry argument names are malformed")
    expected_entry = len(case.entry_arguments) + int(case.append_stream)
    if len(entry_names) != expected_entry:
        raise ValueError("CuTe entry arity differs from the semantic positional harness")
    return {
        "device_function": "scrc_k0",
        "device_arguments": normalized_device,
        "entry_function": "scrc_entry",
        "entry_arguments": [
            argument.kind for argument in case.entry_arguments
        ]
        + (["stream"] if case.append_stream else []),
    }


def _schedule_view(case: CuteCompileCase) -> dict[str, Any]:
    return {
        **case.schedule,
        "grid": case.grid,
        "block": case.block,
        "smem_bytes": case.smem_bytes,
    }


class CuteGate1Runner:
    """Same normal/masked protocol as :class:`Gate1Runner`, for CuTe MLIR/PTX."""

    def __init__(
        self,
        *,
        compiler: CuteCompiler,
        writer: ReplayWriter,
        comparator: CuteComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.writer = writer
        self.comparator = comparator or CuteComparator()
        self.config = config or Gate1Config()

    def _fingerprint(
        self,
        artifact: Gate1Artifact,
        result: CuteCompileResult,
        case: CuteCompileCase,
    ) -> CuteFingerprint:
        if result.mlir is None or result.ptx is None:
            raise ValueError("CuTe compiler result lacks MLIR/PTX")
        return fingerprint_cute(
            result.mlir,
            result.ptx,
            python_interface=_opaque_cute_interface(
                artifact, result.python_interface, case
            ),
            schedule=_schedule_view(case),
        )

    def _arm(
        self,
        artifact: Gate1Artifact,
        source_fingerprint: CuteFingerprint,
        case: CuteCompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = self.writer.generate(artifact, masked=masked)
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                None,
                None,
                None,
                None,
                f"generation failed: {type(exc).__name__}: {exc}",
            )
        entry = generation.entry_function_name
        if entry is None:
            return Gate1ArmEvidence(
                mode,
                generation,
                None,
                None,
                None,
                "CuTe Replay did not declare its compile entry",
            )
        replay_source = CuteKernelSource.from_code(
            generation.code,
            generation.function_name,
            entry,
            trusted=False,
        )
        compile_result = self.compiler.compile(replay_source, case)
        if not compile_result.ok:
            return Gate1ArmEvidence(
                mode,
                generation,
                compile_result,
                None,
                None,
                compile_result.diagnostic or "CuTe replay compilation failed",
            )
        try:
            replay_fingerprint = self._fingerprint(
                artifact, compile_result, case
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                compile_result,
                None,
                None,
                f"replay observation failed: {type(exc).__name__}: {exc}",
            )
        comparison = self.comparator.compare(
            source_fingerprint, replay_fingerprint
        )
        return Gate1ArmEvidence(
            mode,
            generation,
            compile_result,
            replay_fingerprint,
            comparison,
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: CuteKernelSource,
        source_case: CuteCompileCase,
    ) -> Gate1ExecutionResult:
        digest = canonical_artifact_digest(
            artifact, observer_level=self.config.observer_level
        )
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                empty,
                None,
                0,
                0,
                observer_profile=CUTE_OBSERVER_PROFILE,
            )
        try:
            source_fingerprint = self._fingerprint(
                artifact, source_compile, source_case
            )
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal",
                None,
                None,
                None,
                None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                None,
                normal,
                None,
                0,
                0,
                observer_profile=CUTE_OBSERVER_PROFILE,
            )
        normal = self._arm(
            artifact,
            source_fingerprint,
            source_case,
            masked=False,
        )
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,  # type: ignore[arg-type]
                source_fingerprint,  # type: ignore[arg-type]
                normal,
                None,
                1,
                0,
                observer_profile=CUTE_OBSERVER_PROFILE,
            )
        masked_evidence = None
        masked_calls = 0
        status = Gate1Status.REPLAY_CLOSED
        if self.config.run_masked_j:
            masked_calls = 1
            masked_evidence = self._arm(
                artifact,
                source_fingerprint,
                source_case,
                masked=True,
            )
        status = (
            Gate1Status.REPLAY_CLOSED
            if not self.config.run_masked_j
            or (masked_evidence is not None and masked_evidence.comparable)
            else Gate1Status.NO_CLOSURE
        )
        return Gate1ExecutionResult(
            status,
            digest,
            source_compile,  # type: ignore[arg-type]
            source_fingerprint,  # type: ignore[arg-type]
            normal,
            masked_evidence,
            1,
            masked_calls,
            observer_profile=CUTE_OBSERVER_PROFILE,
        )


__all__ = ["CuteGate1Runner"]
