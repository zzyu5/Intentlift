"""Gate1 runners for CUDA C++ and TileLang public frontends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from intentir.ir.canonical_j import j_elaboration

from .cuda_compiler import (
    CudaCompileCase,
    CudaCompileResult,
    CudaCompiler,
    CudaKernelSource,
    FrontendParameter,
)
from .model import Gate1Artifact, Gate1Status, ParameterKind, ScheduleSlotKind
from .normalize import NORMALIZER_PROTOCOL
from .native_semantic import (
    compare_cuda_jc,
    compare_tilelang_jc,
    semantic_normalization_receipt,
)
from .orchestrate import (
    Gate1ArmEvidence,
    Gate1Config,
    Gate1ExecutionResult,
    canonical_artifact_digest,
)
from .pi import canonical_program_coordinate_projection
from .profile import CUDA_OBSERVER_PROFILE, TILELANG_OBSERVER_PROFILE
from .ptx import PTXComparator, PTXFingerprint, fingerprint_ptx
from .replay import ReplayWriter
from .tilelang import TileLangComparator, TileLangFingerprint, fingerprint_tilelang
from .tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompileResult,
    TileLangCompiler,
    TileLangKernelSource,
)


def _identity_normalization(
    artifact: Gate1Artifact, *, frontend: str, observer_profile: str
) -> dict[str, Any]:
    """Receipt for a unary, peer-blind native identity normalizer.

    Native fingerprints are already frontend-specific structural observations.
    Until a typed J/C anchor matcher exists, every field remains hard and pi may
    not select, erase, or rewrite one of them.
    """

    program_projection = canonical_program_coordinate_projection(artifact.pi)
    return {
        "protocol": NORMALIZER_PROTOCOL,
        "profile_ok": True,
        "frontend": frontend,
        "observer_profile": observer_profile,
        "unary": True,
        "peer_observation_visible": False,
        "transformed_occurrences": 0,
        "active_rules": [],
        "hard_relations": (
            [program_projection.family] if program_projection.has_mapping else []
        ),
        "hard_preservation_check": "passed",
        "idempotence_check": "passed",
        "unknown_atoms": "hard",
    }


def _normalization_receipt(
    artifact: Gate1Artifact,
    *,
    frontend: str,
    observer_profile: str,
    semantic_projection: bool,
) -> dict[str, Any]:
    if semantic_projection:
        return semantic_normalization_receipt(
            artifact,
            frontend=frontend,
            observer_profile=observer_profile,
        )
    return _identity_normalization(
        artifact,
        frontend=frontend,
        observer_profile=observer_profile,
    )


def _dtype(value: str | None) -> str:
    aliases = {
        "bool": "i1",
        "float": "f32",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "half": "f16",
        "int": "i32",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "index": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    key = str(value or "").removeprefix("torch.").lower()
    return aliases.get(key, key)


def _artifact_interface(
    artifact: Gate1Artifact,
    parameters: tuple[FrontendParameter, ...],
) -> dict[str, Any]:
    signature = artifact.j.signature.parameters
    if len(signature) != len(parameters):
        raise ValueError("frontend Harness parameter count differs from J.signature")
    expected_kind = {
        ParameterKind.POINTER: "pointer",
        ParameterKind.TENSOR: "pointer",
        ParameterKind.SCALAR: "scalar",
        ParameterKind.OPAQUE: "opaque",
    }
    normalized: list[dict[str, Any]] = []
    for index, (semantic, harness) in enumerate(zip(signature, parameters, strict=True)):
        if semantic.kind is ParameterKind.CONSTEXPR:
            raise ValueError(
                "CUDA/TileLang runtime ABI may not expose J constexpr parameters; use C"
            )
        kind = expected_kind[semantic.kind]
        if harness.kind != kind:
            raise ValueError(f"frontend parameter {index} kind differs from J.signature")
        if _dtype(semantic.dtype) != _dtype(harness.dtype):
            raise ValueError(f"frontend parameter {index} dtype differs from J.signature")
        if semantic.rank is not None and semantic.rank != harness.rank:
            raise ValueError(f"frontend parameter {index} rank differs from J.signature")
        if semantic.mode.value != harness.mode:
            raise ValueError(f"frontend parameter {index} mode differs from J.signature")
        normalized.append(
            {
                "position": index,
                "kind": kind,
                "dtype": _dtype(semantic.dtype),
                "rank": semantic.rank,
                "mode": semantic.mode.value,
            }
        )
    return {"parameters": normalized}


def _validate_compiled_interface(
    artifact: Gate1Artifact,
    observed: Mapping[str, Any],
    *,
    frontend: str,
    expected_parameters: tuple[FrontendParameter, ...] | None = None,
) -> None:
    """Fail closed when compiled native ABI differs from its owned contract.

    CUDA C++ has a physical declaration ABI and a separate logical J/Harness
    ABI.  The former is checked against ``expected_parameters`` (the trusted
    source view for a module file, or the logical replay view for Replay).  The
    latter is built independently by ``_artifact_interface`` below.  This
    prevents a compiler result from passing by simply echoing the Harness case,
    while still allowing a trusted CUDA source to expose a CUtensorMap slot.

    TileLang's PrimFunc observer already reports the logical tensor interface,
    so its existing rank/shape/mode checks remain unchanged.
    """

    if frontend == "cuda":
        if expected_parameters is None:
            raise ValueError("CUDA compiled interface requires an expected ABI view")
        raw_parameters = observed.get("parameters")
        if not isinstance(raw_parameters, list):
            raise ValueError("CUDA compiler did not report a parameter interface")
        if len(raw_parameters) != len(expected_parameters):
            raise ValueError(
                "CUDA compiled physical parameter count differs from its ABI view"
            )
        expected_kind = {
            ParameterKind.POINTER: "pointer",
            ParameterKind.TENSOR: "pointer",
            ParameterKind.SCALAR: "scalar",
            ParameterKind.OPAQUE: "opaque",
        }
        for index, (expected, raw) in enumerate(
            zip(expected_parameters, raw_parameters, strict=True)
        ):
            if not isinstance(raw, Mapping):
                raise ValueError(f"CUDA compiled parameter {index} is not an object")
            if raw.get("kind") != expected_kind[expected.kind]:
                raise ValueError(
                    f"CUDA compiled physical parameter {index} kind differs from its ABI view"
                )
            if _dtype(str(raw.get("dtype"))) != _dtype(expected.dtype):
                raise ValueError(
                    f"CUDA compiled physical parameter {index} dtype differs from its ABI view"
                )
            if expected.kind in {ParameterKind.POINTER, ParameterKind.TENSOR} and not isinstance(
                raw.get("pointee_const"), bool
            ):
                raise ValueError(
                    f"CUDA compiled pointer parameter {index} lacks pointee constness"
                )
        return

    raw_parameters = observed.get("parameters")
    if not isinstance(raw_parameters, list):
        raise ValueError(f"{frontend} compiler did not report a parameter interface")
    signature = artifact.j.signature.parameters
    if len(raw_parameters) != len(signature):
        raise ValueError(f"{frontend} compiled parameter count differs from J.signature")
    # The native compiler interface may validate input declarations directly,
    # but output dtype/rank must come from the one canonical output/return/ABI
    # record rather than a second ``intent.outputs`` projection.
    tensors = {
        **{item.name: item for item in artifact.j.intent.inputs},
        **{
            binding.output: binding.output_type
            for binding in j_elaboration(artifact.j).output_bindings
        },
    }
    expected_kind = {
        ParameterKind.POINTER: "pointer",
        ParameterKind.TENSOR: "pointer",
        ParameterKind.SCALAR: "scalar",
        ParameterKind.OPAQUE: "opaque",
    }
    for index, (semantic, raw) in enumerate(zip(signature, raw_parameters, strict=True)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{frontend} compiled parameter {index} is not an object")
        if semantic.kind is ParameterKind.CONSTEXPR:
            raise ValueError(f"{frontend} compiled ABI may not expose J constexpr")
        expected = expected_kind[semantic.kind]
        if str(raw.get("kind")) != expected:
            raise ValueError(f"{frontend} compiled parameter {index} kind differs from J.signature")
        if _dtype(str(raw.get("dtype"))) != _dtype(semantic.dtype):
            raise ValueError(f"{frontend} compiled parameter {index} dtype differs from J.signature")
        if raw.get("mode") != semantic.mode.value:
            raise ValueError(f"{frontend} compiled parameter {index} mode differs from J.signature")
        raw_rank = raw.get("rank")
        if semantic.rank is not None and raw_rank != semantic.rank:
            raise ValueError(f"{frontend} compiled parameter {index} rank differs from J.signature")
        tensor = tensors.get(str(semantic.logical_ref))
        raw_shape = raw.get("shape")
        if tensor is None or not isinstance(raw_shape, list):
            continue
        if len(raw_shape) != len(tensor.shape):
            raise ValueError(f"{frontend} compiled parameter {index} shape rank differs from J")
        for axis, (logical, physical) in enumerate(zip(tensor.shape, raw_shape, strict=True)):
            if isinstance(logical, int) and physical != logical:
                raise ValueError(
                    f"{frontend} compiled parameter {index} static extent {axis} "
                    "differs from J.signature"
                )


def _validate_schedule(
    artifact: Gate1Artifact,
    case: CudaCompileCase | TileLangCompileCase,
) -> None:
    """Bind C to native Harness schedule and explicit grid-enable slots.

    Native compile cases historically expose non-unit block extents as their
    opaque schedule. A Replay that uses a non-leading grid coordinate receives
    that implementation binding through C. ``launch_grid_{x,y,z}`` identifies
    the coordinate and its value is checked against the real Harness grid.
    """

    observed = dict(case.schedule)
    grid_roles = {
        "launch_grid_x": 0,
        "launch_grid_y": 1,
        "launch_grid_z": 2,
    }
    for slot in artifact.c.slots:
        axis = grid_roles.get(slot.role)
        if axis is None:
            continue
        if slot.kind is not ScheduleSlotKind.COMPILER:
            raise ValueError("native launch-grid C slot must be compiler-kind")
        if slot.value_type.casefold() not in {"i1", "bool", "boolean"}:
            raise ValueError("native launch-grid C slot must be boolean")
        # The bit denotes that the coordinate is part of the authored launch
        # topology, even when this finite valuation happens to bind its extent
        # to one.  Gate1's compiled program-coordinate observation validates
        # that the Source actually exposes the declared axis.
        observed[slot.name] = True
    if artifact.c.values != observed:
        raise ValueError("frontend Harness schedule differs from the accepted C")


def _replay_writers(
    writer: ReplayWriter | tuple[ReplayWriter, ...],
) -> tuple[ReplayWriter, ...]:
    writers = writer if isinstance(writer, tuple) else (writer,)
    if not writers:
        raise ValueError("Gate1 requires at least one Replay writer")
    if len({id(item) for item in writers}) != len(writers):
        raise ValueError("Gate1 Replay fan-out requires fresh independent writers")
    return writers


def _normal_arms(
    runner: Any,
    artifact: Gate1Artifact,
    source_fingerprint: PTXFingerprint | TileLangFingerprint,
    case: CudaCompileCase | TileLangCompileCase,
) -> tuple[Gate1ArmEvidence, ...]:
    if len(runner.writers) == 1:
        return (
            runner._arm(
                runner.writers[0],
                artifact,
                source_fingerprint,
                case,
                masked=False,
            ),
        )
    with ThreadPoolExecutor(max_workers=len(runner.writers)) as executor:
        futures = [
            executor.submit(
                runner._arm,
                writer,
                artifact,
                source_fingerprint,
                case,
                masked=False,
            )
            for writer in runner.writers
        ]
        return tuple(future.result() for future in futures)


def _selected_normal_index(candidates: tuple[Gate1ArmEvidence, ...]) -> int:
    return next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate.hard_closed
        ),
        0,
    )


class CudaGate1Runner:
    def __init__(
        self,
        *,
        compiler: CudaCompiler,
        writer: ReplayWriter | tuple[ReplayWriter, ...],
        masked_writer: ReplayWriter | None = None,
        comparator: PTXComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.config = config or Gate1Config()
        self.writers = _replay_writers(writer)
        if self.config.run_masked_j:
            if masked_writer is None:
                raise ValueError("masked-J requires a fresh independent Replay writer")
            if id(masked_writer) in {id(item) for item in self.writers}:
                raise ValueError("masked-J writer must differ from every normal writer")
        elif masked_writer is not None:
            raise ValueError("masked-J writer was provided while the audit is disabled")
        self.writer = self.writers[0]
        self.masked_writer = masked_writer
        self.comparator = comparator or PTXComparator()
        self._semantic_projection = comparator is None

    def _fingerprint(
        self,
        artifact: Gate1Artifact,
        result: CudaCompileResult,
        case: CudaCompileCase,
        *,
        source_kind: str,
    ) -> PTXFingerprint:
        if result.ptx is None or result.entry is None:
            raise ValueError("CUDA compile result lacks PTX/entry")
        if source_kind not in {"source", "replay"}:
            raise ValueError(f"unknown CUDA observation source kind {source_kind!r}")
        expected_parameters = (
            case.physical_parameters
            if source_kind == "source" and case.physical_parameters is not None
            else case.parameters
        )
        _validate_compiled_interface(
            artifact,
            result.interface,
            frontend="cuda",
            expected_parameters=expected_parameters,
        )
        interface = _artifact_interface(artifact, case.parameters)
        return fingerprint_ptx(
            result.ptx,
            entry=result.entry,
            interface=interface,
            target={"arch": case.arch},
            schedule=artifact.c.values,
        )

    def _arm(
        self,
        writer: ReplayWriter,
        artifact: Gate1Artifact,
        source_fingerprint: PTXFingerprint,
        case: CudaCompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = writer.generate(artifact, masked=masked)
        except Exception as exc:
            attempt = getattr(exc, "attempt", None)
            return Gate1ArmEvidence(
                mode,
                None,
                None,
                None,
                None,
                f"generation failed: {type(exc).__name__}: {exc}",
                getattr(attempt, "attempts", 0),
            )
        try:
            replay_source = CudaKernelSource.from_replay(
                generation.code, generation.function_name
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                None,
                None,
                None,
                f"Replay policy failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        compiled = self.compiler.compile(replay_source, case)
        if not compiled.ok:
            return Gate1ArmEvidence(
                mode,
                generation,
                compiled,
                None,
                None,
                compiled.diagnostic or "CUDA Replay compilation failed",
                generation.attempts,
            )
        try:
            replay_fingerprint = self._fingerprint(
                artifact,
                compiled,
                case,
                source_kind="replay",
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                compiled,
                None,
                None,
                f"Replay observation failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        comparison = (
            compare_cuda_jc(artifact, source_fingerprint, replay_fingerprint)
            if self._semantic_projection
            else self.comparator.compare(source_fingerprint, replay_fingerprint)
        )
        return Gate1ArmEvidence(
            mode,
            generation,
            compiled,
            replay_fingerprint,
            comparison,
            physical_calls=generation.attempts,
            normalization=_normalization_receipt(
                artifact,
                frontend="cuda",
                observer_profile=CUDA_OBSERVER_PROFILE.profile_version,
                semantic_projection=self._semantic_projection,
            ),
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: CudaKernelSource,
        source_case: CudaCompileCase,
    ) -> Gate1ExecutionResult:
        if artifact.c.slots:
            artifact.c.validate_complete_values(path="Gate1 active C")
        digest = canonical_artifact_digest(artifact, observer_level=self.config.observer_level)
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,
                None,
                empty,
                None,
                0,
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
            )
        try:
            _validate_schedule(artifact, source_case)
            source_fingerprint = self._fingerprint(
                artifact,
                source_compile,
                source_case,
                source_kind="source",
            )
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                None,
                normal,
                None,
                0,
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
            )
        normal_candidates = _normal_arms(
            self, artifact, source_fingerprint, source_case
        )
        selected_normal_index = _selected_normal_index(normal_candidates)
        normal = normal_candidates[selected_normal_index]
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                source_fingerprint,
                normal,
                None,
                len(normal_candidates),
                0,
                observer_profile=CUDA_OBSERVER_PROFILE,
                normal_candidates=normal_candidates,
                selected_normal_index=selected_normal_index,
                source_normalization=_normalization_receipt(
                    artifact,
                    frontend="cuda",
                    observer_profile=CUDA_OBSERVER_PROFILE.profile_version,
                    semantic_projection=self._semantic_projection,
                ),
            )
        masked_evidence = None
        masked_calls = 0
        if self.config.run_masked_j:
            assert self.masked_writer is not None
            masked_calls = 1
            masked_evidence = self._arm(
                self.masked_writer,
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
            source_compile,
            source_fingerprint,
            normal,
            masked_evidence,
            len(normal_candidates),
            masked_calls,
            observer_profile=CUDA_OBSERVER_PROFILE,
            normal_candidates=normal_candidates,
            selected_normal_index=selected_normal_index,
            source_normalization=_normalization_receipt(
                artifact,
                frontend="cuda",
                observer_profile=CUDA_OBSERVER_PROFILE.profile_version,
                semantic_projection=self._semantic_projection,
            ),
        )


class TileLangGate1Runner:
    def __init__(
        self,
        *,
        compiler: TileLangCompiler,
        writer: ReplayWriter | tuple[ReplayWriter, ...],
        masked_writer: ReplayWriter | None = None,
        comparator: TileLangComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        self.config = config or Gate1Config()
        self.writers = _replay_writers(writer)
        if self.config.run_masked_j:
            if masked_writer is None:
                raise ValueError("masked-J requires a fresh independent Replay writer")
            if id(masked_writer) in {id(item) for item in self.writers}:
                raise ValueError("masked-J writer must differ from every normal writer")
        elif masked_writer is not None:
            raise ValueError("masked-J writer was provided while the audit is disabled")
        self.writer = self.writers[0]
        self.masked_writer = masked_writer
        self.comparator = comparator or TileLangComparator()
        self._semantic_projection = comparator is None

    def _fingerprint(
        self,
        artifact: Gate1Artifact,
        result: TileLangCompileResult,
        case: TileLangCompileCase,
    ) -> TileLangFingerprint:
        if result.tir is None or result.ptx is None:
            raise ValueError("TileLang compile result lacks TIR/PTX")
        _validate_compiled_interface(artifact, result.interface, frontend="tilelang")
        interface = _artifact_interface(artifact, case.parameters)
        return fingerprint_tilelang(
            result.tir,
            result.ptx,
            interface=interface,
            target={"name": case.target},
            schedule=artifact.c.values,
        )

    def _arm(
        self,
        writer: ReplayWriter,
        artifact: Gate1Artifact,
        source_fingerprint: TileLangFingerprint,
        case: TileLangCompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = writer.generate(artifact, masked=masked)
        except Exception as exc:
            attempt = getattr(exc, "attempt", None)
            return Gate1ArmEvidence(
                mode, None, None, None, None,
                f"generation failed: {type(exc).__name__}: {exc}",
                getattr(attempt, "attempts", 0),
            )
        try:
            replay_source = TileLangKernelSource.from_replay(
                generation.code, generation.function_name
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, None, None, None,
                f"Replay policy failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        compiled = self.compiler.compile(replay_source, case)
        if not compiled.ok:
            return Gate1ArmEvidence(
                mode, generation, compiled, None, None,
                compiled.diagnostic or "TileLang Replay compilation failed",
                generation.attempts,
            )
        try:
            replay_fingerprint = self._fingerprint(artifact, compiled, case)
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, compiled, None, None,
                f"Replay observation failed: {type(exc).__name__}: {exc}",
                generation.attempts,
            )
        comparison = (
            compare_tilelang_jc(artifact, source_fingerprint, replay_fingerprint)
            if self._semantic_projection
            else self.comparator.compare(source_fingerprint, replay_fingerprint)
        )
        return Gate1ArmEvidence(
            mode,
            generation,
            compiled,
            replay_fingerprint,
            comparison,
            physical_calls=generation.attempts,
            normalization=_normalization_receipt(
                artifact,
                frontend="tilelang",
                observer_profile=TILELANG_OBSERVER_PROFILE.profile_version,
                semantic_projection=self._semantic_projection,
            ),
        )

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: TileLangKernelSource,
        source_case: TileLangCompileCase,
    ) -> Gate1ExecutionResult:
        if artifact.c.slots:
            artifact.c.validate_complete_values(path="Gate1 active C")
        digest = canonical_artifact_digest(artifact, observer_level=self.config.observer_level)
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,
                None,
                empty,
                None,
                0,
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
            )
        try:
            _validate_schedule(artifact, source_case)
            source_fingerprint = self._fingerprint(artifact, source_compile, source_case)
        except Exception as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"harness binding failed: {type(exc).__name__}: {exc}",
            )
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                None,
                normal,
                None,
                0,
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
            )
        normal_candidates = _normal_arms(
            self, artifact, source_fingerprint, source_case
        )
        selected_normal_index = _selected_normal_index(normal_candidates)
        normal = normal_candidates[selected_normal_index]
        if not normal.hard_closed:
            return Gate1ExecutionResult(
                Gate1Status.NO_CLOSURE,
                digest,
                source_compile,
                source_fingerprint,
                normal,
                None,
                len(normal_candidates),
                0,
                observer_profile=TILELANG_OBSERVER_PROFILE,
                normal_candidates=normal_candidates,
                selected_normal_index=selected_normal_index,
                source_normalization=_normalization_receipt(
                    artifact,
                    frontend="tilelang",
                    observer_profile=TILELANG_OBSERVER_PROFILE.profile_version,
                    semantic_projection=self._semantic_projection,
                ),
            )
        masked_evidence = None
        masked_calls = 0
        if self.config.run_masked_j:
            assert self.masked_writer is not None
            masked_calls = 1
            masked_evidence = self._arm(
                self.masked_writer,
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
            source_compile,
            source_fingerprint,
            normal,
            masked_evidence,
            len(normal_candidates),
            masked_calls,
            observer_profile=TILELANG_OBSERVER_PROFILE,
            normal_candidates=normal_candidates,
            selected_normal_index=selected_normal_index,
            source_normalization=_normalization_receipt(
                artifact,
                frontend="tilelang",
                observer_profile=TILELANG_OBSERVER_PROFILE.profile_version,
                semantic_projection=self._semantic_projection,
            ),
        )


__all__ = ["CudaGate1Runner", "TileLangGate1Runner"]
