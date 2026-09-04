"""Gate1 realization closure with an independent optional masked-J audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .compare import TTIRComparator, TTIRComparison
from .compiler import CompileCase, CompileResult, KernelSource, TritonCompiler
from .model import (
    Frontend,
    Gate1Artifact,
    Gate1Status,
    JNecessityStatus,
    ParameterKind,
    ScheduleSlotKind,
    opaque_replay_abi,
)
from .normalize import normalize_ttir_for_gate1
from .prompt import build_rewriter_payload
from .replay import ReplayGeneration, ReplayWriter, replay_source_ingress
from .profile import ObserverProfile, TRITON_OBSERVER_PROFILE
from .ttir import TTIRFingerprint, TTIRParseError, fingerprint_ttir


GATE1_PROTOCOL = "intentir.j-only-replay.jc-structural-gate1.v2"
SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX = (
    "observer unsupported: source TTIR"
)


@dataclass(frozen=True)
class Gate1Config:
    observer_level: int = 2
    run_masked_j: bool = True

    def __post_init__(self) -> None:
        if self.observer_level != 2:
            raise ValueError("the active strong Gate1 paths require observer level 2")


@dataclass(frozen=True)
class Gate1ArmEvidence:
    mode: str
    generation: ReplayGeneration | None
    compile_result: CompileResult | None
    fingerprint: TTIRFingerprint | None
    comparison: TTIRComparison | None
    diagnostic: str | None = None
    physical_calls: int = 0
    normalization: Mapping[str, Any] | None = None

    @property
    def hard_closed(self) -> bool:
        return self.comparison is not None and self.comparison.hard_pass


@dataclass(frozen=True)
class Gate1ExecutionResult:
    status: Gate1Status
    artifact_digest: str
    source_compile: CompileResult
    source_fingerprint: TTIRFingerprint | None
    normal: Gate1ArmEvidence
    masked_j: Gate1ArmEvidence | None
    normal_calls: int
    masked_j_calls: int
    protocol: str = GATE1_PROTOCOL
    observer_profile: ObserverProfile = TRITON_OBSERVER_PROFILE
    normal_candidates: tuple[Gate1ArmEvidence, ...] = ()
    selected_normal_index: int = 0
    source_normalization: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.status is Gate1Status.REPLAY_CLOSED

    @property
    def j_necessity_status(self) -> JNecessityStatus:
        """Report masked-J separately from the main Gate1 status."""

        if self.masked_j is None or self.masked_j.comparison is None:
            return JNecessityStatus.INCONCLUSIVE
        if self.masked_j.hard_closed:
            return JNecessityStatus.NOT_PROVED
        return JNecessityStatus.PROVED

    @property
    def replay_code(self) -> str | None:
        return self.normal.generation.code if self.normal.generation is not None else None

    def to_json(self) -> dict[str, Any]:
        def arm(value: Gate1ArmEvidence | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {
                "mode": value.mode,
                "generation": (
                    {
                        "model": value.generation.model,
                        "function_name": value.generation.function_name,
                        "entry_function_name": value.generation.entry_function_name,
                        "response_model": value.generation.response_model,
                        "response_id": value.generation.response_id,
                        "attempts": value.generation.attempts,
                        "elapsed_ms": value.generation.elapsed_ms,
                        "context_receipt": dict(value.generation.context_receipt),
                        "usage": dict(value.generation.usage),
                    }
                    if value.generation is not None else None
                ),
                "compile": (
                    value.compile_result.to_payload()
                    if value.compile_result is not None else None
                ),
                "fingerprint": (
                    value.fingerprint.to_dict()
                    if value.fingerprint is not None else None
                ),
                "comparison": (
                    value.comparison.to_dict()
                    if value.comparison is not None else None
                ),
                "diagnostic": value.diagnostic,
                "physical_calls": value.physical_calls,
                "normalization": (
                    dict(value.normalization)
                    if value.normalization is not None else None
                ),
            }

        normal_fanout = None
        if len(self.normal_candidates) > 1:
            normal_fanout = {
                "width": len(self.normal_candidates),
                "selected_index": self.selected_normal_index,
                "selection_rule": "exact-pass-then-gate1-rank-v1",
                "source_blind_independent_replays": True,
                "candidates": [arm(item) for item in self.normal_candidates],
            }
        return {
            "protocol": self.protocol,
            "observer_profile": self.observer_profile.to_json(),
            "status": self.status.value,
            "j_necessity_status": self.j_necessity_status.value,
            "artifact_digest": self.artifact_digest,
            "source_compile": self.source_compile.to_payload(),
            "source_fingerprint": (
                self.source_fingerprint.to_dict()
                if self.source_fingerprint is not None else None
            ),
            "normal": arm(self.normal),
            "masked_j": arm(self.masked_j),
            "llm_calls": {
                "normal": self.normal_calls,
                "masked_j": self.masked_j_calls,
            },
            "normal_fanout": normal_fanout,
            "source_normalization": (
                dict(self.source_normalization)
                if self.source_normalization is not None else None
            ),
        }


def canonical_artifact_digest(
    artifact: Gate1Artifact, *, observer_level: int = 2
) -> str:
    """Compute an audit identifier; it is not a verdict or protocol gate."""

    encoded = artifact.to_json()
    payload = {
        **encoded,
        "observer_level": observer_level,
        "protocol": GATE1_PROTOCOL,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _opaque_python_interface(
    artifact: Gate1Artifact,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Alpha-normalize Python signature spelling by declaration position."""

    if not metadata:
        return {}
    result = dict(metadata)
    raw_names = result.get("arg_names")
    if not isinstance(raw_names, (list, tuple)) or not all(
        isinstance(name, str) for name in raw_names
    ):
        raise ValueError("compiler metadata lacks an ordered Python signature")
    signature_parameters = artifact.j.signature.parameters
    c_constexprs = [
        slot for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.CONSTEXPR
    ]
    if len(raw_names) != len(signature_parameters) + len(c_constexprs):
        raise ValueError("compiled Python signature arity differs from artifact harness")
    constexpr_indices = result.get("constexpr_indices", [])
    if not isinstance(constexpr_indices, (list, tuple)) or not all(
        isinstance(index, int) and not isinstance(index, bool)
        for index in constexpr_indices
    ):
        raise ValueError("compiler constexpr indices are malformed")
    opaque_function, parameter_names, schedule_names = opaque_replay_abi(artifact)
    authoritative_names = {
        **parameter_names,
        **{slot.name: schedule_names[slot.name] for slot in c_constexprs},
    }
    if set(raw_names) == set(authoritative_names):
        # Triton canonicalizes interleaved semantic constexprs by moving them
        # behind runtime arguments.  The p*/c* names are authoritative, so
        # recover identity and constexpr ownership by name rather than by the
        # compiler's reordered declaration position.
        constexpr_names = {
            parameter.name
            for parameter in signature_parameters
            if parameter.kind is ParameterKind.CONSTEXPR
        } | {slot.name for slot in c_constexprs}
        expected_constexpr = [
            index for index, name in enumerate(raw_names)
            if name in constexpr_names
        ]
        aliases = [authoritative_names[name] for name in raw_names]
    else:
        # Compatibility for trusted fixtures that predate opaque p*/c* names.
        signature_constexpr = [
            index
            for index, parameter in enumerate(signature_parameters)
            if parameter.kind is ParameterKind.CONSTEXPR
        ]
        expected_constexpr = [
            *signature_constexpr,
            *range(
                len(signature_parameters),
                len(signature_parameters) + len(c_constexprs),
            ),
        ]
        aliases = [
            parameter_names[parameter.name]
            for parameter in signature_parameters
        ] + [
            schedule_names[slot.name] for slot in c_constexprs
        ]
    if sorted(constexpr_indices) != expected_constexpr:
        raise ValueError("compiled constexpr identities differ from artifact harness")
    # TTIR exposes function arguments as positional ``argN`` names even when
    # the Python compiler metadata retains the authored/opaque parameter names
    # (p*/c*).  Bridge those two namespaces by declaration position.  The
    # fingerprint consumes this map only for leaf naming and deliberately
    # excludes it from the hard public-interface payload.  Specialized scalar
    # carriers may disappear from TTIR; the remaining function arguments still
    # retain their compiler declaration order.
    result["ttir_argument_aliases"] = {
        f"arg{index}": alias for index, alias in enumerate(aliases)
    }
    result["arg_names"] = aliases
    result["constexpr_indices"] = expected_constexpr
    result["constexprs"] = [aliases[index] for index in expected_constexpr]
    if "function_name" in result:
        result["function_name"] = opaque_function
    return result


def _replay_case(
    artifact: Gate1Artifact,
    source_case: CompileCase,
    source_compile: CompileResult,
) -> CompileCase:
    _function, parameter_names, schedule_names = opaque_replay_abi(artifact)
    signature_parameters = artifact.j.signature.parameters
    runtime_parameters = [
        item for item in signature_parameters
        if item.kind is not ParameterKind.CONSTEXPR
    ]
    semantic_constexprs = [
        item for item in signature_parameters
        if item.kind is ParameterKind.CONSTEXPR
    ]
    c_constexprs = [
        slot for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.CONSTEXPR
    ]
    metadata_names = source_compile.python_interface.get("arg_names")
    constexpr_indices = source_compile.python_interface.get("constexpr_indices")
    if metadata_names:
        if not isinstance(metadata_names, (list, tuple)) or not all(
            isinstance(name, str) for name in metadata_names
        ):
            raise ValueError("source compiler returned malformed argument names")
        if len(metadata_names) != len(signature_parameters) + len(c_constexprs):
            raise ValueError("source signature arity differs from artifact harness")
        if not isinstance(constexpr_indices, (list, tuple)):
            raise ValueError("source compiler returned malformed constexpr positions")
        constexpr_positions = {int(index) for index in constexpr_indices}
        source_runtime_names = [
            name for index, name in enumerate(metadata_names)
            if index not in constexpr_positions
        ]
        source_constexpr_names = [
            name for index, name in enumerate(metadata_names)
            if index in constexpr_positions
        ]
        if set(source_runtime_names) != set(source_case.arguments):
            raise ValueError("source runtime bindings differ from compiled signature")
        argument_values = [source_case.arguments[name] for name in source_runtime_names]
    else:
        argument_values = list(source_case.arguments.values())
        source_constexpr_names = list(source_case.schedule)[:
            len(semantic_constexprs) + len(c_constexprs)
        ]
    if len(runtime_parameters) != len(argument_values):
        raise ValueError(
            "source invocation arity differs from artifact positional harness"
        )
    dtype_aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "index": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
        "i1": "bool",
    }

    def dtype_key(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.removeprefix("torch.")
        return dtype_aliases.get(stripped, stripped)

    for parameter, value in zip(runtime_parameters, argument_values, strict=True):
        if parameter.kind in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            if value.kind != "tensor":
                raise ValueError(
                    f"source position for {parameter.name!r} must be a tensor"
                )
            if parameter.rank is not None and len(value.shape or ()) != parameter.rank:
                raise ValueError(
                    f"source rank for {parameter.name!r} differs from artifact harness"
                )
        elif parameter.kind is ParameterKind.SCALAR and value.kind != "scalar":
            raise ValueError(
                f"source position for {parameter.name!r} must be a scalar"
            )
        if (
            parameter.dtype is not None
            and value.dtype is not None
            and dtype_key(parameter.dtype) != dtype_key(value.dtype)
        ):
            raise ValueError(
                f"source dtype for {parameter.name!r} differs from artifact harness"
            )
    arguments = {
        parameter_names[parameter.name]: value
        for parameter, value in zip(runtime_parameters, argument_values, strict=True)
    }
    slots = {slot.name: slot for slot in artifact.c.slots}
    if len(source_constexpr_names) != len(semantic_constexprs) + len(c_constexprs):
        raise ValueError("source constexpr bindings differ from artifact harness")
    semantic_aliases = [
        parameter_names[parameter.name] for parameter in semantic_constexprs
    ]
    c_aliases = [schedule_names[slot.name] for slot in c_constexprs]
    if set(source_constexpr_names) == set(semantic_aliases) | set(c_aliases):
        # Source-anonymized real kernels can interleave semantic constexprs and
        # pure-C constexprs in their original ABI.  Their p*/c* names are
        # authoritative, so do not recover ownership from source position.
        semantic_source_names = semantic_aliases
        c_source_names = c_aliases
    else:
        # Compatibility for trusted, non-anonymized source fixtures whose names
        # predate the p*/c* compiler view.  Those fixtures use the historical
        # semantic-then-C ordering contract.
        semantic_source_names = source_constexpr_names[: len(semantic_constexprs)]
        c_source_names = source_constexpr_names[len(semantic_constexprs):]
    schedule: dict[str, Any] = {}
    consumed_source_schedule: set[str] = set()
    for parameter, source_name in zip(
        semantic_constexprs,
        semantic_source_names,
        strict=True,
    ):
        if source_name not in source_case.schedule:
            raise ValueError(f"source constexpr {source_name!r} is unbound")
        # Semantic compile-time values are Harness bindings.  Their logical
        # consequence is already in J; no concrete value enters C or Replay.
        expected = source_case.schedule[source_name]
        schedule[parameter_names[parameter.name]] = expected
        consumed_source_schedule.add(source_name)
    for slot, source_name in zip(c_constexprs, c_source_names, strict=True):
        if source_name not in source_case.schedule:
            raise ValueError(f"source C constexpr {source_name!r} is unbound")
        expected = artifact.c.values[slot.name]
        if source_case.schedule[source_name] != expected:
            raise ValueError(
                f"source constexpr position for {slot.name!r} differs from C"
            )
        schedule[schedule_names[slot.name]] = expected
        consumed_source_schedule.add(source_name)

    source_compiler_names = [
        name for name in source_case.schedule if name not in consumed_source_schedule
    ]
    compiler_slot_names = [
        slot.name for slot in artifact.c.slots
        if slot.kind is ScheduleSlotKind.COMPILER
    ]
    if len(source_compiler_names) != len(compiler_slot_names):
        raise ValueError("source compiler-option bindings differ from C")
    compiler_roles: set[str] = set()
    for source_name, slot_name in zip(
        source_compiler_names, compiler_slot_names, strict=True
    ):
        slot = slots[slot_name]
        if slot.kind is not ScheduleSlotKind.COMPILER:
            raise ValueError("artifact compiler slot has the wrong C kind")
        if slot.role in compiler_roles:
            raise ValueError("active compiler C roles must be unique")
        if slot.role in arguments or slot.role in schedule:
            raise ValueError(
                f"compiler C role {slot.role!r} collides with a Replay binding"
            )
        compiler_roles.add(slot.role)
        expected = artifact.c.values[slot_name]
        if source_case.schedule[source_name] != expected:
            raise ValueError(
                f"source compiler-option position for {slot_name!r} differs from C"
            )
        schedule[slot.role] = expected
    return CompileCase(
        arguments=arguments,
        grid=source_case.grid,
        schedule=schedule,
        device=source_case.device,
    )


class Gate1Runner:
    def __init__(
        self,
        *,
        compiler: TritonCompiler,
        writer: ReplayWriter | tuple[ReplayWriter, ...],
        comparator: TTIRComparator | None = None,
        config: Gate1Config | None = None,
    ) -> None:
        self.compiler = compiler
        writers = writer if isinstance(writer, tuple) else (writer,)
        if not writers:
            raise ValueError("Gate1 requires at least one Replay writer")
        if len({id(item) for item in writers}) != len(writers):
            raise ValueError("Gate1 Replay fan-out requires fresh independent writers")
        self.writers = writers
        self.writer = writers[0]
        self.comparator = comparator or TTIRComparator()
        self.config = config or Gate1Config()

    def _arm(
        self,
        writer: ReplayWriter,
        artifact: Gate1Artifact,
        source_fingerprint: TTIRFingerprint,
        replay_case: CompileCase,
        *,
        masked: bool,
    ) -> Gate1ArmEvidence:
        mode = "masked_j" if masked else "normal"
        try:
            generation = writer.generate(artifact, masked=masked)
        except Exception as exc:
            attempt = getattr(exc, "attempt", None)
            physical_calls = getattr(attempt, "physical_provider_calls", 0)
            if not physical_calls:
                physical_calls = getattr(attempt, "attempts", 0)
            return Gate1ArmEvidence(
                mode, None, None, None, None,
                f"generation failed: {type(exc).__name__}: {exc}",
                physical_calls,
            )
        physical_calls = (
            generation.physical_provider_calls or generation.attempts
        )

        try:
            packet = build_rewriter_payload(
                artifact,
                frontend=Frontend.TRITON,
                masked=masked,
            )
            function_name = str(packet["u_f"]["replay_function"])
            ingress = replay_source_ingress(
                generation.code,
                artifact=artifact,
                frontend=Frontend.TRITON,
                function_name=function_name,
                packet=packet,
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode,
                generation,
                None,
                None,
                None,
                f"replay ingress failed: {type(exc).__name__}: {exc}",
                physical_calls,
            )
        if ingress.status != "pass" or ingress.code is None:
            return Gate1ArmEvidence(
                mode,
                generation,
                None,
                None,
                None,
                "replay ingress failed: "
                + json.dumps(
                    list(ingress.diagnostics),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
                physical_calls,
            )
        generation = replace(
            generation,
            code=ingress.code,
            function_name=function_name,
            entry_function_name=ingress.entry_function_name,
        )
        replay_source = KernelSource.from_replay(
            generation.code, generation.function_name
        )
        compile_result = self.compiler.compile(replay_source, replay_case)
        if not compile_result.ok or compile_result.ttir is None:
            return Gate1ArmEvidence(
                mode, generation, compile_result, None, None,
                compile_result.diagnostic or "replay compilation failed",
                physical_calls,
            )
        try:
            normalized = normalize_ttir_for_gate1(compile_result.ttir, artifact)
            replay_fingerprint = fingerprint_ttir(
                normalized.text,
                python_interface=_opaque_python_interface(
                    artifact, compile_result.python_interface
                ),
                schedule=artifact.c.values,
                representation_anchor_exemptions=(
                    normalized.representation_anchor_exemptions
                ),
                runtime_values={
                    name: argument.value
                    for name, argument in replay_case.arguments.items()
                    if argument.kind == "scalar"
                    and isinstance(argument.value, (bool, int))
                },
            )
        except Exception as exc:
            return Gate1ArmEvidence(
                mode, generation, compile_result, None, None,
                f"replay observation failed: {type(exc).__name__}: {exc}",
                physical_calls,
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
            physical_calls=physical_calls,
            normalization=normalized.evidence(),
        )

    def _normal_arms(
        self,
        artifact: Gate1Artifact,
        source_fingerprint: TTIRFingerprint,
        replay_case: CompileCase,
    ) -> tuple[Gate1ArmEvidence, ...]:
        if len(self.writers) == 1:
            return (
                self._arm(
                    self.writers[0],
                    artifact,
                    source_fingerprint,
                    replay_case,
                    masked=False,
                ),
            )
        # Replay workers receive identical J/optional-C packets and no Source or
        # Gate1 feedback.  pi_source remains host-owned by the two independent
        # Gate1 normalizer calls and is not mounted in a Replay episode. Running
        # them concurrently changes only completeness and latency; the ordered
        # result list remains deterministic.
        with ThreadPoolExecutor(max_workers=len(self.writers)) as executor:
            futures = [
                executor.submit(
                    self._arm,
                    writer,
                    artifact,
                    source_fingerprint,
                    replay_case,
                    masked=False,
                )
                for writer in self.writers
            ]
            return tuple(future.result() for future in futures)

    @staticmethod
    def _normal_rank(arm: Gate1ArmEvidence) -> tuple[int | float, ...]:
        if arm.comparison is None:
            return (-1, -1, -1.0, -10**18, -1, -10**18)
        return tuple(arm.comparison.rank_tuple)

    def run(
        self,
        artifact: Gate1Artifact,
        *,
        source: KernelSource,
        source_case: CompileCase,
    ) -> Gate1ExecutionResult:
        if artifact.c.slots:
            artifact.c.validate_complete_values(path="Gate1 active C")
        digest = canonical_artifact_digest(
            artifact, observer_level=self.config.observer_level
        )
        source_compile = self.compiler.compile(source, source_case)
        empty = Gate1ArmEvidence("normal", None, None, None, None)
        if not source_compile.ok or source_compile.ttir is None:
            return Gate1ExecutionResult(
                Gate1Status.COMPILE_FAILED,
                digest,
                source_compile,
                None,
                empty,
                None,
                0,
                0,
            )
        try:
            replay_case = _replay_case(artifact, source_case, source_compile)
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
            )
        source_normalization: Mapping[str, Any] | None = None
        try:
            normalized_source = normalize_ttir_for_gate1(
                source_compile.ttir, artifact
            )
            source_normalization = normalized_source.evidence()
            source_fingerprint = fingerprint_ttir(
                normalized_source.text,
                python_interface=_opaque_python_interface(
                    artifact, source_compile.python_interface
                ),
                schedule=artifact.c.values,
                representation_anchor_exemptions=(
                    normalized_source.representation_anchor_exemptions
                ),
                runtime_values={
                    name: argument.value
                    for name, argument in source_case.arguments.items()
                    if argument.kind == "scalar"
                    and isinstance(argument.value, (bool, int))
                },
            )
        except TTIRParseError as exc:
            normal = Gate1ArmEvidence(
                "normal", None, None, None, None,
                f"{SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX}: "
                f"TTIRParseError: {exc}",
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
            )

        normal_candidates = self._normal_arms(
            artifact, source_fingerprint, replay_case
        )
        selected_normal_index = max(
            range(len(normal_candidates)),
            key=lambda index: (
                self._normal_rank(normal_candidates[index]),
                -index,
            ),
        )
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
                normal_candidates=normal_candidates,
                selected_normal_index=selected_normal_index,
                source_normalization=source_normalization,
            )

        masked_evidence: Gate1ArmEvidence | None = None
        masked_calls = 0
        status = Gate1Status.REPLAY_CLOSED
        if self.config.run_masked_j:
            masked_calls = 1
            masked_evidence = self._arm(
                self.writers[selected_normal_index],
                artifact,
                source_fingerprint,
                replay_case,
                masked=True,
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
            normal_candidates=normal_candidates,
            selected_normal_index=selected_normal_index,
            source_normalization=source_normalization,
        )


__all__ = [
    "GATE1_PROTOCOL", "Gate1ArmEvidence", "Gate1Config",
    "Gate1ExecutionResult", "Gate1Runner",
    "SOURCE_OBSERVER_UNSUPPORTED_DIAGNOSTIC_PREFIX",
    "canonical_artifact_digest",
]
