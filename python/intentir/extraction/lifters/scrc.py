"""Source-aware Lift producing backend-neutral J plus Source C/pi evidence."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from intentir.extraction.providers.client import ProviderError, call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.dim_expr import validate_dimension_instantiation
from intentir.ir.intent_op_contract import KNOWN_INTENT_OPS, STRUCTURED_INTENT_OPS
from intentir.ir.layer_c import (
    LayerCInterface,
    binding_interface_digest,
    source_layer_c_interface,
)
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.verify.gate1.model import (
    AliasSpec,
    Gate1Artifact,
    ImplementationConfig,
    ParameterKind,
    ParameterMode,
    ScheduleSlotKind,
    SchemaError,
    _nested_root_owned_field_paths,
    implementation_value_matches,
)
from intentir.verify.gate1.pi import NORMALIZATION_WITNESS_SCHEMA
from intentir.verify.typecheck import TYPECHECK_PROTOCOL, type_check

from .scrc_schema import (
    scrc_lift_consistency_only_text_format,
    scrc_lift_core_json_schema,
    scrc_lift_core_text_format,
)


SCRC_RELIFT_PROTOCOL = "llm_direct_pi_structured_source_aware_relift_v2"
SCRC_DECODE_PROTOCOL = "intentir.scrc.decode.v18"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
# Lift has no framework-owned wall-clock deadline.  Keep these public names for
# callers that import the historical policy constants, but make their value
# explicit: ``None`` is the unbounded provider-turn contract.  A caller may
# still pass a positive value to ``resolve_source_aware_lift_timeout`` when it
# intentionally wants an experiment-local limit.
SOURCE_AWARE_LIFT_TURN_TIMEOUT_SECONDS: float | None = None
SOURCE_AWARE_LIFT_XHIGH_TURN_TIMEOUT_SECONDS: float | None = None
# A direct no-reasoning/low-effort Lift still has an output budget so a response
# cannot grow without bound.  This is not a wall-clock or CandidateCheck limit.
SOURCE_AWARE_LIFT_DIRECT_OUTPUT_TOKEN_BUDGET = 16000
# This is an authoring hint only; it does not create a second acceptance path.
# It is used exclusively by the diagnostic Source-to-J consistency workflow.
CONSISTENCY_ONLY_AUTHORING_PROFILE = "consistency-only"
# Keep the historical public names for compatibility with existing readers.
LIFT_RESPONSES_TIMEOUT_SECONDS: float | None = None
LIFT_RESPONSES_XHIGH_TIMEOUT_SECONDS: float | None = None
_CANDIDATE_DIAGNOSTIC_LIMIT = 20
_CANDIDATE_DIAGNOSTIC_TEXT_LIMIT = 2048
# A rejected JSON draft is useful only as an adjacent repair starting point.
# Keep it bounded so a malformed/verbose response cannot make the next
# source-aware request grow without limit.  Complete accepted artifacts are
# not subject to this limit; they remain the formal J/C/pi evidence.
MAX_RELIFT_CANDIDATE_BYTES = 256 * 1024
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
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "bool": "i1",
    "str": "string",
    "string": "string",
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


def _provider_error_is_timeout(error: ProviderError) -> bool:
    """Classify only transport timeouts as retryable Lift failures."""

    failure_class = getattr(error, "failure_class", None)
    if failure_class == "provider_timeout":
        return True
    if isinstance(failure_class, str) and failure_class:
        return False
    message = str(error).casefold()
    return "timeout" in message or "timed out" in message


def _provider_error_failure_class(error: ProviderError) -> str:
    """Return a closed, secret-free class for one provider failure."""

    failure_class = getattr(error, "failure_class", None)
    if isinstance(failure_class, str) and failure_class:
        return failure_class
    message = str(error).casefold()
    if "timeout" in message or "timed out" in message:
        return "provider_timeout"
    if "empty completion" in message:
        return "provider_no_candidate"
    return "provider_transport_error"


def _canonical_dtype(value: str | None) -> str | None:
    # Raw provider JSON reaches the additive surface-diagnostic pass before
    # the strict decoder has established field types.  Treat a malformed
    # spelling as an unmatched dtype instead of allowing ``casefold`` to turn
    # a candidate error into an uncaught host exception.
    if not isinstance(value, str):
        return None
    spelling = value.casefold().removeprefix("torch.")
    return _DTYPE_ALIASES.get(spelling, spelling)


def canonical_scalar_binding_dtype(value: bool | int | float | str) -> str:
    """Return the closed scalar dtype for one trusted Source binding.

    This is deliberately shared by Source-C slots and bound semantic
    ``tl.constexpr`` carriers.  Triton's ``tl.constexpr`` annotation establishes
    compile-time ownership but does not itself carry an integer width, so the
    Source/Harness boundary uses one deterministic representation rule instead
    of letting C and the Source manifest independently choose ``i32`` or
    ``i64``.  An explicit trusted type may still override this default when an
    upstream frontend provides one.
    """

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


@dataclass(frozen=True)
class SCRCExtraction:
    artifact: Gate1Artifact
    provenance: SCRCProvenance
    harness_binding: SCRCHarnessBinding
    source: str = field(repr=False)
    layer_c_interface: LayerCInterface | None = None

    def __post_init__(self) -> None:
        expected = source_layer_c_interface(self.artifact.j, self.artifact.c)
        if self.layer_c_interface is None:
            object.__setattr__(self, "layer_c_interface", expected)
        elif self.layer_c_interface != expected:
            raise ValueError(
                "SCRC extraction Layer-C interface differs from Source-C "
                "declarations"
            )

    @property
    def layer_c_interface_digest(self) -> str:
        assert self.layer_c_interface is not None
        return binding_interface_digest(self.layer_c_interface)

    @property
    def layer_c_interface_payload(self) -> dict[str, Any]:
        """Return the standalone, value-free Lift sidecar payload."""

        assert self.layer_c_interface is not None
        return self.layer_c_interface.to_dict()


@dataclass(frozen=True)
class SCRCLiftRequest:
    """Separated semantic prompt and private Source-binding material."""

    instructions: str = field(repr=False)
    user_input: str = field(repr=False)
    prompt_sha256: str
    harness_binding: SCRCHarnessBinding
    anonymous_compiler_view: Mapping[str, Any] | None = field(
        default=None, repr=False
    )
    semantic_compiler_view: Mapping[str, Any] | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True)
class SCRCCandidateCheckContext:
    """Value-free context visible to the local CandidateCheck."""

    expected_parameters: tuple[str, ...]
    semantic_compiler_view: Mapping[str, Any] | None = field(
        default=None, repr=False
    )
    schema_version: str = Gate1Artifact.SCHEMA
    decode_protocol: str = SCRC_DECODE_PROTOCOL
    typecheck_protocol: str = TYPECHECK_PROTOCOL

    def __post_init__(self) -> None:
        if any(not re.fullmatch(r"p[0-9]+", item) for item in self.expected_parameters):
            raise ValueError("CandidateCheck parameters must be anonymous p* handles")
        if self.semantic_compiler_view is not None:
            if not isinstance(self.semantic_compiler_view, Mapping):
                raise TypeError("CandidateCheck compiler view must be a mapping")
            value = dict(self.semantic_compiler_view)
            json.dumps(value, sort_keys=True, allow_nan=False)
            object.__setattr__(self, "semantic_compiler_view", value)


def scrc_candidate_check_context(
    request: SCRCLiftRequest,
) -> SCRCCandidateCheckContext:
    """Project a Lift request to the exact source-free checker context."""

    return SCRCCandidateCheckContext(
        expected_parameters=tuple(
            opaque
            for _source, opaque in request.harness_binding.source_to_opaque
            if re.fullmatch(r"p[0-9]+", opaque)
        ),
        semantic_compiler_view=request.semantic_compiler_view,
        typecheck_protocol=TYPECHECK_PROTOCOL,
    )


def _unbound_lift_visible_payload(value: Any) -> Any:
    """Remove Source-C values from every Gate1 artifact visible to Lift.

    Feedback can contain an artifact at the root or nested inside a diagnostic
    envelope.  Only objects with the closed Gate1 shape are changed; an
    unrelated diagnostic field named ``c`` is left untouched.  The returned
    object is a JSON-deep-copy, so persisted proof artifacts are never edited.
    """

    payload = json.loads(json.dumps(value, allow_nan=False))

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        result = {str(key): visit(child) for key, child in item.items()}
        schema = result.get("schema")
        c = result.get("c")
        if (
            isinstance(schema, str)
            and schema.startswith("intentir.scrc.artifact.")
            and {"schema", "j", "c", "pi"}.issubset(result)
            and isinstance(c, dict)
            and isinstance(c.get("slots"), list)
            and isinstance(c.get("values"), dict)
        ):
            c["values"] = {}
        return result

    return visit(payload)


def _unbound_lift_artifact_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one value-free prior Gate1 artifact for a Lift continuation."""

    payload = _unbound_lift_visible_payload(value)
    if not isinstance(payload, dict):
        raise TypeError("Lift artifact payload must be an object")
    return payload


def _value_free_lift_candidate_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one rejected draft without accepting or completing it.

    CandidateCheck can reject an object before it is a complete Gate1 artifact,
    so the normal artifact-only projection cannot be used as the continuation
    carrier.  This helper keeps the exact JSON structure, removes every
    candidate-authored ``c.values`` object, and applies a hard size bound.  It
    never allocates handles, inserts fields, or changes J semantics.
    """

    if not isinstance(value, Mapping):
        raise TypeError("Lift candidate draft must be a mapping")
    copied = json.loads(json.dumps(dict(value), allow_nan=False))

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        result = {str(key): visit(child) for key, child in item.items()}
        c = result.get("c")
        if isinstance(c, dict) and "values" in c:
            c["values"] = {}
        return result

    normalized = visit(copied)
    if not isinstance(normalized, dict):
        raise TypeError("Lift candidate draft must remain an object")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_RELIFT_CANDIDATE_BYTES:
        raise ValueError(
            "Lift candidate draft exceeds the bounded relift continuation size"
        )
    return normalized


_RELIFT_ENVELOPE_KEYS = frozenset({
    "unresolved_gate1",
    "unresolved_gate2",
})


def _prune_relift_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the current diagnostic frontier without recursive old envelopes.

    A relift may carry the previous Gate1/Gate2 frontier while the current
    attempt is being checked.  Retaining that frontier is useful; wrapping it
    again on every failed round is not.  Only nested ``unresolved_*`` envelopes
    are removed.  Current category counts, local frontiers, effect-order
    witnesses, and all other diagnostic fields are preserved byte-for-byte in
    the JSON projection.
    """

    if not isinstance(value, Mapping):
        raise TypeError("relift diagnostics must be a mapping")
    copied = json.loads(json.dumps(dict(value), allow_nan=False))

    def visit(item: Any, depth: int) -> Any:
        if isinstance(item, list):
            return [visit(child, depth + 1) for child in item]
        if not isinstance(item, dict):
            return item
        result: dict[str, Any] = {}
        for key, child in item.items():
            key = str(key)
            if depth > 0 and key in _RELIFT_ENVELOPE_KEYS:
                continue
            result[key] = visit(child, depth + 1)
        return result

    normalized = visit(copied, 0)
    if not isinstance(normalized, dict):
        raise TypeError("relift diagnostics must remain an object")
    return normalized


@dataclass(frozen=True)
class SCRCReliftContext:
    """Structured TypeCheck/Gate1/Gate2 evidence visible only to Lift."""

    relift_round: int
    failure_kind: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    previous_artifact: Mapping[str, Any] | None = None
    previous_candidate: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.relift_round < 1:
            raise ValueError("relift_round must be positive")
        if self.failure_kind not in {
            "lift",
            "typecheck",
            "gate1",
            "gate2",
        }:
            raise ValueError("unsupported relift failure kind")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("relift diagnostics must be a mapping")
        diagnostics = _unbound_lift_visible_payload(dict(self.diagnostics))
        if not isinstance(diagnostics, dict):
            raise TypeError("relift diagnostics must remain an object")
        diagnostics = _prune_relift_diagnostics(diagnostics)
        # The payload is sent only to Lift, but it must still be deterministic
        # JSON data rather than an executable callback or arbitrary object.
        json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.previous_artifact is not None:
            if not isinstance(self.previous_artifact, Mapping):
                raise TypeError("previous relift artifact must be a mapping")
            previous = _unbound_lift_artifact_payload(self.previous_artifact)
            json.dumps(previous, sort_keys=True, allow_nan=False)
            object.__setattr__(self, "previous_artifact", previous)
        if self.previous_candidate is not None:
            if self.previous_artifact is not None:
                raise ValueError(
                    "relift context cannot carry both a previous artifact and "
                    "a rejected candidate"
                )
            previous_candidate = _value_free_lift_candidate_payload(
                self.previous_candidate
            )
            object.__setattr__(self, "previous_candidate", previous_candidate)

    def to_json(self) -> dict[str, Any]:
        return {
            "relift_round": self.relift_round,
            "failure_kind": self.failure_kind,
            "diagnostics": dict(self.diagnostics),
            "previous_artifact": (
                dict(self.previous_artifact)
                if self.previous_artifact is not None else None
            ),
            "previous_candidate": (
                dict(self.previous_candidate)
                if self.previous_candidate is not None else None
            ),
        }


@dataclass(frozen=True)
class SCRCReliftRecord:
    relift_round: int
    extraction: SCRCExtraction | None = None
    diagnostic: str | None = None
    rejected_attempts: tuple["SCRCLiftAttempt", ...] = ()
    backend_evidence: Mapping[str, Any] | None = field(default=None, repr=False)
    continuation_candidate: Mapping[str, Any] | None = field(
        default=None, repr=False
    )


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
    # Keep this field after the pre-existing positional fields.  Older callers
    # used the fifth positional argument for ``value``; inserting strides before
    # it silently reinterpreted a scalar value as a physical layout.
    # Physical strides are Harness/source-realization facts and never belong to
    # logical J shape or target-neutral Kernel IR.
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
        if self.kind != "tensor" and strides is not None:
            raise ValueError("only tensor compiler arguments have physical strides")
        if strides is not None:
            if len(strides) != len(shape) or any(stride < 0 for stride in strides):
                raise ValueError(
                    "compiler argument physical strides must match rank and be non-negative"
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
            if self.strides is not None:
                result["strides"] = list(self.strides)
            if self.shape_contract is not None:
                result["shape_contract"] = list(self.shape_contract)
        elif self.kind == "scalar":
            result["value"] = self.value
            if self.shape_coordinate is not None:
                result["shape_coordinate"] = self.shape_coordinate
        if self.mode is not None:
            result["mode"] = self.mode
        return result


@dataclass(frozen=True)
class SCRCCompilerView:
    """Trusted invocation facts supplied to Lift, never a semantic ABI."""

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
    # Kept after the established positional fields so legacy callers that use
    # positional construction do not reinterpret a shape-coordinate map as a
    # type map.
    bound_constexpr_value_types: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.schedule_values)
        value_types = dict(self.schedule_value_types)
        bound_constexpr_values = dict(self.bound_constexpr_values)
        bound_constexpr_value_types = dict(self.bound_constexpr_value_types)
        bound_constexpr_shape_coordinates = dict(
            self.bound_constexpr_shape_coordinates
        )
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
            if not implementation_value_matches(value_type, value):
                raise ValueError(
                    f"schedule value {name!r} does not match its declared C type"
                )
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
        compiler_set = set(slots)
        expected_schedule_order = tuple(
            name for name in values if name not in compiler_set
        ) + slots
        if tuple(values) != expected_schedule_order:
            raise ValueError(
                "schedule values must declare Source-C slots followed by compiler slots"
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
        if parameter_order:
            if set(parameter_order) != runtime_names | constexpr_names:
                raise ValueError(
                    "parameter order must cover runtime and constexpr inputs"
                )
            expected_runtime_order = tuple(
                name for name in parameter_order if name not in constexpr_names
            )
            expected_constexpr_order = tuple(
                name for name in parameter_order if name in constexpr_names
            )
            if tuple(item.name for item in self.arguments) != expected_runtime_order:
                raise ValueError(
                    "runtime arguments must follow parameter declaration order"
                )
            if constexpr_parameters != expected_constexpr_order:
                raise ValueError(
                    "constexpr parameters must follow parameter declaration order"
                )
        elif runtime_names or constexpr_names:
            raise ValueError(
                "parameter order is required when compiler arguments are declared"
            )
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
            if not implementation_value_matches(value_type, value):
                raise ValueError(
                    f"bound constexpr value {name!r} does not match its declared type"
                )
            ordered_bound_constexpr_value_types[name] = value_type
        ordered_bound_constexpr_shape_coordinates = {
            name: bound_constexpr_shape_coordinates[name]
            for name in constexpr_parameters
            if name in bound_constexpr_shape_coordinates
        }
        object.__setattr__(
            self, "bound_constexpr_values", ordered_bound_constexpr_values
        )
        object.__setattr__(
            self,
            "bound_constexpr_value_types",
            ordered_bound_constexpr_value_types,
        )
        object.__setattr__(
            self,
            "bound_constexpr_shape_coordinates",
            ordered_bound_constexpr_shape_coordinates,
        )
        schedule_constexprs = constexpr_names - bound_names
        if schedule_constexprs - set(values):
            raise ValueError("schedule constexpr parameters require C values")
        if schedule_constexprs & compiler_set:
            raise ValueError(
                "source constexpr parameters cannot be compiler-only slots"
            )
        c_source_names = tuple(name for name in values if name not in compiler_set)
        if set(c_source_names) != schedule_constexprs:
            raise ValueError(
                "schedule values must cover exactly unbound source constexprs and compiler slots"
            )

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
class SCRCLiftAttempt:
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    backend: str = "responses"
    termination_status: str | None = None
    trajectory: Mapping[str, Any] | None = field(default=None, repr=False)


class SCRCLiftError(RuntimeError):
    """Rejected Responses Lift output with an explicit feedback boundary.

    ``feedback`` is safe to return to a semantic Lift owner.  A terminal error
    denotes a rejection that happened only after the value-free candidate had
    been accepted, so callers must not start another owner round from it.
    """

    def __init__(
        self,
        attempt: SCRCLiftAttempt,
        *,
        feedback: Mapping[str, Any] | None = None,
        failure_class: str = "candidate_check_rejected",
        terminal: bool = False,
        candidate_draft: Mapping[str, Any] | None = None,
    ) -> None:
        self.attempts = (attempt,)
        if not isinstance(failure_class, str) or not failure_class:
            raise ValueError("SCRC Lift failure_class must be non-empty")
        if not isinstance(terminal, bool):
            raise TypeError("SCRC Lift terminal flag must be boolean")
        safe_feedback = dict(feedback or {"diagnostic": attempt.diagnostic})
        json.dumps(safe_feedback, sort_keys=True, allow_nan=False)
        self.feedback = safe_feedback
        self.failure_class = failure_class
        self.terminal = terminal
        self.candidate_draft = (
            _value_free_lift_candidate_payload(candidate_draft)
            if candidate_draft is not None
            else None
        )
        super().__init__(f"SCRC Lift response rejected: {attempt.diagnostic}")


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


class _ModuleBindingCollector(ast.NodeVisitor):
    """Collect bindings in module execution scope without entering local scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)


def _intent_kernel_decorator_aliases(
    tree: ast.Module,
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve only imports that can spell the public Intent kernel decorator.

    The projection deliberately does not classify decorators by a generic
    ``.kernel`` suffix: Triton and other frontend helpers must remain visible.
    Star imports are rejected because a later bare ``@kernel`` cannot be tied
    to the Intent package without guessing about rebinding.
    """

    module_aliases: set[str] = set()
    direct_aliases: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "intent":
                    module_aliases.add(alias.asname or "intent")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "intent":
            for alias in statement.names:
                if alias.name == "*":
                    raise ValueError(
                        "single-kernel Lift cannot safely project an Intent star import"
                    )
                if alias.name == "kernel":
                    direct_aliases.add(alias.asname or alias.name)
    protected_aliases = module_aliases | direct_aliases
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            rebound = {
                alias.asname or alias.name.split(".", 1)[0]
                for alias in statement.names
                if alias.name != "intent"
            }
        elif isinstance(statement, ast.ImportFrom):
            rebound = {
                alias.asname or alias.name
                for alias in statement.names
                if not (
                    statement.module == "intent"
                    and alias.name in {"kernel", "*"}
                )
            }
        else:
            collector = _ModuleBindingCollector()
            collector.visit(statement)
            rebound = collector.names
        collision = rebound & protected_aliases
        if collision:
            raise ValueError(
                "single-kernel Lift cannot prove an Intent decorator alias after "
                "module-level rebinding: " + ", ".join(sorted(collision))
            )
    return frozenset(module_aliases), frozenset(direct_aliases)


def _is_intent_kernel_definition(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    module_aliases: frozenset[str],
    direct_aliases: frozenset[str],
) -> bool:
    for original in function.decorator_list:
        decorator = original.func if isinstance(original, ast.Call) else original
        if isinstance(decorator, ast.Name) and decorator.id in direct_aliases:
            return True
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "kernel"
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id in module_aliases
        ):
            return True
    return False


def _project_selected_intent_kernel(
    tree: ast.Module,
    source_symbol: str,
) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Remove unrelated top-level ``@intent.kernel`` variants before Lift.

    This is a source-selection projection, not dependency pruning.  Ordinary
    helpers, ``@intent.fn`` functions, imports, constants, classes and enums
    remain available to the selected kernel.  In particular, decorated Triton
    helpers are not mistaken for sibling Intent entries.
    """

    if not isinstance(source_symbol, str) or not source_symbol.isidentifier():
        raise ValueError("Lift source_symbol must be a Python identifier")
    top_level_functions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    selected = [
        function for function in top_level_functions if function.name == source_symbol
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Lift source_symbol {source_symbol!r} must resolve exactly once at "
            f"module scope; found {len(selected)} definitions"
        )
    primary = selected[0]
    module_aliases, direct_aliases = _intent_kernel_decorator_aliases(tree)
    runtime_kernels = {
        id(function): function.name
        for function in top_level_functions
        if _is_intent_kernel_definition(
            function,
            module_aliases=module_aliases,
            direct_aliases=direct_aliases,
        )
    }
    if runtime_kernels and id(primary) not in runtime_kernels:
        raise ValueError(
            "Lift source_symbol must select a top-level @intent.kernel entry "
            "when the module declares Intent runtime kernels"
        )
    siblings = {
        function_id: function_name
        for function_id, function_name in runtime_kernels.items()
        if function_id != id(primary)
    }
    if siblings:
        sibling_names = set(siblings.values())
        retained_statements = [
            statement for statement in tree.body if id(statement) not in siblings
        ]
        directly_referenced = {
            node.id
            for statement in retained_statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in sibling_names
        }
        if directly_referenced:
            raise ValueError(
                "selected Intent projection retains a reference to another "
                "runtime kernel entry: "
                + ", ".join(sorted(directly_referenced))
            )
        tree.body = retained_statements
        ast.fix_missing_locations(tree)
    return tree, primary


def anonymize_source(
    source: str,
    *,
    source_symbol: str | None = None,
    bound_constexpr_parameters: tuple[str, ...] = (),
) -> tuple[str, SCRCHarnessBinding, dict[str, str]]:
    """Alpha-rename source identifiers before the Lift provider sees them.

    ``source_symbol`` is a privileged source-selection coordinate.  It chooses
    the callable whose positional interface becomes ``scrc_k0`` but is never
    placed in J, C, pi, or the provider prompt.  This is required for real
    library modules that contain helpers and several kernels; declaration order
    in a Python file is not a semantic kernel identity.
    """

    tree = ast.parse(source)
    projected_primary: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if source_symbol is not None:
        tree, projected_primary = _project_selected_intent_kernel(
            tree, source_symbol
        )
    functions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    if not functions:
        raise ValueError("Lift source contains no function")
    if source_symbol is not None:
        assert projected_primary is not None
        primary = projected_primary
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
    primary_parameters = (
        *primary.args.posonlyargs,
        *primary.args.args,
        *primary.args.kwonlyargs,
    )
    if primary.args.vararg is not None or primary.args.kwarg is not None:
        raise ValueError(
            "selected source ABI must be expanded before anonymization; "
            "variadic parameters are not stable Harness formals"
        )
    annotated_constexprs = {
        argument.arg
        for argument in primary_parameters
        if _annotation_is_constexpr(argument.annotation)
    }
    unknown_bound_constexprs = bound_constexprs - annotated_constexprs
    if unknown_bound_constexprs:
        raise ValueError(
            "bound constexpr parameters are absent from the selected ABI: "
            + ", ".join(sorted(unknown_bound_constexprs))
        )

    runtime_index = 0
    constexpr_index = 0
    for argument in primary_parameters:
        if _annotation_is_constexpr(argument.annotation):
            if argument.arg in bound_constexprs:
                opaque = f"p{runtime_index}"
                runtime_index += 1
            else:
                opaque = f"c{constexpr_index}"
                constexpr_index += 1
        else:
            opaque = f"p{runtime_index}"
            runtime_index += 1
        mapping.setdefault(argument.arg, opaque)

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

    # Keep lexical Source locals out of J's ``v*`` SSA namespace.  The Lift
    # owner sees this source projection and must be able to distinguish a
    # source temporary from a J-authored operation result.  The old reuse of
    # ``v*`` on both sides made a local such as ``v0`` look like an already
    # allocated J value and was the direct cause of candidates declaring
    # source temporaries as logical tensors.  Generated names are also kept
    # away from imported/public names so alpha-renaming cannot introduce a
    # second lexical binding.
    source_local_index = 0

    def next_source_local_handle() -> str:
        nonlocal source_local_index
        while True:
            candidate = f"s{source_local_index}"
            source_local_index += 1
            if candidate not in reserved and candidate not in mapping.values():
                return candidate

    for node in local_nodes:
        if node.id not in mapping:
            mapping[node.id] = next_source_local_handle()

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
                mapping[argument.arg] = next_source_local_handle()
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
    """Alpha-rename a selected native device projection without Python AST.

    Native source is already reduced to one verified device-entry projection by
    its frontend adapter.  This routine only rewrites declared ABI identifiers
    and the selected entry symbol; it never parses CUDA/C++ as Python and never
    invents a semantic mapping for local expressions.
    """

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
        re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(source_symbol)}(?![A-Za-z0-9_])", masked)
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


_SEMANTIC_LIFT_VIEW_FIELDS = frozenset(
    {
        "arguments",
        "aliases",
        "shape_coordinates",
        "bound_constexpr_parameters",
        "bound_constexpr_value_types",
        "bound_constexpr_shape_coordinates",
        "constexpr_bindings",
        "parameter_order",
        "constexpr_parameters",
    }
)
_SEMANTIC_ARGUMENT_FIELDS = frozenset(
    {
        "name",
        "names",
        "kind",
        "dtype",
        "rank",
        "shape_contract",
        "shape_coordinate",
        "mode",
    }
)
_SEMANTIC_CONSTEXPR_BINDING_FIELDS = frozenset(
    {"name", "dtype", "shape_coordinate"}
)


def _semantic_lift_handle(value: Any, path: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"p[0-9]+", value) is None:
        raise ValueError(f"{path} must be a p* handle")
    return value


def _semantic_shape_coordinate_order(
    arguments: Sequence[Mapping[str, Any]],
    constexpr_shape_coordinates: Mapping[str, str],
) -> list[str]:
    """Recover the only dense h* order permitted by the semantic ABI.

    The trusted compiler view establishes a shape coordinate at its first
    appearance in a tensor shape contract.  Scalar and constexpr coordinate
    attachments must point back to that same coordinate.  Looking at all three
    locations here keeps the compact transport self-contained while retaining
    the original first-occurrence rule if a future frontend exposes the latter
    two locations first.
    """

    result: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            result.append(value)

    for argument in arguments:
        contract = argument.get("shape_contract")
        if isinstance(contract, list):
            for extent in contract:
                add(extent)
        add(argument.get("shape_coordinate"))
    for coordinate in constexpr_shape_coordinates.values():
        add(coordinate)
    return result


def _expand_semantic_lift_view(
    semantic_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand the compact model-visible ABI view into its closed legacy form.

    ``semantic_lift_view`` is a value-free transport projection, not a second
    semantic graph.  Its abbreviated fields are accepted only where the full
    value has one deterministic reconstruction.  This function is deliberately
    fail-closed so a malformed compact request cannot cause CandidateCheck and
    final Decode to construct different ABI facts.
    """

    if not isinstance(semantic_view, Mapping):
        raise TypeError("semantic ABI view must be an object")
    unknown = sorted(set(semantic_view) - _SEMANTIC_LIFT_VIEW_FIELDS)
    if unknown:
        raise ValueError(
            "semantic ABI view has unsupported field(s): " + ", ".join(unknown)
        )

    raw_arguments = semantic_view.get("arguments", [])
    if not isinstance(raw_arguments, list):
        raise TypeError("semantic ABI arguments must be an array")
    arguments: list[dict[str, Any]] = []
    argument_names: set[str] = set()
    for index, raw_argument in enumerate(raw_arguments):
        path = f"semantic ABI arguments[{index}]"
        if not isinstance(raw_argument, Mapping):
            raise TypeError(f"{path} must be an object")
        unknown_fields = sorted(set(raw_argument) - _SEMANTIC_ARGUMENT_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{path} has unsupported field(s): " + ", ".join(unknown_fields)
            )
        has_name = "name" in raw_argument
        has_names = "names" in raw_argument
        if has_name == has_names:
            raise ValueError(f"{path} must contain exactly one of name or names")
        if has_name:
            names = [_semantic_lift_handle(raw_argument["name"], f"{path}.name")]
        else:
            raw_names = raw_argument["names"]
            if not isinstance(raw_names, list) or not raw_names:
                raise TypeError(f"{path}.names must be a non-empty array")
            names = [
                _semantic_lift_handle(name, f"{path}.names[{ordinal}]")
                for ordinal, name in enumerate(raw_names)
            ]
            if len(names) != len(set(names)):
                raise ValueError(f"{path}.names must not repeat a p* handle")
        duplicate_names = sorted(set(names) & argument_names)
        if duplicate_names:
            raise ValueError(
                f"{path} repeats semantic ABI argument(s): "
                + ", ".join(duplicate_names)
            )
        argument_names.update(names)
        common = {
            key: value
            for key, value in raw_argument.items()
            if key not in {"name", "names"}
        }
        if common.get("kind") == "tensor" and "shape_contract" in common:
            contract = common["shape_contract"]
            if not isinstance(contract, list):
                raise TypeError(f"{path}.shape_contract must be an array")
            if "rank" in common:
                rank = common["rank"]
                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank != len(contract)
                ):
                    raise ValueError(
                        f"{path}.rank must equal the shape_contract length"
                    )
            else:
                common["rank"] = len(contract)
        for name in names:
            arguments.append({"name": name, **common})

    raw_aliases = semantic_view.get("aliases", [])
    if not isinstance(raw_aliases, list):
        raise TypeError("semantic ABI aliases must be an array")

    compact_constexpr = "constexpr_bindings" in semantic_view
    legacy_constexpr_fields = {
        "bound_constexpr_parameters",
        "bound_constexpr_value_types",
        "bound_constexpr_shape_coordinates",
    }
    if compact_constexpr and legacy_constexpr_fields & set(semantic_view):
        raise ValueError(
            "compact constexpr_bindings cannot coexist with legacy bound "
            "constexpr fields"
        )
    bound_constexpr_parameters: list[str]
    bound_constexpr_value_types: dict[str, str]
    bound_constexpr_shape_coordinates: dict[str, str]
    if compact_constexpr:
        raw_bindings = semantic_view["constexpr_bindings"]
        if not isinstance(raw_bindings, list):
            raise TypeError("semantic constexpr_bindings must be an array")
        bound_constexpr_parameters = []
        bound_constexpr_value_types = {}
        bound_constexpr_shape_coordinates = {}
        for index, raw_binding in enumerate(raw_bindings):
            path = f"semantic constexpr_bindings[{index}]"
            if not isinstance(raw_binding, Mapping):
                raise TypeError(f"{path} must be an object")
            unknown_fields = sorted(
                set(raw_binding) - _SEMANTIC_CONSTEXPR_BINDING_FIELDS
            )
            if unknown_fields:
                raise ValueError(
                    f"{path} has unsupported field(s): "
                    + ", ".join(unknown_fields)
                )
            if "name" not in raw_binding or "dtype" not in raw_binding:
                raise ValueError(f"{path} requires name and dtype")
            name = _semantic_lift_handle(raw_binding["name"], f"{path}.name")
            if name in bound_constexpr_value_types:
                raise ValueError(f"{path}.name duplicates a constexpr binding")
            dtype = raw_binding["dtype"]
            if not isinstance(dtype, str) or not dtype:
                raise TypeError(f"{path}.dtype must be a non-empty string")
            bound_constexpr_parameters.append(name)
            bound_constexpr_value_types[name] = dtype
            if "shape_coordinate" in raw_binding:
                coordinate = raw_binding["shape_coordinate"]
                if (
                    not isinstance(coordinate, str)
                    or re.fullmatch(r"h[0-9]+", coordinate) is None
                ):
                    raise ValueError(f"{path}.shape_coordinate must be an h* handle")
                bound_constexpr_shape_coordinates[name] = coordinate
    else:
        raw_parameters = semantic_view.get("bound_constexpr_parameters", [])
        if not isinstance(raw_parameters, list):
            raise TypeError("bound constexpr parameters must be an array of names")
        bound_constexpr_parameters = [
            _semantic_lift_handle(
                name, f"bound constexpr parameters[{index}]"
            )
            for index, name in enumerate(raw_parameters)
        ]
        if len(bound_constexpr_parameters) != len(set(bound_constexpr_parameters)):
            raise ValueError("bound constexpr parameters must be unique")
        raw_value_types = semantic_view.get("bound_constexpr_value_types", {})
        if not isinstance(raw_value_types, Mapping):
            raise TypeError("bound constexpr value types must be an object")
        bound_constexpr_value_types = {}
        for name, dtype in raw_value_types.items():
            handle = _semantic_lift_handle(name, "bound constexpr value type key")
            if not isinstance(dtype, str) or not dtype:
                raise TypeError("bound constexpr value types must be non-empty strings")
            bound_constexpr_value_types[handle] = dtype
        raw_shape_coordinates = semantic_view.get(
            "bound_constexpr_shape_coordinates", {}
        )
        if not isinstance(raw_shape_coordinates, Mapping):
            raise TypeError(
                "bound constexpr shape-coordinate bindings must be an object"
            )
        bound_constexpr_shape_coordinates = {}
        for name, coordinate in raw_shape_coordinates.items():
            handle = _semantic_lift_handle(
                name, "bound constexpr shape-coordinate key"
            )
            if (
                not isinstance(coordinate, str)
                or re.fullmatch(r"h[0-9]+", coordinate) is None
            ):
                raise ValueError(
                    "bound constexpr shape-coordinate values must be h* handles"
                )
            bound_constexpr_shape_coordinates[handle] = coordinate

    if "constexpr_parameters" not in semantic_view:
        constexpr_parameters = list(bound_constexpr_parameters)
    else:
        raw_constexpr_parameters = semantic_view["constexpr_parameters"]
        if not isinstance(raw_constexpr_parameters, list):
            raise TypeError("semantic ABI constexpr_parameters must be an array")
        constexpr_parameters = [
            _semantic_lift_handle(
                name, f"semantic ABI constexpr_parameters[{index}]"
            )
            for index, name in enumerate(raw_constexpr_parameters)
        ]
        if len(constexpr_parameters) != len(set(constexpr_parameters)):
            raise ValueError("semantic ABI constexpr_parameters must be unique")
    missing_constexpr_declarations = sorted(
        set(bound_constexpr_parameters) - set(constexpr_parameters)
    )
    if missing_constexpr_declarations:
        raise ValueError(
            "bound constexpr parameters are absent from constexpr_parameters: "
            + ", ".join(missing_constexpr_declarations)
        )

    if "shape_coordinates" not in semantic_view:
        shape_coordinates = _semantic_shape_coordinate_order(
            arguments, bound_constexpr_shape_coordinates
        )
    else:
        raw_shape_coordinates = semantic_view["shape_coordinates"]
        if not isinstance(raw_shape_coordinates, list) or any(
            not isinstance(item, str) for item in raw_shape_coordinates
        ):
            raise TypeError("semantic ABI shape_coordinates must be an array of names")
        shape_coordinates = list(raw_shape_coordinates)
    if len(shape_coordinates) != len(set(shape_coordinates)):
        raise ValueError("semantic ABI shape_coordinates must be unique")
    if shape_coordinates != [f"h{index}" for index in range(len(shape_coordinates))]:
        raise ValueError("semantic ABI shape_coordinates must be dense h* handles")

    if "parameter_order" not in semantic_view:
        parameter_order = [
            *(argument["name"] for argument in arguments),
            *constexpr_parameters,
        ]
    else:
        raw_parameter_order = semantic_view["parameter_order"]
        if not isinstance(raw_parameter_order, list):
            raise TypeError("semantic ABI parameter_order must be an array of names")
        parameter_order = [
            _semantic_lift_handle(
                name, f"semantic ABI parameter_order[{index}]"
            )
            for index, name in enumerate(raw_parameter_order)
        ]
    if len(parameter_order) != len(set(parameter_order)):
        raise ValueError("semantic ABI parameter_order must be unique")

    return {
        "arguments": arguments,
        "aliases": list(raw_aliases),
        "shape_coordinates": shape_coordinates,
        "bound_constexpr_parameters": bound_constexpr_parameters,
        "bound_constexpr_value_types": bound_constexpr_value_types,
        "bound_constexpr_shape_coordinates": bound_constexpr_shape_coordinates,
        "parameter_order": parameter_order,
        "constexpr_parameters": constexpr_parameters,
    }


def _compact_semantic_lift_view(
    semantic_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a lossless, model-visible projection of the semantic ABI view.

    The private Lift/BindSource view remains fully expanded.  This projection
    removes only values reconstructed by :func:`_expand_semantic_lift_view`:
    adjacent argument names with identical metadata, tensor rank implied by a
    shape contract, the three bound-constexpr duplicate collections, dense h*
    order, and the default physical parameter order.
    """

    expanded = _expand_semantic_lift_view(semantic_view)
    compact_arguments: list[dict[str, Any]] = []
    group_names: list[str] = []
    group_metadata: dict[str, Any] | None = None

    def flush_group() -> None:
        nonlocal group_names, group_metadata
        if group_metadata is None:
            return
        if len(group_names) == 1:
            compact_arguments.append({"name": group_names[0], **group_metadata})
        else:
            compact_arguments.append({"names": group_names, **group_metadata})
        group_names = []
        group_metadata = None

    for argument in expanded["arguments"]:
        name = argument["name"]
        metadata = {key: value for key, value in argument.items() if key != "name"}
        if metadata.get("kind") == "tensor" and isinstance(
            metadata.get("shape_contract"), list
        ):
            # ``_expand_semantic_lift_view`` has already proved equality.
            metadata.pop("rank", None)
        if group_metadata is not None and metadata == group_metadata:
            group_names.append(name)
            continue
        flush_group()
        group_names = [name]
        group_metadata = metadata
    flush_group()

    constexpr_bindings: list[dict[str, Any]] = []
    for name in expanded["bound_constexpr_parameters"]:
        binding: dict[str, Any] = {
            "name": name,
            "dtype": expanded["bound_constexpr_value_types"][name],
        }
        coordinate = expanded["bound_constexpr_shape_coordinates"].get(name)
        if coordinate is not None:
            binding["shape_coordinate"] = coordinate
        constexpr_bindings.append(binding)

    compact: dict[str, Any] = {
        "arguments": compact_arguments,
        "aliases": list(expanded["aliases"]),
        "constexpr_bindings": constexpr_bindings,
        "constexpr_parameters": list(expanded["constexpr_parameters"]),
    }
    derived_shape_coordinates = _semantic_shape_coordinate_order(
        expanded["arguments"], expanded["bound_constexpr_shape_coordinates"]
    )
    if expanded["shape_coordinates"] != derived_shape_coordinates:
        compact["shape_coordinates"] = list(expanded["shape_coordinates"])
    default_parameter_order = [
        *(argument["name"] for argument in expanded["arguments"]),
        *expanded["constexpr_parameters"],
    ]
    if expanded["parameter_order"] != default_parameter_order:
        compact["parameter_order"] = list(expanded["parameter_order"])
    if _expand_semantic_lift_view(compact) != expanded:
        raise AssertionError("semantic Lift view compact projection is not lossless")
    return compact


def _semantic_abi_contract(
    semantic_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one mechanical Source-ABI-to-J contract for all consumers.

    The returned private indexes are used by the final Decode and by the
    source-free CandidateCheck diagnostics.  The nested ``lock`` is the
    value-free subset exposed to the Lift owner.  Keeping both projections
    derived from this function prevents the prompt, diagnostic, and accepting
    validator from silently assigning different ``p*``/``t*``/``x*`` handles.
    """

    semantic_view = _expand_semantic_lift_view(semantic_view)
    raw_arguments = semantic_view.get("arguments", [])
    if not isinstance(raw_arguments, list):
        raise TypeError("semantic ABI arguments must be an array")

    parameter_order = semantic_view.get("parameter_order", [])
    if not isinstance(parameter_order, list) or any(
        not isinstance(item, str) for item in parameter_order
    ):
        raise TypeError("semantic ABI parameter_order must be an array of names")
    if len(parameter_order) != len(set(parameter_order)) or any(
        re.fullmatch(r"p[0-9]+", item) is None for item in parameter_order
    ):
        raise ValueError("semantic ABI parameter_order must be unique p* handles")
    if parameter_order != [f"p{index}" for index in range(len(parameter_order))]:
        raise ValueError("semantic ABI parameter_order must be dense p* handles")

    shape_coordinates = semantic_view.get("shape_coordinates", [])
    if not isinstance(shape_coordinates, list) or any(
        not isinstance(item, str) for item in shape_coordinates
    ):
        raise TypeError("semantic ABI shape_coordinates must be an array of names")
    if len(shape_coordinates) != len(set(shape_coordinates)):
        raise ValueError("semantic ABI shape_coordinates must be unique")
    if shape_coordinates != [
        f"h{index}" for index in range(len(shape_coordinates))
    ]:
        raise ValueError("semantic ABI shape_coordinates must be dense h* handles")
    shape_dimension_bindings = [
        {
            "source_coordinate": coordinate,
            "j_dimension": f"x{index}",
        }
        for index, coordinate in enumerate(shape_coordinates)
    ]
    coordinate_to_dimension = {
        item["source_coordinate"]: item["j_dimension"]
        for item in shape_dimension_bindings
    }

    tensor_arguments: list[Mapping[str, Any]] = []
    argument_names: set[str] = set()
    for argument in raw_arguments:
        if not isinstance(argument, Mapping):
            raise TypeError("semantic ABI argument must be an object")
        name = argument.get("name")
        kind = argument.get("kind")
        if not isinstance(name, str) or re.fullmatch(r"p[0-9]+", name) is None:
            raise ValueError("semantic ABI argument names must be p* handles")
        if name in argument_names:
            raise ValueError("semantic ABI argument names must be unique")
        argument_names.add(name)
        if kind == "tensor":
            mode = argument.get("mode")
            # A tensor with no explicit ABI role is not safe to project into J:
            # treating it as an input would turn an unresolved physical p*
            # parameter into a logical t* input and make every downstream
            # consumer agree on the same wrong layout.  The source adapter is
            # responsible for supplying input/output/inout, so fail closed at
            # this ownership boundary instead of guessing.
            if mode not in {"input", "output", "inout"}:
                raise ValueError(
                    "semantic tensor ABI requires an explicit input, output, "
                    "or inout role"
                )
            rank = argument.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError("semantic tensor ABI rank must be non-negative")
            contract = argument.get("shape_contract")
            if contract is not None:
                if not isinstance(contract, list) or len(contract) != rank:
                    raise ValueError(
                        "semantic tensor ABI shape_contract must match its rank"
                    )
                for extent in contract:
                    if isinstance(extent, str):
                        if extent not in coordinate_to_dimension:
                            raise ValueError(
                                "semantic tensor ABI shape_contract references an "
                                f"undeclared source coordinate {extent!r}"
                            )
                    elif isinstance(extent, bool) or not isinstance(extent, int):
                        raise ValueError(
                            "semantic tensor ABI shape_contract entries must be "
                            "source coordinates or static integers"
                        )
            tensor_arguments.append(argument)

    # Keep the Source physical tensor role order separate from the final J
    # logical-tensor numbering.  Output-only tensors cannot receive a final t*
    # here: a scalar/constexpr that Lift proves to be logical rank-0 data is
    # inserted before outputs.  Assigning output t* handles in this value-free
    # baseline was the source of the old t6/t7 vs t4/t5/t6 split-brain ABI.
    # Tensor roles are closed above, so the logical order can be derived only
    # from explicit Source ABI declarations.  Do not retain an unresolved
    # entry: that would make a physical p* parameter look like a J input.
    known_inputs = [
        item for item in tensor_arguments if item.get("mode") in {"input", "inout"}
    ]
    known_outputs = [
        item for item in tensor_arguments if item.get("mode") == "output"
    ]
    ordered_tensors = [*known_inputs, *known_outputs]
    logical_tensor_by_parameter: dict[str, str | None] = {}
    logical_tensor_order: list[dict[str, Any]] = []
    for item in ordered_tensors:
        parameter = item["name"]
        mode = item.get("mode")
        logical_ref = (
            f"t{len(logical_tensor_order)}"
            if mode != "output"
            else None
        )
        logical_tensor_by_parameter[parameter] = logical_ref
        container = {
            "input": "inputs",
            "inout": "inputs+outputs",
            "output": "outputs",
        }[mode]
        raw_contract = item.get("shape_contract")
        logical_shape = (
            [
                coordinate_to_dimension[extent]
                if isinstance(extent, str)
                else extent
                for extent in raw_contract
            ]
            if isinstance(raw_contract, list)
            else None
        )
        logical_tensor_order.append(
            {
                "logical_tensor": logical_ref,
                "physical_parameter": parameter,
                "container": container,
                "role_known": True,
                "logical_ref_policy": (
                    "fixed_declaration_order"
                    if logical_ref is not None
                    else "deferred_after_optional_scalar_values"
                ),
                "mode": mode,
                "dtype": item.get("dtype"),
                "rank": item.get("rank"),
                "shape": logical_shape,
            }
        )

    physical_bindings: list[dict[str, Any]] = []
    tensor_container_by_parameter = {
        item["physical_parameter"]: item["container"]
        for item in logical_tensor_order
    }
    for argument in raw_arguments:
        parameter = argument["name"]
        kind = argument.get("kind")
        mode = argument.get("mode")
        if kind == "tensor":
            logical_ref = logical_tensor_by_parameter[parameter]
            # Output-only tensor skeleton entries intentionally carry
            # logical_tensor=null until optional rank-0 scalar decisions are
            # applied.  Looking them up by logical_ref therefore aliases every
            # output to the first null entry.  Physical parameter identity is
            # stable at this stage and is the only safe lookup key.
            try:
                container = tensor_container_by_parameter[parameter]
            except KeyError as exc:
                raise ValueError(
                    f"semantic tensor ABI parameter {parameter!r} has no "
                    "physical-role descriptor"
                ) from exc
            physical_bindings.append(
                {
                    "physical_parameter": parameter,
                    "kind": "tensor",
                    "mode": mode,
                    "logical_ref": logical_ref,
                    "logical_container": container,
                    "logical_ref_policy": (
                        "fixed_declaration_order"
                        if logical_ref is not None
                        else "deferred_after_optional_scalar_values"
                    ),
                }
            )
        elif kind == "scalar" and isinstance(
            argument.get("shape_coordinate"), str
        ):
            coordinate = argument["shape_coordinate"]
            if coordinate not in coordinate_to_dimension:
                raise ValueError(
                    "semantic scalar ABI references an undeclared source coordinate"
                )
            physical_bindings.append(
                {
                    "physical_parameter": parameter,
                    "kind": "scalar",
                    "mode": mode,
                    "logical_ref": coordinate_to_dimension[coordinate],
                    "logical_kind": "dimension",
                    "source_coordinate": coordinate,
                }
            )
        elif kind == "scalar":
            physical_bindings.append(
                {
                    "physical_parameter": parameter,
                    "kind": "scalar",
                    "mode": mode,
                    "logical_ref": None,
                    "logical_kind": "source_use_dependent_rank0_or_representation",
                }
            )
        else:
            physical_bindings.append(
                {
                    "physical_parameter": parameter,
                    "kind": kind,
                    "mode": mode,
                    "logical_ref": None,
                    "logical_kind": "representation_only",
                }
            )

    bound_constexpr_parameters = semantic_view.get(
        "bound_constexpr_parameters", []
    )
    if not isinstance(bound_constexpr_parameters, list) or any(
        not isinstance(item, str) for item in bound_constexpr_parameters
    ):
        raise TypeError("bound constexpr parameters must be an array of names")
    if not set(bound_constexpr_parameters) <= set(parameter_order):
        raise ValueError(
            "bound constexpr parameters must be present in semantic parameter_order"
        )
    ordered_bound_constexpr_parameters = [
        name for name in parameter_order if name in set(bound_constexpr_parameters)
    ]
    raw_bound_constexpr_value_types = semantic_view.get(
        "bound_constexpr_value_types", {}
    )
    if not isinstance(raw_bound_constexpr_value_types, Mapping):
        raise TypeError("bound constexpr value types must be an object")
    if set(raw_bound_constexpr_value_types) != set(
        ordered_bound_constexpr_parameters
    ):
        raise ValueError(
            "bound constexpr value types must cover exactly bound constexpr parameters"
        )
    bound_constexpr_value_types: dict[str, str] = {}
    for name in ordered_bound_constexpr_parameters:
        value_type = raw_bound_constexpr_value_types[name]
        if not isinstance(value_type, str) or not value_type:
            raise TypeError("bound constexpr value types must be non-empty strings")
        bound_constexpr_value_types[name] = value_type
    raw_bound_constexpr_shape_coordinates = semantic_view.get(
        "bound_constexpr_shape_coordinates", {}
    )
    if not isinstance(raw_bound_constexpr_shape_coordinates, Mapping):
        raise TypeError("bound constexpr shape-coordinate bindings must be an object")
    if not set(raw_bound_constexpr_shape_coordinates) <= set(
        ordered_bound_constexpr_parameters
    ):
        raise ValueError(
            "bound constexpr shape-coordinate bindings may name only bound constexpr parameters"
        )
    bound_constexpr_shape_coordinates: dict[str, str] = {}
    bound_constexpr_dimension_by_parameter: dict[str, str] = {}
    for name, coordinate in raw_bound_constexpr_shape_coordinates.items():
        if not isinstance(name, str) or not isinstance(coordinate, str):
            raise TypeError("bound constexpr shape-coordinate binding is invalid")
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "bound constexpr shape-coordinate binding names an undeclared source coordinate"
            )
        bound_constexpr_shape_coordinates[name] = coordinate
        bound_constexpr_dimension_by_parameter[name] = dimension
    if set(argument_names) & set(bound_constexpr_parameters):
        raise ValueError(
            "bound constexpr parameters must not also appear as runtime arguments"
        )
    expected_parameter_names = argument_names | set(
        bound_constexpr_parameters
    )
    if set(parameter_order) != expected_parameter_names:
        missing = sorted(expected_parameter_names - set(parameter_order))
        extra = sorted(set(parameter_order) - expected_parameter_names)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "semantic ABI parameter_order must cover exactly runtime and bound "
            "constexpr arguments" + (" (" + "; ".join(detail) + ")" if detail else "")
        )
    scalar_logical_value_candidates: list[dict[str, Any]] = []
    for binding in physical_bindings:
        if binding["kind"] != "scalar":
            continue
        argument = next(
            item
            for item in raw_arguments
            if item["name"] == binding["physical_parameter"]
        )
        candidate: dict[str, Any] = {
            "physical_parameter": binding["physical_parameter"],
            "kind": "scalar",
            "dtype": argument.get("dtype"),
            "logical_ref": (
                binding["logical_ref"]
                if binding.get("logical_kind") == "dimension"
                else None
            ),
            "logical_policy": (
                "dimension_or_rank0_t_with_explicit_eq"
                if binding.get("logical_kind") == "dimension"
                else "rank0_t_or_representation_null"
            ),
        }
        if isinstance(binding.get("source_coordinate"), str):
            candidate["source_coordinate"] = binding["source_coordinate"]
        scalar_logical_value_candidates.append(candidate)
    for name in ordered_bound_constexpr_parameters:
        dimension = bound_constexpr_dimension_by_parameter.get(name)
        candidate: dict[str, Any] = {
            "physical_parameter": name,
            "kind": "constexpr",
            "dtype": bound_constexpr_value_types[name],
            "logical_ref": dimension,
            "logical_policy": (
                "dimension_or_rank0_t_with_explicit_eq"
                if dimension is not None
                else "rank0_t_or_constexpr_null"
            ),
        }
        if dimension is not None:
            candidate["source_coordinate"] = bound_constexpr_shape_coordinates[name]
        scalar_logical_value_candidates.append(candidate)

    fixed_input_order = [
        item["logical_tensor"]
        for item in logical_tensor_order
        if item["container"] in {"inputs", "inputs+outputs"}
        and item["logical_tensor"] is not None
    ]
    fixed_output_order = [
        item["physical_parameter"]
        for item in logical_tensor_order
        if item["container"] == "outputs"
    ]

    lock = {
        "source_local_namespace": "s* (Source lexical temporary only)",
        "j_ssa_namespace": "v* (J operation result only)",
        "physical_parameter_order": list(parameter_order),
        "physical_to_logical": physical_bindings,
        # This is a stable physical-role skeleton.  Output-only entries have
        # logical_tensor=null until _semantic_abi_tensor_layout applies the
        # candidate's optional rank-0 scalar decisions.
        "logical_tensor_order": logical_tensor_order,
        "scalar_logical_value_candidates": scalar_logical_value_candidates,
        "j_fixed_input_tensor_order": fixed_input_order,
        "j_fixed_output_tensor_order": fixed_output_order,
        "j_value_dimension_relation_contract": {
            "value_ref": "rank-0 t*",
            "dimension_ref": "x*",
            "relation": "eq",
        },
        "tensor_declaration_order": (
            "trusted tensor inputs/inouts, then source-use-dependent rank-0 t* "
            "scalar values in physical p* order, then output-only tensors; "
            "output-only t* handles are assigned only by the final layout"
        ),
        "shape_dimension_order": shape_dimension_bindings,
        "semantic_constexpr_parameters": ordered_bound_constexpr_parameters,
        "semantic_constexpr_value_types": bound_constexpr_value_types,
        "semantic_constexpr_shape_coordinates": bound_constexpr_shape_coordinates,
    }
    return {
        "lock": lock,
        "coordinate_to_dimension": coordinate_to_dimension,
        "dimension_to_coordinate": {
            item["j_dimension"]: item["source_coordinate"]
            for item in shape_dimension_bindings
        },
        "physical_to_logical": physical_bindings,
        "logical_tensor_order": logical_tensor_order,
        "parameter_order": list(parameter_order),
        "bound_constexpr_parameters": ordered_bound_constexpr_parameters,
        "bound_constexpr_value_types": bound_constexpr_value_types,
        "bound_constexpr_shape_coordinates": bound_constexpr_shape_coordinates,
        "bound_constexpr_dimension_by_parameter": (
            bound_constexpr_dimension_by_parameter
        ),
    }


def _anonymous_semantic_lift_view(
    view: SCRCCompilerView,
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Project only Source ABI/shape facts allowed to constrain J=A+B.

    Concrete tensor extents and scalar values are intentionally absent.  A
    symbolic ``h*`` coordinate or an explicitly static integer may constrain J
    only through ``shape_contract``; the current Harness valuation of that
    contract belongs to the Source-realization projection below.
    """

    arguments: list[dict[str, Any]] = []
    for item in view.arguments:
        argument: dict[str, Any] = {
            "name": mapping[item.name],
            "kind": item.kind,
            "dtype": item.dtype,
        }
        if item.kind == "tensor":
            argument["rank"] = len(item.shape)
            if item.shape_contract is not None:
                argument["shape_contract"] = list(item.shape_contract)
        elif item.kind == "scalar" and item.shape_coordinate is not None:
            argument["shape_coordinate"] = item.shape_coordinate
        if item.mode is not None:
            argument["mode"] = item.mode
        arguments.append(argument)

    # Keep the model-visible semantic projection strictly J-owned.  The full
    # Source/Harness declaration order is retained in the private realization
    # view below, where BindSource needs it to validate the physical ABI.  If a
    # C slot is left in this list, a Lift owner can mistake it for a J
    # signature parameter even though the prose contract says otherwise.
    semantic_parameter_order = [
        mapping[name]
        for name in view.parameter_order
        if re.fullmatch(r"p[0-9]+", mapping[name]) is not None
    ]
    semantic_constexpr_parameters = [
        mapping[name]
        for name in view.constexpr_parameters
        if re.fullmatch(r"p[0-9]+", mapping[name]) is not None
    ]
    semantic_projection = {
        "arguments": arguments,
        "aliases": [
            {
                "left": mapping[item.left],
                "right": mapping[item.right],
                "kind": item.kind.value,
            }
            for item in view.aliases
        ],
        "shape_coordinates": list(view.shape_coordinates),
        "bound_constexpr_parameters": [
            mapping[name] for name in view.bound_constexpr_values
        ],
        "bound_constexpr_value_types": {
            mapping[name]: value_type
            for name, value_type in view.bound_constexpr_value_types.items()
        },
        "bound_constexpr_shape_coordinates": {
            mapping[name]: coordinate
            for name, coordinate in view.bound_constexpr_shape_coordinates.items()
        },
        "parameter_order": semantic_parameter_order,
        "constexpr_parameters": semantic_constexpr_parameters,
    }
    return semantic_projection


def _anonymous_source_realization_view(
    view: SCRCCompilerView,
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Project exact Source/Harness bindings for C_source and Gate evidence."""

    return {
        "schedule_values": {
            mapping[name]: value for name, value in view.schedule_values.items()
        },
        "schedule_value_types": {
            mapping[name]: value_type
            for name, value_type in view.schedule_value_types.items()
        },
        "bound_constexpr_values": {
            mapping[name]: value
            for name, value in view.bound_constexpr_values.items()
        },
        "bound_constexpr_value_types": {
            mapping[name]: value_type
            for name, value_type in view.bound_constexpr_value_types.items()
        },
        "compiler_slots": [mapping[name] for name in view.compiler_slots],
        "compiler_roles": {
            mapping[name]: name for name in view.compiler_slots
        },
        "argument_bindings": [
            {
                "name": mapping[item.name],
                **(
                    {
                        "shape": list(item.shape),
                        **(
                            {"strides": list(item.strides)}
                            if item.strides is not None
                            else {}
                        ),
                    }
                    if item.kind == "tensor"
                    else {"value": item.value}
                    if item.kind == "scalar"
                    else {}
                ),
            }
            for item in view.arguments
        ],
        "shape_coordinate_values": dict(view.shape_coordinates),
        "grid": list(view.grid),
        "grid_contract": [
            _replace_names(item, mapping) for item in view.grid_contract
        ],
        "grid_shape_coordinates": list(view.grid_shape_coordinates),
        # Compiler/constexpr C slots are not callable parameters. Their
        # canonical order is the declaration order of schedule_values, which
        # is also the order consumed by BindSource and the source manifest.
        "c_slot_order": [mapping[name] for name in view.schedule_values],
        "parameter_order": [mapping[name] for name in view.parameter_order],
        "constexpr_parameters": [
            mapping[name] for name in view.constexpr_parameters
        ],
    }


def _anonymous_semantic_prompt_view(
    anonymous_view: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Expose no actual Source/Harness valuation to the Lift model.

    The model sees ABI/type/shape relations and the closed set of Source-C
    handles, but no schedule, runtime scalar, concrete shape/grid, or compiler
    option value.  The complete trusted view remains private to Decode and
    ``BindSource``.
    """

    if anonymous_view is None:
        return None
    semantic, realization = _partitioned_compiler_views(anonymous_view)
    schedule_values = realization.get("schedule_values", {})
    schedule_value_types = realization.get("schedule_value_types", {})
    compiler_slots = realization.get("compiler_slots", [])
    compiler_roles = realization.get("compiler_roles", {})
    parameter_order = realization.get("parameter_order", [])
    constexpr_parameters = realization.get("constexpr_parameters", [])
    c_slot_order = realization.get("c_slot_order", list(schedule_values))
    if not isinstance(schedule_values, Mapping):
        raise TypeError("Source realization schedule_values must be an object")
    if not isinstance(schedule_value_types, Mapping):
        raise TypeError("Source realization schedule value types must be an object")
    if not isinstance(compiler_slots, list):
        raise TypeError("Source realization compiler_slots must be an array")
    if not isinstance(compiler_roles, Mapping):
        raise TypeError("Source realization compiler roles must be an object")
    if not isinstance(parameter_order, list) or not isinstance(
        constexpr_parameters, list
    ) or not isinstance(c_slot_order, list):
        raise TypeError("Source realization parameter declarations must be arrays")
    if c_slot_order != list(schedule_values):
        raise ValueError(
            "Source realization C slot order must match schedule declaration order"
        )
    compiler_set = set(compiler_slots)
    if set(schedule_value_types) != set(schedule_values):
        raise ValueError("Source-C value-free types must cover every active slot")
    if set(compiler_roles) != compiler_set:
        raise ValueError("Source-C compiler roles must cover every compiler slot")
    slots: list[dict[str, Any]] = []
    for name in schedule_values:
        declaration: dict[str, Any] = {
            "slot": name,
            "kind": "compiler" if name in compiler_set else "constexpr",
            "value_type": schedule_value_types[name],
            "required": True,
        }
        if name in compiler_set:
            declaration["role"] = compiler_roles[name]
        slots.append(declaration)
    return {
        # The model receives the reversible compact projection only.  Decode
        # keeps the paired expanded projection private so BindSource and final
        # Source validation retain the exact legacy ABI representation.
        "semantic_lift_view": _compact_semantic_lift_view(semantic),
        # The slot list is already ordered and each row carries its own kind;
        # slot_order/constexpr_slots/compiler_slots would only repeat the same
        # declarations in three additional arrays.
        "layer_c_declarations": {
            "slots": slots,
        },
    }


def _anonymous_compiler_view(
    view: SCRCCompilerView | None, mapping: dict[str, str]
) -> dict[str, Any] | None:
    """Return partitioned Lift facts plus the host-private compatibility view.

    Flat fields remain available only to trusted Source binding and retained
    local checker contexts. They are not serialized into the model request.
    New Lift/Decode code treats the two named projections as the ownership
    boundary and never lets Source-realization values authorize J.
    """

    if view is None:
        return None
    next_c = 0
    used = set(mapping.values())
    for name in [*view.schedule_values, *view.compiler_slots]:
        if (
            name in mapping
            and re.fullmatch(r"c[0-9]+", mapping[name]) is not None
        ):
            continue
        while f"c{next_c}" in used:
            next_c += 1
        mapping[name] = f"c{next_c}"
        used.add(mapping[name])

    semantic_view = _anonymous_semantic_lift_view(view, mapping)
    realization_view = _anonymous_source_realization_view(view, mapping)
    compatibility_view = {
        "schedule_values": {
            mapping[name]: value for name, value in view.schedule_values.items()
        },
        "schedule_value_types": {
            mapping[name]: value_type
            for name, value_type in view.schedule_value_types.items()
        },
        "bound_constexpr_values": {
            mapping[name]: value
            for name, value in view.bound_constexpr_values.items()
        },
        "bound_constexpr_value_types": {
            mapping[name]: value_type
            for name, value_type in view.bound_constexpr_value_types.items()
        },
        "bound_constexpr_shape_coordinates": {
            mapping[name]: coordinate
            for name, coordinate in view.bound_constexpr_shape_coordinates.items()
        },
        "compiler_slots": [mapping[name] for name in view.compiler_slots],
        "compiler_roles": {
            mapping[name]: name for name in view.compiler_slots
        },
        "aliases": list(semantic_view["aliases"]),
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
    return {
        **compatibility_view,
        "semantic_lift_view": semantic_view,
        "source_realization_view": realization_view,
    }


def _legacy_semantic_lift_view(
    anonymous_view: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the semantic projection from a retained flat context."""

    arguments: list[dict[str, Any]] = []
    raw_arguments = anonymous_view.get("arguments", [])
    if isinstance(raw_arguments, list):
        for raw in raw_arguments:
            if not isinstance(raw, Mapping):
                continue
            argument = {
                key: raw[key]
                for key in ("name", "kind", "dtype", "mode", "shape_coordinate")
                if key in raw
            }
            if raw.get("kind") == "tensor":
                shape = raw.get("shape")
                if isinstance(shape, list):
                    argument["rank"] = len(shape)
                if "shape_contract" in raw:
                    argument["shape_contract"] = raw["shape_contract"]
            arguments.append(argument)
    shape_coordinates = anonymous_view.get("shape_coordinates", {})
    parameter_order = list(anonymous_view.get("parameter_order", []))
    constexpr_parameters = list(
        anonymous_view.get("constexpr_parameters", [])
    )
    bound_constexpr_values = anonymous_view.get("bound_constexpr_values", {})
    raw_bound_constexpr_value_types = anonymous_view.get(
        "bound_constexpr_value_types"
    )
    if isinstance(raw_bound_constexpr_value_types, Mapping):
        bound_constexpr_value_types: Mapping[str, Any] = (
            raw_bound_constexpr_value_types
        )
    elif isinstance(bound_constexpr_values, Mapping):
        bound_constexpr_value_types = {
            str(name): canonical_scalar_binding_dtype(value)
            for name, value in bound_constexpr_values.items()
            if isinstance(value, (bool, int, float, str))
        }
    else:
        bound_constexpr_value_types = {}
    return {
        "arguments": arguments,
        "aliases": anonymous_view.get("aliases", []),
        "shape_coordinates": (
            list(shape_coordinates)
            if isinstance(shape_coordinates, Mapping)
            else list(shape_coordinates)
            if isinstance(shape_coordinates, list)
            else []
        ),
        "bound_constexpr_parameters": list(bound_constexpr_values),
        "bound_constexpr_value_types": bound_constexpr_value_types,
        "bound_constexpr_shape_coordinates": anonymous_view.get(
            "bound_constexpr_shape_coordinates", {}
        ),
        "parameter_order": [
            name
            for name in parameter_order
            if isinstance(name, str)
            and re.fullmatch(r"p[0-9]+", name) is not None
        ],
        "constexpr_parameters": [
            name
            for name in constexpr_parameters
            if isinstance(name, str)
            and re.fullmatch(r"p[0-9]+", name) is not None
        ],
    }


def _legacy_source_realization_view(
    anonymous_view: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the Source-realization projection from a retained flat context."""

    raw_arguments = anonymous_view.get("arguments", [])
    argument_bindings: list[dict[str, Any]] = []
    if isinstance(raw_arguments, list):
        for raw in raw_arguments:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
                continue
            binding: dict[str, Any] = {"name": raw["name"]}
            if raw.get("kind") == "tensor" and "shape" in raw:
                binding["shape"] = raw["shape"]
                if "strides" in raw:
                    binding["strides"] = raw["strides"]
            elif raw.get("kind") == "scalar" and "value" in raw:
                binding["value"] = raw["value"]
            argument_bindings.append(binding)
    schedule_values = anonymous_view.get("schedule_values", {})
    raw_value_types = anonymous_view.get("schedule_value_types")
    if isinstance(raw_value_types, Mapping):
        schedule_value_types: Mapping[str, Any] = raw_value_types
    elif isinstance(schedule_values, Mapping):
        schedule_value_types = {
            str(name): canonical_scalar_binding_dtype(value)
            for name, value in schedule_values.items()
            if isinstance(value, (bool, int, float, str))
        }
    else:
        schedule_value_types = {}
    bound_constexpr_values = anonymous_view.get("bound_constexpr_values", {})
    raw_bound_constexpr_value_types = anonymous_view.get(
        "bound_constexpr_value_types"
    )
    if isinstance(raw_bound_constexpr_value_types, Mapping):
        bound_constexpr_value_types: Mapping[str, Any] = (
            raw_bound_constexpr_value_types
        )
    elif isinstance(bound_constexpr_values, Mapping):
        bound_constexpr_value_types = {
            str(name): canonical_scalar_binding_dtype(value)
            for name, value in bound_constexpr_values.items()
            if isinstance(value, (bool, int, float, str))
        }
    else:
        bound_constexpr_value_types = {}
    return {
        "schedule_values": schedule_values,
        "schedule_value_types": schedule_value_types,
        "bound_constexpr_values": bound_constexpr_values,
        "bound_constexpr_value_types": bound_constexpr_value_types,
        "compiler_slots": anonymous_view.get("compiler_slots", []),
        "compiler_roles": anonymous_view.get("compiler_roles", {}),
        "argument_bindings": argument_bindings,
        "shape_coordinate_values": anonymous_view.get("shape_coordinates", {}),
        "grid": anonymous_view.get("grid", []),
        "grid_contract": anonymous_view.get("grid_contract", []),
        "grid_shape_coordinates": anonymous_view.get(
            "grid_shape_coordinates", []
        ),
        "c_slot_order": (
            list(schedule_values)
            if isinstance(schedule_values, Mapping)
            else []
        ),
        "parameter_order": anonymous_view.get("parameter_order", []),
        "constexpr_parameters": anonymous_view.get("constexpr_parameters", []),
    }


def _partitioned_compiler_views(
    anonymous_view: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the two owned projections and reject compatibility-copy drift."""

    nested_semantic = anonymous_view.get("semantic_lift_view")
    nested_realization = anonymous_view.get("source_realization_view")
    has_semantic = isinstance(nested_semantic, Mapping)
    has_realization = isinstance(nested_realization, Mapping)
    if has_semantic != has_realization:
        raise ValueError(
            "partitioned compiler view requires both semantic_lift_view and "
            "source_realization_view"
        )
    legacy_semantic = _legacy_semantic_lift_view(anonymous_view)
    legacy_realization = _legacy_source_realization_view(anonymous_view)
    if not has_semantic:
        return legacy_semantic, legacy_realization
    assert isinstance(nested_semantic, Mapping)
    assert isinstance(nested_realization, Mapping)
    if dict(nested_semantic) != dict(legacy_semantic):
        raise ValueError(
            "semantic_lift_view differs from its retained flat compatibility copy"
        )
    if dict(nested_realization) != dict(legacy_realization):
        raise ValueError(
            "source_realization_view differs from its retained flat "
            "compatibility copy"
        )
    return nested_semantic, nested_realization


def _semantic_abi_view(
    compiler_view: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select the semantic projection without requiring private realization.

    Final Decode receives the paired semantic/source-realization view and must
    verify their retained compatibility copy.  CandidateCheck intentionally
    receives only the value-free semantic projection, however: requiring its
    unavailable private sibling would make a valid fresh candidate fail after
    it passed the same public response schema.  Both paths still feed the
    exact selected semantic mapping into the shared ABI contract.
    """

    nested_semantic = compiler_view.get("semantic_lift_view")
    nested_realization = compiler_view.get("source_realization_view")
    if nested_semantic is not None:
        if not isinstance(nested_semantic, Mapping):
            raise TypeError("semantic_lift_view must be an object")
        if nested_realization is None:
            return _expand_semantic_lift_view(nested_semantic)
        if not isinstance(nested_realization, Mapping):
            raise TypeError("source_realization_view must be an object")
        semantic, _realization = _partitioned_compiler_views(compiler_view)
        return _expand_semantic_lift_view(semantic)
    if nested_realization is not None:
        raise ValueError(
            "source_realization_view requires a paired semantic_lift_view"
        )
    semantic, _realization = _partitioned_compiler_views(compiler_view)
    return _expand_semantic_lift_view(semantic)


def _consistency_only_lift_instructions() -> str:
    return (
        """You are the sole source-aware Lift owner for one selected anonymous Source kernel.
Return exactly one complete compact JSON object matching the supplied schema.
The root keys are exactly {schema,j,c,pi}; emit no prose, markdown, patch,
second kernel, or convenience field. The supplied schema is authoritative for
all field names, enums, required fields, operation contracts, and relations.
The schema-derived output structure map appended below is input-only mechanical
guidance: use it to keep the envelope hierarchy complete, but never copy the
map itself into the candidate.

FAST SOURCE-TO-J CONSISTENCY PASS
Make one bounded direct pass over the selected Source and the mounted semantic
view. Author the smallest complete J graph that preserves every reachable
Source operation, order, control/effect dependency, access, output, and return.
Do not explore target implementations, alternate graphs, Replay, Gate1, or
program-axis normalization. Submit as soon as the complete graph and declared
C slots are accounted for. This shortcut changes only the authoring prompt;
CandidateCheck, BindSource, TypeCheck, and the public Source-vs-J probe remain
the existing fixed consumers.

ENVELOPE SHAPE
Keep the hierarchy literal: root.schema, root.j, root.c, and root.pi are four
sibling keys. root.c and root.pi must never be nested under root.j or
root.j.intent. root.j contains exactly the required containers kernel_id,
intent, node_ids, boundary, signature, scopes, storage, dimension_constraints,
value_dimension_relations, and dimension_relations; root.j.intent contains
intent_function, whose required children are name, inputs, outputs, returns,
and body. root.c contains slots and values. root.pi contains schema and the
required empty program_axes. Fill each required container even when its array
or object is empty.

SOURCE FIDELITY AND OWNERSHIP
J owns all Source-visible semantics: logical tensors, dtypes, shapes and
dimensions, constants, casts and promotion, operation order, loops/recurrences,
reductions/scans/contractions, branches, early/terminal control, access
indices, predicates/fills, reads/writes/effect order, inout pre/post state,
outputs, returns, and boundary relations. Preserve explicit sum/divide,
casts, masks, fills, recurrence, and access multiplicity; do not substitute a
merely value-equivalent shortcut. J/C/pi must not be derived from launch/grid,
stride/layout, block/thread/warp mapping, fusion, pipeline, target compiler,
TTIR, hardware capability, or target source. C_source has only the declared
Source/Harness realization slots. pi is only a directly Source-grounded
normalization witness over existing J/C anchors.

ABI, HANDLES, AND SSA
Use dense anonymous handles only: scrc_k0, t*, n*, v*, x*, d*, a*, e*, q*,
b*, p*, c*. J.signature.parameters contains exactly the p* physical Source
ABI parameters in order. c* exists only in C.slots and C attachments: never
put c* in J.signature, logical_ref, operation operands/results, boundary, or
pi. Logical refs are declared t*/x* only. Declare logical inputs before
output-only tensors; an inout repeats one tensor handle. An output-only tensor
must never be a body operand. Every v* is single-assignment with one dtype and
shape, defined before use in its scope; use a fresh result for each operation.

NAMESPACE LOCK
The anonymous Source lexical temporary namespace is s* only; J operation SSA
results are v* only; J logical tensor/rank-0 value handles are t* only; J
symbolic dimensions are x* only; physical Source ABI parameters are p* only;
portable scopes are q* only; J realization authorizations are u* only; and
Source-side shape coordinates are h* only. d*/a*/e*/b*/n* keep their stated J
domain/access/effect/storage/node roles. Never copy an s* Source temporary into
intent_function.inputs, outputs, operands, results, or logical_ref. Never copy
p* into a logical_ref, h* into a J shape, q* into a value, u* into a C slot, or
c* into J. A spelling collision between these namespaces is a contract error,
not a reason to invent or rename a semantic handle.

SHAPES, CONTROL, AND BOUNDARY
Reuse x* only for the same Source-grounded coordinate; keep static extents
literal and map symbolic h* coordinates using the semantic view, not hidden
concrete values. Preserve all established dimension and value-dimension
relations, or use [] when none exist. A value_dimension_relations entry is
always {value_ref:t*, dimension_ref:x*, relation:eq}: it never uses a physical
parameter p*, a Source coordinate h*, or a v* result. A shape-coordinate
parameter may either bind directly to its x* dimension or bind to a rank-0 t*
with that explicit equality; fixed code never invents the rank-0 value or the
relation. Declare complete function outputs and returns in order. Keep scopes
faithful to Source control. Every boundary access has one matching effect with
identical kind/target/node/access; retain guard predicates and fills. An inout
has matching pre-load/post-store, read/write effects, and output binding. Use
updates only for a real along-axis scatter. Do not encode Source semantics as
a target schedule.

WIRE SHAPE ANCHORS
boundary.domains is a flat map from d* to one ContractExpr string, for example
{"d0":"(0 <= k0) and (k0 < x0)"}; never use {"shape":[...]} or a tuple/list
as a domain value. The function scope q0 definitions must include every logical
input/output t* and every top-level v* definition exactly once; branch/loop
scopes list their own definitions and explicit captures. A cast uses exactly
the canonical attrs key to_dtype. For a shape-coordinate parameter, follow the
mounted semantic view: do not choose x* merely because the physical parameter
looks like an extent; if the view says rank-0 logical data, use a t* logical_ref
and its explicit value-dimension equality.

ABI CONSTRUCTION CHECKLIST
Use the model-visible semantic_lift_view as the mechanical ABI fact. An
arguments row with names:[p*,...] expands in that written order; tensor rank is
the shape_contract length when omitted; constexpr_bindings is the complete
bound constexpr declaration; omitted shape_coordinates and parameter_order use
only their stated deterministic first-occurrence/default order. Fixed tensor
inputs/inouts are declared first. Optional
rank-0 scalar/constexpr values, if Source use proves that they are logical J
data, are inserted as additional input t* handles after all fixed tensor inputs
and in physical p* order; output-only tensors then receive the next dense t*
handles in that stable order. If no such logical scalar use is proved, keep the
carrier's logical_ref null. Never use an output-only physical parameter as a
body operand, never use p* as an input/output tensor name,
    and never let an optional scalar renumber an existing fixed input. The
    signature still contains every physical p* in physical_parameter_order
    regardless of whether its logical_ref is null.

STRUCTURED CONTROL COMPLETENESS
When Source has a loop, use the closed loop shape exactly: a non-empty result
list, non-empty carry list, trip, scan list, non-empty body, and yield list.
Each carry item must contain state_id, init_value, body_argument, next_value,
result_value, and type; result must equal the ordered carry result_value list,
yield must equal the ordered carry next_value list, and the body must define
each next_value in the same scope. A static trip is {axis:x*}; a dynamic trip
is {upper_bound:t*, index:...} and must use the local induction value. For
effect_if/value_if, keep both explicit arms and their scope_id/body; a
value_if additionally needs a yield_value in each arm. Do not emit a partial
loop or replace a loop carry with an output tensor.
Use the exact operation names effect_if and value_if; there is no generic if
operation. effect_if has only predicate, positive, and negative in addition to
op/id, and has no operands, attrs, type, result, or yield. value_if has only
predicate, positive, negative, result, and type in addition to op/id, and its
arms each carry yield_value. In j.node_ids, each n* maps to its structural
body path such as body[0] or body[1].positive.body[0], never to a v* result.

C_source AND pi
Emit exactly the active C slots in Harness declaration order with identical
slot/kind/value_type/required/role metadata. Copy trusted compiler roles and
leave c.values exactly {}. For kind=compiler, copy only the trusted role and
omit subject_ref and axis_ref entirely; a compiler slot is not attached to a J
anchor. For kind=constexpr, add subject_ref/axis_ref only when the semantic view
directly establishes that exact source-neutral attachment and the declaration
itself exposes that attachment; otherwise omit both fields. Never use p*,
Source names, v*, physical axes, or explanations in a C slot. The complete pi object
is still required, but keep program_axes and every other pi family empty unless
the anonymous Source semantic view directly establishes an exact relation to
an existing J/C anchor. Never invent a Gate1 row, SSA relation, route,
iteration, target recipe, tolerance, or repair permission.

RELIFT
If a previous candidate is supplied, edit that exact value-free draft and
retain all correct Source semantics. Feedback is diagnostic evidence only.
Never let fixed code fill, reorder, infer, select, or repair J/C/pi. Finish
with one complete compact JSON object and c.values={}.
"""
        + _lift_output_structure_contract()
    )


def build_scrc_lift_instructions(
    *, authoring_profile: str | None = None
) -> str:
    """Return the compact Source-to-J authoring contract.

    The JSON schema and focused mounted references carry field-level details.
    This prompt therefore states only the ownership and semantic invariants
    that are easy to lose when a source-aware repair is repeated. Keeping this
    contract short leaves relift context for Source and the current draft.
    """

    if authoring_profile == CONSISTENCY_ONLY_AUTHORING_PROFILE:
        return _consistency_only_lift_instructions()
    if authoring_profile is not None:
        raise ValueError(f"unsupported Lift authoring profile: {authoring_profile!r}")

    instructions = """You are the sole source-aware Lift owner for one selected anonymous Source kernel.
Return exactly one complete compact JSON object matching the supplied schema.
Do not pretty-print or add explanatory whitespace; the root keys are exactly
{schema,j,c,pi}; never output prose, a patch, a second kernel,
or convenience fields. The schema is authoritative for all field names, enums,
operation contracts, and relation shapes. The schema-derived output structure
map appended below is input-only mechanical guidance: use it to keep the
envelope hierarchy complete, but never copy the map itself into the candidate.

DELIVERY DISCIPLINE
Use one direct bounded pass over the selected Source and emit the complete
artifact as soon as its reachable semantics and ABI are accounted for. Do not
explore alternative implementations, write a plan, or re-derive unchanged
contract text; fixed consumers will reject any missing or inconsistent fact.
For a straight-line Source, construct once in this order: logical ABI and
returns, body operations/results, boundary domains/accesses/effects, exact J
scopes, then the declared C slots and only directly grounded pi rows. Keep
every optional array empty unless the Source and an existing J/C anchor require
that relation; an empty array is the canonical answer for an inapplicable
optional feature.

OWNERSHIP
J is the only owner of Source logic and observable semantics: operations and
their order, operands/results, constants, logical shapes and dimensions,
casts and promotion, loops/recurrences, effect_if/value_if, early terminal
control, reductions/scans/contractions, access indices and predicates/fills,
reads/writes/effect order, inout pre/post state, outputs, returns, and logical
boundary relations. C_source contains only the declared Source/Harness
implementation-selection slots and their source-neutral attachments. pi is
only a closed Source-grounded normalization witness over anchors already
authored in J/C. C_source and pi are not CUDA/RISC-V/CPU target lowering.

SOURCE FIDELITY
Read the anonymous Source as the semantic authority. Preserve every reachable
selected-kernel stage and control/effect dependency. Do not replace
sum-plus-divide with mean, a loop recurrence with an equivalent global
contraction, an explicit cast sequence with a convenient inferred dtype, or a
predicate/fill/effect with an observationally similar shortcut. Preserve
Source access multiplicity and consumer-local views. A later target compiler,
TTIR, launch/grid, stride, layout, vector width, block/thread/warp mapping,
fusion, pipeline, or hardware capability must never create or alter J, C_source,
or pi. Supporting helpers contribute their literal reachable memory/effect
order; sibling kernels and variants are out of scope.

ABI AND HANDLES
Use only dense declaration-order handles: scrc_k0, t*, n*, v*, x*, d*, a*,
e*, q*, b*, p*, c*. Declare all logical inputs before output-only tensors;
inout repeats one tensor handle in inputs and outputs. p* is the physical ABI,
not a logical reference. J.signature.parameters contains only the p* physical
parameters in Source ABI order; c* compiler slots belong only to C.slots and
source-neutral C attachments and must never appear in J.signature, a J
logical_ref, an operation operand/result, a boundary, or pi. A J logical_ref
is only a declared t* or x*. Keep parameter order, dtype, rank, mode, alias,
and shape contracts exact. Runtime or data-bearing scalar parameters are
rank-0 t* values; extent-only parameters are x*; pure compile-time selectors
may be null only when Source proves they have no data, shape, index, mask,
arithmetic, predicate, or result use. Do not copy Source names into the
artifact and do not use r*/i* as authorization ids. The semantic ABI view keeps
the physical p* sequence separate from the C-slot declarations and the
input/output roles. A physical parameter marked output is output-only: its
logical t* may be returned or bound by an output mapping, but it must never be
an operand of a body operation. Body operands may use logical input or inout t*
handles and values produced earlier in the same J scope; they may not use a
later output-only t* or a result defined in another region. Do not infer a new
input merely because an output buffer is present. If Source proves a scalar or
bound constexpr is logical rank-0 data, place its new t* after all fixed tensor
inputs and before output-only tensors, in physical p* order; then assign the
output-only tensors the next dense t* handles. Keep optional representation-only
carriers null and never use a physical p* as a logical tensor handle.
The mounted semantic_lift_view is a reversible compact ABI declaration: an
arguments row may group ordered names:[p*,...]; a tensor with shape_contract
omits rank only when that length is its rank; constexpr_bindings replaces the
three bound-constexpr lists; omitted shape_coordinates and parameter_order use
their declared deterministic first-occurrence/default reconstruction. Do not
invent a missing ABI fact or reorder a grouped name sequence.

NAMESPACE LOCK
The anonymous Source lexical temporary namespace is s* only; J operation SSA
results are v* only; J logical tensor/rank-0 value handles are t* only; J
symbolic dimensions are x* only; physical Source ABI parameters are p* only;
portable scopes are q* only; J realization authorizations are u* only; and
Source-side shape coordinates are h* only. d*/a*/e*/b*/n* keep their stated J
domain/access/effect/storage/node roles. Never copy an s* Source temporary into
intent_function.inputs, outputs, operands, results, or logical_ref. Never copy
p* into a logical_ref, h* into a J shape, q* into a value, u* into a C slot, or
c* into J. A spelling collision between these namespaces is a contract error,
not a reason to invent or rename a semantic handle.
SSA TYPE DISCIPLINE
Every v* is one single-assignment value with exactly one dtype and shape. Give
each operation a fresh result handle (or the explicitly required result list),
never redefine or reuse a v* for another scalar/vector value, and use a result
only after it is defined in the same scope or explicitly yielded/captured by
that region. A broadcasted map result keeps its authored broadcast shape; a
reduction result has its reduced shape. Do not rely on a later use to change a
previous v* from rank 0 to rank 1 or vice versa. Before submission, check that
every v* has one consistent type across its definition and all uses.

STRUCTURED CONTROL COMPLETENESS
When Source has a loop, use the closed loop shape exactly: a non-empty result
list, non-empty carry list, trip, scan list, non-empty body, and yield list.
Each carry item must contain state_id, init_value, body_argument, next_value,
result_value, and type; result must equal the ordered carry result_value list,
yield must equal the ordered carry next_value list, and the body must define
each next_value in the same scope. A static trip is {axis:x*}; a dynamic trip
is {upper_bound:t*, index:...} and must use the local induction value. For
effect_if/value_if, keep both explicit arms and their scope_id/body; a
value_if additionally needs a yield_value in each arm. Do not emit a partial
loop or replace a loop carry with an output tensor.
Use the exact operation names effect_if and value_if; there is no generic if
operation. effect_if has only predicate, positive, and negative in addition to
op/id, and has no operands, attrs, type, result, or yield. value_if has only
predicate, positive, negative, result, and type in addition to op/id, and its
arms each carry yield_value. In j.node_ids, each n* maps to its structural
body path such as body[0] or body[1].positive.body[0], never to a v* result.

SHAPES AND DIMENSIONS
Assign x* by first appearance in the complete logical tensor ABI and reuse the
same x* only for the same Source-grounded coordinate. Static literals remain
static; symbolic h* coordinates become x* according to the value-free semantic
view, never according to hidden concrete values or coincident extents. Author
all Source-established dimension_relations and value_dimension_relations, or
[] when none exist. Every value_dimension_relations entry is exactly
{value_ref:t*, dimension_ref:x*, relation:eq}; a physical p*, Source h*, or
SSA v* is never a legal value_ref. A shape-coordinate parameter binds directly
to x* or to a rank-0 t* with that explicit equality. Legal dimension expressions
are only the supplied closed
identity/add_const/sub_const/mul_const/floordiv_const/mod_const/
ceil_div_const/round_up_const forms. Fixed code validates these facts; it never
invents a dimension, relation, shape, or graph.

WIRE SHAPE ANCHORS
boundary.domains is a flat map from d* to one ContractExpr string, for example
{"d0":"(0 <= k0) and (k0 < x0)"}; never use {"shape":[...]} or a tuple/list
as a domain value. The function scope q0 definitions must include every logical
input/output t* and every top-level v* definition exactly once; branch/loop
scopes list their own definitions and explicit captures. A cast uses exactly
the canonical attrs key to_dtype. For a shape-coordinate parameter, follow the
mounted semantic view: do not choose x* merely because the physical parameter
looks like an extent; if the view says rank-0 logical data, use a t* logical_ref
and its explicit value-dimension equality.

J CONTROL, BOUNDARY, AND EFFECTS
Declare the complete intent_function outputs and complete returns list in the
same order; output_mappings are additional sparse boundary anchors, never a
replacement for returns. Every boundary access has exactly one matching
effect with identical kind/target/node_ref/access_ref. Accesses are attached
to their consuming node and are not silently propagated to siblings. A guarded
load has its authored predicate/fill; a fill is never replaced by a downstream
neutral value. An inout tensor requires a matching pre-load and post-store,
read/write effects, and output binding. Use boundary.updates only for an actual
along_axis scatter and match its node and index-bounds exactly. Preserve
effect_if arms, loop carriers, early returns, and terminal control as authored
J scopes; do not encode them as target schedules.

KNOWN OPERATIONS
Use only the closed operation grammar and exact canonical attrs. In particular,
map_binary uses fn, contraction uses subscripts, map_unary uses a closed
function spelling (logical_not for boolean complement), and no known op gets
extra dtype, accumulation, source, schedule, or backend metadata. Explicit
broadcast_in_dim carries its own broadcast_dimensions. boundary.broadcasts is
only an implicit operand broadcast of map_binary/select, naming that exact
node and operand; it is not a generic shape annotation.

C_source AND pi
Emit exactly the declared active C slots in Harness declaration order, with the
same slot/kind/value_type/required/role metadata. Copy trusted compiler roles
exactly; for a Lift-authored constexpr role use one concise lower_snake_case
source-derived scheduling label or constexpr. Leave c.values exactly {}. A
compiler slot has no subject_ref/axis_ref and is not a J value or pi row. A
source-neutral attachment may be emitted only when the declaration exposes that
exact attachment, and may reference only an existing t*/n*/d*/a*/e*/q*/u*/b*
anchor and the exact logical axis; never a p*, Source name, v*, physical axis,
or free-form explanation. u* names J realization authorizations only.

pi may contain only the supplied program, iteration, terminal/materialization,
and access-representation relation families, bound to existing J/C anchors.
program_axes is only the simple direct domain-to-launch-axis form. A
program_bindings row is the complete complex form and must carry that same
observed physical axis together with direct/product_linearized/guarded_union
logical_axes; never split the relation from its physical axis or emit both
families. Keep required program_axes empty when neither mapping is directly
established. pi cannot contain SSA ids, regions/instances/routes/order, numeric
permission, target recipes, free text, tolerances, or exceptions. Its r*/i*
namespaces are local to pi; they do not change J u* authorization names. If a
relation is not directly established by anonymous Source semantics and existing
anchors, omit it.

REPAIR DISCIPLINE
On relift, if previous_candidate is supplied, patch that exact draft locally
and retain all correct Source semantics; do not regenerate a new artifact from
the diagnostic. If previous_artifact is supplied, use it as the prior complete
J/C/pi draft. Feedback is diagnostic evidence only, never a patch or lowering
recipe. Candidate c.values must remain {}. CandidateCheck/Decode/TypeCheck and
formal Replay are fixed consumers: they validate and reject, but never fill,
reorder, infer, select, or repair J/C/pi. Finish with one complete JSON object.
    """
    return instructions + _lift_output_structure_contract()


def _lift_output_structure() -> dict[str, Any]:
    """Project the authoritative schema into a small input-only shape map.

    The Responses relay may treat ``text.format`` as advisory.  This projection
    keeps the structural contract visible in the ordinary user packet without
    copying the full schema or constructing any candidate field.  It is derived
    on every request so the prompt, wire schema, and local CandidateCheck cannot
    silently drift at the envelope boundary.
    """

    root = scrc_lift_core_json_schema()
    root_properties = root.get("properties")
    if not isinstance(root_properties, Mapping):
        raise TypeError("Lift schema root properties must be a mapping")

    def object_shape(value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or value.get("type") != "object":
            raise TypeError(f"Lift schema {path} must be an object")
        properties = value.get("properties")
        required = value.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise TypeError(f"Lift schema {path} lacks object shape metadata")
        required_names = set(required)
        optional = [name for name in properties if name not in required_names]
        shape = {
            "required": list(required),
        }
        if optional:
            shape["optional"] = optional
        return shape

    j = root_properties.get("j")
    if not isinstance(j, Mapping):
        raise TypeError("Lift schema root.j must be an object")
    j_properties = j.get("properties")
    if not isinstance(j_properties, Mapping):
        raise TypeError("Lift schema root.j properties must be a mapping")
    intent = j_properties.get("intent")
    if not isinstance(intent, Mapping):
        raise TypeError("Lift schema root.j.intent must be an object")
    intent_properties = intent.get("properties")
    if not isinstance(intent_properties, Mapping):
        raise TypeError("Lift schema root.j.intent properties must be a mapping")
    function = intent_properties.get("intent_function")
    if not isinstance(function, Mapping):
        raise TypeError(
            "Lift schema root.j.intent.intent_function must be an object"
        )
    function_properties = function.get("properties")
    if not isinstance(function_properties, Mapping):
        raise TypeError(
            "Lift schema intent_function properties must be a mapping"
        )

    return {
        "schema_values": {
            "root.schema": root_properties["schema"].get("enum", []),
            "pi.schema": root_properties["pi"]["properties"]["schema"].get(
                "const"
            ),
        },
        "root": object_shape(root, "root"),
        "j": object_shape(j, "root.j"),
        "j.intent": object_shape(intent, "root.j.intent"),
        "j.intent.intent_function": object_shape(
            function, "root.j.intent.intent_function"
        ),
        "j.boundary": object_shape(
            j_properties["boundary"], "root.j.boundary"
        ),
        "j.signature": object_shape(
            j_properties["signature"], "root.j.signature"
        ),
        "j.scope_item": object_shape(
            j_properties["scopes"]["items"], "root.j.scopes[]"
        ),
        "j.tensor_item": object_shape(
            function_properties["inputs"]["items"],
            "root.j.intent.intent_function.inputs[]",
        ),
        "c": object_shape(root_properties["c"], "root.c"),
        "pi": object_shape(root_properties["pi"], "root.pi"),
    }


def _lift_output_structure_contract() -> str:
    """Render the schema-derived shape map for every Lift authoring backend."""

    return (
        "\n\nOUTPUT STRUCTURE MAP (input-only; never emit this map):\n"
        + json.dumps(
            _lift_output_structure(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _build_user_input(
    anonymous_source: str,
    compiler_view: dict[str, Any] | None,
    relift_context: SCRCReliftContext | None,
    source_kind: str,
    authoring_profile: str | None,
) -> str:
    payload: dict[str, Any] = {
        "anonymous_source": anonymous_source,
        "source_kind": source_kind,
    }
    if authoring_profile is not None:
        payload["authoring_profile"] = authoring_profile
    if compiler_view is not None:
        # This key is model-visible.  It contains only semantic ABI facts and
        # value-free Layer-C handle declarations.  The private complete view is
        # retained separately on SCRCLiftRequest for Source observation/binding.
        payload["harness_compiler_view"] = compiler_view
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
    source_symbol: str | None = None,
    *,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    authoring_profile: str | None = None,
) -> tuple[
    str,
    str,
    SCRCHarnessBinding,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
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
    anonymous_view = _anonymous_compiler_view(compiler_view, mapping)
    semantic_view = _anonymous_semantic_prompt_view(anonymous_view)
    instructions = build_scrc_lift_instructions(
        authoring_profile=authoring_profile
    )
    user = _build_user_input(
        anonymous_source,
        semantic_view,
        relift_context,
        source_kind,
        authoring_profile,
    )
    return instructions, user, binding, anonymous_view, semantic_view


def prepare_scrc_lift(
    source: str,
    *,
    compiler_view: SCRCCompilerView | None = None,
    relift_context: SCRCReliftContext | None = None,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    authoring_profile: str | None = None,
) -> SCRCLiftRequest:
    instructions, user, binding, anonymous_view, semantic_view = _prompt_material(
        source,
        compiler_view,
        relift_context,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        authoring_profile=authoring_profile,
    )
    return SCRCLiftRequest(
        instructions,
        user,
        hashlib.sha256((instructions + "\0" + user).encode()).hexdigest(),
        binding,
        anonymous_view,
        semantic_view,
    )


def scrc_lift_prompt_sha256(
    source: str,
    name_hint: str | None = None,
    *,
    compiler_view: SCRCCompilerView | None = None,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    authoring_profile: str | None = None,
) -> str:
    del name_hint
    instructions, user, _binding, _view, _semantic_view = _prompt_material(
        source,
        compiler_view,
        None,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        authoring_profile=authoring_profile,
    )
    return hashlib.sha256((instructions + "\0" + user).encode()).hexdigest()


def scrc_relift_prompt_sha256(
    source: str,
    context: SCRCReliftContext,
    *,
    compiler_view: SCRCCompilerView | None = None,
    source_symbol: str | None = None,
    source_kind: str = "python",
    source_parameter_order: tuple[str, ...] = (),
    source_to_opaque: Mapping[str, str] | None = None,
    authoring_profile: str | None = None,
) -> str:
    instructions, user, _binding, _view, _semantic_view = _prompt_material(
        source,
        compiler_view,
        context,
        source_symbol,
        source_kind=source_kind,
        source_parameter_order=source_parameter_order,
        source_to_opaque=source_to_opaque,
        authoring_profile=authoring_profile,
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


def _candidate_draft_for_relift(response_text: str) -> dict[str, Any] | None:
    """Extract a bounded value-free rejected draft for the next Lift round.

    Parsing failure means there is no safe continuation object.  In that case
    the next owner round receives only the diagnostic and Source context.  A
    parsed object is never decoded, completed, type-checked, or treated as a
    formal artifact here.
    """

    try:
        return _value_free_lift_candidate_payload(_json_object(response_text))
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def decode_scrc_lift_artifact(
    response_text: str,
    request: SCRCLiftRequest,
) -> Gate1Artifact:
    """Apply the one canonical J/C/pi decoder to any Lift backend output."""

    expected_parameters = tuple(
        opaque
        for _source, opaque in request.harness_binding.source_to_opaque
        if re.fullmatch(r"p[0-9]+", opaque)
    )
    artifact = _decode_scrc_lift_artifact_core(
        response_text,
        expected_parameters=expected_parameters,
        anonymous_compiler_view=request.anonymous_compiler_view,
        required_schema=Gate1Artifact.SCHEMA,
    )
    _validate_source_identifier_privacy(artifact, request.harness_binding)
    return artifact


def _decode_scrc_lift_artifact_core(
    response_text: str,
    *,
    expected_parameters: tuple[str, ...],
    anonymous_compiler_view: Mapping[str, Any] | None,
    required_schema: str,
) -> Gate1Artifact:
    """Decode, BindSource, and validate the persisted Gate artifact."""

    payload = _json_object(response_text)
    _require_compact_lift_pi(payload)
    _require_fresh_lift_j_surface(payload)
    bound_payload = _bind_source_artifact_payload(
        payload, anonymous_compiler_view
    )
    artifact = canonicalize_gate1_artifact(
        Gate1Artifact.from_json(bound_payload)
    )
    if artifact.wire_schema != required_schema:
        raise ValueError(
            f"Lift artifact uses {artifact.wire_schema!r}; "
            f"this round requires {required_schema!r}"
        )
    _validate_anonymous_lift_core(
        artifact,
        expected_parameters,
        semantic_compiler_view=anonymous_compiler_view,
    )
    _validate_compiler_view(artifact, anonymous_compiler_view)
    # CandidateCheck has already checked the same J/C declarations/pi with
    # value-free type sentinels.  Re-run TypeCheck after BindSource so a future
    # value-dependent C constraint cannot escape this private boundary as a
    # relift diagnostic.  Deliberately omit individual diagnostics here: any
    # delta from CandidateCheck is evidence about the trusted realization.
    bound_report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.wire_schema == artifact.SCHEMA,
    )
    if not bound_report.passed:
        raise ValueError(
            "bound Source realization is incompatible with the accepted "
            "value-free Lift contract"
        )
    return artifact


def _value_free_layer_c_declarations(
    declarations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize the trusted value-free Layer-C declaration sequence.

    The declaration is the only source of truth for the active C interface at
    CandidateCheck time.  In particular, an omitted attachment field means
    that the candidate must omit that field too; it is not an invitation to
    project or silently discard a candidate attachment.
    """

    raw_declarations = declarations.get("slots", [])
    if not isinstance(raw_declarations, list):
        raise TypeError("Layer-C declarations must be an array")
    expected_slots: list[dict[str, Any]] = []
    allowed = {
        "slot",
        "kind",
        "value_type",
        "required",
        "role",
        "subject_ref",
        "axis_ref",
    }
    required = {"slot", "kind", "value_type", "required"}
    seen: set[str] = set()
    for index, declaration in enumerate(raw_declarations):
        if not isinstance(declaration, Mapping):
            raise TypeError(f"Layer-C declaration {index} must be an object")
        unknown = set(declaration) - allowed
        if unknown:
            raise TypeError(
                f"Layer-C declaration {index} has unsupported field(s): "
                + ", ".join(sorted(unknown))
            )
        missing = required - set(declaration)
        if missing:
            raise TypeError(
                f"Layer-C declaration {index} is missing field(s): "
                + ", ".join(sorted(missing))
            )
        slot = declaration["slot"]
        if not isinstance(slot, str) or not slot:
            raise TypeError(f"Layer-C declaration {index}.slot must be a name")
        if slot in seen:
            raise ValueError(f"Layer-C declaration slot {slot!r} is duplicated")
        seen.add(slot)
        kind = declaration["kind"]
        if kind not in {"constexpr", "compiler"}:
            raise ValueError(
                f"Layer-C declaration {index}.kind must be constexpr or compiler"
            )
        if kind == "compiler" and "role" not in declaration:
            raise TypeError(
                f"Layer-C compiler declaration {index} requires its trusted role"
            )
        # Preserve omission versus presence for optional attachment fields.
        # CandidateCheck must compare this exact shape, not a projected subset.
        expected_slots.append(
            {
                key: declaration[key]
                for key in (
                    "slot",
                    "kind",
                    "value_type",
                    "required",
                    "role",
                    "subject_ref",
                    "axis_ref",
                )
                if key in declaration
            }
        )

    raw_slot_order = declarations.get("slot_order")
    if raw_slot_order is not None:
        if not isinstance(raw_slot_order, list) or any(
            not isinstance(name, str) for name in raw_slot_order
        ):
            raise TypeError("Layer-C slot_order must be an array of names")
        declared_slot_order = [item["slot"] for item in expected_slots]
        if raw_slot_order != declared_slot_order:
            raise ValueError(
                "Layer-C declaration metadata disagrees about the active slot order"
            )
    return expected_slots


def _validate_value_free_layer_c_slots(
    slots: Sequence[ScheduleSlot],
    expected_slots: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Require candidate C slots to match trusted declarations exactly.

    A constexpr role is Lift-authored when the trusted declaration does not
    expose one, so that one field remains intentionally open.  Every trusted
    field, including the presence/absence and value of ``subject_ref`` and
    ``axis_ref``, is otherwise exact.  This prevents a candidate attachment
    from disappearing merely because the trusted declaration omitted it.
    """

    if len(slots) != len(expected_slots):
        raise ValueError(
            f"{context} differ from the value-free active interface: expected "
            f"{len(expected_slots)} slot(s), got {len(slots)}"
        )
    for index, (slot, expected) in enumerate(zip(slots, expected_slots, strict=True)):
        actual = slot.to_json()
        if actual.get("name") != expected.get("slot"):
            raise ValueError(
                f"{context} slot {index} has name "
                f"{actual.get('name')!r}, expected {expected.get('slot')!r}"
            )
        for expected_key, actual_key in (
            ("kind", "kind"),
            ("value_type", "value_type"),
            ("required", "required"),
        ):
            if actual.get(actual_key) != expected.get(expected_key):
                raise ValueError(
                    f"{context} slot {index}.{actual_key} differs from the "
                    "trusted declaration"
                )
        if "role" in expected and actual.get("role") != expected["role"]:
            raise ValueError(
                f"{context} slot {index}.role differs from the trusted declaration"
            )
        for key in ("subject_ref", "axis_ref"):
            expected_has = key in expected
            actual_has = key in actual
            if expected_has != actual_has or (
                expected_has and actual.get(key) != expected.get(key)
            ):
                raise ValueError(
                    f"{context} slot {index}.{key} differs from the trusted "
                    "declaration (including field presence)"
                )


def _decode_value_free_lift_candidate(
    response_text: str,
    *,
    expected_parameters: tuple[str, ...],
    semantic_compiler_view: Mapping[str, Any] | None,
    required_schema: str,
) -> Gate1Artifact:
    """Decode one draft without consulting Source realization or actual C values.

    ``Gate1Artifact`` requires complete active values, so CandidateCheck uses a
    deterministic type-only sentinel for each declared slot. The sentinels are
    derived solely from the public slot declaration and are never returned,
    persisted, sent to Replay, or treated as Source/target bindings.
    """

    payload = _json_object(response_text)
    _require_compact_lift_pi(payload)
    _require_fresh_lift_j_surface(payload)
    raw_c = payload.get("c")
    if not isinstance(raw_c, Mapping):
        raise TypeError("Lift artifact c must be an object")
    authored = ImplementationConfig.from_json(dict(raw_c))
    if authored.values:
        raise ValueError("Lift candidate must leave C_source.values empty")
    if semantic_compiler_view is None:
        if authored.slots:
            raise ValueError(
                "Lift declared Source-C slots without value-free declarations"
            )
        semantic_view: Mapping[str, Any] | None = None
        declarations: Mapping[str, Any] = {"slots": []}
    else:
        semantic_view = _semantic_abi_view(semantic_compiler_view)
        declarations = semantic_compiler_view.get("layer_c_declarations")
        if not isinstance(semantic_view, Mapping) or not isinstance(
            declarations, Mapping
        ):
            raise TypeError(
                "CandidateCheck requires semantic_lift_view and "
                "layer_c_declarations"
            )
    expected_slots = _value_free_layer_c_declarations(declarations)
    _validate_value_free_layer_c_slots(
        authored.slots,
        expected_slots,
        context="CandidateCheck Layer-C declarations",
    )

    def sentinel(value_type: str) -> bool | int | float | str:
        normalized = value_type.casefold()
        if normalized in {"i1", "bool"}:
            return False
        if normalized.startswith(("i", "u")) or normalized in {
            "int", "integer", "index"
        }:
            return 1
        if normalized.startswith("f") or normalized in {
            "bf16", "float", "number", "double"
        }:
            return 1.0
        if normalized in {"str", "string", "layout", "enum"}:
            return "candidate-check"
        raise ValueError(f"unsupported Layer-C scalar type {value_type!r}")

    provisional = ImplementationConfig(
        slots=[item.to_json() for item in authored.slots],
        values={
            item.name: sentinel(item.value_type)
            for item in authored.slots
        },
    )
    bound_payload = json.loads(json.dumps(payload, allow_nan=False))
    bound_payload["c"] = provisional.to_json()
    artifact = canonicalize_gate1_artifact(
        Gate1Artifact.from_json(bound_payload)
    )
    if artifact.wire_schema != required_schema:
        raise ValueError(
            f"Lift artifact uses {artifact.wire_schema!r}; "
            f"this round requires {required_schema!r}"
        )
    _validate_anonymous_lift_core(
        artifact,
        expected_parameters,
        semantic_compiler_view=semantic_compiler_view,
    )
    if semantic_view is not None:
        _validate_source_semantic_view(artifact, semantic_view)
    return artifact


def _require_compact_lift_pi(payload: Mapping[str, Any]) -> None:
    """Require the one pi subwire accepted at fresh Lift ingress.

    ``Gate1Artifact.from_json`` also reads retained historical Gate evidence,
    including complete legacy realization graphs.  That compatibility is
    deliberately broader than the fresh provider contract: accepting one of
    those graphs from a new Lift reply would make final Decode disagree with
    CandidateCheck's closed response schema.  Keep the archival parser broad,
    but make both fresh ingress paths require the same explicit compact
    normalization witness before Source binding or sentinel construction.
    """

    pi = payload.get("pi")
    if not isinstance(pi, Mapping):
        raise TypeError("Lift artifact pi must be an object")
    if pi.get("schema") != NORMALIZATION_WITNESS_SCHEMA:
        raise ValueError(
            "Lift artifact pi must use the compact normalization witness "
            "required by CandidateCheck"
        )


def _require_fresh_lift_j_surface(payload: Mapping[str, Any]) -> None:
    """Reject redundant J loop capture lists at both fresh Lift ingress paths."""

    raw_j = payload.get("j")
    intent = raw_j.get("intent") if isinstance(raw_j, Mapping) else None
    function = (
        intent.get("intent_function") if isinstance(intent, Mapping) else None
    )
    if not isinstance(function, Mapping):
        return

    def visit(body: Any, path: str) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            operation_path = f"{path}[{index}]"
            if operation.get("op") == "loop" and "operands" in operation:
                raise ValueError(
                    f"{operation_path}.operands duplicates loop capture facts; "
                    "fresh J represents them through its body and scope"
                )
            visit(operation.get("body"), f"{operation_path}.body")
            for arm_name in ("positive", "negative"):
                arm = operation.get(arm_name)
                if isinstance(arm, Mapping):
                    visit(arm.get("body"), f"{operation_path}.{arm_name}.body")

    visit(function.get("body"), "j.intent.intent_function.body")


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
    schema_diagnostics = _candidate_json_schema_diagnostics(response_text)
    if schema_diagnostics:
        occupied_paths = {item["path"] for item in schema_diagnostics}
        for item in _safe_candidate_surface_diagnostics(response_text, context):
            if len(schema_diagnostics) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                break
            if item["path"] in occupied_paths:
                continue
            schema_diagnostics.append(item)
            occupied_paths.add(item["path"])
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": schema_diagnostics,
        }

    try:
        artifact = _decode_value_free_lift_candidate(
            response_text,
            expected_parameters=context.expected_parameters,
            semantic_compiler_view=context.semantic_compiler_view,
            required_schema=context.schema_version,
        )
    except Exception as exc:
        diagnostics = _candidate_decode_diagnostics(exc)
        occupied_paths = {item["path"] for item in diagnostics}
        for item in _safe_candidate_surface_diagnostics(response_text, context):
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
    report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=artifact.wire_schema == artifact.SCHEMA,
    )
    return {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass" if report.passed else "fail",
        "diagnostics": [
            {
                "rule_code": item.code.value,
                "path": _bounded_candidate_diagnostic_text(item.path),
                "observed_refs_or_types": _candidate_typecheck_observations(
                    item.ref,
                    item.detail,
                ),
            }
            for item in report.diagnostics[:_CANDIDATE_DIAGNOSTIC_LIMIT]
        ],
    }


def _safe_candidate_surface_diagnostics(
    response_text: str,
    context: SCRCCandidateCheckContext,
) -> list[dict[str, Any]]:
    """Keep the additive candidate-diagnostic pass total and non-authoritative.

    The strict schema/decode path has already rejected the response before
    this helper runs.  Surface diagnostics may inspect the malformed object,
    but they must never be able to escape as an infrastructure exception or
    alter acceptance.  A bounded diagnostic-pass failure is therefore itself
    reported as candidate evidence; the original rejection remains decisive.
    """

    try:
        return _candidate_surface_diagnostics(response_text, context)
    except Exception as exc:  # pragma: no cover - defensive untrusted-input boundary
        return [
            {
                "rule_code": "candidate_surface_diagnostic_failed",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            }
        ]


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
    def useful_errors(error: Any) -> list[Any]:
        """Prefer the active discriminated branch over a huge oneOf summary."""

        context = getattr(error, "context", ())
        if not context:
            return [error]
        instance = getattr(error, "instance", None)
        op = instance.get("op") if isinstance(instance, Mapping) else None
        branch_index = {
            # The recursive IntentOp schema puts the ordinary semantic branch
            # first.  Select it for a known ordinary discriminator as well;
            # otherwise jsonschema expands every structured alternative and
            # reports unrelated loop/if fields for a simple op such as
            # map_binary.  Unknown/malformed discriminators intentionally keep
            # the complete fail-closed context so no useful error is hidden.
            **{
                operation: 0
                for operation in KNOWN_INTENT_OPS - STRUCTURED_INTENT_OPS
            },
            "loop": 1,
            "effect_if": 2,
            "value_if": 3,
        }.get(op)
        if branch_index is not None:
            selected = [
                child
                for child in context
                if list(getattr(child, "schema_path", ()))
                and list(child.schema_path)[0] == branch_index
            ]
            if selected:
                context = selected
        leaves: list[Any] = []
        for child in context:
            leaves.extend(useful_errors(child))
            if len(leaves) >= _CANDIDATE_DIAGNOSTIC_LIMIT:
                break
        return leaves or [error]

    diagnostics: list[dict[str, Any]] = []
    expanded_errors = [
        leaf
        for error in errors
        for leaf in useful_errors(error)
    ]
    for error in expanded_errors[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = ""
        for item in error.absolute_path:
            if isinstance(item, int):
                path += f"[{item}]"
            else:
                path += ("." if path else "") + str(item)
        absolute_path = tuple(str(item) for item in error.absolute_path)
        contract_note: str | None = None
        if (
            "c" in absolute_path
            and absolute_path
            and absolute_path[-1] == "subject_ref"
        ):
            contract_note = (
                "C subject_ref is a J subject t*/n*/d*/a*/e*/q*/b*, never x*; "
                "compiler slots must omit subject_ref and axis_ref, and a "
                "constexpr slot may attach only when its declaration supplies "
                "that exact attachment"
            )
        elif (
            "value_dimension_relations" in absolute_path
            and absolute_path[-1] == "value_ref"
        ):
            contract_note = (
                "value_dimension_relations.value_ref must be a rank-0 logical "
                "t*; physical p*, Source h*, dimensions x*, and SSA v* are "
                "invalid here"
            )
        elif (
            "intent_function" in absolute_path
            and absolute_path[-1] == "name"
            and any(item in {"inputs", "outputs"} for item in absolute_path)
        ):
            contract_note = (
                "logical tensor names are dense t* handles; physical p* handles "
                "belong only in j.signature.parameters"
            )
        elif "node_ids" in absolute_path:
            contract_note = (
                "j.node_ids values are structural body paths such as body[0] or "
                "body[1].positive.body[0], never v* result handles"
            )
        observed = [_bounded_candidate_diagnostic_text(error.message)]
        if contract_note is not None:
            observed.append(_bounded_candidate_diagnostic_text(contract_note))
        diagnostics.append(
            {
                "rule_code": "schema_rejected",
                "path": _bounded_candidate_diagnostic_text(
                    path or "<candidate>"
                ),
                "observed_refs_or_types": observed,
            }
        )
    return diagnostics


def _candidate_surface_diagnostics(
    response_text: str,
    context: SCRCCandidateCheckContext,
) -> list[dict[str, Any]]:
    """Report independent authored-surface mistakes beside a Decode failure.

    The strict decoder remains the only acceptance boundary.  This pass reads
    raw JSON only after that decoder has already rejected the candidate and
    checks closed naming/ABI facts that do not depend on repairing or
    interpreting an invalid graph.  It never mutates, canonicalizes, or fills
    a field, and its diagnostics cannot turn rejection into acceptance.
    """

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
                    _bounded_candidate_diagnostic_text(
                        "expected=" + json.dumps(
                            expected,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    ),
                    _bounded_candidate_diagnostic_text(
                        "actual=" + json.dumps(
                            actual,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    ),
                ],
            }
        )

    j = payload.get("j")
    if not isinstance(j, Mapping):
        return diagnostics
    nested_root_owned = _nested_root_owned_field_paths(j, path="j")
    for nested_path in nested_root_owned:
        mismatch(
            "root_envelope_nesting",
            nested_path,
            "root-level sibling of j",
            "nested under J",
        )
    intent = j.get("intent")
    fn = intent.get("intent_function") if isinstance(intent, Mapping) else None
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    if isinstance(parameters, list) and all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str)
        for item in parameters
    ):
        actual_parameters = [str(item["name"]) for item in parameters]
        if actual_parameters != list(context.expected_parameters):
            mismatch(
                "construction_handle_sequence",
                "j.signature.parameters[*].name",
                list(context.expected_parameters),
                actual_parameters,
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
    if (
        tensor_names
        and all(isinstance(item, str) for item in tensor_names)
        and not isinstance(context.semantic_compiler_view, Mapping)
    ):
        actual_tensors = list(dict.fromkeys(tensor_names))
        expected_tensors = [f"t{index}" for index in range(len(actual_tensors))]
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
            if (
                isinstance(extent, str)
                and extent != "?"
                and extent not in dimensions
            ):
                dimensions.append(extent)
    expected_dimensions = [f"x{index}" for index in range(len(dimensions))]
    if dimensions != expected_dimensions:
        mismatch(
            "construction_handle_sequence",
            "j.intent.intent_function tensor shape dimensions",
            expected_dimensions,
            dimensions,
        )

    def authored_op_paths(
        values: Any,
        *,
        prefix: str,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        """Mirror the canonical J body traversal without using JSON key order."""

        result: list[tuple[str, Mapping[str, Any]]] = []
        if not isinstance(values, list):
            return result
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                continue
            path = f"{prefix}[{index}]"
            result.append((path, item))
            result.extend(
                authored_op_paths(item.get("body"), prefix=f"{path}.body")
            )
            for arm_name in ("positive", "negative"):
                arm = item.get(arm_name)
                if isinstance(arm, Mapping):
                    result.extend(
                        authored_op_paths(
                            arm.get("body"),
                            prefix=f"{path}.{arm_name}.body",
                        )
                    )
        return result

    body = fn.get("body") if isinstance(fn, Mapping) else None
    operation_paths = authored_op_paths(body, prefix="body")
    operations = [operation for _path, operation in operation_paths]
    node_ids = [item.get("id") for item in operations]
    if node_ids and all(isinstance(item, str) for item in node_ids):
        expected_nodes = [f"n{index}" for index in range(len(node_ids))]
        if node_ids != expected_nodes:
            mismatch(
                "construction_handle_sequence",
                "j.intent.intent_function.body recursive node ids",
                expected_nodes,
                node_ids,
            )
    authored_node_map = j.get("node_ids")
    if isinstance(authored_node_map, Mapping) and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in authored_node_map.items()
    ):
        path_order = {
            path: index for index, (path, _operation) in enumerate(operation_paths)
        }
        # The node map is a JSON object, so insertion order cannot represent
        # declaration order.  Only compare its explicit body paths after every
        # one resolves; malformed paths remain for the strict decoder to
        # diagnose without a serializer-dependent secondary error.
        if set(authored_node_map.values()) <= set(path_order):
            ordered_keys = [
                key
                for key, _value in sorted(
                    authored_node_map.items(),
                    key=lambda item: path_order[item[1]],
                )
            ]
            expected_map_keys = [
                f"n{index}" for index in range(len(ordered_keys))
            ]
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
        expected = [f"{prefix}{index}" for index in range(len(actual))]
        if actual != expected:
            mismatch(
                "construction_handle_sequence",
                f"j.{field_name}" + (f"[*].{id_field}" if id_field else " keys"),
                expected,
                actual,
            )

    pi = payload.get("pi")
    if isinstance(pi, Mapping):
        for field_name, prefix in (("regions", "r"), ("instances", "i")):
            values = pi.get(field_name)
            if not isinstance(values, list) or not all(
                isinstance(item, Mapping) and isinstance(item.get("id"), str)
                for item in values
            ):
                continue
            actual = [str(item["id"]) for item in values]
            expected = [f"{prefix}{index}" for index in range(len(actual))]
            if actual != expected:
                mismatch(
                    "construction_handle_sequence",
                    f"pi.{field_name}[*].id",
                    expected,
                    actual,
                )
        order = pi.get("order")
        if isinstance(order, list):
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in order
            ]
            duplicates = sorted(
                item for item in set(encoded) if encoded.count(item) > 1
            )
            if duplicates:
                mismatch(
                    "construction_duplicate_relation",
                    "pi.order",
                    "unique order edges",
                    [json.loads(item) for item in duplicates],
                )

    view = context.semantic_compiler_view
    if isinstance(view, Mapping) and isinstance(parameters, list):
        try:
            semantic_view = _semantic_abi_view(view)
            abi_contract = _semantic_abi_contract(semantic_view)
        except (TypeError, ValueError) as exc:
            mismatch(
                "candidate_check_context_mismatch",
                "harness_compiler_view.semantic_lift_view",
                "closed semantic ABI contract",
                f"{type(exc).__name__}: {exc}",
            )
            return diagnostics
        expected_lock = dict(abi_contract["lock"])
        expected_lock["j_signature_parameter_order"] = list(
            abi_contract["parameter_order"]
        )
        expected_lock["j_constexpr_parameter_order"] = list(
            abi_contract["bound_constexpr_parameters"]
        )
        expected_lock["j_input_tensor_parameters"] = [
            item["physical_parameter"]
            for item in abi_contract["logical_tensor_order"]
            if item["container"] in {"inputs", "inputs+outputs"}
        ]
        expected_lock["j_output_only_tensor_parameters"] = [
            item["physical_parameter"]
            for item in abi_contract["logical_tensor_order"]
            if item["container"] == "outputs"
        ]
        declarations = view.get("layer_c_declarations")
        expected_c_slot_order: list[str] = []
        if isinstance(declarations, Mapping):
            raw_c_slot_order = declarations.get("slot_order")
            if isinstance(raw_c_slot_order, list) and all(
                isinstance(name, str) for name in raw_c_slot_order
            ):
                expected_c_slot_order = list(raw_c_slot_order)
            else:
                raw_c_slots = declarations.get("slots")
                if isinstance(raw_c_slots, list) and all(
                    isinstance(item, Mapping)
                    and isinstance(item.get("slot"), str)
                    for item in raw_c_slots
                ):
                    expected_c_slot_order = [
                        str(item["slot"]) for item in raw_c_slots
                    ]
        expected_lock["c_slot_order"] = expected_c_slot_order
        supplied_lock = view.get("abi_lock")
        if supplied_lock is None:
            # The compact model-visible view intentionally omits the lock:
            # every lock field is a deterministic projection of
            # semantic_lift_view plus the ordered C slot rows.  CandidateCheck
            # derives that projection locally, so the model need not receive
            # a second copy of the same ABI facts.
            pass
        elif not isinstance(supplied_lock, Mapping):
            mismatch(
                "candidate_check_context_mismatch",
                "harness_compiler_view.abi_lock",
                "mechanical ABI lock",
                supplied_lock,
            )
        else:
            for field_name in (
                "physical_parameter_order",
                "physical_to_logical",
                "logical_tensor_order",
                "scalar_logical_value_candidates",
                "j_fixed_input_tensor_order",
                "j_fixed_output_tensor_order",
                "j_value_dimension_relation_contract",
                "shape_dimension_order",
                "semantic_constexpr_parameters",
                "semantic_constexpr_value_types",
                "semantic_constexpr_shape_coordinates",
                "j_signature_parameter_order",
                "j_constexpr_parameter_order",
                "j_input_tensor_parameters",
                "j_output_only_tensor_parameters",
                "c_slot_order",
            ):
                if supplied_lock.get(field_name) != expected_lock.get(field_name):
                    mismatch(
                        "candidate_check_context_mismatch",
                        f"harness_compiler_view.abi_lock.{field_name}",
                        expected_lock.get(field_name),
                        supplied_lock.get(field_name),
                    )
        expected_parameters = abi_contract["parameter_order"]
        if list(context.expected_parameters) != expected_parameters:
            mismatch(
                "candidate_check_context_mismatch",
                "<context>.expected_parameters",
                expected_parameters,
                list(context.expected_parameters),
            )
        by_name = {
            item.get("name"): item
            for item in parameters
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        arguments = semantic_view.get("arguments")
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
                if source_kind == "tensor" and isinstance(argument.get("rank"), int):
                    expected_fields["rank"] = argument["rank"]
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
        physical_bindings = abi_contract["physical_to_logical"]
        logical_input_names = [
            item.get("name")
            for item in (fn.get("inputs", []) if isinstance(fn, Mapping) else [])
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ]
        logical_tensors = {
            item.get("name"): item
            for item in tensors
            if isinstance(item.get("name"), str)
        }
        try:
            layout = _semantic_abi_tensor_layout(
                by_name,
                abi_contract,
                logical_input_names=logical_input_names,
                logical_tensors=logical_tensors,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # This function is an additive diagnostic pass after the strict
            # schema/decode boundary has already rejected the candidate.  A
            # malformed authored layout must therefore remain a candidate
            # diagnostic; it must never escape as an infrastructure failure
            # and terminate the bounded Lift continuation.  In particular,
            # an output-only t* inserted before an optional scalar can make
            # the shared layout derivation reject the draft.  Preserve that
            # fact without repairing or interpreting the candidate further.
            mismatch(
                "construction_abi_mismatch",
                "j.signature.parameters and j.intent.intent_function.inputs+outputs",
                "a trusted dense logical ABI layout",
                f"{type(exc).__name__}: {exc}",
            )
            return diagnostics
        for binding in physical_bindings:
            physical_parameter = binding["physical_parameter"]
            parameter = by_name.get(physical_parameter)
            if not isinstance(parameter, Mapping):
                mismatch(
                    "construction_abi_mismatch",
                    f"j.signature.parameters[{physical_parameter}]",
                    "one parameter with the trusted physical handle",
                    None,
                )
                continue
            kind = binding["kind"]
            if kind == "tensor":
                expected_descriptor = layout["descriptor_by_parameter"].get(
                    physical_parameter
                )
                expected_ref = (
                    expected_descriptor.get("logical_tensor")
                    if isinstance(expected_descriptor, Mapping)
                    else None
                )
                actual_ref = parameter.get("logical_ref")
                if not isinstance(expected_ref, str) or actual_ref != expected_ref:
                    mismatch(
                        "construction_logical_tensor_mapping",
                        f"j.signature.parameters[{physical_parameter}].logical_ref",
                        expected_ref,
                        actual_ref,
                    )
                expected_mode = binding.get("mode")
                if expected_mode is not None:
                    actual_mode = parameter.get("mode")
                    if actual_mode != expected_mode:
                        mismatch(
                            "construction_abi_mismatch",
                            f"j.signature.parameters[{physical_parameter}].mode",
                            expected_mode,
                            actual_mode,
                        )
            elif binding.get("logical_kind") == "dimension":
                expected_dimension = binding["logical_ref"]
                actual_ref = parameter.get("logical_ref")
                relation_values = j.get("value_dimension_relations", [])
                relation_ok = any(
                    isinstance(item, Mapping)
                    and item.get("value_ref") == actual_ref
                    and item.get("dimension_ref") == expected_dimension
                    and item.get("relation") == "eq"
                    for item in relation_values
                ) if isinstance(relation_values, list) else False
                scalar_tensor = next(
                    (
                        item
                        for item in tensors
                        if item.get("name") == actual_ref
                        and item.get("shape") == []
                    ),
                    None,
                )
                if actual_ref != expected_dimension and not (
                    isinstance(actual_ref, str)
                    and re.fullmatch(r"t[0-9]+", actual_ref)
                    and scalar_tensor is not None
                    and relation_ok
                ):
                    mismatch(
                        "construction_shape_coordinate_mapping",
                        f"j.signature.parameters[{physical_parameter}].logical_ref",
                        [
                            expected_dimension,
                            "rank-0 t* plus value_dimension_relations eq",
                        ],
                        actual_ref,
                    )
            elif kind == "scalar":
                actual_ref = parameter.get("logical_ref")
                if actual_ref is not None and (
                    not isinstance(actual_ref, str)
                    or re.fullmatch(r"t[0-9]+", actual_ref) is None
                ):
                    mismatch(
                        "construction_representation_mapping",
                        f"j.signature.parameters[{physical_parameter}].logical_ref",
                        ["rank-0 t*", None],
                        actual_ref,
                    )
            elif kind in {"opaque", "none"} and parameter.get("logical_ref") is not None:
                mismatch(
                    "construction_representation_mapping",
                    f"j.signature.parameters[{physical_parameter}].logical_ref",
                    None,
                    parameter.get("logical_ref"),
                )

        for name in abi_contract["bound_constexpr_parameters"]:
            parameter = by_name.get(name)
            if isinstance(parameter, Mapping):
                expected_dtype = abi_contract["bound_constexpr_value_types"][name]
                actual_dtype = parameter.get("dtype")
                if _canonical_dtype(actual_dtype) != _canonical_dtype(expected_dtype):
                    mismatch(
                        "construction_abi_mismatch",
                        f"j.signature.parameters[{name}].dtype",
                        _canonical_dtype(expected_dtype),
                        _canonical_dtype(actual_dtype),
                    )
                actual_ref = parameter.get("logical_ref")
                expected_dimension = abi_contract[
                    "bound_constexpr_dimension_by_parameter"
                ].get(name)
                if expected_dimension is not None:
                    relation_values = j.get("value_dimension_relations", [])
                    relation_ok = any(
                        isinstance(item, Mapping)
                        and item.get("value_ref") == actual_ref
                        and item.get("dimension_ref") == expected_dimension
                        and item.get("relation") == "eq"
                        for item in relation_values
                    ) if isinstance(relation_values, list) else False
                    scalar_tensor = next(
                        (
                            item
                            for item in tensors
                            if item.get("name") == actual_ref
                            and item.get("shape") == []
                        ),
                        None,
                    )
                    if actual_ref != expected_dimension and not (
                        isinstance(actual_ref, str)
                        and re.fullmatch(r"t[0-9]+", actual_ref)
                        and scalar_tensor is not None
                        and relation_ok
                    ):
                        mismatch(
                            "construction_shape_coordinate_mapping",
                            f"j.signature.parameters[{name}].logical_ref",
                            [
                                expected_dimension,
                                "rank-0 t* plus value_dimension_relations eq",
                            ],
                            actual_ref,
                        )
                    continue
                if actual_ref is not None and (
                    not isinstance(actual_ref, str)
                    or re.fullmatch(r"t[0-9]+", actual_ref) is None
                ):
                    mismatch(
                        "construction_representation_mapping",
                        f"j.signature.parameters[{name}].logical_ref",
                        ["rank-0 t*", None],
                        actual_ref,
                    )

        logical_order = layout["logical_tensor_order"]
        expected_input_names = layout["expected_input_names"]
        expected_output_names = layout["expected_output_names"]
        expected_logical_names = layout["expected_unique_names"]
        candidate_tensor_by_name = {
            item.get("name"): item
            for item in tensors
            if isinstance(item.get("name"), str)
        }
        for expected in logical_order:
            logical_name = expected["logical_tensor"]
            physical_parameter = expected["physical_parameter"]
            if not isinstance(logical_name, str):
                continue
            tensor = candidate_tensor_by_name.get(logical_name)
            if not isinstance(tensor, Mapping):
                continue
            expected_dtype = expected.get("dtype")
            if isinstance(expected_dtype, str) and _canonical_dtype(
                tensor.get("dtype")
            ) != _canonical_dtype(expected_dtype):
                mismatch(
                    "construction_logical_tensor_contract",
                    f"j.intent.intent_function.tensor[{logical_name}].dtype",
                    _canonical_dtype(expected_dtype),
                    tensor.get("dtype"),
                )
            if tensor.get("shape") is not None:
                expected_shape = expected.get("shape")
                if isinstance(expected_shape, list) and tensor.get("shape") != expected_shape:
                    mismatch(
                        "construction_shape_coordinate_mapping",
                        f"j.intent.intent_function.tensor[{logical_name}].shape",
                        expected_shape,
                        tensor.get("shape"),
                    )

        candidate_unique_tensors = list(dict.fromkeys(
            item.get("name") for item in tensors
            if isinstance(item.get("name"), str)
        ))
        if candidate_unique_tensors != expected_logical_names:
            mismatch(
                "construction_logical_tensor_order",
                "j.intent.intent_function.inputs+outputs[*].name",
                expected_logical_names,
                candidate_unique_tensors,
            )
        raw_inputs = fn.get("inputs") if isinstance(fn, Mapping) else None
        raw_outputs = fn.get("outputs") if isinstance(fn, Mapping) else None
        actual_inputs = [
            item.get("name") for item in raw_inputs
        ] if isinstance(raw_inputs, list) and all(
            isinstance(item, Mapping) and isinstance(item.get("name"), str)
            for item in raw_inputs
        ) else None
        actual_outputs = [
            item.get("name") for item in raw_outputs
        ] if isinstance(raw_outputs, list) and all(
            isinstance(item, Mapping) and isinstance(item.get("name"), str)
            for item in raw_outputs
        ) else None
        if actual_inputs is not None and actual_inputs != expected_input_names:
            mismatch(
                "construction_logical_tensor_container",
                "j.intent.intent_function.inputs[*].name",
                expected_input_names,
                actual_inputs,
            )
        if actual_outputs is not None and actual_outputs != expected_output_names:
            mismatch(
                "construction_logical_tensor_container",
                "j.intent.intent_function.outputs[*].name",
                expected_output_names,
                actual_outputs,
            )
        c = payload.get("c")
        if isinstance(c, Mapping) and c.get("values") != {}:
            mismatch(
                "construction_abi_mismatch",
                "c.values",
                {},
                c.get("values"),
            )
        if isinstance(c, Mapping):
            raw_slots = c.get("slots")
            if isinstance(raw_slots, list):
                for index, raw_slot in enumerate(raw_slots):
                    if not isinstance(raw_slot, Mapping):
                        continue
                    if raw_slot.get("kind") != "compiler":
                        continue
                    attached = {
                        key: raw_slot[key]
                        for key in ("subject_ref", "axis_ref")
                        if key in raw_slot
                    }
                    if attached:
                        mismatch(
                            "construction_layer_c_attachment",
                            f"c.slots[{index}]",
                            "compiler slot without subject_ref/axis_ref",
                            attached,
                        )
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
            declarations = view.get("layer_c_declarations")
            expected_slot_order: list[str] | None = None
            if isinstance(declarations, Mapping):
                raw_slot_order = declarations.get("slot_order")
                if isinstance(raw_slot_order, list) and all(
                    isinstance(name, str) for name in raw_slot_order
                ):
                    expected_slot_order = list(raw_slot_order)
                else:
                    raw_declarations = declarations.get("slots")
                    if isinstance(raw_declarations, list) and all(
                        isinstance(item, Mapping)
                        and isinstance(item.get("slot"), str)
                        for item in raw_declarations
                    ):
                        expected_slot_order = [
                            str(item["slot"]) for item in raw_declarations
                        ]
            if (
                expected_slot_order is not None
                and isinstance(raw_slots, list)
                and all(
                    isinstance(item, Mapping)
                    and isinstance(item.get("name"), str)
                    for item in raw_slots
                )
            ):
                actual_slot_order = [str(item["name"]) for item in raw_slots]
                if actual_slot_order != expected_slot_order:
                    mismatch(
                        "construction_layer_c_slot_order",
                        "c.slots[*].name",
                        expected_slot_order,
                        actual_slot_order,
                    )
            bound_non_c = semantic_view.get("bound_constexpr_parameters")
            if isinstance(bound_non_c, list):
                for name in bound_non_c:
                    if name in slots_by_name:
                        mismatch(
                            "construction_semantic_constexpr_in_c",
                            f"c.slots[{name}]",
                            "bound non-C constexpr carrier in J.signature only",
                            slots_by_name[name],
                        )
        actual_aliases = boundary.get("aliases") if isinstance(boundary, Mapping) else None
        if actual_aliases != semantic_view.get("aliases"):
            mismatch(
                "construction_abi_mismatch",
                "j.boundary.aliases",
                semantic_view.get("aliases"),
                actual_aliases,
            )

    return diagnostics


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


def _candidate_decode_diagnostics(exc: Exception) -> list[dict[str, Any]]:
    """Expose only independent errors already found by the strict decoder."""

    if not isinstance(exc, SchemaError) or not exc.issues:
        return [
            {
                "rule_code": "decode_rejected",
                "path": "<candidate>",
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            }
        ]

    diagnostics: list[dict[str, Any]] = []
    for issue in exc.issues[:_CANDIDATE_DIAGNOSTIC_LIMIT]:
        path = issue.path
        if path == "LogicalSignature" or path.startswith("LogicalSignature."):
            path = "j.signature" + path.removeprefix("LogicalSignature")
        elif path == "SemanticArchive" or path.startswith("SemanticArchive."):
            path = "j" + path.removeprefix("SemanticArchive")
        diagnostics.append(
            {
                "rule_code": "decode_rejected",
                "path": _bounded_candidate_diagnostic_text(path),
                "observed_refs_or_types": [
                    _bounded_candidate_diagnostic_text(
                        f"SchemaError: {issue.detail}"
                    )
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


def _value_dimension_relation_exists(
    artifact: Gate1Artifact,
    value_ref: str,
    dimension_ref: str,
) -> bool:
    return any(
        relation.value_ref == value_ref
        and relation.dimension_ref == dimension_ref
        and relation.relation == "eq"
        for relation in artifact.j.value_dimension_relations
    )


def _semantic_abi_tensor_layout(
    signature: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    logical_input_names: Sequence[str] | None = None,
    logical_tensors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the dense logical tensor layout after optional scalar values.

    Tensor handles are assigned by first declaration in J, not by physical
    parameter position.  A scalar/constexpr parameter that the Lift owner
    elects to expose as a rank-0 logical value therefore consumes a new input
    ``t*`` before output-only tensors.  This helper is shared by provisional
    CandidateCheck diagnostics and final Decode so neither path accepts the
    non-dense ``t4,...,t2`` layout that used to cross the boundary.
    """

    def field(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    if not isinstance(signature, Mapping):
        raise TypeError("semantic ABI signature must be a parameter mapping")
    parameter_order = list(contract.get("parameter_order", []))
    if len(parameter_order) != len(set(parameter_order)):
        raise ValueError("semantic ABI parameter_order must be unique")
    missing_parameters = [
        name for name in parameter_order if name not in signature
    ]
    if missing_parameters:
        raise ValueError(
            "semantic ABI signature is missing parameter(s): "
            + ", ".join(missing_parameters)
        )

    # ``contract.logical_tensor_order`` is deliberately only the stable
    # physical-role skeleton.  In particular, output-only entries have no
    # baseline logical_ref.  This helper is the sole place where candidate
    # logical scalar choices are combined with that skeleton to produce final
    # dense t* handles.
    logical_order = list(contract["logical_tensor_order"])
    fixed_input_tensors = [
        item
        for item in logical_order
        if item["container"] in {"inputs", "inputs+outputs"}
    ]
    output_only_tensors = [
        item for item in logical_order if item["container"] == "outputs"
    ]
    known_roles = True

    fixed_input_names = [item.get("logical_tensor") for item in fixed_input_tensors]
    expected_fixed_names = [f"t{index}" for index in range(len(fixed_input_names))]
    if fixed_input_names != expected_fixed_names:
        raise ValueError(
            "fixed logical tensor entries must use dense declaration-order t* "
            "handles"
        )
    if len({item.get("physical_parameter") for item in logical_order}) != len(
        logical_order
    ):
        raise ValueError(
            "semantic ABI logical tensor entries must use unique physical parameters"
        )

    scalar_parameter_names: list[str] = []
    for binding in contract["physical_to_logical"]:
        if binding["kind"] == "scalar":
            scalar_parameter_names.append(binding["physical_parameter"])
    scalar_parameter_names.extend(contract["bound_constexpr_parameters"])
    parameter_position = {
        name: index for index, name in enumerate(contract["parameter_order"])
    }
    ordered_scalar_names = sorted(
        set(scalar_parameter_names),
        key=lambda name: parameter_position[name],
    )
    scalar_refs = [
        (name, field(signature[name], "logical_ref"))
        for name in ordered_scalar_names
        if name in signature
        and isinstance(field(signature[name], "logical_ref"), str)
        and re.fullmatch(r"t[0-9]+", field(signature[name], "logical_ref"))
    ]
    scalar_ref_names = [ref for _name, ref in scalar_refs]
    if len(scalar_ref_names) != len(set(scalar_ref_names)):
        raise ValueError(
            "optional scalar/constexpr logical values must use unique t* handles"
        )
    expected_scalar_refs = [
        f"t{len(fixed_input_tensors) + index}"
        for index in range(len(scalar_refs))
    ]
    if scalar_ref_names != expected_scalar_refs:
        raise ValueError(
            "optional scalar/constexpr logical values must use dense t* handles "
            "after fixed tensor inputs"
        )
    if logical_input_names is not None:
        logical_input_set = set(logical_input_names)
        for physical_parameter, logical_ref in scalar_refs:
            if logical_ref not in logical_input_set:
                raise ValueError(
                    f"optional scalar/constexpr parameter {physical_parameter!r} "
                    f"must bind a J input, got {logical_ref!r}"
                )
    for physical_parameter, logical_ref in scalar_refs:
        parameter = signature[physical_parameter]
        rank = field(parameter, "rank")
        if rank != 0:
            raise ValueError(
                f"optional scalar/constexpr parameter {physical_parameter!r} "
                "must have signature rank=0"
            )
        if logical_tensors is not None:
            tensor = logical_tensors.get(logical_ref)
            if tensor is None:
                raise ValueError(
                    f"optional scalar/constexpr parameter {physical_parameter!r} "
                    f"references undeclared logical tensor {logical_ref!r}"
                )
            shape = field(tensor, "shape")
            if shape is None or list(shape) != []:
                raise ValueError(
                    f"optional scalar/constexpr parameter {physical_parameter!r} "
                    "must bind a rank-0 J tensor with shape=[]"
                )
            parameter_dtype = _canonical_dtype(field(parameter, "dtype"))
            tensor_dtype = _canonical_dtype(field(tensor, "dtype"))
            if parameter_dtype != tensor_dtype:
                raise ValueError(
                    f"optional scalar/constexpr parameter {physical_parameter!r} "
                    "dtype differs from its logical rank-0 tensor"
                )
    expected_output_refs = [
        f"t{len(fixed_input_tensors) + len(scalar_refs) + index}"
        for index in range(len(output_only_tensors))
    ]
    output_ref_by_parameter = {
        item["physical_parameter"]: expected_ref
        for item, expected_ref in zip(
            output_only_tensors,
            expected_output_refs,
            strict=True,
        )
    }
    expected_input_names = fixed_input_names + expected_scalar_refs
    fixed_ref_by_parameter = {
        item["physical_parameter"]: item["logical_tensor"]
        for item in fixed_input_tensors
    }
    output_tensor_parameters = [
        binding["physical_parameter"]
        for binding in contract["physical_to_logical"]
        if binding.get("kind") == "tensor"
        and binding.get("mode") in {"output", "inout"}
    ]
    output_names_by_parameter = {
        **fixed_ref_by_parameter,
        **output_ref_by_parameter,
    }
    expected_output_parameters = {
        item["physical_parameter"]
        for item in fixed_input_tensors
        if item["container"] == "inputs+outputs"
    } | set(output_ref_by_parameter)
    if (
        len(output_tensor_parameters) != len(expected_output_parameters)
        or set(output_tensor_parameters) != expected_output_parameters
    ):
        raise ValueError(
            "semantic ABI output parameter order does not cover the declared "
            "inout and output-only tensor roles"
        )
    try:
        expected_output_names = [
            output_names_by_parameter[parameter]
            for parameter in output_tensor_parameters
        ]
    except KeyError as exc:
        raise ValueError(
            "semantic ABI output parameter order references an unknown tensor "
            f"parameter {exc.args[0]!r}"
        ) from exc
    expected_unique_names = list(
        dict.fromkeys([*expected_input_names, *expected_output_names])
    )
    if expected_unique_names != [f"t{index}" for index in range(len(expected_unique_names))]:
        raise ValueError(
            "final semantic ABI logical tensor handles must be dense in "
            "first-declaration order"
        )
    # Materialize one final descriptor list.  All consumers that need a
    # logical tensor's dtype/rank/shape must use this list rather than the
    # baseline skeleton; otherwise an output after an optional scalar can
    # accidentally validate against the preceding output's metadata.
    final_logical_order: list[dict[str, Any]] = []
    descriptor_by_parameter: dict[str, dict[str, Any]] = {}
    descriptor_by_logical_ref: dict[str, dict[str, Any]] = {}

    def add_descriptor(item: Mapping[str, Any], logical_ref: str) -> None:
        if not isinstance(logical_ref, str) or re.fullmatch(r"t[0-9]+", logical_ref) is None:
            raise ValueError(
                "semantic ABI logical tensor descriptors require t* handles"
            )
        if logical_ref in descriptor_by_logical_ref:
            raise ValueError(
                f"semantic ABI logical tensor {logical_ref!r} is described more than once"
            )
        descriptor = dict(item)
        descriptor["logical_tensor"] = logical_ref
        final_logical_order.append(descriptor)
        physical_parameter = descriptor.get("physical_parameter")
        if isinstance(physical_parameter, str):
            if physical_parameter in descriptor_by_parameter:
                raise ValueError(
                    f"semantic ABI physical parameter {physical_parameter!r} is "
                    "described more than once"
                )
            descriptor_by_parameter[physical_parameter] = descriptor
        descriptor_by_logical_ref[logical_ref] = descriptor

    for item in fixed_input_tensors:
        logical_ref = item.get("logical_tensor")
        if not isinstance(logical_ref, str):
            raise ValueError(
                "fixed logical tensor entries must have declaration-order t* handles"
            )
        add_descriptor(item, logical_ref)

    for (physical_parameter, logical_ref), expected_ref in zip(
        scalar_refs,
        expected_scalar_refs,
        strict=True,
    ):
        parameter = signature.get(physical_parameter)
        dtype = field(parameter, "dtype") if parameter is not None else None
        add_descriptor(
            {
                "logical_tensor": expected_ref,
                "physical_parameter": physical_parameter,
                "container": "inputs",
                "role_known": True,
                "logical_ref_policy": "optional_rank0_value",
                "mode": "value",
                "dtype": dtype,
                "rank": 0,
                "shape": [],
            },
            expected_ref,
        )

    for item, expected_ref in zip(
        output_only_tensors,
        expected_output_refs,
        strict=True,
    ):
        add_descriptor(item, expected_ref)

    # A final descriptor must exist for every physical tensor/scalar logical
    # binding.  This catches a contract/layout mismatch at the shared helper
    # boundary instead of letting one consumer silently fall back to a
    # different numbering.
    for item in logical_order:
        if item["container"] == "outputs":
            parameter = item["physical_parameter"]
            if parameter not in descriptor_by_parameter:
                raise ValueError(
                    f"output-only parameter {parameter!r} has no final logical t*"
                )

    tensor_binding_names = {
        binding["physical_parameter"]
        for binding in contract["physical_to_logical"]
        if binding.get("kind") == "tensor"
    }
    descriptor_tensor_names = {
        item["physical_parameter"]
        for item in logical_order
    }
    if tensor_binding_names != descriptor_tensor_names:
        raise ValueError(
            "semantic ABI tensor bindings and logical tensor descriptors do not "
            "cover the same physical parameters"
        )

    return {
        "known_roles": known_roles,
        "fixed_input_tensors": fixed_input_tensors,
        "output_only_tensors": output_only_tensors,
        "scalar_parameter_names": ordered_scalar_names,
        "scalar_refs": scalar_refs,
        "scalar_ref_names": scalar_ref_names,
        "expected_scalar_refs": expected_scalar_refs,
        "output_ref_by_parameter": output_ref_by_parameter,
        "expected_input_names": expected_input_names,
        "expected_output_names": expected_output_names,
        "expected_unique_names": expected_unique_names,
        "logical_tensor_order": final_logical_order,
        "descriptor_by_parameter": descriptor_by_parameter,
        "descriptor_by_logical_ref": descriptor_by_logical_ref,
    }


def _declared_abi_tensor_layout(artifact: Gate1Artifact) -> dict[str, Any]:
    """Derive the same dense tensor layout from an already decoded artifact.

    A caller without a compiler semantic sidecar still has enough information
    in ``J.signature`` to distinguish fixed tensor inputs, source-use-dependent
    rank-0 logical scalar values, and output-only tensors.  Use the shared
    layout routine rather than falling back to the old ``t0..tN`` count-only
    check, which could accept an output handle in the scalar insertion gap.
    """

    parameters = list(artifact.j.signature.parameters)
    tensor_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind in {ParameterKind.TENSOR, ParameterKind.POINTER}
    ]
    tensor_by_logical_ref: dict[str, Any] = {}
    for parameter in tensor_parameters:
        logical_ref = parameter.logical_ref
        if not isinstance(logical_ref, str):
            continue
        if logical_ref in tensor_by_logical_ref:
            previous = tensor_by_logical_ref[logical_ref]
            raise ValueError(
                "J logical tensor handle "
                f"{logical_ref!r} is bound by both physical parameters "
                f"{previous.name!r} and {parameter.name!r}"
            )
        tensor_by_logical_ref[logical_ref] = parameter
    declared_tensors = {
        item.name: item
        for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    # J input declaration order is authoritative for the fixed logical tensor
    # prefix.  It is intentionally not reconstructed from physical p* order:
    # a source ABI may place output-only parameters before an inout parameter,
    # while J must still declare the inout tensor before optional rank-0 values
    # and output-only tensors.
    fixed_input_names: list[str] = []
    for tensor in artifact.j.intent.inputs:
        parameter = tensor_by_logical_ref.get(tensor.name)
        if parameter is None:
            continue
        if parameter.mode in {ParameterMode.INPUT, ParameterMode.INOUT}:
            fixed_input_names.append(tensor.name)
    fixed_inputs = [
        tensor_by_logical_ref[name]
        for name in fixed_input_names
    ]
    output_only = [
        parameter
        for parameter in tensor_parameters
        if parameter.mode is ParameterMode.OUTPUT
    ]
    if len(fixed_input_names) != len(fixed_inputs):
        raise ValueError("J fixed input tensor declarations are incomplete")
    logical_order: list[dict[str, Any]] = []
    for index, (logical_name, parameter) in enumerate(
        zip(fixed_input_names, fixed_inputs, strict=True)
    ):
        tensor = declared_tensors.get(logical_name)
        if tensor is None:
            raise ValueError(
                f"fixed logical tensor {logical_name!r} is not declared in J"
            )
        logical_order.append(
            {
                "logical_tensor": f"t{index}",
                "physical_parameter": parameter.name,
                "container": (
                    "inputs+outputs"
                    if parameter.mode is ParameterMode.INOUT
                    else "inputs"
                ),
                "role_known": True,
                "mode": parameter.mode.value,
                "dtype": parameter.dtype,
                "rank": parameter.rank,
                "shape": list(tensor.shape),
            }
        )
    for index, parameter in enumerate(output_only):
        logical_order.append(
            {
                # The final output handle depends on optional rank-0 scalar
                # values and is assigned by _semantic_abi_tensor_layout.
                "logical_tensor": None,
                "physical_parameter": parameter.name,
                "container": "outputs",
                "role_known": True,
                "logical_ref_policy": "deferred_after_optional_scalar_values",
                "mode": parameter.mode.value,
                "dtype": parameter.dtype,
                "rank": parameter.rank,
                "shape": list(
                    declared_tensors[parameter.logical_ref].shape
                )
                if isinstance(parameter.logical_ref, str)
                and parameter.logical_ref in declared_tensors
                else None,
            }
        )
    physical_bindings: list[dict[str, Any]] = []
    for parameter in parameters:
        if parameter.kind in {ParameterKind.TENSOR, ParameterKind.POINTER}:
            if parameter.mode in {ParameterMode.INPUT, ParameterMode.INOUT}:
                logical_ref = parameter.logical_ref
                container = (
                    "inputs+outputs"
                    if parameter.mode is ParameterMode.INOUT
                    else "inputs"
                )
            else:
                logical_ref = None
                container = "outputs"
            physical_bindings.append(
                {
                    "physical_parameter": parameter.name,
                    "kind": "tensor",
                    "mode": parameter.mode.value,
                    "logical_ref": logical_ref,
                    "logical_container": container,
                }
            )
            continue
        if parameter.kind in {ParameterKind.SCALAR, ParameterKind.CONSTEXPR}:
            logical_kind = (
                "dimension"
                if isinstance(parameter.logical_ref, str)
                and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
                else None
            )
            physical_bindings.append(
                {
                    "physical_parameter": parameter.name,
                    "kind": "scalar",
                    "mode": parameter.mode.value,
                    "logical_ref": parameter.logical_ref,
                    **(
                        {"logical_kind": logical_kind}
                        if logical_kind is not None
                        else {}
                    ),
                }
            )
            continue
        physical_bindings.append(
            {
                "physical_parameter": parameter.name,
                "kind": parameter.kind.value,
                "mode": parameter.mode.value,
                "logical_ref": parameter.logical_ref,
            }
        )
    contract = {
        "logical_tensor_order": logical_order,
        "physical_to_logical": physical_bindings,
        "bound_constexpr_parameters": [],
        "parameter_order": [parameter.name for parameter in parameters],
    }
    layout = _semantic_abi_tensor_layout(
        {parameter.name: parameter for parameter in parameters},
        contract,
        logical_input_names=[item.name for item in artifact.j.intent.inputs],
        logical_tensors=declared_tensors,
    )
    for parameter in tensor_parameters:
        descriptor = layout["descriptor_by_parameter"].get(parameter.name)
        expected_ref = (
            descriptor.get("logical_tensor")
            if isinstance(descriptor, Mapping)
            else None
        )
        if parameter.logical_ref != expected_ref:
            raise ValueError(
                f"physical tensor parameter {parameter.name!r} maps to "
                f"{parameter.logical_ref!r}, expected final logical {expected_ref!r}"
            )
    actual_unique = list(
        dict.fromkeys(
            item.name
            for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
        )
    )
    expected_dense = [f"t{index}" for index in range(len(actual_unique))]
    if actual_unique != expected_dense:
        raise ValueError(
            "J logical tensor handles must be dense in first-declaration order"
        )
    return layout


def _validate_semantic_abi_contract(
    artifact: Gate1Artifact,
    semantic_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only the ABI mapping derived by ``_semantic_abi_contract``.

    This is deliberately shared by value-free CandidateCheck's provisional
    decode and the final Decode after BindSource.  It checks the physical
    parameter order, p* to logical tensor mapping, tensor container roles, and
    h* to x* shape mapping without using any concrete Source realization value.
    """

    contract = _semantic_abi_contract(semantic_view)
    signature = {item.name: item for item in artifact.j.signature.parameters}
    expected_parameters = contract["parameter_order"]
    actual_parameters = [item.name for item in artifact.j.signature.parameters]
    if actual_parameters != expected_parameters:
        raise ValueError(
            "J.signature parameters differ from the trusted physical ABI order"
        )

    inputs = list(artifact.j.intent.inputs)
    outputs = list(artifact.j.intent.outputs)
    input_names = [item.name for item in inputs]
    output_names = [item.name for item in outputs]
    tensors = {
        item.name: item
        for item in [*inputs, *outputs]
    }
    layout = _semantic_abi_tensor_layout(
        signature,
        contract,
        logical_input_names=input_names,
        logical_tensors=tensors,
    )
    known_roles = layout["known_roles"]
    descriptor_by_parameter = layout["descriptor_by_parameter"]
    for binding in contract["physical_to_logical"]:
        physical_parameter = binding["physical_parameter"]
        parameter = signature.get(physical_parameter)
        if parameter is None:
            raise ValueError(
                f"J.signature is missing trusted physical parameter {physical_parameter!r}"
            )
        kind = binding["kind"]
        if kind == "tensor":
            if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
                raise ValueError(
                    f"physical tensor parameter {physical_parameter!r} changed kind"
                )
            container = binding.get("logical_container")
            expected = descriptor_by_parameter.get(physical_parameter)
            if expected is None:
                raise ValueError(
                    f"physical tensor parameter {physical_parameter!r} has no "
                    "shared final logical layout entry"
                )
            expected_ref = expected.get("logical_tensor")
            if not isinstance(expected_ref, str):
                raise ValueError(
                    f"physical tensor parameter {physical_parameter!r} has no "
                    "final logical t* handle"
                )
            if parameter.logical_ref != expected_ref:
                raise ValueError(
                    f"physical tensor parameter {physical_parameter!r} must map to "
                    f"{expected_ref!r}"
                )
            expected_mode = expected.get("mode")
            if expected_mode is not None and parameter.mode.value != expected_mode:
                raise ValueError(
                    f"physical tensor parameter {physical_parameter!r} changed mode"
                )
            actual_tensor_ref = (
                parameter.logical_ref
            )
            tensor = tensors.get(actual_tensor_ref)
            if tensor is None:
                raise ValueError(
                    f"trusted logical tensor {actual_tensor_ref!r} is not declared"
                )
            expected_dtype = expected.get("dtype")
            if not isinstance(expected_dtype, str) or _canonical_dtype(
                tensor.dtype
            ) != _canonical_dtype(expected_dtype):
                raise ValueError(
                    f"logical tensor {actual_tensor_ref!r} changed trusted dtype"
                )
            if expected.get("rank") != len(tensor.shape):
                raise ValueError(
                    f"logical tensor {actual_tensor_ref!r} changed trusted rank"
                )
            expected_shape = expected.get("shape")
            if isinstance(expected_shape, list) and tensor.shape != expected_shape:
                raise ValueError(
                    f"logical tensor {actual_tensor_ref!r} changed the h* to x* shape mapping"
                )
            continue
        if kind == "scalar" and binding.get("logical_kind") == "dimension":
            expected_dimension = binding["logical_ref"]
            if parameter.kind not in {ParameterKind.SCALAR, ParameterKind.CONSTEXPR}:
                raise ValueError(
                    f"shape-coordinate parameter {physical_parameter!r} changed kind"
                )
            actual_ref = parameter.logical_ref
            relation_ok = (
                isinstance(actual_ref, str)
                and re.fullmatch(r"t[0-9]+", actual_ref) is not None
                and actual_ref in tensors
                and not tensors[actual_ref].shape
                and _value_dimension_relation_exists(
                    artifact,
                    actual_ref,
                    expected_dimension,
                )
            )
            if actual_ref != expected_dimension and not relation_ok:
                raise ValueError(
                    f"shape-coordinate parameter {physical_parameter!r} must map to "
                    f"{expected_dimension!r} or an explicitly equal rank-0 t*"
                )
            continue
        if kind == "scalar":
            actual_ref = parameter.logical_ref
            if actual_ref is not None and (
                not isinstance(actual_ref, str)
                or re.fullmatch(r"t[0-9]+", actual_ref) is None
            ):
                raise ValueError(
                    f"source-use-dependent scalar parameter {physical_parameter!r} "
                    "must map to rank-0 t* or remain unbound"
                )
            if actual_ref is not None and (
                actual_ref not in tensors or tensors[actual_ref].shape
            ):
                raise ValueError(
                    f"scalar parameter {physical_parameter!r} must map to a rank-0 t*"
                )
            continue
        if kind in {"opaque", "none"} and parameter.logical_ref is not None:
            raise ValueError(
                f"representation-only parameter {physical_parameter!r} cannot map "
                "to a J logical handle"
            )

    for name in contract["bound_constexpr_parameters"]:
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError(
                f"J.signature is missing bound constexpr parameter {name!r}"
            )
        if parameter.kind is not ParameterKind.CONSTEXPR:
            raise ValueError(
                f"bound constexpr parameter {name!r} must remain a constexpr carrier"
            )
        expected_dtype = contract["bound_constexpr_value_types"][name]
        if _canonical_dtype(parameter.dtype) != _canonical_dtype(expected_dtype):
            raise ValueError(
                f"bound constexpr parameter {name!r} changed its Source ABI dtype"
            )
        actual_ref = parameter.logical_ref
        expected_dimension = contract["bound_constexpr_dimension_by_parameter"].get(
            name
        )
        if expected_dimension is not None:
            relation_ok = (
                isinstance(actual_ref, str)
                and re.fullmatch(r"t[0-9]+", actual_ref) is not None
                and actual_ref in tensors
                and not tensors[actual_ref].shape
                and _value_dimension_relation_exists(
                    artifact,
                    actual_ref,
                    expected_dimension,
                )
            )
            if actual_ref != expected_dimension and not relation_ok:
                raise ValueError(
                    f"bound constexpr parameter {name!r} must map to "
                    f"{expected_dimension!r} or an explicitly equal rank-0 t*"
                )
            continue
        if actual_ref is not None and (
            not isinstance(actual_ref, str)
            or re.fullmatch(r"t[0-9]+", actual_ref) is None
            or actual_ref not in tensors
            or tensors[actual_ref].shape
        ):
            raise ValueError(
                f"bound constexpr parameter {name!r} must map to rank-0 t* or "
                "remain unbound"
            )
    actual_scalar_refs = layout["scalar_ref_names"]
    expected_scalar_refs = layout["expected_scalar_refs"]
    if actual_scalar_refs != expected_scalar_refs:
        raise ValueError(
            "optional scalar/constexpr logical values must use dense t* handles "
            "after fixed tensor inputs"
        )
    expected_input_names = layout["expected_input_names"]
    expected_output_names = layout["expected_output_names"]
    if known_roles and (
        input_names != expected_input_names
        or output_names != expected_output_names
    ):
        raise ValueError(
            "J logical tensor input/output containers differ from the trusted ABI "
            "and optional scalar-value order"
        )
    actual_unique_tensors = list(dict.fromkeys([*input_names, *output_names]))
    expected_unique_tensors = layout["expected_unique_names"]
    if actual_unique_tensors != expected_unique_tensors:
        raise ValueError(
            "J logical tensor declarations differ from the trusted physical ABI "
            "and optional scalar-value mapping"
        )

    return contract


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


def _validate_source_semantic_view(
    artifact: Gate1Artifact,
    semantic_view: Mapping[str, Any],
) -> None:
    """Validate J=A+B only against Source semantic declarations.

    This boundary deliberately has no access to schedule/grid facts or to any
    concrete tensor/scalar valuation.  It can therefore reject ABI, alias and
    symbolic-shape-contract drift without specializing J to one Harness run.
    """

    semantic_view = _expand_semantic_lift_view(semantic_view)
    abi_contract = _validate_semantic_abi_contract(artifact, semantic_view)
    signature = {item.name: item for item in artifact.j.signature.parameters}
    j_tensors = {
        item.name: item
        for item in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
    }
    if [item.to_json() for item in artifact.j.boundary.aliases] != semantic_view.get(
        "aliases", []
    ):
        raise ValueError("Lift changed trusted alias relation")

    coordinate_to_dimension = dict(abi_contract["coordinate_to_dimension"])
    dimension_to_coordinate = dict(abi_contract["dimension_to_coordinate"])

    def bind_shape_contract(
        declared: list[int | str],
        contract: Any,
        *,
        parameter_name: str,
    ) -> None:
        if not isinstance(contract, list) or len(declared) != len(contract):
            raise ValueError(
                "trusted shape contract rank differs from its declared ABI rank"
            )
        for axis, (expected, coordinate) in enumerate(
            zip(declared, contract, strict=True)
        ):
            if isinstance(coordinate, str):
                expected_dimension = coordinate_to_dimension.get(coordinate)
                if expected_dimension is None:
                    raise ValueError(
                        "trusted shape contract references an undeclared source "
                        f"coordinate {coordinate!r}"
                    )
                if not isinstance(expected, str):
                    raise ValueError(
                        "Lift specialized source-grounded shape coordinate "
                        f"{coordinate!r} at {parameter_name}[{axis}] to a literal"
                    )
                if expected != expected_dimension:
                    raise ValueError(
                        "Lift changed the deterministic h* to x* shape mapping"
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
            elif isinstance(coordinate, bool) or not isinstance(coordinate, int):
                raise ValueError(
                    "trusted shape contract entries must be h* or static integers"
                )
            elif expected != coordinate:
                raise ValueError(
                    "Lift parameterized a Harness-declared static shape extent "
                    f"at {parameter_name}[{axis}]"
                )

    raw_arguments = semantic_view.get("arguments", ())
    if not isinstance(raw_arguments, list):
        raise TypeError("semantic lift arguments must be an array")
    for argument in raw_arguments:
        if not isinstance(argument, Mapping):
            raise TypeError("semantic lift argument must be an object")
        name = argument.get("name")
        kind = argument.get("kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise TypeError("semantic lift argument requires name/kind")
        if re.fullmatch(r"c[0-9]+", name) is not None:
            # C_source declaration/completeness belongs to the independent
            # Source-realization validator below.
            continue
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a trusted compiler argument")
        expected_kind = {
            "tensor": ParameterKind.POINTER,
            "scalar": ParameterKind.SCALAR,
            "opaque": ParameterKind.OPAQUE,
            "none": ParameterKind.OPAQUE,
        }.get(kind)
        if expected_kind is None:
            raise ValueError("semantic lift argument has an unsupported kind")
        allowed_kinds = {expected_kind}
        if expected_kind is ParameterKind.POINTER:
            allowed_kinds.add(ParameterKind.TENSOR)
        if parameter.kind not in allowed_kinds:
            raise ValueError("Lift changed a trusted compiler argument kind")
        dtype = argument.get("dtype")
        if not isinstance(dtype, str) or _canonical_dtype(
            parameter.dtype
        ) != _canonical_dtype(dtype):
            raise ValueError("Lift changed a trusted compiler argument dtype")
        if kind == "tensor":
            rank = argument.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError("semantic tensor ABI requires an integer rank")
            if parameter.rank != rank:
                raise ValueError("Lift changed a trusted compiler argument rank")
            if parameter.logical_ref in j_tensors and "shape_contract" in argument:
                bind_shape_contract(
                    j_tensors[parameter.logical_ref].shape,
                    argument["shape_contract"],
                    parameter_name=name,
                )
        expected_mode = argument.get("mode")
        if expected_mode is not None and parameter.mode.value != expected_mode:
            raise ValueError("Lift changed a trusted compiler argument mode")

    expected_shape_coordinates = set(semantic_view.get("shape_coordinates", ()))
    if set(coordinate_to_dimension) != expected_shape_coordinates:
        missing = sorted(expected_shape_coordinates - set(coordinate_to_dimension))
        raise ValueError(
            "Lift omitted source-grounded shape coordinate(s) from J tensor shapes: "
            + ", ".join(missing)
        )

    def has_value_dimension_relation(logical_ref: str, dimension: str) -> bool:
        return any(
            relation.value_ref == logical_ref
            and relation.dimension_ref == dimension
            and relation.relation == "eq"
            for relation in artifact.j.value_dimension_relations
        )

    for argument in raw_arguments:
        if not isinstance(argument, Mapping):
            continue
        coordinate = argument.get("shape_coordinate")
        name = argument.get("name")
        if not isinstance(coordinate, str) or not isinstance(name, str):
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a scalar-bound source-grounded shape coordinate"
            )
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a trusted compiler argument")
        if parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and has_value_dimension_relation(logical_ref, dimension)
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness scalar-to-shape-coordinate equality"
        )

    bound_coordinates = semantic_view.get(
        "bound_constexpr_shape_coordinates", {}
    )
    if not isinstance(bound_coordinates, Mapping):
        raise TypeError(
            "bound constexpr shape-coordinate declarations must be an object"
        )
    for name, coordinate in bound_coordinates.items():
        if not isinstance(name, str) or not isinstance(coordinate, str):
            raise TypeError("bound constexpr shape-coordinate declaration is invalid")
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a bound-constexpr source-grounded shape coordinate"
            )
        parameter = signature.get(name)
        if parameter is not None and parameter.logical_ref == dimension:
            continue
        logical_ref = parameter.logical_ref if parameter is not None else None
        if (
            isinstance(logical_ref, str)
            and logical_ref in j_tensors
            and not j_tensors[logical_ref].shape
            and has_value_dimension_relation(logical_ref, dimension)
        ):
            continue
        raise ValueError(
            "Lift did not preserve the Harness "
            "bound-constexpr-to-shape-coordinate equality"
        )


def _validate_source_realization_config(
    artifact: Gate1Artifact,
    semantic_view: Mapping[str, Any],
    realization_view: Mapping[str, Any],
    layer_c_declarations: Mapping[str, Any] | None = None,
) -> None:
    """Validate exact C_source/pi_source and one Harness instantiation."""

    bound_values = realization_view.get("bound_constexpr_values", {})
    if not isinstance(bound_values, Mapping):
        raise TypeError("Source realization bound constexpr values must be an object")
    bound_constexprs = set(bound_values)
    slots = {item.name: item for item in artifact.c.slots}
    misplaced_bound_constexprs = sorted(bound_constexprs & set(slots))
    if misplaced_bound_constexprs:
        raise ValueError(
            "Lift moved bound non-C constexpr carrier(s) into C_source: "
            + ", ".join(misplaced_bound_constexprs)
        )
    parameter_order = realization_view.get("parameter_order", [])
    if not isinstance(parameter_order, list) or any(
        not isinstance(name, str)
        or re.fullmatch(r"(?:p|c)[0-9]+", name) is None
        for name in parameter_order
    ):
        raise TypeError(
            "Source realization parameter_order must be an ordered p*/c* array"
        )
    expected_p = [
        name for name in parameter_order
        if re.fullmatch(r"p[0-9]+", name) is not None
    ]
    expected_c = [
        name for name in parameter_order
        if re.fullmatch(r"c[0-9]+", name) is not None
    ]
    actual_p = [item.name for item in artifact.j.signature.parameters]
    actual_c = [item.name for item in artifact.c.slots]
    if actual_p != expected_p or actual_c != expected_c:
        raise ValueError("Lift changed Harness-declared logical/C handles")
    schedule_values = realization_view.get("schedule_values", {})
    if not isinstance(schedule_values, Mapping):
        raise TypeError("Source realization schedule_values must be an object")
    if artifact.c.values != schedule_values:
        raise ValueError("Lift changed trusted C_source values")
    actual_compiler_slots = [
        item.name for item in artifact.c.slots
        if item.kind is ScheduleSlotKind.COMPILER
    ]
    compiler_slots = realization_view.get("compiler_slots", [])
    if not isinstance(compiler_slots, list) or any(
        not isinstance(name, str)
        or re.fullmatch(r"c[0-9]+", name) is None
        for name in compiler_slots
    ) or len(compiler_slots) != len(set(compiler_slots)):
        raise TypeError("Source realization compiler_slots must be unique c* handles")
    if actual_compiler_slots != compiler_slots:
        raise ValueError("Lift changed trusted compiler slots")
    compiler_roles = realization_view.get("compiler_roles", {})
    if not isinstance(compiler_roles, Mapping):
        raise TypeError("Source realization compiler roles must be an object")
    if set(compiler_roles) != set(compiler_slots):
        raise ValueError(
            "Source realization compiler_roles must cover exactly compiler_slots"
        )
    if layer_c_declarations is not None:
        _validate_value_free_layer_c_slots(
            artifact.c.slots,
            _value_free_layer_c_declarations(layer_c_declarations),
            context="Source realization Layer-C declarations",
        )
    elif any(
        slot.subject_ref is not None or slot.axis_ref is not None
        for slot in artifact.c.slots
    ):
        raise ValueError(
            "Source realization contains a C slot attachment without a trusted "
            "declaration that exposes it"
        )
    for name, role in compiler_roles.items():
        slot = slots.get(name)
        if (
            slot is None
            or slot.kind is not ScheduleSlotKind.COMPILER
            or slot.role != role
        ):
            raise ValueError("Lift changed a public compiler-option C role")
    raw_constexpr_parameters = realization_view.get("constexpr_parameters", [])
    if not isinstance(raw_constexpr_parameters, list) or any(
        not isinstance(name, str)
        or re.fullmatch(r"(?:p|c)[0-9]+", name) is None
        for name in raw_constexpr_parameters
    ):
        raise TypeError(
            "Source realization constexpr_parameters must be an ordered p*/c* array"
        )
    expected_constexprs = set(raw_constexpr_parameters)
    semantic_bound_constexprs = semantic_view.get(
        "bound_constexpr_parameters", []
    )
    if not isinstance(semantic_bound_constexprs, list) or any(
        not isinstance(name, str)
        or re.fullmatch(r"p[0-9]+", name) is None
        for name in semantic_bound_constexprs
    ):
        raise TypeError(
            "semantic bound constexpr parameters must be an ordered p* array"
        )
    if bound_constexprs != set(semantic_bound_constexprs):
        raise ValueError(
            "Source realization bound constexpr values do not match the trusted "
            "semantic declaration"
        )
    semantic_bound_constexpr_value_types = semantic_view.get(
        "bound_constexpr_value_types", {}
    )
    realization_bound_constexpr_value_types = realization_view.get(
        "bound_constexpr_value_types", {}
    )
    if not isinstance(semantic_bound_constexpr_value_types, Mapping) or not isinstance(
        realization_bound_constexpr_value_types, Mapping
    ):
        raise TypeError("bound constexpr value types must be objects")
    if set(semantic_bound_constexpr_value_types) != bound_constexprs or set(
        realization_bound_constexpr_value_types
    ) != bound_constexprs:
        raise ValueError(
            "bound constexpr value types must cover exactly bound constexpr values"
        )
    if dict(realization_bound_constexpr_value_types) != dict(
        semantic_bound_constexpr_value_types
    ):
        raise ValueError(
            "Source realization constexpr types differ from the semantic ABI"
        )
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
    bound_constexpr_shape_coordinates = semantic_view.get(
        "bound_constexpr_shape_coordinates", {}
    )
    for name, value in bound_values.items():
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a bound semantic constexpr")
        expected_dtype = semantic_bound_constexpr_value_types[name]
        if _canonical_dtype(parameter.dtype) != _canonical_dtype(expected_dtype):
            raise ValueError(
                "Harness specialization type differs from its constexpr carrier dtype"
            )
        if not implementation_value_matches(expected_dtype, value):
            raise ValueError(
                "Harness specialization value differs from its declared Source dtype"
            )
        if isinstance(value, str) and parameter.logical_ref is not None:
            raise ValueError(
                "A string specialization carrier cannot name a J logical value"
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

    def physical_strides(
        binding: Mapping[str, Any],
        shape: list[Any],
        *,
        parameter_name: str,
    ) -> tuple[int, ...] | None:
        """Validate the physical view without giving it J ownership.

        Strides are intentionally read only from the private Source
        realization binding.  They are checked for rank and non-negativity,
        then retained solely as an ABI fact used by the Source/Replay
        compilers.  This function never writes them into a J shape, relation,
        access, or pi field.
        """

        if "strides" not in binding:
            return None
        raw = binding["strides"]
        if not isinstance(raw, list):
            raise TypeError(
                f"Source realization tensor binding {parameter_name!r} strides "
                "must be an array"
            )
        if len(raw) != len(shape):
            raise ValueError(
                f"Source realization tensor binding {parameter_name!r} stride "
                "rank differs from its shape"
            )
        result: list[int] = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Source realization tensor physical strides must be "
                    f"non-negative integers ({parameter_name!r}[{index}])"
                )
            result.append(value)
        return tuple(result)

    raw_bindings = realization_view.get("argument_bindings", [])
    if not isinstance(raw_bindings, list):
        raise TypeError("Source realization argument bindings must be an array")
    bindings: dict[str, Mapping[str, Any]] = {}
    for binding in raw_bindings:
        if not isinstance(binding, Mapping) or not isinstance(
            binding.get("name"), str
        ):
            raise TypeError("Source realization argument binding requires a name")
        name = str(binding["name"])
        if name in bindings:
            raise ValueError("Source realization argument bindings must be unique")
        unknown = set(binding) - {"name", "shape", "strides", "value"}
        if unknown:
            raise ValueError(
                f"Source realization argument binding {name!r} has unsupported "
                "field(s): " + ", ".join(sorted(unknown))
            )
        bindings[name] = binding

    semantic_arguments = semantic_view.get("arguments", [])
    if not isinstance(semantic_arguments, list):
        raise TypeError("semantic lift arguments must be an array")
    expected_binding_names = {
        str(argument.get("name"))
        for argument in semantic_arguments
        if isinstance(argument, Mapping)
        and re.fullmatch(r"p[0-9]+", str(argument.get("name"))) is not None
    }
    if set(bindings) != expected_binding_names:
        missing = sorted(expected_binding_names - set(bindings))
        extra = sorted(set(bindings) - expected_binding_names)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "Source realization argument bindings must cover exactly semantic "
            "runtime arguments" + (" (" + "; ".join(details) + ")" if details else "")
        )
    strides_by_logical: dict[str, tuple[int, ...] | None] = {}
    for argument in semantic_arguments:
        if not isinstance(argument, Mapping):
            raise TypeError("semantic lift argument must be an object")
        name = argument.get("name")
        kind = argument.get("kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise TypeError("semantic lift argument requires name/kind")
        if re.fullmatch(r"c[0-9]+", name) is not None:
            if name not in slots:
                raise ValueError("Lift omitted a trusted C argument")
            continue
        parameter = signature.get(name)
        if parameter is None:
            raise ValueError("Lift omitted a trusted compiler argument")
        binding = bindings.get(name)
        if binding is None:
            raise ValueError("Source realization omitted a trusted argument binding")
        if kind == "tensor":
            if set(binding) - {"name", "shape", "strides"} or not isinstance(
                binding.get("shape"), list
            ):
                raise TypeError(
                    f"Source realization tensor binding {name!r} requires only a "
                    "shape and optional strides"
                )
        elif kind == "scalar":
            if set(binding) != {"name", "value"}:
                raise TypeError(
                    f"Source realization scalar binding {name!r} requires exactly "
                    "name and value"
                )
            if not implementation_value_matches(
                str(argument.get("dtype")), binding.get("value")
            ):
                raise ValueError(
                    f"Source realization scalar binding {name!r} does not match "
                    "its declared dtype"
                )
        elif set(binding) != {"name"}:
            raise TypeError(
                f"Source realization {kind} binding {name!r} must contain only name"
            )
        if kind == "tensor" and parameter.logical_ref in j_tensors:
            shape = binding.get("shape")
            if not isinstance(shape, list):
                raise TypeError("Source realization tensor binding requires shape")
            strides = physical_strides(
                binding,
                shape,
                parameter_name=name,
            )
            logical_ref = str(parameter.logical_ref)
            previous_strides = strides_by_logical.setdefault(logical_ref, strides)
            if previous_strides != strides:
                raise ValueError(
                    "Source and Replay must share one physical stride valuation "
                    f"for logical tensor {logical_ref!r}"
                )
            bind_shape(
                j_tensors[parameter.logical_ref].shape,
                shape,
                argument.get("shape_contract"),
                parameter_name=name,
            )
        elif kind != "tensor" and "strides" in binding:
            raise ValueError(
                "Source realization physical strides are legal only for tensor "
                f"bindings ({name!r})"
            )
        if (
            kind == "scalar"
            and isinstance(parameter.logical_ref, str)
            and re.fullmatch(r"x[0-9]+", parameter.logical_ref)
        ):
            bind_dimension(parameter.logical_ref, binding.get("value"))
        if (
            kind == "scalar"
            and isinstance(parameter.logical_ref, str)
            and parameter.logical_ref in j_tensors
            and not j_tensors[parameter.logical_ref].shape
        ):
            logical_values[parameter.logical_ref] = binding.get("value")
    for name, value in bound_values.items():
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
            "Lift did not preserve the Harness "
            "bound-constexpr-to-shape-coordinate equality"
        )


    expected_shape_coordinates = set(
        realization_view.get("shape_coordinate_values", {})
    )
    if set(coordinate_to_dimension) != expected_shape_coordinates:
        missing = sorted(expected_shape_coordinates - set(coordinate_to_dimension))
        raise ValueError(
            "Lift omitted source-grounded shape coordinate(s) from J tensor shapes: "
            + ", ".join(missing)
        )

    for argument in semantic_arguments:
        if not isinstance(argument, Mapping):
            continue
        coordinate = argument.get("shape_coordinate")
        if not isinstance(coordinate, str):
            continue
        dimension = coordinate_to_dimension.get(coordinate)
        if dimension is None:
            raise ValueError(
                "Lift omitted a scalar-bound source-grounded shape coordinate"
            )
        name = argument.get("name")
        if not isinstance(name, str) or name not in signature:
            raise ValueError("Lift omitted a trusted compiler argument")
        parameter = signature[name]
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


def _bind_source_config(
    authored: ImplementationConfig,
    anonymous_view: Mapping[str, Any] | None,
) -> ImplementationConfig:
    """Bind private trusted Source-C values after semantic Lift authorship.

    A fresh Lift candidate must declare only the value-free Source-C interface.
    Fixed code copies no guessed value and authors no slot: it only binds the
    exact anonymous compiler-view map to the already-declared c* handles.  A
    legacy/no-view caller keeps its submitted C unchanged.
    """

    if authored.values:
        raise ValueError(
            "Lift candidate must leave C_source.values empty; BindSource owns "
            "actual trusted values"
        )
    if anonymous_view is None:
        if authored.slots:
            raise ValueError(
                "Lift declared Source-C slots without a trusted BindSource view"
            )
        return authored
    _semantic_view, realization_view = _partitioned_compiler_views(
        anonymous_view
    )
    raw_values = realization_view.get("schedule_values", {})
    raw_value_types = realization_view.get("schedule_value_types", {})
    raw_compiler_slots = realization_view.get("compiler_slots", [])
    raw_compiler_roles = realization_view.get("compiler_roles", {})
    if not isinstance(raw_values, Mapping) or any(
        not isinstance(name, str) for name in raw_values
    ):
        raise TypeError("Source realization schedule_values must be an object")
    if not isinstance(raw_value_types, Mapping) or set(raw_value_types) != set(
        raw_values
    ):
        raise TypeError(
            "Source realization schedule_value_types must cover active slots"
        )
    if not isinstance(raw_compiler_slots, list) or any(
        not isinstance(name, str) for name in raw_compiler_slots
    ):
        raise TypeError("Source realization compiler_slots must be an array")
    if not isinstance(raw_compiler_roles, Mapping) or set(
        raw_compiler_roles
    ) != set(raw_compiler_slots):
        raise TypeError(
            "Source realization compiler_roles must cover compiler slots"
        )
    raw_declarations = anonymous_view.get("layer_c_declarations")
    if isinstance(raw_declarations, Mapping):
        _validate_value_free_layer_c_slots(
            authored.slots,
            _value_free_layer_c_declarations(raw_declarations),
            context="BindSource Layer-C declarations",
        )
    elif any(
        slot.subject_ref is not None or slot.axis_ref is not None
        for slot in authored.slots
    ):
        raise ValueError(
            "BindSource received a C slot attachment without a trusted "
            "declaration that exposes it"
        )
    expected_names = list(raw_values)
    actual_names = [item.name for item in authored.slots]
    if actual_names != expected_names:
        raise ValueError(
            "Lift Source-C declarations differ from the trusted active slot order"
        )
    compiler_slots = set(raw_compiler_slots)
    for slot in authored.slots:
        expected_kind = (
            ScheduleSlotKind.COMPILER
            if slot.name in compiler_slots
            else ScheduleSlotKind.CONSTEXPR
        )
        if slot.kind is not expected_kind:
            raise ValueError(
                f"Lift Source-C slot {slot.name!r} has the wrong kind"
            )
        expected_type = raw_value_types[slot.name]
        if slot.value_type != expected_type:
            raise ValueError(
                f"Lift Source-C slot {slot.name!r} has the wrong value_type"
            )
        if not slot.required:
            raise ValueError(
                f"active Source-C slot {slot.name!r} must be required"
            )
        if slot.kind is ScheduleSlotKind.COMPILER and (
            slot.role != raw_compiler_roles[slot.name]
            or slot.subject_ref is not None
            or slot.axis_ref is not None
        ):
            raise ValueError(
                f"Lift compiler Source-C slot {slot.name!r} changed its role or attachment"
            )
    bound = ImplementationConfig(
        slots=[item.to_json() for item in authored.slots],
        values=dict(raw_values),
    )
    bound.validate_complete_values(path="BindSource.C_source")
    return bound


def _bind_source_artifact_payload(
    payload: Mapping[str, Any],
    anonymous_view: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create the complete persisted Gate1 payload after local BindSource.

    ``Gate1Artifact`` intentionally rejects missing required C values, so the
    model-authored value-free transport cannot be instantiated as that proof
    object.  This narrow boundary first parses only the closed C declaration
    object, binds trusted values, then lets the ordinary full Gate1 parser
    validate J/C/pi together.  No J or pi field is edited or synthesized.
    """

    if set(payload) != {"schema", "j", "c", "pi"}:
        raise SchemaError(
            "Lift artifact root must contain exactly schema, j, c, and pi"
        )
    raw_j = payload.get("j")
    if isinstance(raw_j, Mapping):
        nested_root_owned = _nested_root_owned_field_paths(
            raw_j,
            path="Lift artifact j",
        )
        if nested_root_owned:
            raise SchemaError(
                f"{nested_root_owned[0]} is a root-owned field nested under J; "
                "root.c and root.pi must be siblings of root.j"
            )
    raw_c = payload.get("c")
    if not isinstance(raw_c, Mapping):
        raise TypeError("Lift artifact c must be an object")
    authored = ImplementationConfig.from_json(dict(raw_c))
    bound = _bind_source_config(authored, anonymous_view)
    result = json.loads(json.dumps(payload, allow_nan=False))
    result["c"] = bound.to_json()
    return result


def _validate_compiler_view(
    artifact: Gate1Artifact,
    anonymous_view: Mapping[str, Any] | None,
) -> None:
    """Compatibility entry point with explicit semantic/realization ownership."""

    if anonymous_view is None:
        return
    semantic_view, realization_view = _partitioned_compiler_views(
        anonymous_view
    )
    _validate_source_semantic_view(artifact, semantic_view)
    _validate_source_realization_config(
        artifact,
        semantic_view,
        realization_view,
        anonymous_view.get("layer_c_declarations")
        if isinstance(anonymous_view.get("layer_c_declarations"), Mapping)
        else None,
    )


def _validate_anonymous_lift_core(
    artifact: Gate1Artifact,
    expected_parameters: tuple[str, ...],
    *,
    semantic_compiler_view: Mapping[str, Any] | None = None,
) -> None:
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
    expected_tensors: list[str]
    if semantic_compiler_view is not None:
        semantic_view = _semantic_abi_view(semantic_compiler_view)
        contract = _semantic_abi_contract(semantic_view)
        layout = _semantic_abi_tensor_layout(
            {
                item.name: item for item in artifact.j.signature.parameters
            },
            contract,
            logical_input_names=[item.name for item in artifact.j.intent.inputs],
            logical_tensors={
                item.name
                : item
                for item in [
                    *artifact.j.intent.inputs,
                    *artifact.j.intent.outputs,
                ]
            },
        )
        expected_tensors = list(layout["expected_unique_names"])
        if layout["known_roles"]:
            actual_inputs = [item.name for item in artifact.j.intent.inputs]
            actual_outputs = [item.name for item in artifact.j.intent.outputs]
            if actual_inputs != layout["expected_input_names"]:
                raise ValueError(
                    "J logical input tensor order differs from the shared "
                    "physical ABI layout after optional scalar values"
                )
            if actual_outputs != layout["expected_output_names"]:
                raise ValueError(
                    "J logical output tensor order differs from the shared "
                    "physical ABI layout after optional scalar values"
                )
    else:
        layout = _declared_abi_tensor_layout(artifact)
        expected_tensors = list(layout["expected_unique_names"])
        if layout["known_roles"]:
            actual_inputs = [item.name for item in artifact.j.intent.inputs]
            actual_outputs = [item.name for item in artifact.j.intent.outputs]
            if actual_inputs != layout["expected_input_names"]:
                raise ValueError(
                    "J logical input tensor order differs from the decoded "
                    "ABI layout after optional scalar values"
                )
            if actual_outputs != layout["expected_output_names"]:
                raise ValueError(
                    "J logical output tensor order differs from the decoded "
                    "ABI layout after optional scalar values"
                )
    if actual_tensors != expected_tensors:
        raise ValueError(
            "J tensors must use their own first-declaration-order handles "
            "(an inout output repeats its input handle) independent "
            f"of p* positions; expected {expected_tensors}, got {actual_tensors}"
        )
    if list(artifact.j.node_ids) != [
        f"n{index}" for index in range(len(artifact.j.node_ids))
    ]:
        raise ValueError("J node ids must use declaration-order handles n0, n1, ...")
    if [item.name for item in artifact.j.signature.parameters] != list(
        expected_parameters
    ):
        raise ValueError("J.signature parameters differ from anonymous source handles")
    dimensions: list[str] = []
    for tensor in tensors:
        for dim in tensor.shape:
            if isinstance(dim, str) and dim != "?" and dim not in dimensions:
                dimensions.append(dim)
    if dimensions != [f"x{index}" for index in range(len(dimensions))]:
        raise ValueError("symbolic dimensions must use x0, x1, ...")
    for prefix, values in (
        ("d", list(artifact.j.boundary.domains)),
        ("a", [item.access_id for item in artifact.j.boundary.accesses]),
        ("e", [item.effect_id for item in artifact.j.boundary.effects]),
        ("q", [item.scope_id for item in artifact.j.scopes]),
        ("b", [item.storage_id for item in artifact.j.storage]),
    ):
        if values != [f"{prefix}{index}" for index in range(len(values))]:
            raise ValueError(f"J {prefix} handles are not declaration ordered")
    if any(not re.fullmatch(r"c[0-9]+", item.name) for item in artifact.c.slots):
        raise ValueError("C_source slots must use c0, c1, ...")


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
    timeout_seconds: float | None = None,
    authoring_profile: str | None = None,
) -> SCRCExtraction:
    """Perform exactly one source-aware provider call for one Lift attempt."""

    del name_hint
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
        authoring_profile=authoring_profile,
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
            # The local parser keeps the full descriptive schema. The provider
            # receives the same machine constraints without repeated prose that
            # is already carried by the authoring contract.
            text_format=(
                scrc_lift_consistency_only_text_format()
                if authoring_profile == CONSISTENCY_ONLY_AUTHORING_PROFILE
                else scrc_lift_core_text_format(compact=True)
            ),
            timeout=resolved_timeout,
            max_output_tokens=(
                SOURCE_AWARE_LIFT_DIRECT_OUTPUT_TOKEN_BUDGET
                if resolved_effort.strip().casefold() in {"none", "low"}
                else None
            ),
        )
    except ProviderError as exc:
        provider_failure = _provider_error_failure_class(exc)
        is_timeout = _provider_error_is_timeout(exc)
        is_no_candidate = provider_failure == "provider_no_candidate"
        if not is_timeout and not is_no_candidate:
            # Authentication, HTTP, and protocol failures are infrastructure
            # failures.  They must not be mislabeled as a bad J candidate or
            # silently spent again through the logical Lift budget.
            raise
        resumable = relift_context is not None and (
            relift_context.previous_candidate is not None
            or relift_context.previous_artifact is not None
        )
        if is_timeout:
            diagnostic = (
                "source-aware Lift provider turn timed out before a candidate "
                + (
                    "was available for another independent request"
                    if not resumable
                    else "was available; the bounded logical Lift budget may "
                    "continue from the existing draft"
                )
            )
            failure_class = "lift_provider_timeout"
            termination_status = "ProviderTimeout"
        else:
            diagnostic = (
                "source-aware Lift provider completed without an assistant "
                "candidate; "
                + (
                    "no independent retry is allowed without an existing draft"
                    if not resumable
                    else "the bounded logical Lift budget may continue from the "
                    "existing draft"
                )
            )
            failure_class = "lift_provider_no_candidate"
            termination_status = "ProviderNoCandidate"
        attempt = SCRCLiftAttempt(
            model=resolved_provider.model,
            diagnostic=diagnostic,
            response_text="",
            request_sha256=exc.request_sha256,
            attempts=exc.attempts,
            elapsed_ms=exc.elapsed_ms,
            termination_status=termination_status,
        )
        feedback = {
            "diagnostic": diagnostic,
            "failure_class": failure_class,
            "retryable": resumable,
            "provider_failure_class": provider_failure,
        }
        if is_timeout:
            feedback["timeout_seconds"] = resolved_timeout
        raise SCRCLiftError(
            attempt,
            feedback=feedback,
            failure_class=failure_class,
            terminal=not resumable,
            candidate_draft=(
                relift_context.previous_candidate
                if relift_context is not None
                else None
            ),
        ) from exc
    candidate_context = scrc_candidate_check_context(request)
    candidate_check = check_scrc_lift_candidate(
        reply.text,
        candidate_context,
    )
    if candidate_check["status"] == "pass":
        try:
            # This privacy check needs the privileged alpha-renaming map, but
            # it still decodes only the value-free candidate.  Its diagnostic
            # exposes an artifact path, never the original Source spelling.
            provisional = _decode_value_free_lift_candidate(
                reply.text,
                expected_parameters=candidate_context.expected_parameters,
                semantic_compiler_view=request.semantic_compiler_view,
                required_schema=Gate1Artifact.SCHEMA,
            )
            _validate_source_identifier_privacy(
                provisional,
                request.harness_binding,
            )
        except SourceIdentifierPrivacyError as exc:
            candidate_check = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "source_identifier_privacy",
                        "path": exc.path,
                        "observed_refs_or_types": [
                            "replace the free-form Source spelling with an "
                            "anonymous handle or source-neutral protocol value"
                        ],
                    }
                ],
            }
    if candidate_check["status"] != "pass":
        candidate_draft = _candidate_draft_for_relift(reply.text)
        attempt = SCRCLiftAttempt(
            model=reply.model,
            diagnostic="value-free CandidateCheck rejected the Lift draft",
            response_text=reply.text,
            response_id=reply.raw.get("id"),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            usage=_response_usage(reply.raw),
            request_sha256=reply.request_sha256,
            response_sha256=(
                reply.response_sha256
                or hashlib.sha256(reply.text.encode()).hexdigest()
            ),
            raw_response_sha256=reply.raw_response_sha256,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            elapsed_ms=reply.elapsed_ms,
        )
        raise SCRCLiftError(
            attempt,
            feedback={
                "diagnostic": attempt.diagnostic,
                "failure_class": "candidate_check_rejected",
                "candidate_check": candidate_check,
            },
            failure_class="candidate_check_rejected",
            candidate_draft=candidate_draft,
        )
    try:
        artifact = decode_scrc_lift_artifact(reply.text, request)
    except Exception as exc:
        attempt = SCRCLiftAttempt(
            model=reply.model,
            diagnostic="host-private BindSource rejected the accepted Lift draft",
            response_text=reply.text,
            response_id=reply.raw.get("id"),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            usage=_response_usage(reply.raw),
            request_sha256=reply.request_sha256,
            response_sha256=(
                reply.response_sha256
                or hashlib.sha256(reply.text.encode()).hexdigest()
            ),
            raw_response_sha256=reply.raw_response_sha256,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            elapsed_ms=reply.elapsed_ms,
        )
        raise SCRCLiftError(
            attempt,
            feedback={
                "diagnostic": attempt.diagnostic,
                "failure_class": "bind_source_rejected",
                "authority": (
                    "host-private BindSource/Gate boundary; no Source "
                    "realization diagnostic is returned to the Lift owner"
                ),
            },
            failure_class="bind_source_rejected",
            terminal=True,
        ) from exc
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
        response_sha256=(
            reply.response_sha256
            or hashlib.sha256(reply.text.encode()).hexdigest()
        ),
        raw_response_sha256=reply.raw_response_sha256,
        response_text=reply.text,
        raw_response=reply.raw_response,
        attempts=reply.attempts,
        elapsed_ms=reply.elapsed_ms,
    )
    return SCRCExtraction(
        artifact,
        provenance,
        request.harness_binding,
        source,
        source_layer_c_interface(artifact.j, artifact.c),
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
    timeout_seconds: float | None = None
    authoring_profile: str | None = None
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

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
        context = SCRCReliftContext(
            round_index,
            failure_kind,
            diagnostics or {},
            previous.artifact.to_json(),
        )
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
                timeout_seconds=self.timeout_seconds,
                authoring_profile=self.authoring_profile,
            )
        except SCRCLiftError as exc:
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                    rejected_attempts=exc.attempts,
                    continuation_candidate=exc.candidate_draft,
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
    timeout_seconds: float | None = None
    authoring_profile: str | None = None
    _records: list[SCRCReliftRecord] = field(default_factory=list, init=False)

    @property
    def records(self) -> tuple[SCRCReliftRecord, ...]:
        return tuple(self._records)

    def lift(self, request: Any, feedback: Any) -> Any:
        source_program = getattr(request, "source", None)
        source = getattr(source_program, "text", None)
        if not isinstance(source, str) or not source:
            raise TypeError("Responses SCRC Lift requires SourceProgram.text")
        resolved_provider = self.provider or get_provider()
        resolved_effort = self.effort or resolved_provider.reasoning_effort
        resolved_timeout = resolve_source_aware_lift_timeout(
            resolved_effort,
            self.timeout_seconds,
        )
        round_index = len(self._records) + 1
        context = None
        if feedback is not None:
            kind = getattr(getattr(feedback, "kind", None), "value", None)
            diagnostics = getattr(feedback, "diagnostics", None)
            previous_artifact = getattr(feedback, "previous_artifact", None)
            previous_candidate = getattr(feedback, "previous_candidate", None)
            context = SCRCReliftContext(
                round_index,
                kind,
                diagnostics if isinstance(diagnostics, Mapping) else {},
                previous_artifact=(
                    previous_artifact.to_json()
                    if isinstance(previous_artifact, Gate1Artifact)
                    else None
                ),
                previous_candidate=(
                    dict(previous_candidate)
                    if isinstance(previous_candidate, Mapping)
                    else None
                ),
            )
        try:
            extraction = extract_scrc_artifact(
                source,
                provider=resolved_provider,
                compiler_view=self.compiler_view,
                effort=resolved_effort,
                relift_context=context,
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
                timeout_seconds=resolved_timeout,
                authoring_profile=self.authoring_profile,
            )
        except SCRCLiftError as exc:
            diagnostic = "; ".join(
                item.diagnostic for item in exc.attempts
            )
            self._records.append(
                SCRCReliftRecord(
                    round_index,
                    diagnostic=diagnostic,
                    rejected_attempts=exc.attempts,
                )
            )
            # Local import avoids making extraction depend on the verify package
            # during module initialization.
            from intentir.verify.scrc_runner import CandidateLiftError

            raise CandidateLiftError(
                exc.feedback,
                physical_calls=max(item.attempts for item in exc.attempts),
                terminal=exc.terminal,
                candidate_draft=exc.candidate_draft,
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
                "response_id": extraction.provenance.response_id,
                "usage": dict(extraction.provenance.usage),
                "elapsed_ms": extraction.provenance.elapsed_ms,
                "lift_timeout_seconds": resolved_timeout,
                "artifact_schema": extraction.artifact.wire_schema,
                "layer_c_interface": extraction.layer_c_interface_payload,
                "layer_c_interface_digest": (
                    extraction.layer_c_interface_digest
                ),
                "decode_protocol": SCRC_DECODE_PROTOCOL,
                "typecheck_protocol": TYPECHECK_PROTOCOL,
            },
        )


__all__ = [
    "ResponsesSCRCLiftAdapter", "ResponsesSCRCRelifter",
    "CANDIDATE_CHECK_PROTOCOL", "SCRC_DECODE_PROTOCOL", "SCRC_RELIFT_PROTOCOL",
    "SCRCCandidateCheckContext", "SCRCCompilerArgument",
    "SCRCCompilerView", "SCRCExtraction", "SCRCHarnessBinding",
    "SCRCLiftAttempt", "SCRCLiftError", "SCRCLiftRequest", "SCRCProvenance", "SCRCReliftContext",
    "SourceIdentifierPrivacyError",
    "SCRCReliftRecord", "anonymize_native_source", "anonymize_source",
    "build_scrc_lift_instructions",
    "canonical_scalar_binding_dtype",
    "check_scrc_lift_candidate", "decode_scrc_lift_artifact", "extract_scrc_artifact", "prepare_scrc_lift", "scrc_candidate_check_context", "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
]
