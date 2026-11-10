"""Backend-neutral, fail-closed contracts for target Layer-C plugins.

Layer A/B remain canonical J.  A target Layer-C plugin is deliberately small:
it binds J-owned extension points to closed scalar values and selects one
opaque backend candidate.  It cannot carry source text, compiler arguments,
passes, commands, or an open-ended schedule object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical_construction import (
    CANONICAL_C_WIRE_FIELDS,
)
from .dim_expr import dimension_expression_references

LayerCScalar = bool | int | float | str
LayerCAxisRef = int | str

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:+-]+")
_CLOSED_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:+/-]*")
_CANDIDATE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]*")
_ROLE = re.compile(r"[a-z][a-z0-9_]*")
_SUBJECT_REF = re.compile(r"[tndaequb][0-9]+")
_AXIS_REF = re.compile(r"x[0-9]+")
_AUTH_REF = re.compile(r"u[0-9]+")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RECORD_ID = re.compile(r"(?:memory_layout|parallel_mapping|pipeline|fusion|split):[0-9]+|lowering")


class LayerCResolutionError(ValueError):
    """Raised when Layer C is incomplete, stale, open-ended, or mismatched."""


def _error(message: str) -> LayerCResolutionError:
    return LayerCResolutionError(message)


def _strict_object(
    value: Any,
    *,
    keys: set[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{context} must be an object")
    actual = set(value)
    if actual != keys or any(not isinstance(key, str) for key in value):
        missing = sorted(keys - actual)
        extra = sorted(actual - keys, key=str)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(map(str, extra)))
        raise _error(
            f"{context} must have exactly {', '.join(sorted(keys))}"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise _error(f"{context} must be a closed identifier")
    return value


def _closed_token(value: Any, context: str) -> str:
    if not isinstance(value, str) or _CLOSED_TOKEN.fullmatch(value) is None:
        raise _error(f"{context} must be a closed token, not free text or source")
    return value


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _assert_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(f"{path} contains a non-finite JSON number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(f"{path} contains a non-string object key")
            _assert_json_value(item, f"{path}.{key}")
        return
    raise _error(f"{path} is not a strict JSON value")


def canonical_layer_c_json(value: Any) -> str:
    """Return the byte-stable JSON spelling used by Layer-C digests."""

    _assert_json_value(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # defensive after the strict walk
        raise _error(f"value cannot be encoded as canonical JSON: {exc}") from exc


def layer_c_content_digest(value: Any) -> str:
    """Return SHA-256 over :func:`canonical_layer_c_json`."""

    return hashlib.sha256(canonical_layer_c_json(value).encode("utf-8")).hexdigest()


def target_artifact_digest(artifact: bytes | bytearray | memoryview | str) -> str:
    """Digest bytes from a real target-lowering artifact.

    The helper intentionally accepts bytes or already-materialized text only.
    Backend callers must obtain those bytes from the compiler-produced MLIR,
    object, shared object, PTX, or cubin before constructing consumption
    evidence; a caller-authored digest is not an accepted substitute.
    """

    if isinstance(artifact, str):
        payload = artifact.encode("utf-8")
    elif isinstance(artifact, (bytes, bytearray, memoryview)):
        payload = bytes(artifact)
    else:
        raise _error("target lowering artifact must be bytes or text")
    if not payload:
        raise _error("target lowering artifact must not be empty")
    return hashlib.sha256(payload).hexdigest()


def capability_snapshot_digest(
    target: Any,
    capabilities: Iterable[str],
) -> str:
    """Digest one exact target identity plus its backend-derived capability set."""

    target_identity = (
        target
        if isinstance(target, TargetIdentity)
        else TargetIdentity.from_target(target)
    )
    values = _canonical_unique(
        capabilities,
        validator=_identifier,
        context="capability_snapshot.capabilities",
    )
    return layer_c_content_digest(
        {
            "target": target_identity.to_dict(),
            "capabilities": list(values),
        }
    )


def _decode_json_object(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in items:
                if key in result:
                    raise _error(f"{context} contains duplicate key {key!r}")
                result[key] = item
            return result

        def constant(token: str) -> None:
            raise _error(f"{context} contains non-finite number {token}")

        try:
            decoded = json.loads(
                value,
                object_pairs_hook=pairs,
                parse_constant=constant,
            )
        except LayerCResolutionError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _error(f"{context} is not valid JSON: {exc}") from exc
        value = decoded
    if not isinstance(value, Mapping):
        raise _error(f"{context} must be a JSON object")
    return value


def _canonical_unique(
    values: Iterable[Any],
    *,
    validator: Any,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise _error(f"{context} must be an array")
    try:
        checked = tuple(validator(item, context) for item in values)
    except TypeError as exc:
        raise _error(f"{context} must be an array") from exc
    if len(checked) != len(set(checked)):
        raise _error(f"{context} must contain unique values")
    return tuple(sorted(checked))


def _value_type(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise _error(f"{context} must name a closed scalar type")
    normalized = value.casefold()
    if normalized in {
        "bool", "i1", "int", "integer", "float", "number",
        "str", "string", "layout", "enum",
    }:
        return normalized
    if re.fullmatch(r"[iu][1-9][0-9]*", normalized) is not None:
        return normalized
    if re.fullmatch(r"f[1-9][0-9]*", normalized) is not None:
        return normalized
    raise _error(f"{context} names an unsupported or open-ended scalar type")


def _scalar(value: Any, context: str) -> LayerCScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(f"{context} must be finite")
        return value
    if isinstance(value, str):
        return _closed_token(value, context)
    raise _error(f"{context} must be a closed bool, integer, number, or token")


def _value_matches(value_type: str, value: LayerCScalar) -> bool:
    normalized = _value_type(value_type, "Layer-C value_type")
    if normalized in {"bool", "i1"}:
        return isinstance(value, bool)
    if normalized in {"int", "integer"} or re.fullmatch(
        r"[iu][1-9][0-9]*", normalized
    ):
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"float", "number"} or re.fullmatch(
        r"f[1-9][0-9]*", normalized
    ):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str) and _CLOSED_TOKEN.fullmatch(value) is not None


@dataclass(frozen=True)
class TargetIdentity:
    """Exact backend identity to which one Layer-C plugin is bound."""

    family: str
    architecture: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, "target.family"))
        object.__setattr__(
            self,
            "architecture",
            _identifier(self.architecture, "target.architecture"),
        )
        object.__setattr__(
            self,
            "features",
            _canonical_unique(
                self.features,
                validator=_identifier,
                context="target.features",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetIdentity":
        data = _strict_object(
            value,
            keys={"family", "architecture", "features"},
            context="TargetIdentity",
        )
        if not isinstance(data["features"], list):
            raise _error("TargetIdentity.features must be an array")
        return cls(
            family=data["family"],
            architecture=data["architecture"],
            features=tuple(data["features"]),
        )

    @classmethod
    def from_json(cls, value: Any) -> "TargetIdentity":
        return cls.from_dict(_decode_json_object(value, "TargetIdentity"))

    @classmethod
    def from_target(cls, value: Any) -> "TargetIdentity":
        """Normalize a target without importing ``backend.protocol``."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        try:
            family = value.family
            architecture = value.architecture
            features = value.features
        except AttributeError as exc:
            raise _error(
                "lowering target must expose family, architecture, and features"
            ) from exc
        if isinstance(features, (str, bytes)):
            raise _error("lowering target features must be a sequence")
        return cls(family, architecture, tuple(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "architecture": self.architecture,
            "features": list(self.features),
        }

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class LayerCExtensionPoint:
    """One J-addressed placeholder declaration, without its actual value."""

    slot: str
    role: str
    value_type: str
    required: bool = True
    subject_ref: str | None = None
    axis_ref: LayerCAxisRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _closed_token(self.slot, "extension.slot"))
        if not isinstance(self.role, str) or _ROLE.fullmatch(self.role) is None:
            raise _error("extension.role must use lower_snake_case")
        object.__setattr__(
            self,
            "value_type",
            _value_type(self.value_type, "extension.value_type"),
        )
        if not isinstance(self.required, bool):
            raise _error("extension.required must be boolean")
        if self.subject_ref is not None:
            if (
                not isinstance(self.subject_ref, str)
                or _SUBJECT_REF.fullmatch(self.subject_ref) is None
            ):
                raise _error(
                    "extension.subject_ref must be a stable J ref "
                    "t*/n*/d*/a*/e*/q*/u*/b*"
                )
        if self.axis_ref is not None:
            if self.subject_ref is None:
                raise _error("extension.axis_ref requires extension.subject_ref")
            if isinstance(self.axis_ref, bool) or not isinstance(
                self.axis_ref, (int, str)
            ):
                raise _error("extension.axis_ref must be non-negative or x*")
            if isinstance(self.axis_ref, int):
                if self.axis_ref < 0:
                    raise _error("extension.axis_ref must be non-negative")
            elif _AXIS_REF.fullmatch(self.axis_ref) is None:
                raise _error("extension.axis_ref symbolic dimensions must use x*")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerCExtensionPoint":
        required = {"slot", "role", "value_type", "required"}
        optional = {"subject_ref", "axis_ref"}
        if not isinstance(value, Mapping):
            raise _error("LayerCExtensionPoint must be an object")
        actual = set(value)
        if not required <= actual or actual - required - optional:
            raise _error(
                "LayerCExtensionPoint must contain slot, role, value_type, required "
                "and only optional subject_ref/axis_ref"
            )
        return cls(
            slot=value["slot"],
            role=value["role"],
            value_type=value["value_type"],
            required=value["required"],
            subject_ref=value.get("subject_ref"),
            axis_ref=value.get("axis_ref"),
        )

    @classmethod
    def from_json(cls, value: Any) -> "LayerCExtensionPoint":
        return cls.from_dict(_decode_json_object(value, "LayerCExtensionPoint"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "slot": self.slot,
            "role": self.role,
            "value_type": self.value_type,
            "required": self.required,
        }
        if self.subject_ref is not None:
            result["subject_ref"] = self.subject_ref
        if self.axis_ref is not None:
            result["axis_ref"] = self.axis_ref
        return result

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class LayerCInterface:
    """The complete, digestible set of A/B extension-point declarations."""

    points: tuple[LayerCExtensionPoint, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.points, (str, bytes)):
            raise _error("LayerCInterface.points must be an array")
        try:
            points = tuple(
                item
                if isinstance(item, LayerCExtensionPoint)
                else LayerCExtensionPoint.from_dict(item)
                for item in self.points
            )
        except TypeError as exc:
            raise _error("LayerCInterface.points must be an array") from exc
        slots = [item.slot for item in points]
        if len(slots) != len(set(slots)):
            raise _error("Layer-C extension slots must be unique")
        # ``points`` is a declaration sequence, not a set.  Backend interface
        # and manifest consumers may depend on declaration order; sorting
        # opaque slot names (especially c10 before c2) would change that
        # contract. Set-like consumers may build keyed views, but the
        # interface itself must preserve declaration order.
        object.__setattr__(self, "points", points)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerCInterface":
        data = _strict_object(
            value,
            keys={"points"},
            context="LayerCInterface",
        )
        if not isinstance(data["points"], list):
            raise _error("LayerCInterface.points must be an array")
        return cls(tuple(LayerCExtensionPoint.from_dict(item) for item in data["points"]))

    @classmethod
    def from_json(cls, value: Any) -> "LayerCInterface":
        return cls.from_dict(_decode_json_object(value, "LayerCInterface"))

    def to_dict(self) -> dict[str, Any]:
        return {"points": [item.to_dict() for item in self.points]}

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


def _interface(value: Any) -> LayerCInterface:
    if isinstance(value, LayerCInterface):
        return value
    if isinstance(value, Mapping):
        return LayerCInterface.from_dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return LayerCInterface(tuple(value))
    raise _error("Layer-C interface must be LayerCInterface or an extension array")


def binding_interface_digest(value: Any) -> str:
    """Digest declarations only; actual binding values are deliberately absent."""

    return layer_c_content_digest(_interface(value).to_dict())


layer_c_interface_digest = binding_interface_digest


@dataclass(frozen=True)
class LayerCBinding:
    """One actual closed value for a separately declared extension slot."""

    slot: str
    value: LayerCScalar

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _closed_token(self.slot, "binding.slot"))
        object.__setattr__(self, "value", _scalar(self.value, "binding.value"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerCBinding":
        data = _strict_object(
            value,
            keys={"slot", "value"},
            context="LayerCBinding",
        )
        return cls(slot=data["slot"], value=data["value"])

    @classmethod
    def from_json(cls, value: Any) -> "LayerCBinding":
        return cls.from_dict(_decode_json_object(value, "LayerCBinding"))

    def to_dict(self) -> dict[str, Any]:
        return {"slot": self.slot, "value": self.value}

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class ImplementationChoice:
    """Opaque registry selection; never a pass string or source-code channel."""

    backend_kind: str
    candidate_key: str
    candidate_digest: str
    authorization_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_kind",
            _identifier(self.backend_kind, "choice.backend_kind"),
        )
        if (
            not isinstance(self.candidate_key, str)
            or _CANDIDATE_KEY.fullmatch(self.candidate_key) is None
            or len(self.candidate_key) > 1024
        ):
            raise _error(
                "choice.candidate_key must be a bounded opaque registry key; "
                "pass text and compiler arguments are forbidden"
            )
        object.__setattr__(
            self,
            "candidate_digest",
            _digest(self.candidate_digest, "choice.candidate_digest"),
        )
        object.__setattr__(
            self,
            "authorization_refs",
            _canonical_unique(
                self.authorization_refs,
                validator=lambda item, context: (
                    item
                    if isinstance(item, str) and _AUTH_REF.fullmatch(item) is not None
                    else (_ for _ in ()).throw(
                        _error(f"{context} entries must be authorization refs u*")
                    )
                ),
                context="choice.authorization_refs",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImplementationChoice":
        data = _strict_object(
            value,
            keys={
                "backend_kind", "candidate_key", "candidate_digest",
                "authorization_refs",
            },
            context="ImplementationChoice",
        )
        if not isinstance(data["authorization_refs"], list):
            raise _error("ImplementationChoice.authorization_refs must be an array")
        return cls(
            backend_kind=data["backend_kind"],
            candidate_key=data["candidate_key"],
            candidate_digest=data["candidate_digest"],
            authorization_refs=tuple(data["authorization_refs"]),
        )

    @classmethod
    def from_json(cls, value: Any) -> "ImplementationChoice":
        return cls.from_dict(_decode_json_object(value, "ImplementationChoice"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "candidate_key": self.candidate_key,
            "candidate_digest": self.candidate_digest,
            "authorization_refs": list(self.authorization_refs),
        }

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


def _optional_contract_subject(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or _SUBJECT_REF.fullmatch(value) is None
    ):
        raise _error(
            f"{context} must be null or a J subject handle "
            "(t*/n*/d*/a*/e*/q*/b*/u*)"
        )
    return value


def _optional_contract_axis(value: Any, context: str) -> LayerCAxisRef | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise _error(f"{context} must be null, a non-negative integer, or x*")
    if isinstance(value, int):
        if value < 0:
            raise _error(f"{context} must be non-negative")
        return value
    if isinstance(value, str) and _AXIS_REF.fullmatch(value) is not None:
        return value
    raise _error(f"{context} must be null, a non-negative integer, or x*")


def _contract_slot(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _closed_token(value, context)


def _positive_contract_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(f"{context} must be a positive integer")
    return value


def _contract_record_list(
    values: Any,
    record_type: Any,
    context: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise _error(f"{context} must be an array")
    try:
        result = tuple(
            item if isinstance(item, record_type) else record_type.from_dict(item)
            for item in values
        )
    except TypeError as exc:
        raise _error(f"{context} must be an array") from exc
    return result


def _contract_integer_value_type(value_type: str) -> bool:
    normalized = _value_type(value_type, "Layer-C contract value_type")
    return normalized in {"int", "integer"} or re.fullmatch(
        r"[iu][1-9][0-9]*", normalized
    ) is not None


def _contract_token_value_type(value_type: str) -> bool:
    normalized = _value_type(value_type, "Layer-C contract value_type")
    return normalized in {"str", "string", "layout", "enum"}


_TARGET_CONTRACT_ROLES = {
    "memory_layout": frozenset(
        {"memory_layout", "tensor_layout", "layout", "memory_space"}
    ),
    "parallel_mapping": frozenset(
        {"parallel_mapping", "parallel_tile", "thread_mapping", "loop_schedule_tile"}
    ),
    "pipeline": frozenset(
        {"pipeline", "pipeline_depth", "stage"}
    ),
    "split": frozenset(
        {"split", "strip_mine", "tile_extent", "reduction_tile_extent", "loop_schedule_tile"}
    ),
}


@dataclass(frozen=True)
class TargetMemoryLayout:
    """One typed memory/layout realization attached to J or an E_C slot."""

    slot: str | None
    subject_ref: str | None
    axis_ref: LayerCAxisRef | None
    layout: str
    memory_space: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _contract_slot(self.slot, "memory_layout.slot"))
        object.__setattr__(
            self,
            "subject_ref",
            _optional_contract_subject(self.subject_ref, "memory_layout.subject_ref"),
        )
        object.__setattr__(
            self,
            "axis_ref",
            _optional_contract_axis(self.axis_ref, "memory_layout.axis_ref"),
        )
        if self.axis_ref is not None and self.subject_ref is None:
            raise _error("memory_layout.axis_ref requires memory_layout.subject_ref")
        object.__setattr__(self, "layout", _closed_token(self.layout, "memory_layout.layout"))
        object.__setattr__(
            self,
            "memory_space",
            _closed_token(self.memory_space, "memory_layout.memory_space"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetMemoryLayout":
        data = _strict_object(
            value,
            keys={"slot", "subject_ref", "axis_ref", "layout", "memory_space"},
            context="TargetMemoryLayout",
        )
        return cls(
            data["slot"],
            data["subject_ref"],
            data["axis_ref"],
            data["layout"],
            data["memory_space"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "subject_ref": self.subject_ref,
            "axis_ref": self.axis_ref,
            "layout": self.layout,
            "memory_space": self.memory_space,
        }


@dataclass(frozen=True)
class TargetParallelMapping:
    """One typed logical-axis to target-parallel mapping."""

    slot: str | None
    subject_ref: str
    axis_ref: LayerCAxisRef
    mapping: str
    factor: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _contract_slot(self.slot, "parallel_mapping.slot"))
        object.__setattr__(
            self,
            "subject_ref",
            _optional_contract_subject(self.subject_ref, "parallel_mapping.subject_ref"),
        )
        if self.subject_ref is None:
            raise _error("parallel_mapping.subject_ref is required")
        axis = _optional_contract_axis(self.axis_ref, "parallel_mapping.axis_ref")
        if axis is None:
            raise _error("parallel_mapping.axis_ref is required")
        object.__setattr__(self, "axis_ref", axis)
        object.__setattr__(self, "mapping", _closed_token(self.mapping, "parallel_mapping.mapping"))
        if self.factor is not None:
            object.__setattr__(
                self,
                "factor",
                _positive_contract_integer(self.factor, "parallel_mapping.factor"),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetParallelMapping":
        data = _strict_object(
            value,
            keys={"slot", "subject_ref", "axis_ref", "mapping", "factor"},
            context="TargetParallelMapping",
        )
        return cls(
            data["slot"],
            data["subject_ref"],
            data["axis_ref"],
            data["mapping"],
            data["factor"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "subject_ref": self.subject_ref,
            "axis_ref": self.axis_ref,
            "mapping": self.mapping,
            "factor": self.factor,
        }


@dataclass(frozen=True)
class TargetPipeline:
    """One closed pipeline stage/depth declaration."""

    stage: str
    depth: int
    slot: str | None = None
    subject_ref: str | None = None
    axis_ref: LayerCAxisRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _closed_token(self.stage, "pipeline.stage"))
        object.__setattr__(self, "depth", _positive_contract_integer(self.depth, "pipeline.depth"))
        object.__setattr__(self, "slot", _contract_slot(self.slot, "pipeline.slot"))
        object.__setattr__(
            self,
            "subject_ref",
            _optional_contract_subject(self.subject_ref, "pipeline.subject_ref"),
        )
        object.__setattr__(
            self,
            "axis_ref",
            _optional_contract_axis(self.axis_ref, "pipeline.axis_ref"),
        )
        if self.axis_ref is not None and self.subject_ref is None:
            raise _error("pipeline.axis_ref requires pipeline.subject_ref")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetPipeline":
        data = _strict_object(
            value,
            keys={"stage", "depth", "slot", "subject_ref", "axis_ref"},
            context="TargetPipeline",
        )
        return cls(
            data["stage"],
            data["depth"],
            data["slot"],
            data["subject_ref"],
            data["axis_ref"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "depth": self.depth,
            "slot": self.slot,
            "subject_ref": self.subject_ref,
            "axis_ref": self.axis_ref,
        }


@dataclass(frozen=True)
class TargetFusion:
    """One closed fusion relation over J subjects."""

    subject_refs: tuple[str, ...]
    mode: str

    def __post_init__(self) -> None:
        refs = tuple(
            _optional_contract_subject(item, "fusion.subject_refs entry")
            for item in self.subject_refs
        )
        if len(refs) < 2 or any(item is None for item in refs):
            raise _error("fusion.subject_refs must contain at least two J subjects")
        if len(refs) != len(set(refs)):
            raise _error("fusion.subject_refs must be unique")
        object.__setattr__(self, "subject_refs", tuple(refs))
        object.__setattr__(self, "mode", _closed_token(self.mode, "fusion.mode"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetFusion":
        data = _strict_object(
            value,
            keys={"subject_refs", "mode"},
            context="TargetFusion",
        )
        if not isinstance(data["subject_refs"], list):
            raise _error("TargetFusion.subject_refs must be an array")
        return cls(tuple(data["subject_refs"]), data["mode"])

    def to_dict(self) -> dict[str, Any]:
        return {"subject_refs": list(self.subject_refs), "mode": self.mode}


@dataclass(frozen=True)
class TargetSplit:
    """One typed logical-axis split/strip-mine declaration."""

    slot: str | None
    subject_ref: str
    axis_ref: LayerCAxisRef
    factor: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _contract_slot(self.slot, "split.slot"))
        object.__setattr__(
            self,
            "subject_ref",
            _optional_contract_subject(self.subject_ref, "split.subject_ref"),
        )
        if self.subject_ref is None:
            raise _error("split.subject_ref is required")
        axis = _optional_contract_axis(self.axis_ref, "split.axis_ref")
        if axis is None:
            raise _error("split.axis_ref is required")
        object.__setattr__(self, "axis_ref", axis)
        object.__setattr__(self, "factor", _positive_contract_integer(self.factor, "split.factor"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetSplit":
        data = _strict_object(
            value,
            keys={"slot", "subject_ref", "axis_ref", "factor"},
            context="TargetSplit",
        )
        return cls(data["slot"], data["subject_ref"], data["axis_ref"], data["factor"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "subject_ref": self.subject_ref,
            "axis_ref": self.axis_ref,
            "factor": self.factor,
        }


@dataclass(frozen=True)
class TargetLowering:
    """The exact registry identity of the target implementation."""

    backend_kind: str
    candidate_key: str
    candidate_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_kind", _identifier(self.backend_kind, "lowering.backend_kind"))
        if (
            not isinstance(self.candidate_key, str)
            or _CANDIDATE_KEY.fullmatch(self.candidate_key) is None
            or len(self.candidate_key) > 1024
        ):
            raise _error("lowering.candidate_key must be a bounded registry key")
        object.__setattr__(
            self,
            "candidate_digest",
            _digest(self.candidate_digest, "lowering.candidate_digest"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetLowering":
        data = _strict_object(
            value,
            keys={"backend_kind", "candidate_key", "candidate_digest"},
            context="TargetLowering",
        )
        return cls(data["backend_kind"], data["candidate_key"], data["candidate_digest"])

    def to_dict(self) -> dict[str, str]:
        return {
            "backend_kind": self.backend_kind,
            "candidate_key": self.candidate_key,
            "candidate_digest": self.candidate_digest,
        }


@dataclass(frozen=True)
class TargetRecordConsumption:
    """One independently auditable application of a target record.

    A record id alone only says that a backend repeated a declaration.  This
    structure also identifies the concrete consumer phase and the compiler
    artifact that carried the application.  ``application_digest`` is derived
    from the exact typed record content, phase, artifact kind, and artifact
    digest, so changing any of those facts invalidates the evidence.
    """

    record_id: str
    consumer: str
    phase: str
    artifact_kind: str
    artifact_digest: str
    application_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or _RECORD_ID.fullmatch(self.record_id) is None:
            raise _error("target record consumption.record_id is invalid")
        object.__setattr__(self, "consumer", _closed_token(self.consumer, "target record consumption.consumer"))
        object.__setattr__(self, "phase", _closed_token(self.phase, "target record consumption.phase"))
        object.__setattr__(self, "artifact_kind", _closed_token(self.artifact_kind, "target record consumption.artifact_kind"))
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, "target record consumption.artifact_digest"),
        )
        object.__setattr__(
            self,
            "application_digest",
            _digest(
                self.application_digest,
                "target record consumption.application_digest",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetRecordConsumption":
        data = _strict_object(
            value,
            keys={
                "record_id",
                "consumer",
                "phase",
                "artifact_kind",
                "artifact_digest",
                "application_digest",
            },
            context="TargetRecordConsumption",
        )
        return cls(
            data["record_id"],
            data["consumer"],
            data["phase"],
            data["artifact_kind"],
            data["artifact_digest"],
            data["application_digest"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "consumer": self.consumer,
            "phase": self.phase,
            "artifact_kind": self.artifact_kind,
            "artifact_digest": self.artifact_digest,
            "application_digest": self.application_digest,
        }


@dataclass(frozen=True)
class TargetRealizationContract:
    """Closed target realization grammar shared by every production backend.

    The records intentionally contain only typed tokens, bounded integers, J
    handles, E_C slot handles, and the exact opaque candidate identity.  Source
    text, compiler commands, pass strings, and arbitrary schedule mappings have
    no representation in this contract.
    """

    memory_layout: tuple[TargetMemoryLayout, ...] = ()
    parallel_mapping: tuple[TargetParallelMapping, ...] = ()
    pipeline: tuple[TargetPipeline, ...] = ()
    fusion: tuple[TargetFusion, ...] = ()
    split: tuple[TargetSplit, ...] = ()
    lowering: TargetLowering | None = None

    def __post_init__(self) -> None:
        for field_name, record_type in (
            ("memory_layout", TargetMemoryLayout),
            ("parallel_mapping", TargetParallelMapping),
            ("pipeline", TargetPipeline),
            ("fusion", TargetFusion),
            ("split", TargetSplit),
        ):
            object.__setattr__(
                self,
                field_name,
                _contract_record_list(
                    getattr(self, field_name),
                    record_type,
                    f"TargetRealizationContract.{field_name}",
                ),
            )
        lowering = self.lowering
        if lowering is not None and not isinstance(lowering, TargetLowering):
            lowering = TargetLowering.from_dict(lowering)
            object.__setattr__(self, "lowering", lowering)

        pipeline_slots = {
            item.slot for item in self.pipeline if item.slot is not None
        }
        if pipeline_slots:
            other_slots = {
                item.slot
                for item in (*self.memory_layout, *self.parallel_mapping, *self.split)
                if item.slot is not None
            }
            reused = sorted(pipeline_slots & other_slots)
            if reused:
                raise _error(
                    "pipeline slot(s) cannot also be used as memory, parallel, "
                    "or split factors: " + ", ".join(reused)
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetRealizationContract":
        data = _strict_object(
            value,
            keys={
                "memory_layout", "parallel_mapping", "pipeline", "fusion",
                "split", "lowering",
            },
            context="TargetRealizationContract",
        )
        if not isinstance(data["lowering"], (Mapping, type(None))):
            raise _error("TargetRealizationContract.lowering must be an object or null")
        return cls(
            tuple(data["memory_layout"]),
            tuple(data["parallel_mapping"]),
            tuple(data["pipeline"]),
            tuple(data["fusion"]),
            tuple(data["split"]),
            TargetLowering.from_dict(data["lowering"])
            if data["lowering"] is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_layout": [item.to_dict() for item in self.memory_layout],
            "parallel_mapping": [item.to_dict() for item in self.parallel_mapping],
            "pipeline": [item.to_dict() for item in self.pipeline],
            "fusion": [item.to_dict() for item in self.fusion],
            "split": [item.to_dict() for item in self.split],
            "lowering": self.lowering.to_dict() if self.lowering is not None else None,
        }

    @property
    def content_digest(self) -> str:
        return layer_c_content_digest(self.to_dict())

    @property
    def slot_refs(self) -> tuple[str, ...]:
        """Return every E_C slot referenced by this typed contract."""

        slots: list[str] = []
        for item in (*self.memory_layout, *self.parallel_mapping, *self.pipeline, *self.split):
            if item.slot is not None:
                slots.append(item.slot)
        # Contract records are declaration sequences.  Preserve the first
        # reference order so a caller cannot accidentally turn a positional
        # realization ABI into lexical slot order.  Consumers that only need
        # membership must explicitly construct a set from this ordered view.
        return tuple(dict.fromkeys(slots))

    def declared_record_ids(self) -> tuple[str, ...]:
        """Return stable identities for every typed realization record.

        Record identity is derived from the contract field and declaration
        order.  It is deliberately not serialized as another field: changing
        an identity must remain impossible without changing the typed contract
        itself.
        """

        record_ids: list[str] = []
        for field_name in (
            "memory_layout",
            "parallel_mapping",
            "pipeline",
            "fusion",
            "split",
        ):
            record_ids.extend(
                f"{field_name}:{index}"
                for index, _record in enumerate(getattr(self, field_name))
            )
        if self.lowering is not None:
            record_ids.append("lowering")
        return tuple(record_ids)

    def _record_payload(self, record_id: str) -> dict[str, Any]:
        """Return the canonical typed payload for one declared record."""

        if not isinstance(record_id, str) or _RECORD_ID.fullmatch(record_id) is None:
            raise _error(f"unknown target realization record id {record_id!r}")
        if record_id == "lowering":
            if self.lowering is None:
                raise _error("target realization record 'lowering' is not declared")
            return {"field": "lowering", "record": self.lowering.to_dict()}
        field_name, raw_index = record_id.split(":", 1)
        records = getattr(self, field_name, None)
        if records is None:
            raise _error(f"unknown target realization record id {record_id!r}")
        index = int(raw_index)
        if index < 0 or index >= len(records):
            raise _error(f"target realization record id {record_id!r} is not declared")
        return {
            "field": field_name,
            "index": index,
            "record": records[index].to_dict(),
        }

    def _application_digest(
        self,
        *,
        record_id: str,
        consumer: str,
        phase: str,
        artifact_kind: str,
        artifact_digest: str,
    ) -> str:
        return layer_c_content_digest(
            {
                "record_id": record_id,
                "record": self._record_payload(record_id),
                "consumer": consumer,
                "phase": phase,
                "artifact_kind": artifact_kind,
                "artifact_digest": artifact_digest,
            }
        )

    def record_consumption_from_artifact(
        self,
        record_id: str,
        *,
        consumer: str,
        phase: str,
        artifact_kind: str,
        artifact: bytes | bytearray | memoryview | str,
    ) -> TargetRecordConsumption:
        """Create evidence from compiler-produced artifact bytes.

        Keeping this constructor on the contract makes every backend use the
        same record payload and digest algorithm.  The backend still owns the
        concrete artifact acquisition; it must pass the bytes returned by its
        real compiler boundary here.
        """

        artifact_digest = target_artifact_digest(artifact)
        application_digest = self._application_digest(
            record_id=record_id,
            consumer=consumer,
            phase=phase,
            artifact_kind=artifact_kind,
            artifact_digest=artifact_digest,
        )
        return TargetRecordConsumption(
            record_id,
            consumer,
            phase,
            artifact_kind,
            artifact_digest,
            application_digest,
        )

    def validate_consumed_records(
        self,
        consumed_record_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Require an exact report of the contract records used by a backend.

        A backend cannot satisfy this boundary by echoing all declarations: the
        report is checked for duplicate, unknown, and missing identities.  The
        returned tuple preserves the backend's execution order for evidence.
        """

        if isinstance(consumed_record_ids, (str, bytes)):
            raise _error("consumed target realization records must be an array")
        try:
            consumed = tuple(consumed_record_ids)
        except TypeError as exc:
            raise _error("consumed target realization records must be an array") from exc
        if any(not isinstance(item, str) or not item for item in consumed):
            raise _error("consumed target realization record ids must be strings")
        if len(consumed) != len(set(consumed)):
            raise _error("consumed target realization records must be unique")
        declared = set(self.declared_record_ids())
        actual = set(consumed)
        unknown = sorted(actual - declared)
        missing = sorted(declared - actual)
        if unknown or missing:
            detail: list[str] = []
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            if missing:
                detail.append("missing " + ", ".join(missing))
            raise _error(
                "target realization record consumption is not exact: "
                + "; ".join(detail)
            )
        return consumed

    def consumption_evidence(
        self,
        applications: Iterable[TargetRecordConsumption | Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Validate complete, artifact-backed target-record consumption."""

        if isinstance(applications, (str, bytes)):
            raise _error(
                "target realization consumption must contain typed applications"
            )
        try:
            checked = tuple(
                item
                if isinstance(item, TargetRecordConsumption)
                else TargetRecordConsumption.from_dict(item)
                for item in applications
            )
        except TypeError as exc:
            raise _error(
                "target realization consumption must contain typed applications"
            ) from exc
        consumed = self.validate_consumed_records(item.record_id for item in checked)
        for item in checked:
            expected_application_digest = self._application_digest(
                record_id=item.record_id,
                consumer=item.consumer,
                phase=item.phase,
                artifact_kind=item.artifact_kind,
                artifact_digest=item.artifact_digest,
            )
            if item.application_digest != expected_application_digest:
                raise _error(
                    "target realization application digest is stale for "
                    + item.record_id
                )
        artifact_owners: dict[str, str] = {}
        for item in checked:
            previous = artifact_owners.get(item.artifact_digest)
            if previous is not None:
                raise _error(
                    "target realization records must have independent compiler "
                    "artifacts; "
                    f"{item.record_id!r} reuses {previous!r}"
                )
            artifact_owners[item.artifact_digest] = item.record_id
        return {
            "schema": "target_record_consumption",
            "supported": True,
            "declared_record_ids": list(self.declared_record_ids()),
            "consumed_record_ids": list(consumed),
            "unconsumed_record_ids": [],
            "applications": [item.to_dict() for item in checked],
            "artifact_digests": sorted({item.artifact_digest for item in checked}),
        }

    def validate_consumption_evidence(
        self,
        evidence: Mapping[str, Any],
    ) -> tuple[TargetRecordConsumption, ...]:
        """Revalidate serialized evidence against this exact contract."""

        if not isinstance(evidence, Mapping):
            raise _error("target realization evidence must be an object")
        required = {
            "schema",
            "supported",
            "declared_record_ids",
            "consumed_record_ids",
            "unconsumed_record_ids",
            "applications",
            "artifact_digests",
        }
        if set(evidence) != required:
            missing = sorted(required - set(evidence))
            extra = sorted(set(evidence) - required, key=str)
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unknown " + ", ".join(extra))
            raise _error(
                "target realization evidence schema is not exact"
                + (": " + "; ".join(detail) if detail else "")
            )
        if evidence["schema"] != "target_record_consumption":
            raise _error("target realization evidence has an unsupported schema")
        if evidence["supported"] is not True:
            raise _error("target realization evidence is not supported")
        applications = evidence["applications"]
        if not isinstance(applications, list):
            raise _error("target realization evidence applications must be an array")
        checked = tuple(TargetRecordConsumption.from_dict(item) for item in applications)
        expected = self.consumption_evidence(checked)
        if dict(evidence) != expected:
            raise _error("target realization evidence is not canonical or is stale")
        return checked

    def validate_choice(self, choice: ImplementationChoice) -> None:
        if not isinstance(choice, ImplementationChoice):
            raise _error("target realization contract requires ImplementationChoice")
        if self.lowering is None:
            raise _error("target realization contract requires a lowering record")
        if (
            self.lowering.backend_kind != choice.backend_kind
            or self.lowering.candidate_key != choice.candidate_key
            or self.lowering.candidate_digest != choice.candidate_digest
        ):
            raise _error(
                "target realization lowering does not exactly match the implementation choice"
            )

    def validate_against(
        self,
        j: Any,
        interface: Any,
        choice: ImplementationChoice,
    ) -> None:
        self.validate_choice(choice)
        interface_value = _interface(interface)
        validate_layer_c_attachments(j, interface_value)
        domain = layer_c_attachment_domain(j)
        declared_points = {item.slot: item for item in interface_value.points}

        def check_attachment(
            *,
            slot: str | None,
            subject_ref: str | None,
            axis_ref: LayerCAxisRef | None,
            context: str,
            record_kind: str | None = None,
            integer_value: bool = False,
            token_value: bool = False,
        ) -> None:
            point = None
            if slot is not None:
                point = declared_points.get(slot)
                if point is None:
                    raise _error(f"{context}.slot {slot!r} is absent from E_C")
                if record_kind is None:
                    raise _error(f"{context} has an untyped E_C slot use")
                if point.role not in _TARGET_CONTRACT_ROLES[record_kind]:
                    raise _error(
                        f"{context}.slot {slot!r} has role {point.role!r}, which "
                        f"is not valid for {record_kind}"
                    )
                if integer_value and not _contract_integer_value_type(point.value_type):
                    raise _error(
                        f"{context}.slot {slot!r} must declare an integer value_type"
                    )
                if token_value and not _contract_token_value_type(point.value_type):
                    raise _error(
                        f"{context}.slot {slot!r} must declare a closed token value_type"
                    )
                # A target contract slot must be J-addressed.  A compiler-only
                # Source slot has no attachment and cannot be silently reused
                # as a target realization point.
                if point.subject_ref is None:
                    raise _error(
                        f"{context}.slot {slot!r} is not attached to a J subject"
                    )
                if point.subject_ref != subject_ref or point.axis_ref != axis_ref:
                    raise _error(
                        f"{context}.slot {slot!r} attachment differs from its E_C declaration"
                    )
            if subject_ref is not None and subject_ref not in domain:
                raise _error(f"{context}.subject_ref {subject_ref!r} is absent from J")
            if axis_ref is not None:
                if subject_ref is None:
                    raise _error(f"{context}.axis_ref requires subject_ref")
                if axis_ref not in domain[subject_ref]:
                    raise _error(
                        f"{context}.axis_ref {axis_ref!r} is invalid for J subject "
                        f"{subject_ref!r}"
                    )

        for index, item in enumerate(self.memory_layout):
            check_attachment(
                slot=item.slot,
                subject_ref=item.subject_ref,
                axis_ref=item.axis_ref,
                context=f"contract.memory_layout[{index}]",
                record_kind="memory_layout",
                token_value=item.slot is not None,
            )
        for index, item in enumerate(self.parallel_mapping):
            check_attachment(
                slot=item.slot,
                subject_ref=item.subject_ref,
                axis_ref=item.axis_ref,
                context=f"contract.parallel_mapping[{index}]",
                record_kind="parallel_mapping",
                integer_value=item.factor is not None,
                token_value=item.factor is None,
            )
        for index, item in enumerate(self.pipeline):
            check_attachment(
                slot=item.slot,
                subject_ref=item.subject_ref,
                axis_ref=item.axis_ref,
                context=f"contract.pipeline[{index}]",
                record_kind="pipeline",
                integer_value=item.slot is not None,
            )
        for index, item in enumerate(self.fusion):
            for subject in item.subject_refs:
                check_attachment(
                    slot=None,
                    subject_ref=subject,
                    axis_ref=None,
                    context=f"contract.fusion[{index}]",
                )
        for index, item in enumerate(self.split):
            check_attachment(
                slot=item.slot,
                subject_ref=item.subject_ref,
                axis_ref=item.axis_ref,
                context=f"contract.split[{index}]",
                record_kind="split",
                integer_value=item.slot is not None,
            )

    def validate_bindings(self, bindings: Sequence[LayerCBinding]) -> None:
        """Check that bound slot values agree with typed contract records."""

        bound: dict[str, LayerCScalar] = {}
        for index, item in enumerate(bindings):
            binding = item if isinstance(item, LayerCBinding) else LayerCBinding.from_dict(item)
            if binding.slot in bound:
                raise _error(f"target bindings duplicate slot {binding.slot!r}")
            bound[binding.slot] = binding.value
        unknown = sorted(set(bound) - set(self.slot_refs))
        if unknown:
            raise _error(
                "target binding slot(s) are not represented by the realization "
                "contract: " + ", ".join(unknown)
            )
        missing = sorted(set(self.slot_refs) - set(bound))
        if missing:
            raise _error(
                "target realization contract references unbound slot(s): "
                + ", ".join(missing)
            )
        for field_name, records in (
            ("parallel_mapping", self.parallel_mapping),
            ("split", self.split),
        ):
            for index, item in enumerate(records):
                if item.slot is None or item.factor is None or item.slot not in bound:
                    continue
                value = bound[item.slot]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise _error(
                        f"contract.{field_name}[{index}].slot {item.slot!r} "
                        "must bind an integer target value"
                    )
                if value != item.factor:
                    raise _error(
                        f"contract.{field_name}[{index}].factor differs from "
                        f"bound target slot {item.slot!r}"
                    )
        for index, item in enumerate(self.pipeline):
            if item.slot is None:
                continue
            value = bound[item.slot]
            if isinstance(value, bool) or not isinstance(value, int):
                raise _error(
                    f"contract.pipeline[{index}].slot {item.slot!r} "
                    "must bind an integer pipeline depth"
                )
            if value != item.depth:
                raise _error(
                    f"contract.pipeline[{index}].depth differs from bound "
                    f"pipeline slot {item.slot!r}"
                )


def _j_object(j: Any) -> Mapping[str, Any]:
    if isinstance(j, Mapping):
        value: Any = j
    else:
        method = getattr(j, "to_json", None)
        if not callable(method):
            raise _error("J must be a canonical mapping or expose to_json()")
        value = method()
        if isinstance(value, str):
            value = _decode_json_object(value, "J")
    if not isinstance(value, Mapping):
        raise _error("J to_json() must return an object")
    _assert_json_value(value, "J")
    return value


def layer_c_attachment_domain(
    j: Any,
) -> dict[str, frozenset[LayerCAxisRef]]:
    """Conservatively derive J-owned schedule subjects and logical axes.

    Empty axis sets still admit subject-only attachments.  Unknown subjects or
    axes are never guessed; callers fail closed through
    :func:`validate_layer_c_attachments`.
    """

    data = _j_object(j)
    # Layer-C is an extension consumer, not a second J elaborator.  Normalize
    # the accepted archive through the same constructor used by Replay,
    # Gate1, and the other backends before deriving attachment subjects.
    # Keep the import local: canonical_j's model-facing imports must not depend
    # on this optional Layer-C extension module during module initialization.
    try:
        from .canonical_j import elaborate_j, j_elaboration
        from intentir.verify.gate1.model import SemanticArchive

        archive = (
            j
            if isinstance(j, SemanticArchive)
            else SemanticArchive.from_json(dict(data))
        )
        archive = elaborate_j(archive)
        elaboration = j_elaboration(archive)
        data = archive.to_json()
        function = archive.intent
        inputs = [item.to_json() for item in function.inputs]
        outputs = [item.to_json() for item in function.outputs]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise _error(f"J cannot be canonically elaborated for Layer-C: {exc}") from exc

    tensor_shapes: dict[str, tuple[Any, ...]] = {}
    dimensions: set[str] = set()
    for index, tensor in enumerate([*inputs, *outputs]):
        if not isinstance(tensor, Mapping):
            raise _error(f"J tensor {index} must be an object")
        name = tensor.get("name")
        shape = tensor.get("shape")
        if not isinstance(name, str) or not isinstance(shape, list):
            raise _error(f"J tensor {index} lacks name/shape")
        tensor_shapes[name] = tuple(shape)
        dimensions.update(
            reference
            for item in shape
            for reference in dimension_expression_references(item)
            if _AXIS_REF.fullmatch(reference) is not None
        )

    operation_records = elaboration.operation_records
    if not operation_records:
        raise _error("J intent has no complete canonical operation table")
    operations = {
        record["path"]: record["wire"]
        for record in operation_records
        if isinstance(record, Mapping)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("wire"), Mapping)
    }
    operations_by_node: dict[str, Mapping[str, Any]] = {}
    for _path, operation in operations.items():
        node = operation.get("id")
        if isinstance(node, str) and re.fullmatch(r"n[0-9]+", node):
            operations_by_node[node] = operation
    node_ids = archive.node_ids
    if not isinstance(node_ids, Mapping):
        raise _error("J.node_ids must be an object")
    for node, path in node_ids.items():
        if not isinstance(node, str) or not isinstance(path, str):
            raise _error("J.node_ids must map string refs to paths")
        operation = operations.get(path)
        if operation is None:
            raise _error(f"J.node_ids path {path!r} is not declared")
        operations_by_node[node] = operation

    # Intermediate SSA shapes belong to the elaborated construction bundle,
    # not to the optional raw ``operation.type`` spelling.  The canonical
    # elaborator may prove a result type from the closed operation contract;
    # reading only the raw wire here would make Layer-C disagree with Replay,
    # Gate1, and the typed materializer for operations that omit ``type``.
    value_shapes = dict(tensor_shapes)
    for name, value_type in elaboration.value_type_by_name.items():
        if not value_type.is_tensor:
            continue
        value_shapes[name] = tuple(value_type.shape)
        dimensions.update(
            reference
            for extent in value_type.shape
            for reference in dimension_expression_references(extent)
            if _AXIS_REF.fullmatch(reference) is not None
        )

    result: dict[str, frozenset[LayerCAxisRef]] = {}

    def add(ref: str, axes: Iterable[LayerCAxisRef]) -> None:
        checked = frozenset(axes)
        previous = result.get(ref)
        if previous is not None and previous != checked:
            raise _error(f"J schedule subject {ref!r} has conflicting declarations")
        result[ref] = checked

    for name, shape in tensor_shapes.items():
        if _SUBJECT_REF.fullmatch(name) is not None:
            add(
                name,
                [
                    *range(len(shape)),
                    *(
                        reference
                        for item in shape
                        for reference in dimension_expression_references(item)
                        if reference in dimensions
                    ),
                ],
            )

    for node, operation in operations_by_node.items():
        axes: set[LayerCAxisRef] = set()
        op = operation.get("op")
        attrs = operation.get("attrs")
        attrs = attrs if isinstance(attrs, Mapping) else {}
        if op == "reduce":
            authored = attrs.get("axes")
            if isinstance(authored, list):
                axes.update(
                    item
                    for item in authored
                    if isinstance(item, int) and not isinstance(item, bool) and item >= 0
                )
        elif op == "loop":
            trip = operation.get("trip")
            trip_axis = trip.get("axis") if isinstance(trip, Mapping) else None
            # Static semantic loops expose the conventional ordinal axis 0
            # and, when present, their J dimension handle.  A dynamic value
            # loop is controlled by a runtime scalar and has no semantic axis;
            # exposing axis 0 here would let a target bind a fabricated tile.
            if isinstance(trip_axis, str) and trip_axis in dimensions:
                axes.add(0)
                axes.add(trip_axis)
        elif op == "contraction":
            subscripts = attrs.get("subscripts")
            operands = operation.get("operands")
            if isinstance(subscripts, str) and isinstance(operands, list):
                try:
                    inputs_text, output_text = subscripts.split("->", 1)
                    terms = inputs_text.split(",")
                except ValueError:
                    terms = []
                    output_text = ""
                if len(terms) == len(operands):
                    contracted: list[str] = []
                    for term in terms:
                        for label in term:
                            if label not in output_text and label not in contracted:
                                contracted.append(label)
                    axes.update(range(len(contracted)))
                    for term, operand in zip(terms, operands, strict=True):
                        shape = value_shapes.get(operand) if isinstance(operand, str) else None
                        if shape is None or len(term) != len(shape):
                            continue
                        for label, extent in zip(term, shape, strict=True):
                            if label in contracted:
                                axes.update(
                                    reference
                                    for reference in dimension_expression_references(extent)
                                    if reference in dimensions
                                )
        add(node, axes)

    boundary = elaboration.boundary_record
    if not isinstance(boundary, Mapping):
        raise _error("J.boundary must be an object")
    domains = boundary.get("domains")
    if not isinstance(domains, Mapping):
        raise _error("J.boundary.domains must be an object")
    for domain, expression in domains.items():
        if not isinstance(domain, str) or not isinstance(expression, str):
            raise _error("J boundary domains must map refs to expressions")
        logical = list(dict.fromkeys(re.findall(r"\bk[0-9]+\b", expression)))
        symbolic = list(dict.fromkeys(re.findall(r"\bx[0-9]+\b", expression)))
        add(domain, [*range(len(logical)), *(item for item in symbolic if item in dimensions)])

    accesses = boundary.get("accesses")
    if not isinstance(accesses, list):
        raise _error("J.boundary.accesses must be an array")
    for access in accesses:
        if not isinstance(access, Mapping):
            raise _error("J boundary access must be an object")
        ref = access.get("access_id")
        tensor = access.get("tensor")
        if not isinstance(ref, str) or not isinstance(tensor, str):
            raise _error("J boundary access lacks access_id/tensor")
        shape = tensor_shapes.get(tensor)
        if shape is None:
            raise _error(f"J access {ref!r} names unknown tensor {tensor!r}")
        add(
            ref,
            [
                *range(len(shape)),
                *(
                    reference
                    for item in shape
                    for reference in dimension_expression_references(item)
                    if reference in dimensions
                ),
            ],
        )

    scopes = data.get("scopes")
    storage = data.get("storage")
    if not isinstance(scopes, list) or not isinstance(storage, list):
        raise _error("J scopes/storage must be arrays")
    for scope in scopes:
        ref = scope.get("scope_id") if isinstance(scope, Mapping) else None
        if not isinstance(ref, str):
            raise _error("J scope lacks scope_id")
        add(ref, ())
    for item in storage:
        ref = item.get("storage_id") if isinstance(item, Mapping) else None
        if not isinstance(ref, str):
            raise _error("J storage lacks storage_id")
        add(ref, ())

    effects = boundary.get("effects")
    if not isinstance(effects, list):
        raise _error("J.boundary.effects must be an array")
    for index, effect in enumerate(effects):
        ref = effect.get("effect_id") if isinstance(effect, Mapping) else None
        if not isinstance(ref, str) or re.fullmatch(r"e[0-9]+", ref) is None:
            raise _error(f"J effect {index} lacks a stable effect_id")
        add(ref, ())

    authorizations = boundary.get("realization_authorizations")
    if not isinstance(authorizations, list):
        raise _error("J.boundary.realization_authorizations must be an array")
    for index, authorization in enumerate(authorizations):
        ref = (
            authorization.get("auth_id")
            if isinstance(authorization, Mapping)
            else None
        )
        if not isinstance(ref, str) or _AUTH_REF.fullmatch(ref) is None:
            raise _error(
                f"J realization authorization {index} lacks a stable auth_id"
            )
        add(ref, ())
    return dict(sorted(result.items()))


def validate_layer_c_attachments(j: Any, declarations: Any) -> None:
    """Validate extension/source-slot attachments against this exact J."""

    if isinstance(declarations, LayerCInterface):
        items: Iterable[Any] = declarations.points
    elif isinstance(declarations, Mapping) and set(declarations) == {"points"}:
        items = LayerCInterface.from_dict(declarations).points
    elif isinstance(declarations, Sequence) and not isinstance(
        declarations, (str, bytes)
    ):
        items = declarations
    else:
        raise _error("Layer-C attachment declarations must be an interface or array")
    domain = layer_c_attachment_domain(j)
    for index, item in enumerate(items):
        if isinstance(item, LayerCExtensionPoint):
            subject_ref = item.subject_ref
            axis_ref = item.axis_ref
            kind: Any = None
        elif isinstance(item, Mapping):
            subject_ref = item.get("subject_ref")
            axis_ref = item.get("axis_ref")
            kind = item.get("kind")
        else:
            subject_ref = getattr(item, "subject_ref", None)
            axis_ref = getattr(item, "axis_ref", None)
            kind = getattr(item, "kind", None)
        kind = getattr(kind, "value", kind)
        if kind == "compiler":
            if subject_ref is not None or axis_ref is not None:
                raise _error(
                    f"Layer-C declaration {index} is compiler-only and cannot attach to J"
                )
            continue
        if axis_ref is not None and subject_ref is None:
            raise _error(f"Layer-C declaration {index} axis_ref requires subject_ref")
        if subject_ref is None:
            continue
        if not isinstance(subject_ref, str) or _SUBJECT_REF.fullmatch(subject_ref) is None:
            raise _error(f"Layer-C declaration {index} has an invalid subject_ref")
        if subject_ref not in domain:
            raise _error(
                f"Layer-C declaration {index} subject_ref {subject_ref!r} is absent from J"
            )
        if axis_ref is not None and axis_ref not in domain[subject_ref]:
            raise _error(
                f"Layer-C declaration {index} axis_ref {axis_ref!r} is invalid for "
                f"J subject {subject_ref!r}"
            )


def _j_authorizations(j: Mapping[str, Any]) -> frozenset[str]:
    boundary = j.get("boundary")
    values = (
        boundary.get("realization_authorizations")
        if isinstance(boundary, Mapping)
        else None
    )
    if not isinstance(values, list):
        raise _error("J.boundary.realization_authorizations must be an array")
    result: set[str] = set()
    for index, item in enumerate(values):
        ref = item.get("auth_id") if isinstance(item, Mapping) else None
        if not isinstance(ref, str) or _AUTH_REF.fullmatch(ref) is None:
            raise _error(f"J realization authorization {index} has invalid auth_id")
        if ref in result:
            raise _error(f"J realization authorization {ref!r} is duplicated")
        result.add(ref)
    return frozenset(result)


def _provided_set(
    values: Iterable[Any],
    *,
    validator: Any,
    context: str,
) -> frozenset[str]:
    return frozenset(
        _canonical_unique(values, validator=validator, context=context)
    )


@dataclass(frozen=True)
class TargetLayerCPlugin:
    """One immutable target plugin used as Lower's hardware-facing input."""

    j_digest: str
    interface_digest: str
    capability_digest: str
    target: TargetIdentity
    interface: LayerCInterface
    bindings: tuple[LayerCBinding, ...]
    choice: ImplementationChoice
    required_capabilities: tuple[str, ...] = ()
    contract: TargetRealizationContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "j_digest", _digest(self.j_digest, "plugin.j_digest"))
        object.__setattr__(
            self,
            "interface_digest",
            _digest(self.interface_digest, "plugin.interface_digest"),
        )
        object.__setattr__(
            self,
            "capability_digest",
            _digest(self.capability_digest, "plugin.capability_digest"),
        )
        target = (
            self.target
            if isinstance(self.target, TargetIdentity)
            else TargetIdentity.from_dict(self.target)
        )
        interface = _interface(self.interface)
        if isinstance(self.bindings, (str, bytes)):
            raise _error("plugin.bindings must be an array")
        try:
            bindings = tuple(
                item
                if isinstance(item, LayerCBinding)
                else LayerCBinding.from_dict(item)
                for item in self.bindings
            )
        except TypeError as exc:
            raise _error("plugin.bindings must be an array") from exc
        slots = [item.slot for item in bindings]
        if len(slots) != len(set(slots)):
            raise _error("plugin bindings must have unique slots")
        choice = (
            self.choice
            if isinstance(self.choice, ImplementationChoice)
            else ImplementationChoice.from_dict(self.choice)
        )
        contract = self.contract
        if contract is not None and not isinstance(contract, TargetRealizationContract):
            contract = TargetRealizationContract.from_dict(contract)
        capabilities = _canonical_unique(
            self.required_capabilities,
            validator=_identifier,
            context="plugin.required_capabilities",
        )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "choice", choice)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "contract", contract)

        actual_interface_digest = binding_interface_digest(interface)
        if self.interface_digest != actual_interface_digest:
            raise _error("plugin.interface_digest is stale or does not cover declarations")
        if choice.backend_kind != target.family:
            raise _error("plugin choice backend_kind does not match target family")
        if contract is not None:
            contract.validate_choice(choice)
        declared = {item.slot: item for item in interface.points}
        bound = {item.slot: item for item in bindings}
        # Optional points are placeholders that J can materialize directly.
        # Only ``required`` declarations demand an actual target value; a
        # present optional binding is still closed and type-checked below.
        required_slots = {
            item.slot for item in interface.points if item.required
        }
        missing = sorted(required_slots - set(bound))
        extra = sorted(set(bound) - set(declared))
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("extra " + ", ".join(extra))
            raise _error("plugin bindings are not complete and exact: " + "; ".join(details))
        expected_order = tuple(
            item.slot for item in interface.points if item.slot in bound
        )
        if tuple(slots) != expected_order:
            raise _error(
                "plugin bindings must follow target interface declaration order"
            )
        mistyped = sorted(
            slot
            for slot, binding in bound.items()
            if not _value_matches(declared[slot].value_type, binding.value)
        )
        if mistyped:
            raise _error(
                "plugin bindings do not match declared types: " + ", ".join(mistyped)
            )

    @classmethod
    def build(
        cls,
        *,
        j: Any,
        target: Any,
        interface: Any,
        bindings: Sequence[LayerCBinding | Mapping[str, Any]],
        choice: ImplementationChoice | Mapping[str, Any],
        required_capabilities: Sequence[str] = (),
        contract: TargetRealizationContract | Mapping[str, Any] | None = None,
        allowed_authorizations: Iterable[str] | None = None,
        available_capabilities: Iterable[str] | None = None,
    ) -> "TargetLayerCPlugin":
        j_data = _j_object(j)
        target_identity = TargetIdentity.from_target(target)
        interface_value = _interface(interface)
        if available_capabilities is None:
            raise _error(
                "target plugin construction requires a backend-derived "
                "capability snapshot"
            )
        available = _canonical_unique(
            available_capabilities,
            validator=_identifier,
            context="available_capabilities",
        )
        plugin = cls(
            j_digest=layer_c_content_digest(j_data),
            interface_digest=binding_interface_digest(interface_value),
            capability_digest=capability_snapshot_digest(
                target_identity,
                available,
            ),
            target=target_identity,
            interface=interface_value,
            bindings=tuple(bindings),
            choice=choice,
            required_capabilities=tuple(required_capabilities),
            contract=contract,
        )
        plugin.validate_against(
            j,
            target,
            expected_interface=interface_value,
            expected_contract=contract,
            allowed_authorizations=allowed_authorizations,
            available_capabilities=available,
        )
        return plugin

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetLayerCPlugin":
        if not isinstance(value, Mapping):
            raise _error("TargetLayerCPlugin must be an object")
        required = {
            "j_digest", "interface_digest", "capability_digest", "target",
            "interface", "bindings", "choice", "required_capabilities",
        }
        optional = {"contract"}
        actual = set(value)
        if not required <= actual or actual - required - optional:
            missing = sorted(required - actual)
            extra = sorted(actual - required - optional, key=str)
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unknown " + ", ".join(map(str, extra)))
            raise _error(
                "TargetLayerCPlugin must contain the canonical fields"
                + (f" ({'; '.join(detail)})" if detail else "")
            )
        data = value
        if not isinstance(data["bindings"], list):
            raise _error("TargetLayerCPlugin.bindings must be an array")
        if not isinstance(data["required_capabilities"], list):
            raise _error("TargetLayerCPlugin.required_capabilities must be an array")
        return cls(
            j_digest=data["j_digest"],
            interface_digest=data["interface_digest"],
            capability_digest=data["capability_digest"],
            target=TargetIdentity.from_dict(data["target"]),
            interface=LayerCInterface.from_dict(data["interface"]),
            bindings=tuple(LayerCBinding.from_dict(item) for item in data["bindings"]),
            choice=ImplementationChoice.from_dict(data["choice"]),
            required_capabilities=tuple(data["required_capabilities"]),
            contract=(
                TargetRealizationContract.from_dict(data["contract"])
                if "contract" in data and data["contract"] is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, value: Any) -> "TargetLayerCPlugin":
        return cls.from_dict(_decode_json_object(value, "TargetLayerCPlugin"))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "j_digest": self.j_digest,
            "interface_digest": self.interface_digest,
            "capability_digest": self.capability_digest,
            "target": self.target.to_dict(),
            "interface": self.interface.to_dict(),
            "bindings": [item.to_dict() for item in self.bindings],
            "choice": self.choice.to_dict(),
            "required_capabilities": list(self.required_capabilities),
        }
        if self.contract is not None:
            result["contract"] = self.contract.to_dict()
        return result

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()

    @property
    def content_digest(self) -> str:
        """Content address of the complete target Layer-C plugin."""

        return layer_c_content_digest(self.to_dict())

    def validate_against(
        self,
        j: Any,
        target: Any,
        *,
        expected_interface: Any | None = None,
        expected_bindings: Any | None = None,
        expected_contract: Any | None = None,
        allowed_authorizations: Iterable[str] | None = None,
        available_capabilities: Iterable[str] | None = None,
    ) -> None:
        """Revalidate every externally rooted fact before target lowering.

        ``expected_bindings`` is retained as a compatibility spelling for the
        expected declaration set; it never means already-bound scalar values.
        """

        j_data = _j_object(j)
        actual_j_digest = layer_c_content_digest(j_data)
        if self.j_digest != actual_j_digest:
            raise _error("plugin.j_digest does not match the lowering J")
        if self.target != TargetIdentity.from_target(target):
            raise _error("plugin target does not match the lowering target")
        if self.interface_digest != binding_interface_digest(self.interface):
            raise _error("plugin.interface_digest is stale")
        if expected_interface is not None and expected_bindings is not None:
            raise _error("provide only expected_interface or expected_bindings")
        expected = expected_interface if expected_interface is not None else expected_bindings
        if expected is not None and self.interface != _interface(expected):
            raise _error("plugin interface does not match the expected A/B extension points")
        validate_layer_c_attachments(j_data, self.interface)
        contract = self.contract
        if expected_contract is not None:
            expected_contract_value = (
                expected_contract
                if isinstance(expected_contract, TargetRealizationContract)
                else TargetRealizationContract.from_dict(expected_contract)
            )
            if contract != expected_contract_value:
                raise _error("plugin target realization contract differs from the expected contract")
        if contract is not None:
            contract.validate_against(j_data, self.interface, self.choice)
            contract.validate_bindings(self.bindings)

        declared_authorizations = _j_authorizations(j_data)
        unknown = sorted(set(self.choice.authorization_refs) - declared_authorizations)
        if unknown:
            raise _error(
                "plugin references authorization(s) absent from J: " + ", ".join(unknown)
            )
        if self.choice.authorization_refs and allowed_authorizations is None:
            raise _error(
                "plugin authorization refs require caller-provided allowed_authorizations"
            )
        if allowed_authorizations is not None:
            allowed = _provided_set(
                allowed_authorizations,
                validator=lambda item, context: (
                    item
                    if isinstance(item, str) and _AUTH_REF.fullmatch(item) is not None
                    else (_ for _ in ()).throw(
                        _error(f"{context} entries must be authorization refs u*")
                    )
                ),
                context="allowed_authorizations",
            )
            denied = sorted(set(self.choice.authorization_refs) - allowed)
            if denied:
                raise _error(
                    "plugin authorization refs were not granted: " + ", ".join(denied)
                )

        if available_capabilities is None:
            raise _error(
                "plugin validation requires backend-derived capabilities"
            )
        available = _provided_set(
            available_capabilities,
            validator=_identifier,
            context="available_capabilities",
        )
        if self.capability_digest != capability_snapshot_digest(
            self.target,
            available,
        ):
            raise _error(
                "plugin.capability_digest is stale or does not match the current "
                "target capability snapshot"
            )
        missing = sorted(set(self.required_capabilities) - available)
        if missing:
            raise _error(
                "plugin requires unavailable capabilities: " + ", ".join(missing)
            )


# Alternate noun order retained for callers that adopted the design draft.
LayerCTargetPlugin = TargetLayerCPlugin


@dataclass(frozen=True)
class ResolvedLayerCScalar:
    role: str
    slot: str
    value: LayerCScalar
    path: str


def resolve_layer_c_scalar(
    layer_c: Mapping[str, Any], role: str
) -> ResolvedLayerCScalar:
    """Resolve a Source/Harness schedule role from canonical ``{slots, values}``.

    This compatibility query remains separate from ``TargetLayerCPlugin``.  It
    accepts only the physical Source-C slot ledger; J-axis associations belong
    to pi, not to this `{slots, values}` object.
    """

    if (
        not isinstance(layer_c, Mapping)
        or set(layer_c) != set(CANONICAL_C_WIRE_FIELDS)
    ):
        raise _error("Layer C must be the canonical {slots, values} object")
    slots = layer_c["slots"]
    values = layer_c["values"]
    if not isinstance(slots, list) or not isinstance(values, Mapping):
        raise _error("Layer C slots/values have invalid types")
    if not isinstance(role, str) or not role:
        raise _error("Layer-C role must be non-empty")
    matches: list[tuple[str, str]] = []
    declared: set[str] = set()
    base = {"name", "kind", "role", "value_type", "required"}
    for index, item in enumerate(slots):
        if not isinstance(item, Mapping):
            raise _error(f"Layer C slot {index} must be an object")
        actual = set(item)
        if not base <= actual or actual - base:
            raise _error("Layer C slot uses a non-canonical schema")
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise _error("Layer C slot name must be non-empty")
        _closed_token(name, f"Layer C slot {index} name")
        if name in declared:
            raise _error("Layer C slot names must be unique")
        declared.add(name)
        kind = item["kind"]
        if kind not in {"constexpr", "compiler"}:
            raise _error("Layer C slot kind must be constexpr or compiler")
        slot_role = item["role"]
        if not isinstance(slot_role, str) or _ROLE.fullmatch(slot_role) is None:
            raise _error("Layer C slot role must use lower_snake_case")
        value_type = _value_type(item["value_type"], f"Layer C slot {index} value_type")
        if not isinstance(item["required"], bool):
            raise _error("Layer C slot required must be boolean")
        if slot_role == role:
            matches.append((name, value_type))
    if any(not isinstance(name, str) for name in values):
        raise _error("Layer C value keys must be strings")
    if set(values) - declared:
        raise _error("Layer C values contain undeclared slots")
    if not matches:
        raise _error(f"Layer-C role {role!r} is absent")
    if len(matches) != 1:
        raise _error(f"Layer-C role {role!r} is ambiguous")
    slot, value_type = matches[0]
    if slot not in values:
        raise _error(f"Layer-C role {role!r} has no value")
    value = _scalar(values[slot], f"Layer C value {slot}")
    if not _value_matches(value_type, value):
        raise _error(f"Layer-C role {role!r} has a mistyped value")
    return ResolvedLayerCScalar(role, slot, value, f"$.values.{slot}")


__all__ = [
    "ImplementationChoice",
    "LayerCAxisRef",
    "LayerCBinding",
    "LayerCExtensionPoint",
    "LayerCInterface",
    "LayerCResolutionError",
    "LayerCScalar",
    "LayerCTargetPlugin",
    "ResolvedLayerCScalar",
    "TargetFusion",
    "TargetIdentity",
    "TargetLowering",
    "TargetLayerCPlugin",
    "TargetMemoryLayout",
    "TargetParallelMapping",
    "TargetPipeline",
    "TargetRecordConsumption",
    "TargetRealizationContract",
    "TargetSplit",
    "binding_interface_digest",
    "capability_snapshot_digest",
    "canonical_layer_c_json",
    "layer_c_attachment_domain",
    "layer_c_content_digest",
    "layer_c_interface_digest",
    "resolve_layer_c_scalar",
    "validate_layer_c_attachments",
]
