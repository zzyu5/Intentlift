"""Gate-neutral backend request accepting immutable J and target Layer C."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from intentir.ir.canonical_j import elaborate_j
from intentir.ir.layer_c import (
    LayerCBinding,
    LayerCInterface,
    TargetRecordConsumption,
    TargetIdentity,
    TargetLayerCPlugin,
    TargetRealizationContract,
    binding_interface_digest,
    layer_c_content_digest,
    validate_layer_c_attachments,
)
from intentir.verify.gate1.model import SemanticArchive


class BackendProtocolError(ValueError):
    pass


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+-]+", value) is None:
        raise BackendProtocolError(f"{context} must be a non-empty identifier")
    return value


@dataclass(frozen=True)
class BackendTarget:
    family: str
    architecture: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _identifier(self.family, "target family"))
        object.__setattr__(self, "architecture", _identifier(self.architecture, "target architecture"))
        features = tuple(_identifier(item, "target feature") for item in self.features)
        if len(features) != len(set(features)):
            raise BackendProtocolError("target features must be unique")
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "architecture": self.architecture,
            "features": list(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendTarget":
        if not isinstance(value, Mapping) or set(value) != {
            "family", "architecture", "features"
        }:
            raise BackendProtocolError(
                "BackendTarget must contain exactly family, architecture, and features"
            )
        if not isinstance(value["features"], list):
            raise BackendProtocolError("BackendTarget.features must be an array")
        return cls(
            family=value["family"],
            architecture=value["architecture"],
            features=tuple(value["features"]),
        )


def _closed_unique_tokens(
    values: Iterable[Any],
    *,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise BackendProtocolError(f"{context} must be an array")
    try:
        result = tuple(_identifier(item, context) for item in values)
    except TypeError as exc:
        raise BackendProtocolError(f"{context} must be an array") from exc
    if len(result) != len(set(result)):
        raise BackendProtocolError(f"{context} must contain unique values")
    return tuple(sorted(result))


def _closed_unique_authorizations(
    values: Iterable[Any],
    *,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise BackendProtocolError(f"{context} must be an array")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise BackendProtocolError(f"{context} must be an array") from exc
    if any(not isinstance(item, str) or re.fullmatch(r"u[0-9]+", item) is None for item in result):
        raise BackendProtocolError(f"{context} must contain only u* authorization refs")
    if len(result) != len(set(result)):
        raise BackendProtocolError(f"{context} must contain unique values")
    return tuple(sorted(result))


@dataclass(frozen=True)
class TargetBindingContext:
    """All target-facing facts required to cross the public BindTarget boundary.

    ``e_c`` is the explicit typed extension interface supplied by the backend
    boundary.  ``interface`` is the target backend's view of that interface.
    Keeping both in the context makes the declaration lineage available to
    BindTarget and prevents a consumer from silently inventing a different
    E_C after the producer boundary has been crossed.
    """

    j: SemanticArchive
    target: BackendTarget
    interface: LayerCInterface | Mapping[str, Any]
    capabilities: tuple[str, ...]
    allowed_authorizations: tuple[str, ...] = ()
    e_c: LayerCInterface | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise BackendProtocolError("target binding context requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        if not isinstance(self.target, BackendTarget):
            raise BackendProtocolError("target binding context requires BackendTarget")
        try:
            interface = (
                self.interface
                if isinstance(self.interface, LayerCInterface)
                else LayerCInterface.from_dict(self.interface)
            )
        except (TypeError, ValueError) as exc:
            raise BackendProtocolError("target binding context has an invalid E_C interface") from exc
        object.__setattr__(self, "interface", interface)
        # ``interface`` is C_target, not automatically shared E_C.  A target
        # declaration may be target-only, so the absence of an
        # explicit producer declaration means an empty shared interface. The
        # target interface is never promoted into Source-C ownership by
        # default; shared declarations must be supplied explicitly at the
        # BindTarget boundary.
        shared_e_c = LayerCInterface() if self.e_c is None else self.e_c
        try:
            shared_e_c = (
                shared_e_c
                if isinstance(shared_e_c, LayerCInterface)
                else LayerCInterface.from_dict(shared_e_c)
            )
            validate_layer_c_attachments(self.j, shared_e_c)
            validate_layer_c_attachments(self.j, interface)
        except (TypeError, ValueError) as exc:
            raise BackendProtocolError(
                "target binding context has an E_C detached from canonical J"
            ) from exc
        object.__setattr__(self, "e_c", shared_e_c)
        object.__setattr__(
            self,
            "capabilities",
            _closed_unique_tokens(self.capabilities, context="target capabilities"),
        )
        object.__setattr__(
            self,
            "allowed_authorizations",
            _closed_unique_authorizations(
                self.allowed_authorizations,
                context="target allowed_authorizations",
            ),
        )


@dataclass(frozen=True)
class TargetExtensionMapping:
    """Explicit mapping for one shared ``E_C`` point to a target point."""

    source_slot: str
    target_slot: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_slot", _identifier(self.source_slot, "E_C mapping.source_slot"))
        object.__setattr__(self, "target_slot", _identifier(self.target_slot, "E_C mapping.target_slot"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetExtensionMapping":
        if not isinstance(value, Mapping) or set(value) != {"source_slot", "target_slot"}:
            raise BackendProtocolError(
                "TargetExtensionMapping must contain exactly source_slot and target_slot"
            )
        return cls(value["source_slot"], value["target_slot"])

    def to_dict(self) -> dict[str, str]:
        return {
            "source_slot": self.source_slot,
            "target_slot": self.target_slot,
        }


def _normalize_extension_mapping(
    values: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None,
) -> tuple[TargetExtensionMapping, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise BackendProtocolError("E_C mapping must be an array")
    try:
        result = tuple(
            item
            if isinstance(item, TargetExtensionMapping)
            else TargetExtensionMapping.from_dict(item)
            for item in values
        )
    except TypeError as exc:
        raise BackendProtocolError("E_C mapping must be an array") from exc
    source_slots = [item.source_slot for item in result]
    target_slots = [item.target_slot for item in result]
    if len(source_slots) != len(set(source_slots)):
        raise BackendProtocolError("E_C mapping source slots must be unique")
    if len(target_slots) != len(set(target_slots)):
        raise BackendProtocolError("E_C mapping target slots must be unique")
    return tuple(sorted(result, key=lambda item: item.source_slot))


def extension_mapping_digest(
    values: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None,
) -> str:
    """Digest the canonical typed ``E_C`` to ``C_target`` mapping."""

    mapping = _normalize_extension_mapping(values)
    return layer_c_content_digest(
        {"mappings": [item.to_dict() for item in mapping]}
    )


def _validate_extension_mapping(
    e_c: LayerCInterface,
    target_interface: LayerCInterface,
    mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None,
) -> tuple[TargetExtensionMapping, ...]:
    """Validate the explicit shared-interface to target-interface relation."""

    normalized = _normalize_extension_mapping(mapping)
    source_points = {item.slot: item for item in e_c.points}
    target_points = {item.slot: item for item in target_interface.points}
    mapped_sources = {item.source_slot for item in normalized}
    mapped_targets = {item.target_slot for item in normalized}
    missing_sources = sorted(set(source_points) - mapped_sources)
    if missing_sources:
        raise BackendProtocolError(
            "E_C declarations must each have an explicit target mapping: "
            + ", ".join(missing_sources)
        )
    for item in normalized:
        source = source_points.get(item.source_slot)
        target = target_points.get(item.target_slot)
        if source is None or target is None:
            raise BackendProtocolError(
                "E_C mapping references an undeclared source or target slot: "
                f"{item.source_slot!r}->{item.target_slot!r}"
            )
        if source.value_type != target.value_type:
            raise BackendProtocolError(
                "E_C mapping changes the extension value type: "
                f"{item.source_slot!r}->{item.target_slot!r}"
            )
        if source.subject_ref != target.subject_ref or source.axis_ref != target.axis_ref:
            raise BackendProtocolError(
                "E_C mapping changes the J subject/axis attachment: "
                f"{item.source_slot!r}->{item.target_slot!r}"
            )
    # A target-specific interface may add points generated from J/target
    # capabilities.  Such points are intentionally allowed to remain unmapped;
    # every point originating in the shared E_C is covered above.
    del mapped_targets
    return normalized


@dataclass(frozen=True)
class TargetPluginArtifact:
    """Public file/CLI envelope connecting ``E_C`` to ``C_target``.

    The plugin's own interface is target-specific, while ``e_c`` is the
    explicit typed extension interface supplied by the backend boundary.
    Keeping both in one canonical envelope makes the handoff auditable
    without deriving a second ABI from Source C.
    """

    e_c: LayerCInterface
    plugin: TargetLayerCPlugin
    e_c_mapping: tuple[TargetExtensionMapping, ...] = ()

    def __post_init__(self) -> None:
        interface = (
            self.e_c
            if isinstance(self.e_c, LayerCInterface)
            else LayerCInterface.from_dict(self.e_c)
        )
        if not isinstance(self.plugin, TargetLayerCPlugin):
            raise BackendProtocolError(
                "TargetPluginArtifact.plugin must be TargetLayerCPlugin"
            )
        mapping = _validate_extension_mapping(
            interface,
            self.plugin.interface,
            self.e_c_mapping,
        )
        object.__setattr__(self, "e_c", interface)
        object.__setattr__(self, "e_c_mapping", mapping)

    @property
    def e_c_digest(self) -> str:
        return binding_interface_digest(self.e_c)

    @property
    def e_c_mapping_digest(self) -> str:
        return extension_mapping_digest(self.e_c_mapping)

    @property
    def target_interface_digest(self) -> str:
        return self.plugin.interface_digest

    @property
    def content_digest(self) -> str:
        return layer_c_content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "e_c": self.e_c.to_dict(),
            "plugin": self.plugin.to_dict(),
            "e_c_mapping": [item.to_dict() for item in self.e_c_mapping],
        }

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()


def bind_target(
    context: TargetBindingContext,
    *,
    bindings: Sequence[LayerCBinding | Mapping[str, Any]],
    choice: Any,
    required_capabilities: Sequence[str] = (),
    contract: TargetRealizationContract | Mapping[str, Any] | None = None,
) -> TargetLayerCPlugin:
    """Construct one complete target plugin from the shared public boundary."""

    if not isinstance(context, TargetBindingContext):
        raise BackendProtocolError("bind_target requires TargetBindingContext")
    if contract is None:
        raise BackendProtocolError(
            "BindTarget requires a typed target realization contract"
        )
    try:
        required = _closed_unique_tokens(
            required_capabilities,
            context="target required_capabilities",
        )
        plugin = TargetLayerCPlugin.build(
            j=context.j,
            target=context.target,
            interface=context.interface,
            bindings=bindings,
            choice=choice,
            required_capabilities=required,
            contract=contract,
            allowed_authorizations=context.allowed_authorizations,
            available_capabilities=context.capabilities,
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(f"BindTarget rejected target plugin: {exc}") from exc
    validate_target_plugin(context, plugin)
    return plugin


def bind_target_artifact(
    context: TargetBindingContext,
    *,
    e_c: LayerCInterface | Mapping[str, Any],
    bindings: Sequence[LayerCBinding | Mapping[str, Any]],
    choice: Any,
    required_capabilities: Sequence[str] = (),
    contract: TargetRealizationContract | Mapping[str, Any] | None = None,
    e_c_mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None = None,
) -> TargetPluginArtifact:
    """Bind a target plugin while retaining the exact public ``E_C`` envelope."""

    if not isinstance(context, TargetBindingContext):
        raise BackendProtocolError("bind_target_artifact requires TargetBindingContext")
    try:
        e_c_value = (
            e_c
            if isinstance(e_c, LayerCInterface)
            else LayerCInterface.from_dict(e_c)
        )
        if e_c_value != context.e_c:
            raise BackendProtocolError(
                "BindTarget E_C differs from the binding context's shared E_C"
            )
        validate_layer_c_attachments(context.j, e_c_value)
        plugin = bind_target(
            context,
            bindings=bindings,
            choice=choice,
            required_capabilities=required_capabilities,
            contract=contract,
        )
        mapping = _validate_extension_mapping(
            e_c_value,
            plugin.interface,
            e_c_mapping,
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(
            f"BindTarget could not close the E_C to C_target bridge: {exc}"
        ) from exc
    return TargetPluginArtifact(
        e_c=e_c_value,
        plugin=plugin,
        e_c_mapping=mapping,
    )


def bind_target_from_e_c(
    *,
    j: SemanticArchive,
    e_c: LayerCInterface | Mapping[str, Any],
    target: BackendTarget,
    target_interface: LayerCInterface | Mapping[str, Any],
    capabilities: Sequence[str],
    bindings: Sequence[LayerCBinding | Mapping[str, Any]],
    choice: Any,
    required_capabilities: Sequence[str] = (),
    contract: TargetRealizationContract | Mapping[str, Any] | None = None,
    allowed_authorizations: Sequence[str] = (),
    e_c_mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None = None,
) -> TargetPluginArtifact:
    """Public production bridge from canonical ``J + E_C`` to ``C_target``."""

    try:
        e_c_value = (
            e_c
            if isinstance(e_c, LayerCInterface)
            else LayerCInterface.from_dict(e_c)
        )
        context = TargetBindingContext(
            j=j,
            target=target,
            interface=target_interface,
            capabilities=tuple(capabilities),
            allowed_authorizations=tuple(allowed_authorizations),
            e_c=e_c_value,
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(
            f"BindTarget received an invalid J/E_C/target context: {exc}"
        ) from exc
    target_interface_value = (
        target_interface
        if isinstance(target_interface, LayerCInterface)
        else LayerCInterface.from_dict(target_interface)
    )
    if e_c_mapping is None and e_c_value == target_interface_value:
        e_c_mapping = tuple(
            TargetExtensionMapping(item.slot, item.slot)
            for item in e_c_value.points
        )
    elif e_c_mapping is None and not e_c_value.points:
        e_c_mapping = ()
    elif e_c_mapping is None:
        raise BackendProtocolError(
            "E_C and target interfaces differ; an explicit e_c_mapping is required"
        )
    return bind_target_artifact(
        context,
        e_c=e_c,
        bindings=bindings,
        choice=choice,
        required_capabilities=required_capabilities,
        contract=contract,
        e_c_mapping=e_c_mapping,
    )


def validate_target_plugin(
    context: TargetBindingContext,
    plugin: TargetLayerCPlugin,
    *,
    expected_choice: Any | None = None,
    expected_bindings: Sequence[LayerCBinding | Mapping[str, Any]] | None = None,
    expected_contract: TargetRealizationContract | Mapping[str, Any] | None = None,
    require_contract: bool = True,
) -> TargetLayerCPlugin:
    """Revalidate a serialized plugin before any backend-specific lowering."""

    if not isinstance(context, TargetBindingContext):
        raise BackendProtocolError("validate_target_plugin requires TargetBindingContext")
    if not isinstance(plugin, TargetLayerCPlugin):
        raise BackendProtocolError("validate_target_plugin requires TargetLayerCPlugin")
    if require_contract and plugin.contract is None:
        raise BackendProtocolError("target plugin is missing its realization contract")
    try:
        plugin.validate_against(
            context.j,
            context.target,
            expected_interface=context.interface,
            expected_contract=expected_contract,
            allowed_authorizations=context.allowed_authorizations,
            available_capabilities=context.capabilities,
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(f"target plugin validation failed: {exc}") from exc
    if expected_choice is not None and plugin.choice != expected_choice:
        raise BackendProtocolError("target plugin choice differs from rebuilt choice")
    if expected_bindings is not None:
        try:
            expected = tuple(
                item
                if isinstance(item, LayerCBinding)
                else LayerCBinding.from_dict(item)
                for item in expected_bindings
            )
        except (TypeError, ValueError) as exc:
            raise BackendProtocolError("expected target bindings are invalid") from exc
        if plugin.bindings != expected:
            raise BackendProtocolError(
                "target plugin bindings differ from the rebuilt exact bindings"
            )
    return plugin


def serialize_target_plugin(plugin: TargetLayerCPlugin) -> dict[str, Any]:
    """Return the backend-neutral JSON object for one target plugin.

    This is the canonical file/CLI boundary between ``BindTarget`` and a
    lowering consumer.  Serializing through JSON also prevents a caller from
    accidentally handing backend code a dataclass carrying non-JSON objects or
    an alias to mutable input state.
    """

    if not isinstance(plugin, TargetLayerCPlugin):
        raise BackendProtocolError(
            "serialize_target_plugin requires TargetLayerCPlugin"
        )
    try:
        return json.loads(
            json.dumps(
                plugin.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(
            f"target plugin is not JSON-serializable: {exc}"
        ) from exc


def serialize_target_artifact(artifact: TargetPluginArtifact) -> dict[str, Any]:
    """Serialize the complete typed ``E_C``/``C_target`` handoff."""

    if not isinstance(artifact, TargetPluginArtifact):
        raise BackendProtocolError(
            "serialize_target_artifact requires TargetPluginArtifact"
        )
    try:
        return json.loads(
            json.dumps(
                artifact.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise BackendProtocolError(
            f"target plugin artifact is not JSON-serializable: {exc}"
        ) from exc


def deserialize_target_plugin(
    context: TargetBindingContext,
    payload: Mapping[str, Any] | str,
    *,
    expected_choice: Any | None = None,
    expected_bindings: Sequence[LayerCBinding | Mapping[str, Any]] | None = None,
    expected_contract: TargetRealizationContract | Mapping[str, Any] | None = None,
    require_contract: bool = True,
) -> TargetLayerCPlugin:
    """Parse and fully revalidate a serialized target plugin.

    The consumer deliberately does not trust the in-memory object that may
    have existed before the boundary.  Parsing reconstructs the target,
    interface, bindings, choice, capability snapshot, and realization
    contract; validation then checks their J/target/interface/capability and
    exact expected identities again.
    """

    if not isinstance(context, TargetBindingContext):
        raise BackendProtocolError(
            "deserialize_target_plugin requires TargetBindingContext"
        )
    try:
        if isinstance(payload, str):
            raw_payload: Any = json.loads(payload)
        elif isinstance(payload, Mapping):
            raw_payload = dict(payload)
        else:
            raise TypeError("serialized target plugin must be an object or JSON text")
        if not isinstance(raw_payload, Mapping):
            raise TypeError("serialized target plugin must decode to an object")
        plugin = TargetLayerCPlugin.from_dict(raw_payload)
        # Reject a non-canonical mapping before it can be used by a backend.
        # ``from_dict`` validates the schema; this equality additionally makes
        # the serialization boundary explicit and catches tuple/list or scalar
        # coercions that would otherwise be hidden by a permissive caller.
        canonical_payload = serialize_target_plugin(plugin)
        if canonical_payload != raw_payload:
            raise ValueError(
                "serialized target plugin is not in canonical JSON form"
            )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BackendProtocolError(
            f"cannot deserialize target plugin: {exc}"
        ) from exc
    return validate_target_plugin(
        context,
        plugin,
        expected_choice=expected_choice,
        expected_bindings=expected_bindings,
        expected_contract=expected_contract,
        require_contract=require_contract,
    )


def deserialize_target_artifact(
    context: TargetBindingContext,
    payload: Mapping[str, Any] | str,
    *,
    expected_e_c: LayerCInterface | Mapping[str, Any] | None = None,
    expected_choice: Any | None = None,
    expected_bindings: Sequence[LayerCBinding | Mapping[str, Any]] | None = None,
    expected_contract: TargetRealizationContract | Mapping[str, Any] | None = None,
    expected_e_c_mapping: Sequence[TargetExtensionMapping | Mapping[str, Any]] | None = None,
) -> TargetPluginArtifact:
    """Parse and validate the public ``E_C``/``C_target`` file boundary."""

    if not isinstance(context, TargetBindingContext):
        raise BackendProtocolError(
            "deserialize_target_artifact requires TargetBindingContext"
        )
    try:
        if isinstance(payload, str):
            raw_payload: Any = json.loads(payload)
        elif isinstance(payload, Mapping):
            raw_payload = dict(payload)
        else:
            raise TypeError("serialized target plugin artifact must be an object")
        if not isinstance(raw_payload, Mapping) or set(raw_payload) != {
            "e_c", "plugin", "e_c_mapping"
        }:
            raise TypeError(
                "serialized target plugin artifact must contain exactly e_c, plugin, and e_c_mapping"
            )
        if not isinstance(raw_payload["e_c_mapping"], list):
            raise TypeError("serialized target plugin artifact e_c_mapping must be an array")
        e_c = LayerCInterface.from_dict(raw_payload["e_c"])
        validate_layer_c_attachments(context.j, e_c)
        expected = context.e_c if expected_e_c is None else (
            expected_e_c
            if isinstance(expected_e_c, LayerCInterface)
            else LayerCInterface.from_dict(expected_e_c)
        )
        if e_c != expected:
            raise ValueError("serialized E_C differs from the expected interface")
        if e_c != context.e_c:
            raise ValueError(
                "serialized E_C differs from the binding context's shared E_C"
            )
        plugin = deserialize_target_plugin(
            context,
            raw_payload["plugin"],
            expected_choice=expected_choice,
            expected_bindings=expected_bindings,
            expected_contract=expected_contract,
        )
        mapping = _normalize_extension_mapping(raw_payload["e_c_mapping"])
        if expected_e_c_mapping is not None and mapping != _normalize_extension_mapping(expected_e_c_mapping):
            raise ValueError("serialized E_C mapping differs from the expected mapping")
        artifact = TargetPluginArtifact(
            e_c=e_c,
            plugin=plugin,
            e_c_mapping=mapping,
        )
        if serialize_target_artifact(artifact) != raw_payload:
            raise ValueError("serialized target plugin artifact is not canonical")
        return artifact
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BackendProtocolError(
            f"cannot deserialize target plugin artifact: {exc}"
        ) from exc


@dataclass(frozen=True)
class LoweringRequest:
    """One backend lowering request over immutable J and target Layer C.

    Gate1's ``ImplementationConfig`` is source-realization evidence.  It is not
    accepted here: a production lower consumes only a target-bound Layer-C
    plugin whose J digest, target identity, bindings, and closed implementation
    choice can be checked again at the backend boundary.
    """

    j: SemanticArchive
    target: BackendTarget
    variant: str
    c: TargetLayerCPlugin | TargetPluginArtifact | None = None
    constexpr_bindings: tuple[tuple[str, bool | int | float], ...] = ()
    e_c: LayerCInterface | None = None
    e_c_mapping: tuple[TargetExtensionMapping, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise BackendProtocolError("lowering request requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        if not isinstance(self.target, BackendTarget):
            raise BackendProtocolError("lowering request requires BackendTarget")
        object.__setattr__(self, "variant", _identifier(self.variant, "lowering variant"))
        artifact = self.c if isinstance(self.c, TargetPluginArtifact) else None
        if artifact is not None:
            if self.e_c is not None or self.e_c_mapping:
                raise BackendProtocolError(
                    "LoweringRequest cannot duplicate E_C beside TargetPluginArtifact"
                )
            object.__setattr__(self, "c", artifact.plugin)
            object.__setattr__(self, "e_c", artifact.e_c)
            object.__setattr__(self, "e_c_mapping", artifact.e_c_mapping)
        if self.c is not None and not isinstance(self.c, TargetLayerCPlugin):
            raise BackendProtocolError(
                "C must be a target-bound Layer-C plugin, not Source/Harness C"
            )
        if self.e_c is not None:
            try:
                e_c = (
                    self.e_c
                    if isinstance(self.e_c, LayerCInterface)
                    else LayerCInterface.from_dict(self.e_c)
                )
                mapping = _validate_extension_mapping(
                    e_c,
                    self.c.interface if self.c is not None else LayerCInterface(),
                    self.e_c_mapping,
                )
            except (TypeError, ValueError) as exc:
                raise BackendProtocolError(
                    f"lowering request has invalid E_C mapping: {exc}"
                ) from exc
            object.__setattr__(self, "e_c", e_c)
            object.__setattr__(self, "e_c_mapping", mapping)
            try:
                validate_layer_c_attachments(self.j, e_c)
            except (TypeError, ValueError) as exc:
                raise BackendProtocolError(
                    f"lowering request E_C is detached from canonical J: {exc}"
                ) from exc
        if self.c is not None:
            expected_j_digest = layer_c_content_digest(self.j.to_json())
            if self.c.j_digest != expected_j_digest:
                raise BackendProtocolError(
                    "target Layer-C plugin is detached from canonical J: "
                    f"expected {expected_j_digest}, got {self.c.j_digest}"
                )
            expected_target = TargetIdentity.from_target(self.target)
            if self.c.target != expected_target:
                raise BackendProtocolError(
                    "target Layer-C plugin is bound to a different target: "
                    f"expected {expected_target.to_dict()}, "
                    f"got {self.c.target.to_dict()}"
                )
            if self.c.contract is not None:
                try:
                    self.c.contract.validate_against(
                        self.j,
                        self.c.interface,
                        self.c.choice,
                    )
                except (TypeError, ValueError) as exc:
                    raise BackendProtocolError(
                        f"target realization contract is invalid: {exc}"
                    ) from exc
        raw_bindings = self.constexpr_bindings
        if isinstance(raw_bindings, Mapping):
            raw_bindings = tuple(raw_bindings.items())
        else:
            try:
                raw_bindings = tuple(raw_bindings)
            except TypeError as exc:
                raise BackendProtocolError(
                    "constexpr_bindings must be a name/value mapping"
                ) from exc
        bindings: list[tuple[str, bool | int | float]] = []
        seen: set[str] = set()
        logical_inputs = {item.name: item for item in self.j.intent.inputs}
        allowed = {
            parameter.logical_ref
            for parameter in self.j.signature.parameters
            if parameter.kind.value == "constexpr"
            and parameter.logical_ref in logical_inputs
        }
        for item in raw_bindings:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise BackendProtocolError(
                    "constexpr_bindings entries must be (name, value) pairs"
                )
            name, value = item
            if not isinstance(name, str) or not name:
                raise BackendProtocolError(
                    "constexpr binding names must be non-empty strings"
                )
            if name in seen:
                raise BackendProtocolError(
                    f"duplicate constexpr binding {name!r}"
                )
            if name not in allowed:
                raise BackendProtocolError(
                    "constexpr binding is not authorized by J.signature: "
                    + name
                )
            if not isinstance(value, (bool, int, float)):
                raise BackendProtocolError(
                    f"constexpr binding {name!r} must be a numeric scalar"
                )
            seen.add(name)
            bindings.append((name, value))
        object.__setattr__(self, "constexpr_bindings", tuple(sorted(bindings)))

class ComparisonKind(str, Enum):
    SOURCE_TO_LOWERING = "source_to_lowering"
    REPLAY_TO_LOWERING = "replay_to_lowering"
    INTENT_TO_LOWERING = "intent_to_lowering"
    ORACLE_TO_LOWERING = "oracle_to_lowering"


@dataclass(frozen=True)
class ComparisonEvidence:
    kind: ComparisonKind
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ComparisonKind):
            object.__setattr__(self, "kind", ComparisonKind(self.kind))
        if not isinstance(self.passed, bool):
            raise BackendProtocolError("comparison passed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BackendConformanceRecord:
    target: BackendTarget
    variant: str
    boundary_consumption: Mapping[str, Any]
    test_domain: Mapping[str, Any]
    comparisons: tuple[ComparisonEvidence, ...]

    def __post_init__(self) -> None:
        if not self.comparisons:
            raise BackendProtocolError("backend conformance requires comparison evidence")
        if not bool(self.boundary_consumption.get("supported", False)):
            raise BackendProtocolError("backend did not consume complete Layer B")
        target_realization = self.boundary_consumption.get("target_realization")
        if not isinstance(target_realization, Mapping):
            raise BackendProtocolError(
                "backend conformance requires typed target realization evidence"
            )
        required_evidence_keys = {
            "schema",
            "supported",
            "declared_record_ids",
            "consumed_record_ids",
            "unconsumed_record_ids",
            "applications",
            "artifact_digests",
        }
        if set(target_realization) != required_evidence_keys:
            raise BackendProtocolError(
                "backend target realization evidence has an incomplete schema"
            )
        if target_realization["schema"] != "target_record_consumption":
            raise BackendProtocolError(
                "backend target realization evidence has an unsupported schema"
            )
        if target_realization["supported"] is not True:
            raise BackendProtocolError(
                "backend target realization evidence is not supported"
            )
        declared = target_realization["declared_record_ids"]
        consumed = target_realization["consumed_record_ids"]
        unconsumed = target_realization["unconsumed_record_ids"]
        applications = target_realization["applications"]
        artifact_digests = target_realization["artifact_digests"]
        if not all(
            isinstance(value, list)
            for value in (declared, consumed, unconsumed, applications, artifact_digests)
        ):
            raise BackendProtocolError(
                "backend target realization evidence has invalid array fields"
            )
        try:
            checked = tuple(
                TargetRecordConsumption.from_dict(item) for item in applications
            )
        except (TypeError, ValueError) as exc:
            raise BackendProtocolError(
                "backend target realization evidence lacks typed applications"
            ) from exc
        application_record_ids = [item.record_id for item in checked]
        if len(application_record_ids) != len(set(application_record_ids)):
            raise BackendProtocolError(
                "backend target realization evidence contains duplicate records"
            )
        if declared != list(dict.fromkeys(declared)) or consumed != list(dict.fromkeys(consumed)):
            raise BackendProtocolError(
                "backend target realization evidence contains duplicate record ids"
            )
        if not all(isinstance(item, str) for item in declared + consumed + unconsumed):
            raise BackendProtocolError(
                "backend target realization evidence record ids must be strings"
            )
        if consumed != application_record_ids or set(declared) != set(consumed):
            raise BackendProtocolError(
                "backend target realization record consumption is not exact"
            )
        if unconsumed or artifact_digests != sorted({item.artifact_digest for item in checked}):
            raise BackendProtocolError(
                "backend target realization artifact evidence is not canonical"
            )
        application_digests = [item.application_digest for item in checked]
        if len(application_digests) != len(set(application_digests)):
            raise BackendProtocolError(
                "backend target realization applications are not unique"
            )
        artifact_owners: dict[str, str] = {}
        for item in checked:
            previous = artifact_owners.get(item.artifact_digest)
            if previous is not None:
                raise BackendProtocolError(
                    "backend target realization records must have independent "
                    "compiler artifacts: "
                    f"{item.record_id!r} reuses {previous!r}"
                )
            artifact_owners[item.artifact_digest] = item.record_id

    def assert_conformant(self) -> None:
        failed = [item.kind.value for item in self.comparisons if not item.passed]
        if failed:
            raise BackendProtocolError("backend comparison failed: " + ", ".join(failed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "variant": self.variant,
            "boundary_consumption": dict(self.boundary_consumption),
            "test_domain": dict(self.test_domain),
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


__all__ = [
    "BackendConformanceRecord", "BackendProtocolError", "BackendTarget",
    "ComparisonEvidence", "ComparisonKind", "LoweringRequest",
    "TargetBindingContext", "TargetExtensionMapping", "TargetPluginArtifact",
    "bind_target",
    "bind_target_artifact", "bind_target_from_e_c", "deserialize_target_artifact",
    "deserialize_target_plugin", "extension_mapping_digest", "serialize_target_artifact",
    "serialize_target_plugin", "validate_target_plugin",
]
