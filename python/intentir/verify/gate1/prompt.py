"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_construction import (
    CANONICAL_C_WIRE_FIELDS,
    CANONICAL_J_WIRE_FIELDS,
    CanonicalHandleAllocator,
)
from intentir.ir.canonical_j import canonicalize_gate1_artifact, j_elaboration
from .codegen import (
    TRITON_REPLAY_ABI_CONTRACT,
    TRITON_REPLAY_API_CONTRACT,
    TRITON_REPLAY_CONTROL_CONTRACT,
    TRITON_REPLAY_STRUCTURE_CONTRACT,
)
from .model import (
    Frontend,
    Gate1Artifact,
    ImplementationConfig,
    SchemaError,
    _nested_root_owned_field_paths,
    opaque_replay_representation,
)
from .pi import per_program_scalar_keepdims_reductions


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v3"
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Write one self-contained public Triton module with exactly one "
            "@triton.jit entry function named scrc_k0. The module must contain "
            "exact `import triton` and `import triton.language as tl` statements; "
            "the only optional import is "
            "`from triton.language.extra.cuda import libdevice`. Its top level may "
            "contain only those imports and `@triton.jit` function definitions, "
            "with no launch, specialization assignment, prose, or Markdown fence. "
            "Its parameters are the declared "
            "J.signature handles, followed in order by complete C slots whose kind "
            "is constexpr and which remain annotated tl.constexpr. If U_f contains "
            "callable_abi_order, use that exact anonymous declaration order instead "
            "of the canonical J-then-C order; it carries ordering only and does not "
            "change J/C ownership. Their C values "
            "are exact compile bindings, not replacements for the callable handles; "
            "kind=compiler slots belong only to the external invocation. When C is "
            "absent, compile the parameterized J directly. J alone supplies semantics. "
            "Physical realization remains the Replay compiler's source-blind target "
            "choice and is checked only outside this packet. If U_f contains a "
            "representation_contract, it is a closed typed ABI/launch binding; "
            "use it exactly and do not infer a replacement from parameter order. "
            + TRITON_REPLAY_API_CONTRACT.strip()
            + " "
            + TRITON_REPLAY_ABI_CONTRACT.strip()
            + " "
            + TRITON_REPLAY_CONTROL_CONTRACT.strip()
            + " "
            + TRITON_REPLAY_STRUCTURE_CONTRACT.strip()
        ),
    },
    "cute": {
        "retrieval_scope": "public NVIDIA CuTe documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry": "scrc_entry",
        "compile_entry_contract": (
            "Define one @cute.jit scrc_entry after the requested @cute.kernel. "
            "Its positional arguments are J tensor inputs, inouts, then outputs, "
            "followed by stream; construct representation-only descriptors inside it. "
            "Optional C values may configure representation or launch choices but "
            "never add device or entry runtime parameters."
        ),
    },
    "tilelang": {
        "retrieval_scope": "public TileLang documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one self-contained Python module using exactly "
            "`import tilelang.language as T` and one `@T.prim_func` named "
            "`scrc_k0`. Express tensor ABI through public `T.Tensor` annotations, "
            "device launch through `T.Kernel`, and realization topology through "
            "public TileLang operations. C values are compile-time schedule "
            "constants, never additional logical runtime parameters."
        ),
    },
    "cuda": {
        "retrieval_scope": "public CUDA C++ programming documentation and examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one standalone CUDA C++ translation unit containing exactly "
            "one `extern \"C\" __global__` kernel named `scrc_k0`. Its positional "
            "runtime parameters follow J.signature; C values are compiled "
            "schedule constants and are not runtime parameters. Use only public "
            "CUDA language constructs and intrinsics. Do not emit host code, "
            "launch syntax, file I/O, inline PTX, build directives, or `main`."
        ),
    },
}

_FORBIDDEN_KEYS = frozenset(
    {
        "source_code",
        "source_span",
        "source_name",
        "location_map",
        "provenance",
        "response_text",
        "candidate",
        "compiler_diff",
        "selected_cards",
        "registry_digest",
        "harness",
        "interface",
        "binding_map",
        "compiler_fingerprint",
        "gate1_diff",
        "prior_replay",
        "conversation",
    }
)

_REPLAY_COMPONENT_KEYS = {
    # Replay transports an accepted J view without the outer artifact
    # envelope.  Its field set must still come from the canonical J wire
    # authority; keeping a second hand-written list here lets Replay accept a
    # shape that CandidateCheck/Gate1 do not understand.
    "j": frozenset(CANONICAL_J_WIRE_FIELDS),
    "c": frozenset(CANONICAL_C_WIRE_FIELDS),
}
_U_F_KEYS = frozenset(
    {
        "protocol",
        "frontend",
        "replay_function",
        "callable_abi_order",
        "retrieval_scope",
        "tool_access",
        "compile_entry",
        "compile_entry_contract",
        "representation_contract",
        "semantic_view",
    }
)
_U_F_PACKET_KEYS = frozenset({"protocol", "frontend", "replay_function"})

_CALLABLE_ABI_HANDLE = re.compile(r"(?:p|c)[0-9]+")

_REPRESENTATION_CONTRACT_KEYS = frozenset(
    {"address_bindings", "program_bindings"}
)
_ADDRESS_BINDING_KEYS = frozenset(
    {"carrier", "access_ref", "tensor", "axis", "role"}
)
_PROGRAM_BINDING_KEYS = frozenset({"domain_ref", "axis"})


def _validate_representation_contract(
    value: Any,
    *,
    j: Mapping[str, Any],
    path: str,
) -> None:
    """Validate the closed source-blind ABI/launch representation view."""

    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    unknown = sorted(set(value) - _REPRESENTATION_CONTRACT_KEYS)
    if unknown:
        raise SchemaError(
            f"{path} has unknown field(s): {unknown}"
        )
    if not any(value.get(key) for key in _REPRESENTATION_CONTRACT_KEYS):
        raise SchemaError(f"{path} must contain at least one binding")

    signature = j.get("signature")
    raw_parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    parameters = {
        str(item.get("name")): item
        for item in raw_parameters or []
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    boundary = j.get("boundary")
    raw_accesses = (
        boundary.get("accesses")
        if isinstance(boundary, Mapping)
        else None
    )
    accesses = {
        str(item.get("access_id")): item
        for item in raw_accesses or []
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }
    raw_domains = boundary.get("domains") if isinstance(boundary, Mapping) else None
    domains = set(raw_domains) if isinstance(raw_domains, Mapping) else set()
    intent = j.get("intent")
    intent_function = (
        intent.get("intent_function")
        if isinstance(intent, Mapping)
        else None
    )
    tensor_ranks = {
        str(item.get("name")): len(item.get("shape", []))
        for key in ("inputs", "outputs")
        for item in (
            intent_function.get(key, [])
            if isinstance(intent_function, Mapping)
            else []
        )
        if isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("shape"), list)
    }

    def require_handle(raw: Any, pattern: str, field_path: str) -> str:
        if not isinstance(raw, str) or re.fullmatch(pattern, raw) is None:
            raise SchemaError(
                f"{field_path} must be an anonymous declaration-order handle"
            )
        return raw

    address_bindings = value.get("address_bindings", [])
    if not isinstance(address_bindings, list):
        raise SchemaError(f"{path}.address_bindings must be an array")
    seen_address_rows: set[tuple[str, str, int, str]] = set()
    seen_access_axes: dict[tuple[str, int], str] = {}
    for index, raw in enumerate(address_bindings):
        item_path = f"{path}.address_bindings[{index}]"
        if not isinstance(raw, Mapping):
            raise SchemaError(f"{item_path} must be an object")
        actual = set(raw)
        if actual != _ADDRESS_BINDING_KEYS:
            missing = sorted(_ADDRESS_BINDING_KEYS - actual)
            extra = sorted(actual - _ADDRESS_BINDING_KEYS)
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"unknown={extra}")
            raise SchemaError(f"{item_path} has invalid fields ({', '.join(detail)})")
        carrier = require_handle(
            raw["carrier"], r"p[0-9]+", f"{item_path}.carrier"
        )
        access_ref = require_handle(
            raw["access_ref"], r"a[0-9]+", f"{item_path}.access_ref"
        )
        tensor = require_handle(
            raw["tensor"], r"t[0-9]+", f"{item_path}.tensor"
        )
        parameter = parameters.get(carrier)
        if (
            parameter is None
            or parameter.get("kind") != "scalar"
            or parameter.get("logical_ref") is not None
        ):
            raise SchemaError(
                f"{item_path}.carrier must be a null-logical-ref scalar parameter"
            )
        access = accesses.get(access_ref)
        if access is None:
            raise SchemaError(f"{item_path}.access_ref references unknown access")
        if access.get("tensor") != tensor:
            raise SchemaError(
                f"{item_path}.tensor must equal the referenced access tensor"
            )
        axis = raw["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise SchemaError(f"{item_path}.axis must be a non-negative integer")
        rank = tensor_ranks.get(tensor)
        if rank is None or axis >= rank:
            raise SchemaError(f"{item_path}.axis exceeds the referenced tensor rank")
        if raw["role"] != "stride":
            raise SchemaError(f"{item_path}.role must be 'stride'")
        row_key = (carrier, access_ref, axis, raw["role"])
        if row_key in seen_address_rows:
            raise SchemaError(f"{item_path} duplicates an address ownership row")
        seen_address_rows.add(row_key)
        access_axis = (access_ref, axis)
        previous = seen_access_axes.get(access_axis)
        if previous is not None and previous != carrier:
            raise SchemaError(
                f"{item_path} conflicts with another carrier for the same access axis"
            )
        seen_access_axes[access_axis] = carrier

    program_bindings = value.get("program_bindings", [])
    if not isinstance(program_bindings, list):
        raise SchemaError(f"{path}.program_bindings must be an array")
    seen_program_domains: set[str] = set()
    for index, raw in enumerate(program_bindings):
        item_path = f"{path}.program_bindings[{index}]"
        if not isinstance(raw, Mapping):
            raise SchemaError(f"{item_path} must be an object")
        actual = set(raw)
        if actual != _PROGRAM_BINDING_KEYS:
            missing = sorted(_PROGRAM_BINDING_KEYS - actual)
            extra = sorted(actual - _PROGRAM_BINDING_KEYS)
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"unknown={extra}")
            raise SchemaError(f"{item_path} has invalid fields ({', '.join(detail)})")
        domain_ref = require_handle(
            raw["domain_ref"], r"d[0-9]+", f"{item_path}.domain_ref"
        )
        if domain_ref not in domains:
            raise SchemaError(
                f"{item_path}.domain_ref references an unknown J domain"
            )
        axis = raw["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
            raise SchemaError(
                f"{item_path}.axis must be a launch axis in [0, 2]"
            )
        if domain_ref in seen_program_domains:
            raise SchemaError(
                f"{item_path}.domain_ref duplicates a program ownership row"
            )
        seen_program_domains.add(domain_ref)


def _validate_callable_abi_order(
    value: Any,
    *,
    j: Mapping[str, Any],
    c: Mapping[str, Any] | None,
    path: str,
) -> None:
    """Validate an anonymous Source-observed callable declaration order.

    J and C keep their semantic/implementation ownership split.  This optional
    sidecar carries only the order in which the already-anonymized callable
    exposes those handles.  It is therefore safe for source-blind Replay while
    still preserving ABIs in which a C constexpr is interleaved with J
    parameters.
    """

    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{path} must be an array")
    handles = list(value)
    if not all(
        isinstance(handle, str) and _CALLABLE_ABI_HANDLE.fullmatch(handle)
        for handle in handles
    ):
        raise SchemaError(
            f"{path} must contain only anonymous p*/c* callable handles"
        )
    if len(handles) != len(set(handles)):
        raise SchemaError(f"{path} must not contain duplicate handles")

    signature = j.get("signature")
    parameters = (
        signature.get("parameters")
        if isinstance(signature, Mapping)
        else None
    )
    expected = [
        str(item.get("name"))
        for item in parameters or []
        if isinstance(item, Mapping)
    ]
    if isinstance(c, Mapping):
        slots = c.get("slots")
        expected.extend(
            str(item.get("name"))
            for item in slots or []
            if isinstance(item, Mapping) and item.get("kind") == "constexpr"
        )
    expected_set = set(expected)
    actual_set = set(handles)
    if actual_set != expected_set or len(handles) != len(expected):
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if len(handles) != len(expected):
            details.append(f"expected_count={len(expected)}")
            details.append(f"actual_count={len(handles)}")
        raise SchemaError(
            f"{path} must exactly cover J.signature and C constexpr handles"
            + (f" ({', '.join(details)})" if details else "")
        )


def _walk(value: Any, path: str = "$" ):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _validate_source_isolated_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate the single current source-isolated Replay packet format."""

    if not isinstance(payload, Mapping):
        raise SchemaError("Replay payload must be an object")
    root = set(payload)
    active_roots = ({"j", "u_f"}, {"j", "c", "u_f"})
    if root not in active_roots:
        raise SchemaError(
            "Replay payload root must contain j and u_f, with c only when "
            "implementation parameters are present"
        )
    j = payload.get("j")
    if isinstance(j, Mapping):
        nested_root_owned = _nested_root_owned_field_paths(
            j,
            path="Replay payload j",
        )
        if nested_root_owned:
            raise SchemaError(
                f"{nested_root_owned[0]} is a root-owned field nested under J; "
                "root.c and root.pi must be siblings of root.j"
            )
    for component, expected in _REPLAY_COMPONENT_KEYS.items():
        if component not in payload:
            if component == "c":
                continue
            raise SchemaError(f"Replay payload is missing {component}")
        value = payload[component]
        actual = set(value) if isinstance(value, Mapping) else set()
        if not isinstance(value, Mapping) or actual != expected:
            raise SchemaError(
                f"Replay payload {component} must contain exactly "
                + ", ".join(sorted(expected))
            )
    c = payload.get("c")
    if c is not None:
        assert isinstance(c, Mapping)
        try:
            config = ImplementationConfig.from_json(dict(c))
        except (SchemaError, TypeError, ValueError) as exc:
            raise SchemaError(f"Replay payload c is invalid: {exc}") from exc
        if not config.slots:
            raise SchemaError(
                "Replay payload c requires at least one slot; omit c for J-only Replay"
            )
        expected_names = list(
            CanonicalHandleAllocator.canonical_names("c", len(config.slots))
        )
        if [slot.name for slot in config.slots] != expected_names:
            raise SchemaError(
                "Replay payload c slots must use declaration-order handles c0, c1, ..."
            )
        config.validate_complete_values(path="Replay payload c")
    u_f = payload["u_f"]
    if not isinstance(u_f, Mapping):
        raise SchemaError("Replay payload u_f must be an object")
    unknown_u_f = sorted(set(u_f) - _U_F_KEYS)
    if unknown_u_f:
        raise SchemaError(
            f"Replay payload u_f has unknown field(s): {unknown_u_f}"
        )
    if "representation_contract" in u_f:
        _validate_representation_contract(
            u_f["representation_contract"],
            j=payload["j"],
            path="Replay payload u_f.representation_contract",
        )
    if "callable_abi_order" in u_f:
        _validate_callable_abi_order(
            u_f["callable_abi_order"],
            j=payload["j"],
            c=c,
            path="Replay payload u_f.callable_abi_order",
        )
    required_u_f = set(_U_F_PACKET_KEYS)
    missing_u_f = sorted(required_u_f - set(u_f))
    if missing_u_f:
        raise SchemaError(
            f"Replay payload u_f is missing field(s): {missing_u_f}"
        )
    if u_f["protocol"] != REPLAY_PACKET_SCHEMA:
        raise SchemaError("Replay payload u_f uses an unsupported protocol")
    try:
        frontend = Frontend(u_f["frontend"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("Replay payload u_f has an unsupported frontend") from exc
    if u_f["replay_function"] != "scrc_k0":
        raise SchemaError("Replay payload u_f must use canonical replay_function")
    if "semantic_view" in u_f and u_f["semantic_view"] != "masked_j":
        raise SchemaError(
            "Replay payload u_f.semantic_view must be 'masked_j' when present"
        )
    fixed_context = PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {})
    missing_context = sorted(set(fixed_context) - set(u_f))
    if missing_context:
        raise SchemaError(
            "Replay payload u_f is missing fixed context field(s): "
            f"{missing_context}"
        )
    modified_context = sorted(
        key for key, value in fixed_context.items() if u_f.get(key) != value
    )
    if modified_context:
        raise SchemaError(
            "Replay payload u_f does not match the fixed public frontend "
            f"context field(s): {modified_context}"
        )
    for path, value in _walk(dict(payload)):
        if isinstance(value, dict):
            forbidden = sorted(set(value) & _FORBIDDEN_KEYS)
            if "source" in value:
                forbidden.append("source")
            if forbidden:
                raise SchemaError(
                    f"Replay payload {path} contains privileged key(s): {forbidden}"
                )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    for token in (
        "Rule" + "Card",
        "Rule" + "Registry",
        "compose_" + "provenance",
        "selected_" + "rule_cards",
    ):
        if token in encoded:
            raise SchemaError("Replay payload contains retired control-plane data")


def assert_source_isolated_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed unless a Replay packet uses the current source-isolated form."""

    _validate_source_isolated_payload(payload)


@dataclass(frozen=True)
class ReplayPacket:
    """Typed source-isolated request; this is not a fourth proof artifact."""

    j: Mapping[str, Any]
    u_f: Mapping[str, Any]
    c: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        assert_source_isolated_payload(self.to_json(validate=False))

    def to_json(self, *, validate: bool = True) -> dict[str, Any]:
        payload = {
            "j": copy.deepcopy(dict(self.j)),
            "u_f": copy.deepcopy(dict(self.u_f)),
        }
        if self.c is not None:
            payload["c"] = copy.deepcopy(dict(self.c))
        if validate:
            assert_source_isolated_payload(payload)
        return payload




def build_rewriter_packet(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
    callable_abi_order: Sequence[str] | None = None,
) -> ReplayPacket:
    """Build J plus the accepted Source/Harness-bound C when present.

    Replay does not derive a target-specific C.  The artifact C has already
    crossed the Source/Harness binding boundary and Gate1 has checked its
    values against the Source compile schedule before this packet is used.
    """

    # Local import avoids a package initialization cycle: TypeCheck depends on
    # the Gate1 data model, while this builder is its first production consumer.
    from intentir.verify.typecheck import type_check

    artifact = canonicalize_gate1_artifact(artifact)
    if artifact.c.slots:
        artifact.c.validate_complete_values(path="Replay active C")
    report = type_check(
        artifact.j,
        artifact.c,
        artifact.pi,
        require_program_axes=True,
    )
    if not report.passed:
        summary = ", ".join(
            f"{item.code.value}@{item.path}" for item in report.diagnostics
        )
        raise SchemaError("Replay requires TypeCheck(J,C,pi)=PASS: " + summary)
    view = artifact.rewriter_view(masked=masked)
    frontend = Frontend(frontend)
    context = copy.deepcopy(PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {}))
    context.update(
        {
            "protocol": REPLAY_PACKET_SCHEMA,
            "frontend": frontend.value,
            "replay_function": "scrc_k0",
        }
    )
    if callable_abi_order is not None:
        context["callable_abi_order"] = list(callable_abi_order)
    if masked:
        # The masked-J diagnostic arm deliberately does not contain a J which
        # can cross normal canonical elaboration. Mark that host-produced
        # projection explicitly so Replay ingress may consume only its
        # rank/ABI-preserving surface without treating an arbitrary malformed
        # normal packet as masked input.
        context["semantic_view"] = "masked_j"
    representation_contract = opaque_replay_representation(artifact)
    if representation_contract:
        context["representation_contract"] = representation_contract
    c_view = view["c"] if artifact.c.slots else None
    return ReplayPacket(j=view["j"], u_f=context, c=c_view)


def build_rewriter_payload(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
    callable_abi_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build exactly ``J + source-bound C + U_f`` when C is present."""

    return build_rewriter_packet(
        artifact,
        frontend=frontend,
        masked=masked,
        callable_abi_order=callable_abi_order,
    ).to_json()


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    callable_abi_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=False,
        callable_abi_order=callable_abi_order,
    )


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    callable_abi_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=True,
        callable_abi_order=callable_abi_order,
    )


def _source_blind_structure_ledger(artifact: Gate1Artifact) -> str:
    """Render a compact, exact structural view derived only from J.

    The full J remains the semantic authority, but its boundary and operation
    records are deliberately distributed across several fields.  Responses
    Replay has no episode-local query tool, so repeat the source-independent
    occurrence ledger in the instruction channel.  This makes distinct
    ``access_ref`` rows mechanically visible without exposing Source, pi, or
    any observation from the other arm.
    """
    # Prompt facts must use the same canonical operation/output table as
    # Replay and Gate1.  Reading only IntentFunction.body here would silently
    # drop loop/conditional descendants and would make output_mappings a
    # second output authority.
    artifact = canonicalize_gate1_artifact(artifact)
    elaboration = j_elaboration(artifact.j)
    j_view = artifact.rewriter_view(masked=False)
    j = j_view.get("j") if isinstance(j_view, Mapping) else None
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    accesses_value = (
        boundary.get("accesses") if isinstance(boundary, Mapping) else None
    )
    effects_value = (
        boundary.get("effects") if isinstance(boundary, Mapping) else None
    )
    accesses = {
        str(item.get("access_id")): item
        for item in accesses_value or []
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }
    ordered_refs: list[str] = []
    for effect in effects_value or []:
        if not isinstance(effect, Mapping):
            continue
        access_ref = effect.get("access_ref")
        if isinstance(access_ref, str) and access_ref in accesses:
            ordered_refs.append(access_ref)
    if not ordered_refs:
        ordered_refs = list(accesses)
    access_rows = [
        {
            "access_ref": access_ref,
            "kind": accesses[access_ref].get("kind"),
            "tensor": accesses[access_ref].get("tensor"),
            "index_expression": accesses[access_ref].get("index_expression"),
            "node_ref": accesses[access_ref].get("node_ref"),
        }
        for access_ref in ordered_refs
    ]

    operation_records = elaboration.operation_records
    reductions = [
        {
            "result": item["results"][0] if item.get("results") else None,
            "operand": (
                item["operands"][0]
                if item.get("operands")
                else None
            ),
            "fn": (item.get("attrs") or {}).get("fn")
            if isinstance(item.get("attrs"), Mapping)
            else None,
            "axes": (item.get("attrs") or {}).get("axes", [])
            if isinstance(item.get("attrs"), Mapping)
            else [],
            "input_shape": (
                (item.get("type") or {}).get("shape", [])
                if isinstance(item.get("type"), Mapping)
                else []
            ),
            "result_shape": [],
        }
        for item in operation_records
        if item.get("op") == "reduce"
        and len(item.get("results", ())) == 1
    ]
    broadcasts = [
        {
            "node_ref": item.get("node_ref"),
            "operand": item.get("operand"),
            "axes": item.get("axes"),
        }
        for item in (boundary.get("broadcasts", []) if isinstance(boundary, Mapping) else [])
        if isinstance(item, Mapping)
    ]
    body_by_result = {
        result: item
        for item in operation_records
        for result in item.get("results", ())
    }
    body_by_id = {
        item["id"]: item
        for item in operation_records
    }
    output_values_by_node: dict[str, set[str]] = {}
    for binding in elaboration.output_bindings:
        if binding.node_ref is not None:
            output_values_by_node.setdefault(binding.node_ref, set()).add(binding.value)
    output_value_by_node = {
        node_ref: next(iter(values))
        for node_ref, values in output_values_by_node.items()
        if len(values) == 1
    }
    effect_position_by_access = {
        str(effect.get("access_ref")): index
        for index, effect in enumerate(
            boundary.get("effects", []) if isinstance(boundary, Mapping) else []
        )
        if isinstance(effect, Mapping)
        and isinstance(effect.get("access_ref"), str)
    }
    for row in access_rows:
        node_ref = row.get("node_ref")
        row["effect_index"] = effect_position_by_access.get(row["access_ref"])
        row["output_value"] = output_value_by_node.get(node_ref)

    def provenance_path(
        value: Any,
        target: Any,
        active: set[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        if value == target:
            return []
        if not isinstance(value, str):
            return None
        active = set() if active is None else set(active)
        if value in active:
            return None
        producer = body_by_result.get(value)
        if producer is None:
            return None
        active.add(value)
        for operand in producer.get("operands", []):
            path = provenance_path(operand, target, active)
            if path is not None:
                return [
                    {
                        "id": producer.get("id"),
                        "op": producer.get("op"),
                        "result": value,
                    },
                    *path,
                ]
        return None

    def first_identity_breaker(
        path: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        # The path is listed from the access toward the consumer below.  Keep
        # this deliberately conservative: zero is preserved by multiplication,
        # while add/sub require the outside identity to be re-materialized.
        for step in path:
            if step.get("op") != "map_binary":
                continue
            operation = body_by_id.get(str(step.get("id")))
            fn = (
                operation.get("attrs", {}).get("fn")
                if isinstance(operation, Mapping)
                and isinstance(operation.get("attrs"), Mapping)
                else None
            )
            if fn in {"add", "sub"}:
                return {**step, "fn": fn}
        return None

    consumer_extensions = []
    for item in (
        boundary.get("consumer_extensions", [])
        if isinstance(boundary, Mapping)
        else []
    ):
        if not isinstance(item, Mapping):
            continue
        access_ref = item.get("access_ref")
        access = accesses.get(access_ref) if isinstance(access_ref, str) else None
        consumer = body_by_id.get(str(item.get("consumer_ref")))
        operand_index = item.get("operand_index")
        operand = None
        if (
            isinstance(consumer, Mapping)
            and isinstance(operand_index, int)
            and not isinstance(operand_index, bool)
            and isinstance(consumer.get("operands"), (list, tuple))
            and 0 <= operand_index < len(consumer["operands"])
        ):
            operand = consumer["operands"][operand_index]
        path = (
            provenance_path(operand, access.get("tensor"))
            if isinstance(access, Mapping)
            else None
        )
        forward_path = list(reversed(path or []))
        identity_breaker = first_identity_breaker(forward_path)
        chain_operations = [
            body_by_id.get(str(step.get("id")))
            for step in forward_path
        ]
        closed_exp_identity = (
            isinstance(access, Mapping)
            and access.get("fill_value") == "-inf"
            and isinstance(item.get("outside"), Mapping)
            and item["outside"].get("value") == 0
            and any(
                isinstance(operation, Mapping)
                and operation.get("op") == "map_binary"
                and isinstance(operation.get("attrs"), Mapping)
                and operation["attrs"].get("fn") == "sub"
                for operation in chain_operations
            )
            and any(
                isinstance(operation, Mapping)
                and operation.get("op") == "map_unary"
                and isinstance(operation.get("attrs"), Mapping)
                and operation["attrs"].get("fn") == "exp"
                for operation in chain_operations
            )
        )
        consumer_extensions.append(
            {
                "access_ref": access_ref,
                "consumer_ref": item.get("consumer_ref"),
                "operand_index": operand_index,
                "consumer_operand": operand,
                "producer_chain": forward_path,
                "identity_breaker": identity_breaker,
                "identity_form": (
                    "closed_neg_inf_to_zero_through_exp"
                    if closed_exp_identity
                    else (
                        "generic_post_consumer_identity"
                        if identity_breaker is not None
                        else "already_closed"
                    )
                ),
                "requires_post_consumer_identity": (
                    identity_breaker is not None and not closed_exp_identity
                ),
                "access_fill_value": (
                    access.get("fill_value")
                    if isinstance(access, Mapping)
                    else None
                ),
                "tile_axis": item.get("tile_axis"),
                "outside": item.get("outside"),
            }
        )
    for reduction in reductions:
        result = reduction.get("result")
        producer = body_by_result.get(str(result))
        if isinstance(producer, Mapping) and isinstance(
            producer.get("type"), Mapping
        ):
            reduction["result_shape"] = producer["type"].get("shape", [])

    payload = {
        "boundary_transfer_count": len(access_rows),
        "boundary_transfers_in_effect_order": access_rows,
        "consumer_extensions_in_J_order": consumer_extensions,
        "broadcasts_in_J_order": broadcasts,
        "reductions_in_J_order": reductions,
    }
    return (
        "Exact source-blind J structural ledger (derived only from the packet; "
        "not a Source recipe): emit one distinct target memory transfer for each "
        "boundary row, in this order. Distinct access_ref values are distinct "
        "occurrences even when tensor and address are equal; never cache, hoist, "
        "or merge one row into another. Emit one target local reduction for each "
        "listed J reduction and preserve its operand/dataflow. For every listed "
        "consumer extension, follow the listed producer_chain, and make its typed "
        "outside identity reach that exact consumer operand. For an entry whose "
        "identity_form is closed_neg_inf_to_zero_through_exp, preserve the "
        "masked load's other=-inf and choose one of these equivalent closed "
        "spellings: feed the exp result directly to the reduction, use "
        "where(mask, exp_result, 0.0) after exp, or use "
        "where(mask, centered_value, -inf) immediately before exp. Never use "
        "where(mask, centered_value, 0.0) before exp, because exp(0)=1 and that "
        "is not the outside identity. For other entries, when the ledger says "
        "requires_post_consumer_identity=true, apply one predicate/select "
        "immediately after the listed identity_breaker and before later producer "
        "steps; do not delay it until after an identity-preserving map. When "
        "false and the typed load fill already equals the outside identity, do "
        "not add a redundant select. A load fill is not a substitute when a later map changes the value, and shared accesses keep "
        "one consumer edge per extension. Preserve every listed broadcast at its "
        "node and operand. The ledger is closed input, not optional advice:\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def _replay_abi_audit(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend,
    callable_abi_order: Sequence[str] | None = None,
) -> str:
    """Render the exact callable ABI audit for the current Replay packet.

    The static frontend contract describes the rule, but a source-blind model
    still has to apply it to the anonymous handles in this particular J/C
    packet.  Keep the concrete declaration-order lists in the instruction
    channel so a ``p*`` handle cannot silently change kind between J and the
    target source.  This is diagnostic guidance only; the ingress checker
    remains authoritative.
    """

    parameters = tuple(artifact.j.signature.parameters)
    if frontend is Frontend.TRITON:
        j_names = [parameter.name for parameter in parameters]
        j_constexpr = [
            parameter.name
            for parameter in parameters
            if getattr(parameter.kind, "value", parameter.kind) == "constexpr"
        ]
        c_constexpr = [
            slot.name
            for slot in artifact.c.slots
            if getattr(slot.kind, "value", slot.kind) == "constexpr"
        ]
        expected = (
            list(callable_abi_order)
            if callable_abi_order is not None
            else [*j_names, *c_constexpr]
        )
        runtime = [name for name in expected if name not in {*j_constexpr, *c_constexpr}]
        representation = opaque_replay_representation(artifact)
        address_bindings = representation.get("address_bindings", [])
        program_bindings = representation.get("program_bindings", [])
        return (
            "\nCRITICAL final ABI audit for this exact Triton packet: the entry "
            "parameter names and order MUST be "
            + json.dumps(expected, ensure_ascii=False)
            + "; the ONLY parameters that may carry `tl.constexpr` are "
            + json.dumps([*j_constexpr, *c_constexpr], ensure_ascii=False)
            + "; these parameters MUST remain unannotated runtime handles: "
            + json.dumps(runtime, ensure_ascii=False)
            + ". Do not infer kind from the p* spelling, use of a value, or "
            "parameter position. The optional callable_abi_order is only an "
            "anonymous Source ABI ordering witness; it does not authorize moving "
            "a handle between J and C. Before returning, inspect the complete function "
            "signature against these three lists; a mismatch is an invalid "
            "Replay source even if the body is otherwise semantically plausible. "
            "The following exact representation rows are mandatory, not examples: "
            "for every address row, use its carrier in the pointer expression for "
            "the named access and multiply the coordinate of the named tensor axis "
            "by that carrier; do not omit a row, substitute an extent, or use the "
            "carrier as a value/control input. Address rows="
            + json.dumps(address_bindings, sort_keys=True, ensure_ascii=False)
            + "; program rows="
            + json.dumps(program_bindings, sort_keys=True, ensure_ascii=False)
            + ".\n"
        )
    if frontend is Frontend.CUDA:
        return (
            "\nCRITICAL final ABI audit for this exact CUDA packet: keep the "
            "kernel's positional runtime parameters exactly in J.signature order; "
            "C slots are compile/launch choices and must not become extra kernel "
            "parameters.\n"
        )
    if frontend is Frontend.TILELANG:
        return (
            "\nCRITICAL final ABI audit for this exact TileLang packet: keep the "
            "declared J.signature handles in order; C slots are compile-time "
            "schedule choices, never extra logical runtime parameters.\n"
        )
    return (
        "\nCRITICAL final ABI audit for this exact CuTe packet: preserve the "
        "declared J.signature positional runtime ABI; C slots configure only "
        "representation or launch and never become device/entry runtime "
        "parameters.\n"
    )


def _concise_instructions(
    *,
    artifact: Gate1Artifact,
    frontend: Frontend,
    masked: bool,
    uses_c: bool,
    structure_ledger: str,
    callable_abi_order: Sequence[str] | None = None,
) -> str:
    output = "CUDA C++" if frontend is Frontend.CUDA else "Python"
    masked_clause = (
        " J is masked for this representation audit; do not reconstruct its "
        "semantic values. The presence of a masked `predicate` field is still "
        "structural: emit a valid boolean mask for that transfer. The presence "
        "of a masked `fill_value` field is also structural: pass a type-correct "
        "`other` only together with that mask; never call `tl.load` with "
        "`other` and no `mask`. An access row without a predicate has neither "
        "a target mask nor an `other` argument. Preserve the masked transfer "
        "count and effect order."
        if masked
        else ""
    )
    if not uses_c:
        implementation_clause = (
            " No C is required: compile the parameterized J directly and retain "
            "its declared ABI placeholders."
        )
    elif frontend is Frontend.TRITON:
        implementation_clause = (
            " Complete C is present: use its values as implementation bindings "
            "while retaining every constexpr c* callable handle."
        )
    else:
        implementation_clause = (
            " Complete C is present: use its values only for implementation and "
            "compile choices according to U_f; do not add logical runtime parameters."
        )
    frontend_clause = {
        Frontend.TRITON: (
            " This episode targets Triton Python; apply the Triton callable and "
            "program-id rules in U_f."
        ),
        Frontend.CUTE: (
            " This episode targets NVIDIA CuTe; use the CuTe callable entry "
            "contract in U_f and do not assume Triton TTIR."
        ),
        Frontend.TILELANG: (
            " This episode targets TileLang, not Triton; emit the canonical "
            "TileLang PrimFunc in U_f and do not use Triton APIs or TTIR."
        ),
        Frontend.CUDA: (
            " This episode targets CUDA C++, not Triton; emit the canonical "
            "extern C __global__ kernel in U_f and do not use Python, tl.* APIs, "
            "or Triton TTIR assumptions."
        ),
    }[frontend]
    scalar_keepdims = per_program_scalar_keepdims_reductions(
        artifact.j,
        artifact.pi,
    )
    projection_clause = (
        " J/pi prove that rowwise keepdims reductions "
        + ",".join(scalar_keepdims)
        + " have one scalar carrier per directly mapped target row. Keep that "
        "carrier scalar; never use tl.reshape to make `(1,)`, and broadcast only "
        "at an actual vector consumer."
        if scalar_keepdims
        else ""
    )
    return (
        "You are one fresh source-blind Replay coding agent. Return one complete "
        f"self-contained {output} source module and no prose. J is the sole semantic "
        "authority. Compile the whole typed J "
        "without adding value semantics. Physical layout, program mapping, loops, "
        "control and target APIs are your source-blind compiler choices, not extra "
        "semantic inputs. Use only public frontend APIs and the canonical "
        "J.signature and optional C handles. Source, Gate feedback, previous Replay and "
        "kernel templates are unavailable. The packet's U_f gives the callable ABI "
        "and, when present, a closed source-blind representation contract. Its "
        "compile_entry_contract is the exact target API, ABI, control and structural "
        "ingress contract; follow it completely. For Triton, audit its exact imports, "
        "top-level-only JIT definitions, and required scrc_k0 entry before returning."
        + projection_clause
        + structure_ledger
        + implementation_clause
        + masked_clause
        + frontend_clause
        + _replay_abi_audit(
            artifact,
            frontend=frontend,
            callable_abi_order=callable_abi_order,
        )
    )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
    callable_abi_order: Sequence[str] | None = None,
) -> tuple[str, str]:
    payload = build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=masked,
        callable_abi_order=callable_abi_order,
    )
    frontend = Frontend(frontend)
    return _concise_instructions(
        artifact=artifact,
        frontend=frontend,
        masked=masked,
        uses_c=bool(artifact.c.slots),
        structure_ledger=_source_blind_structure_ledger(artifact),
        callable_abi_order=callable_abi_order,
    ), json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=False,
    )


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=True,
    )


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "REPLAY_PACKET_SCHEMA",
    "ReplayPacket", "assert_source_isolated_payload",
    "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_rewriter_packet", "build_rewriter_prompt",
]
