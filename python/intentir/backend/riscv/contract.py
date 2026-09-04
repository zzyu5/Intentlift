"""Fail-closed Layer-B ContractView for the native RISC-V backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from intentir.backend.contract_capabilities import derive_search_capabilities
from intentir.backend.contract_updates import (
    canonical_node_table,
    consume_exact_internal_overwrite_update,
)
from intentir.backend.materialize import BackendSearchState
from intentir.ir.intent_op_contract import RAGGED_OPS
from intentir.ir.model import (
    validate_ragged_region_boundaries,
    walk_intent_op_paths,
)
from intentir.verify.gate1.model import RefinementKind


CONTRACT_VIEW_VERSION = "intentir.backend.riscv.contract-view.v1"


class RiscvContractError(ValueError):
    pass


@dataclass(frozen=True)
class RiscvContractView:
    target: Mapping[str, Any]
    consumed: tuple[str, ...]
    unsupported: tuple[str, ...]
    capabilities: tuple[str, ...]
    version: str = CONTRACT_VIEW_VERSION
    ragged_ops: tuple[str, ...] = ()
    dynamic_ragged_member_output: bool = False
    ragged_lowering: str | None = None

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": dict(self.target),
            "supported": self.supported,
            "consumed": list(self.consumed),
            "unsupported": list(self.unsupported),
            "capabilities": list(self.capabilities),
            "ragged_ops": list(self.ragged_ops),
            "dynamic_ragged_member_output": self.dynamic_ragged_member_output,
            "ragged_lowering": self.ragged_lowering,
        }


def _riscv_vlen_bits(state: BackendSearchState) -> int:
    declared: list[int] = []
    malformed: list[str] = []
    for feature in state.target.features:
        normalized = feature.casefold()
        if not normalized.startswith("vlen"):
            continue
        suffix = normalized.removeprefix("vlen")
        if not suffix.isdigit() or suffix.startswith("0"):
            malformed.append(feature)
            continue
        declared.append(int(suffix))
    if malformed:
        raise RiscvContractError(
            "RISC-V target contains malformed VLEN feature(s): "
            + ", ".join(sorted(malformed))
        )
    if len(declared) != 1:
        raise RiscvContractError(
            "RISC-V target must declare exactly one non-conflicting vlenN feature"
        )
    value = declared[0]
    if value < 64 or value % 8:
        raise RiscvContractError(
            "RISC-V target vlenN feature must be a byte-aligned VLEN of at least 64"
        )
    return value


def validate_riscv_target_contract(state: BackendSearchState) -> int:
    """Bind the logical target to this process's real LLVM RVV toolchain.

    Plugin-declared capabilities are deliberately absent here.  The backend
    derives authority from the lowering state plus the configured TRIPLE and
    MATTR that are actually passed to ``opt``/``llc``.
    """

    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V target validation requires BackendSearchState")
    # Local import avoids a contract<->compiler import cycle while still
    # validating the exact process configuration consumed by opt/llc.
    from intentir.backend.riscv.compile import MATTR, TRIPLE

    target = state.target
    if target.family != "riscv":
        raise RiscvContractError("RISC-V backend requires target family riscv")
    architecture = target.architecture.casefold()
    if not architecture.startswith("rv64"):
        raise RiscvContractError(
            "RISC-V backend requires an rv64 target architecture"
        )
    triple_arch = TRIPLE.split("-", 1)[0].casefold()
    if not triple_arch.startswith("riscv64"):
        raise RiscvContractError(
            "configured INTENT_RISCV_TRIPLE must identify a riscv64 target"
        )
    mattr = {
        item.strip().casefold()
        for item in MATTR.split(",")
        if item.strip()
    }
    if "+v" not in mattr or "-v" in mattr:
        raise RiscvContractError(
            "configured INTENT_RISCV_MATTR must enable the RVV +v extension"
        )
    single_letter_extensions = architecture.split("_", 1)[0][4:]
    if "v" not in single_letter_extensions:
        raise RiscvContractError(
            "RISC-V target architecture must declare the vector extension"
        )
    for extension in ("zfh", "zvfh"):
        architecture_declares = extension in architecture
        mattr_enables = f"+{extension}" in mattr and f"-{extension}" not in mattr
        if architecture_declares != mattr_enables:
            raise RiscvContractError(
                f"RISC-V target architecture and MATTR disagree on {extension}"
            )
    return _riscv_vlen_bits(state)


def riscv_allowed_authorizations(
    state: BackendSearchState,
) -> tuple[str, ...]:
    """Derive the authorization refs this backend may honor from J itself."""

    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V authorization derivation requires BackendSearchState")
    validate_riscv_target_contract(state)
    allowed_kinds = {
        RefinementKind.SIGNED_WIDEN,
        RefinementKind.UNSIGNED_WIDEN,
        RefinementKind.NUMERIC_CAST,
        RefinementKind.ACCUMULATION_DTYPE,
        RefinementKind.REASSOCIATION,
    }
    return tuple(
        item.auth_id
        for item in state.base.j.boundary.realization_authorizations
        if item.kind in allowed_kinds
    )


def _riscv_ragged_facts(
    j: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ragged_ops: set[str] = set()
    consumed: list[str] = []

    for path, operation in walk_intent_op_paths(j.intent.body):
        node_ref = operation.node_id or path
        if operation.op in RAGGED_OPS:
            ragged_ops.add(operation.op)
            consumed.append(f"ragged_op:{operation.op}:{node_ref}")
    return tuple(sorted(ragged_ops)), tuple(sorted(consumed))


def derive_riscv_contract_view(
    state: BackendSearchState,
) -> RiscvContractView:
    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V ContractView requires BackendSearchState")
    validate_riscv_target_contract(state)
    j = state.base.j
    try:
        validate_ragged_region_boundaries(j.intent.body)
    except ValueError as exc:
        raise RiscvContractError(str(exc)) from exc
    ragged_ops, ragged_consumed = _riscv_ragged_facts(j)
    dynamic_ragged_member_output = "ragged_member" in ragged_ops
    ragged_lowering = "shared_carrier_tensor" if ragged_ops else None
    projection = state.base.boundary_projection
    consumed = list(projection.consumed)
    projected_consumed = set(projection.consumed)
    unsupported: list[str] = list(projection.unsupported_lowering)
    if j.boundary.aliases:
        unsupported.append("alias_relation")
    nodes = canonical_node_table(j)
    for update in j.boundary.updates:
        update_record = consume_exact_internal_overwrite_update(
            j,
            update,
            nodes=nodes,
        )
        if update_record is None:
            unsupported.append("logical_update_relation")
        else:
            consumed.append(update_record)
    if j.boundary.access_guard_relations:
        guards_consumed = all(
            any(
                ref.startswith(
                    f"access_guard:{item.access_ref}:{item.value_ref}:"
                )
                and ref.endswith(f":{item.polarity.value}")
                for ref in projected_consumed
            )
            for item in j.boundary.access_guard_relations
        )
        if not guards_consumed:
            # The shared typed boundary materializer is the only authority that
            # may turn an authored access-local guard into an explicit select.
            # Consume its record when complete; otherwise fail closed instead
            # of reconstructing or guessing the guard in the RVV backend.
            unsupported.append("access_guard_materialization_not_implemented")
    for storage in j.storage:
        reference = f"storage:{storage.storage_id}"
        if (
            storage.mutability.value == "mutable"
            and reference not in projected_consumed
        ):
            unsupported.append(
                f"general_mutable_storage:{storage.storage_id}"
            )
        else:
            consumed.append(reference)
    consumed.extend(
        f"effect:{item.effect_id or index}:{item.kind.value}"
        for index, item in enumerate(j.boundary.effects)
    )
    consumed.extend(
        f"reduction:{item.node_ref}" for item in j.boundary.reductions
    )
    consumed.extend(
        f"result_relation:{item.output}:{item.kind.value}"
        for item in j.boundary.result_relations
    )
    consumed.extend(
        f"output:{item.output}" for item in j.boundary.output_mappings
    )
    consumed.extend(f"scope:{item.scope_id}" for item in j.scopes)
    consumed.extend(state.base.consumed_structures)
    consumed.extend(ragged_consumed)
    if ragged_ops:
        consumed.append("ragged:shared_carrier_tensor_lowering")
    if dynamic_ragged_member_output:
        consumed.append("ragged:dynamic_member_output")
    view = RiscvContractView(
        target=state.target.to_dict(),
        consumed=tuple(sorted(set(consumed))),
        unsupported=tuple(sorted(set(unsupported))),
        capabilities=derive_search_capabilities(state.base),
        ragged_ops=ragged_ops,
        dynamic_ragged_member_output=dynamic_ragged_member_output,
        ragged_lowering=ragged_lowering,
    )
    if not view.supported:
        raise RiscvContractError(
            "RISC-V backend cannot consume complete Layer B: "
            + ", ".join(view.unsupported)
        )
    return view


__all__ = [
    "CONTRACT_VIEW_VERSION",
    "RiscvContractError",
    "RiscvContractView",
    "derive_riscv_contract_view",
    "riscv_allowed_authorizations",
    "validate_riscv_target_contract",
]
