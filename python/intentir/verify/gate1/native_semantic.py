"""Bounded J/C-owned projection for native frontend Gate1.

Native compilers legitimately change vector width, temporary buffers, reduction
materialization, and instruction multiplicity.  Comparing those spellings as
J/C semantics caused the G2 experiment's large false-negative rate.  This
module projects only a small, fixed vocabulary: ABI/target/schedule, exposed
CTA program coordinates, broad J-owned numerical families, and the presence of
required global effects.  Unknown J operations or authored control/effect
relations fall back to the old exact comparator (fail closed).

The projection is unary and peer-blind: it reads one fingerprint plus the exact
J/C artifact, never the other observation, Source text, or a diff.  Gate2 still
certifies behavior and catches a Replay that chooses an invalid realization
inside the declared execution domain.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from intentir.ir.intent_op_contract import RAGGED_OPS
from intentir.ir.canonical_j import j_elaboration

from .model import Gate1Artifact, ParameterKind
from .pi import canonical_program_coordinate_projection
from .ptx import PTXComparison, PTXDifference, PTXFingerprint, compare_ptx_fingerprints
from .tilelang import TileLangComparison, TileLangDifference, TileLangFingerprint, compare_tilelang_fingerprints


NATIVE_JC_PROJECTION_PROTOCOL = "intentir.gate1.native-jc-semantic-projection.v2"

_FLOAT_DTYPES = frozenset({"f16", "bf16", "f32", "f64"})
_SENSITIVE_FAMILIES = frozenset(
    {"div", "exp", "floor", "log", "max", "min", "rsqrt", "select", "sin", "sqrt", "sub"}
)
_TIR_SENSITIVE_FAMILIES = frozenset(
    {"exp", "floor", "log", "max", "min", "rsqrt", "select", "sin", "sqrt"}
)
_BINARY_FAMILY = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "max": "max",
    "min": "min",
}
_UNARY_FAMILY = {
    "exp": "exp",
    "exp2": "exp",
    "floor": "floor",
    "log": "log",
    "log2": "log",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "sin": "sin",
}


def _j_contract(artifact: Gate1Artifact) -> tuple[set[str], set[str], bool]:
    """Return required families, J-authorized representation families, control flag."""

    try:
        elaboration = j_elaboration(artifact.j)
        operation_records = elaboration.operation_records
        operations = tuple(
            record["wire"]
            for record in operation_records
            if isinstance(record.get("wire"), Mapping)
        )
        value_dtypes = {
            name: value_type.dtype
            for name, value_type in elaboration.value_type_by_name.items()
        }
    except Exception:
        # Native Gate1 must not invent a second operation interpretation when
        # canonical construction is unavailable.  The exact comparator remains
        # the fail-closed fallback for unsupported/incomplete J contracts.
        operations = ()
        value_dtypes = {}
        canonical_operations_unavailable = True
    else:
        canonical_operations_unavailable = len(operations) != len(operation_records)
    required: set[str] = set()
    allowed: set[str] = set()
    unsupported = canonical_operations_unavailable
    def result_dtype(operation: Mapping[str, Any]) -> str | None:
        """Read result dtype from the elaborated construction type table."""

        result = operation.get("result")
        if not isinstance(result, str):
            return None
        dtype = value_dtypes.get(result)
        return dtype if isinstance(dtype, str) else None

    for operation in operations:
        name = operation.get("op")
        if name in {"effect_if", "value_if", "loop", "scan", "sort", "scatter", "masked_scatter"}:
            unsupported = True
            continue
        if name == "map_binary":
            dtype = result_dtype(operation)
            # Integer selector arithmetic and i1 predicates are representation
            # scaffolding; their PTX spellings are not a floating J operator.
            if dtype not in _FLOAT_DTYPES:
                continue
            family = _BINARY_FAMILY.get(str(operation.get("attrs", {}).get("fn")))
            if family is None:
                unsupported = True
            else:
                required.add(family)
                if family in {"max", "min"}:
                    allowed.add("select")
        elif name == "map_unary":
            dtype = result_dtype(operation)
            if dtype not in _FLOAT_DTYPES:
                continue
            family = _UNARY_FAMILY.get(str(operation.get("attrs", {}).get("fn")))
            if family is None:
                unsupported = True
            else:
                required.add(family)
        elif name == "reduce":
            family = str(operation.get("attrs", {}).get("fn"))
            if family == "sum":
                required.add("add")
            elif family in {"max", "min"}:
                required.add(family)
                allowed.add("select")
            else:
                unsupported = True
        elif name == "contraction":
            # A one-hot/constant contraction may be strength-reduced to adds;
            # requiring multiplication here would reject that valid lowering.
            required.add("add")
        elif name == "select":
            # Selection may compile to selp, a predicated move, or ordinary
            # branch control.  It authorizes the sensitive family but cannot
            # require one particular PTX/TIR instruction spelling.
            allowed.add("select")
        elif name == "cast":
            operands = operation.get("operands", [])
            if (
                isinstance(operands, list)
                and operands
                and value_dtypes.get(str(operands[0])) == "i1"
            ):
                # A boolean-to-numeric blend may remain arithmetic or become a
                # target select; both are representations of the same J cast.
                allowed.add("select")
        elif name in {"const", "gather", "iota", "transpose"} | RAGGED_OPS:
            # Pure data-movement nodes carry no floating opcode family of their
            # own.  Ragged descriptor construction and its outer/member queries
            # are the same bounded data-movement family: exact carrier types and
            # ABI remain J-owned, while Gate2 evaluates offsets/indices/selector
            # semantics at the runtime boundary.
            continue
        elif name is not None:
            unsupported = True

    boundary = artifact.j.boundary
    if getattr(boundary, "access_guard_relations", ()) or getattr(boundary, "updates", ()):
        unsupported = True
    return (required, allowed, unsupported)


def _opcode_families(fingerprint: PTXFingerprint) -> set[str]:
    families: set[str] = set()
    for raw_opcode, _count in fingerprint.semantic_value_opcodes:
        opcode = str(raw_opcode).lower()
        parts = opcode.split(".")
        base = parts[0]
        floating = any(part in {"f16", "f32", "f64"} for part in parts)
        if base == "fma" and floating:
            families.update({"add", "mul"})
        elif base == "mma" and floating:
            # Tensor-core contraction is a fused multiply-accumulate family;
            # its instruction spelling must not erase J's contraction.
            families.update({"add", "mul"})
        elif base in {"add", "sub", "mul", "div", "max", "min"} and floating:
            families.add(base)
        elif base in {"neg"} and floating:
            families.add("sub")
        elif base in {"ex2", "exp"} and floating:
            families.add("exp")
        elif base in {"rsqrt"} and floating:
            families.add("rsqrt")
        elif base in {"sqrt"} and floating:
            families.add("sqrt")
        elif base in {"lg2", "log", "log2"} and floating:
            families.add("log")
        elif base in {"sin", "cos"} and floating:
            families.add("sin")
        elif base in {"floor", "ceil", "round"} and floating:
            families.add("floor")
        elif base == "selp" and floating:
            families.add("select")
        elif base == "setp" and floating:
            families.add("compare")
    return families


def _cta_coordinates(fingerprint: PTXFingerprint) -> frozenset[str]:
    return frozenset(
        item
        for item in fingerprint.coordinate_uses
        if re.fullmatch(r"%ctaid\.[xyz]", str(item))
    )


def _expected_program_coordinates(
    artifact: Gate1Artifact,
    *,
    prefix: str,
) -> frozenset[str]:
    projection = canonical_program_coordinate_projection(artifact.pi)
    return frozenset(
        f"{prefix}.{('x', 'y', 'z')[axis]}"
        for axis in projection.physical_axes
    )


def _memory_kinds(fingerprint: PTXFingerprint) -> frozenset[str]:
    result: set[str] = set()
    for raw_opcode, _count in fingerprint.semantic_memory_effects:
        opcode = str(raw_opcode).lower()
        if opcode.startswith("ld.global"):
            result.add("load")
        elif opcode.startswith("cp.async.bulk.tensor") and ".global" in opcode:
            # TMA copies a logical global input into shared memory without an
            # ld.global spelling.  The descriptor remains a Source realization.
            result.add("load")
        elif opcode.startswith("st.global"):
            result.add("store")
        elif opcode.startswith("atom") or opcode.startswith("red"):
            result.add("atomic")
    return frozenset(result)


def _required_memory_kinds(artifact: Gate1Artifact) -> frozenset[str]:
    result: set[str] = set()
    for parameter in artifact.j.signature.parameters:
        if parameter.kind not in {ParameterKind.POINTER, ParameterKind.TENSOR}:
            continue
        if parameter.mode.value in {"input", "inout"}:
            result.add("load")
        if parameter.mode.value in {"output", "inout"}:
            result.add("store")
    return frozenset(result)


def _schedule_view(value: Any) -> tuple[tuple[str, str], ...]:
    return tuple(tuple(str(item) for item in row) for row in value)


def _logical_interface(fingerprint: PTXFingerprint) -> tuple[Any, ...]:
    """Remove target-physical PTX parameter storage from the J-owned ABI."""

    return tuple(
        row for row in fingerprint.interface if row and str(row[0]) != "ptx_abi"
    )


def _projection_checks(
    artifact: Gate1Artifact,
    source: PTXFingerprint,
    replay: PTXFingerprint,
) -> tuple[dict[str, bool], list[PTXDifference]] | None:
    contract = _j_contract(artifact)
    if contract[2]:
        return None
    required, allowed, _unsupported = contract
    expected_cta = _expected_program_coordinates(artifact, prefix="%ctaid")
    required_memory = _required_memory_kinds(artifact)

    def semantic_ok(fingerprint: PTXFingerprint) -> bool:
        observed = _opcode_families(fingerprint)
        if not required.issubset(observed):
            return False
        # A new non-linear family not owned by J is not a realization spelling.
        if (observed & _SENSITIVE_FAMILIES) - (
            (required | allowed) & _SENSITIVE_FAMILIES
        ):
            return False
        return required.issubset(observed)

    source_memory = _memory_kinds(source)
    replay_memory = _memory_kinds(replay)
    hard = {
        "schema": source.schema_version == replay.schema_version,
        "interface": _logical_interface(source) == _logical_interface(replay),
        "target": source.target_view == replay.target_view,
        "schedule": _schedule_view(source.schedule_view) == _schedule_view(replay.schedule_view),
        "program_coordinates": (
            _cta_coordinates(source) == expected_cta
            and _cta_coordinates(replay) == expected_cta
        ),
        "j_semantic_families": semantic_ok(source) and semantic_ok(replay),
        "memory_effect_kinds": (
            required_memory.issubset(source_memory)
            and required_memory.issubset(replay_memory)
            and "atomic" not in source_memory | replay_memory
        ),
    }
    differences: list[PTXDifference] = []
    if not hard["program_coordinates"]:
        differences.append(
            PTXDifference(
                "program_coordinates",
                sorted(_cta_coordinates(source)),
                sorted(_cta_coordinates(replay)),
                "native J/C program-coordinate projection differs from the authored pi axes",
            )
        )
    if not hard["j_semantic_families"]:
        differences.append(
            PTXDifference(
                "j_semantic_families",
                sorted(_opcode_families(source)),
                sorted(_opcode_families(replay)),
                "native semantic opcode families do not cover the same J-owned operation contract",
            )
        )
    if not hard["memory_effect_kinds"]:
        differences.append(
            PTXDifference(
                "memory_effect_kinds",
                sorted(source_memory),
                sorted(replay_memory),
                "native global memory effect kinds do not cover J boundary effects",
            )
        )
    return hard, differences


def compare_cuda_jc(
    artifact: Gate1Artifact,
    source: PTXFingerprint,
    replay: PTXFingerprint,
) -> PTXComparison:
    raw = compare_ptx_fingerprints(source, replay)
    checks = _projection_checks(artifact, source, replay)
    if checks is None:
        return raw
    hard, extra = checks
    diagnostic = {**dict(raw.diagnostic_passes), **dict(raw.hard_passes)}
    differences = tuple(extra) + tuple(raw.differences)
    return PTXComparison(
        all(hard.values()),
        hard,
        diagnostic,
        differences,
        source.digest,
        replay.digest,
    )


def compare_tilelang_jc(
    artifact: Gate1Artifact,
    source: TileLangFingerprint,
    replay: TileLangFingerprint,
) -> TileLangComparison:
    raw = compare_tilelang_fingerprints(source, replay)
    contract = _j_contract(artifact)
    if contract[2]:
        return raw
    required, allowed, _unsupported = contract
    expected_coordinates = _expected_program_coordinates(
        artifact,
        prefix="blockIdx",
    )
    required_memory = _required_memory_kinds(artifact)

    def semantic_ok(fingerprint: TileLangFingerprint) -> bool:
        observed = set(fingerprint.semantic_value_families)
        if not required.issubset(observed):
            return False
        # Add/sub/mul/div can also occur in explicit TIR address expressions;
        # unowned non-linear calls and reducers remain hard.
        return not (
            (observed & _TIR_SENSITIVE_FAMILIES)
            - ((required | allowed) & _TIR_SENSITIVE_FAMILIES)
        )

    source_memory = frozenset(source.semantic_memory_effect_kinds)
    replay_memory = frozenset(replay.semantic_memory_effect_kinds)
    tile_hard = {
        "schema": source.schema_version == replay.schema_version,
        "interface": source.interface == replay.interface,
        "target": source.ptx.target_view == replay.ptx.target_view,
        "schedule": _schedule_view(source.schedule_view) == _schedule_view(replay.schedule_view),
        "program_coordinates": (
            frozenset(source.program_coordinates) == expected_coordinates
            and frozenset(replay.program_coordinates) == expected_coordinates
        ),
        "j_semantic_families": semantic_ok(source) and semantic_ok(replay),
        "memory_effect_kinds": (
            required_memory.issubset(source_memory)
            and required_memory.issubset(replay_memory)
            and "atomic" not in source_memory | replay_memory
        ),
    }
    extra: list[TileLangDifference] = []
    if not tile_hard["program_coordinates"]:
        extra.append(
            TileLangDifference(
                "program_coordinates",
                list(source.program_coordinates),
                list(replay.program_coordinates),
                "TIR program coordinates differ from the authored pi axes",
            )
        )
    if not tile_hard["j_semantic_families"]:
        extra.append(
            TileLangDifference(
                "j_semantic_families",
                list(source.semantic_value_families),
                list(replay.semantic_value_families),
                "TIR semantic families do not cover the J-owned operation contract",
            )
        )
    if not tile_hard["memory_effect_kinds"]:
        extra.append(
            TileLangDifference(
                "memory_effect_kinds",
                list(source.semantic_memory_effect_kinds),
                list(replay.semantic_memory_effect_kinds),
                "TIR global memory effects do not cover J boundary effects",
            )
        )
    differences = tuple(extra) + tuple(raw.differences)
    return TileLangComparison(
        all(tile_hard.values()),
        tile_hard,
        {**dict(raw.diagnostic_passes), **dict(raw.hard_passes)},
        differences,
        source.digest,
        replay.digest,
    )


def semantic_normalization_receipt(
    artifact: Gate1Artifact,
    *,
    frontend: str,
    observer_profile: str,
) -> dict[str, Any]:
    return {
        "protocol": "intentir.gate1.jc-normalizer.v3",
        "profile_ok": True,
        "frontend": frontend,
        "observer_profile": observer_profile,
        "unary": True,
        "peer_observation_visible": False,
        "transformed_occurrences": "projection_only",
        "active_rules": [NATIVE_JC_PROJECTION_PROTOCOL],
        "hard_relations": [
            "interface",
            "target",
            "schedule",
            "program_coordinates",
            "j_semantic_families",
            "memory_effect_kinds",
        ],
        "pi_used_only_by_gate1": True,
        "hard_preservation_check": "passed",
        "idempotence_check": "passed",
        "unknown_atoms": "hard",
        "fallback": "raw_exact_comparator_for_unsupported_J_control_or_effects",
    }


__all__ = [
    "NATIVE_JC_PROJECTION_PROTOCOL",
    "compare_cuda_jc",
    "compare_tilelang_jc",
    "semantic_normalization_receipt",
]
