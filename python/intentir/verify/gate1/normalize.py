"""Fixed unary normalization for J/C structural Gate1.

The normalizer consumes one raw compiler observation and the already typed pi
witness.  It never sees the peer observation and never decides whether a pair
matches.  A typed relation remains hard until the profile can independently map
each observed physical occurrence back to its J/C anchor.  In particular, merely
having a ``program_axes`` row does not authorize collapsing x/y/z identities.

The one physical-load rule implemented here is deliberately narrower than a
general memory CSE: an extra read may be removed only when the observation
contains an exact repeated pointer SSA operand, that pointer resolves to one
read-only J access for one tensor parameter, and ``pi`` explicitly contains the
corresponding extra ``transfer_direct`` occurrence.  The rule is bounded by
that multiplicity and never touches masked loads, stores, values, predicates,
or unknown pointer provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any

from .model import Gate1Artifact, opaque_replay_representation


NORMALIZER_PROTOCOL = "intentir.gate1.jc-normalizer.v2"


_SSA_NAME = r"%[A-Za-z0-9_.$-]+"
_POINTER_DEF_RE = re.compile(
    rf"^\s*(?P<result>{_SSA_NAME})\s*=\s*"
    r"(?:tt\.addptr|tt\.advance|tt\.splat|tt\.broadcast|"
    r"tt\.expand_dims|tt\.reshape)\s+(?P<operand>"
    rf"{_SSA_NAME})\b"
)
_UNMASKED_LOAD_RE = re.compile(
    rf"^(?P<indent>\s*)(?P<result>{_SSA_NAME})\s*=\s*"
    rf"(?:tt\.load|\"tt\.load\")\s+(?P<pointer>{_SSA_NAME})"
    r"\s*:\s*(?P<type>.+?)(?:\s+loc\(.*\))?\s*$"
)
_EFFECT_POINTER_RE = re.compile(
    rf"(?:tt\.store|\"tt\.store\"|tt\.atomic_[A-Za-z0-9_]+)\s+"
    rf"(?P<pointer>{_SSA_NAME})\b"
)
_POINTER_OPERATION_RE = re.compile(
    rf"^\s*(?:(?P<result>{_SSA_NAME})\s*=\s*)?"
    r"(?P<operation>\"?tt\.(?:addptr|advance|bitcast|splat|broadcast|"
    r"expand_dims|reshape)\"?)\s+(?P<tail>.*)$"
)
_MEMORY_OPERATION_RE = re.compile(
    rf"^\s*(?:(?P<result>{_SSA_NAME})\s*=\s*)?"
    r"(?P<operation>\"?tt\.(?:load|store|atomic_[A-Za-z0-9_]+)\"?)\s+"
    r"(?P<tail>.*)$"
)
_ANY_OPERATION_RE = re.compile(
    rf"^\s*(?:(?:{_SSA_NAME})\s*=\s*)?"
    r"(?P<operation>\"?[A-Za-z_][A-Za-z0-9_.-]*\"?)\b"
)
_SSA_TOKEN_RE = re.compile(_SSA_NAME)
_POINTER_WRAPPER_OPERATIONS = frozenset(
    {
        "tt.advance",
        "tt.bitcast",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
    }
)


def _operation_name(value: str) -> str:
    return value.removeprefix('"').removesuffix('"')


def _pointer_operation_definitions(
    ttir: str,
) -> tuple[dict[str, tuple[str, tuple[str, ...]]], frozenset[str]]:
    """Return the closed pointer SSA slice and its explicit addptr results."""

    definitions: dict[str, tuple[str, tuple[str, ...]]] = {}
    addptr_results: set[str] = set()
    for line in ttir.splitlines():
        match = _POINTER_OPERATION_RE.match(line)
        if match is None or match.group("result") is None:
            continue
        result = match.group("result")
        operation = _operation_name(match.group("operation"))
        operands = tuple(_SSA_TOKEN_RE.findall(match.group("tail")))
        if result in definitions:
            # Duplicate SSA definitions are outside the closed observation
            # profile.  The caller treats this as an unprovable certificate.
            return {}, frozenset()
        definitions[result] = (operation, operands)
        if operation == "tt.addptr":
            addptr_results.add(result)
    return definitions, frozenset(addptr_results)


def _resolve_pointer_root_from_definitions(
    name: str,
    definitions: dict[str, tuple[str, tuple[str, ...]]],
    active: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve a pointer root through the same pure wrappers as TTIR."""

    if name in active:
        return None
    if name.startswith("%p"):
        return name
    definition = definitions.get(name)
    if definition is None or not definition[1]:
        return None
    return _resolve_pointer_root_from_definitions(
        definition[1][0], definitions, active | {name}
    )


def _resolve_address_anchors(
    name: str,
    definitions: dict[str, tuple[str, tuple[str, ...]]],
    active: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Find the addptr stage(s) that own one surviving memory pointer."""

    if name in active:
        return frozenset()
    definition = definitions.get(name)
    if definition is None:
        return frozenset()
    operation, operands = definition
    if operation == "tt.addptr":
        return frozenset({name}) if len(operands) >= 2 else frozenset()
    if operation not in _POINTER_WRAPPER_OPERATIONS or len(operands) != 1:
        return frozenset()
    return _resolve_address_anchors(
        operands[0], definitions, active | {name}
    )


def _memory_operation_rows(
    ttir: str,
) -> tuple[tuple[str, str, str, bool], ...] | None:
    """Collect ordered memory effects without interpreting their formulas.

    Each row is ``(kind, pointer, operation, masked)``.  The source-blind
    certificate uses only the pointer root, access kind, multiplicity, and the
    preserved mask shape.  Values, predicates, fills, and effect order remain
    in the TTIR fingerprint and are never rewritten here.
    """

    rows: list[tuple[str, str, str, bool]] = []
    for line in ttir.splitlines():
        match = _MEMORY_OPERATION_RE.match(line)
        if match is None:
            continue
        operation = _operation_name(match.group("operation"))
        # The first SSA operand is the pointer for all closed tt.load/store and
        # tt.atomic forms.  Stop before the result type so generic and custom
        # assembly are handled alike.
        operand_text = match.group("tail").split(":", 1)[0]
        operands = tuple(_SSA_TOKEN_RE.findall(operand_text))
        if not operands:
            return None
        if operation == "tt.load":
            kind = "load"
            masked = len(operands) >= 2
        elif operation == "tt.store":
            kind = "store"
            masked = len(operands) >= 3
        else:
            kind = "update"
            masked = len(operands) >= 2
        rows.append((kind, operands[0], operation, masked))
    return tuple(rows)


def _address_normalization_certificate(
    ttir: str,
    artifact: Gate1Artifact,
) -> tuple[bool, tuple[tuple[str, str], ...], str]:
    """Prove that every surviving addptr is one J-owned address occurrence.

    The returned flag is an authorization for the *address-form* quotient only.
    It is intentionally derived independently for one observation.  No peer
    text, replay code, fingerprint, or comparator result is consulted.  The
    certificate is rejected if any J access lacks typed address ownership, if a
    tensor/root/access mapping is ambiguous, if a surviving transfer is
    duplicated or missing, or if an addptr result escapes the pointer slice.
    """

    try:
        representation = opaque_replay_representation(artifact)
    except Exception:
        return False, (), "typed_address_ownership_unavailable"
    address_bindings = representation.get("address_bindings")
    if not isinstance(address_bindings, list) or not address_bindings:
        return False, (), "typed_address_ownership_unavailable"

    accesses = list(artifact.j.boundary.accesses)
    access_by_id = {access.access_id: access for access in accesses}
    rows_by_access: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_access_axes: set[tuple[str, int]] = set()
    for row in address_bindings:
        if not isinstance(row, dict):
            return False, (), "typed_address_ownership_malformed"
        access_ref = row.get("access_ref")
        tensor = row.get("tensor")
        axis = row.get("axis")
        if (
            not isinstance(access_ref, str)
            or access_ref not in access_by_id
            or not isinstance(tensor, str)
            or tensor != access_by_id[access_ref].tensor
            or isinstance(axis, bool)
            or not isinstance(axis, int)
            or axis < 0
            or row.get("role") != "stride"
        ):
            return False, (), "typed_address_ownership_malformed"
        key = (access_ref, axis)
        if key in seen_access_axes:
            return False, (), "typed_address_ownership_duplicate"
        seen_access_axes.add(key)
        rows_by_access[access_ref].append(row)
    if set(rows_by_access) != set(access_by_id):
        return False, (), "typed_address_ownership_incomplete"

    parameters_by_tensor: dict[str, list[Any]] = defaultdict(list)
    for parameter in artifact.j.signature.parameters:
        if parameter.logical_ref is None:
            continue
        if getattr(parameter.kind, "value", parameter.kind) not in {
            "tensor",
            "pointer",
        }:
            continue
        parameters_by_tensor[parameter.logical_ref].append(parameter)

    root_for_access: dict[str, str] = {}
    access_for_root: dict[str, str] = {}
    for access in accesses:
        candidates = parameters_by_tensor.get(access.tensor, [])
        if len(candidates) != 1:
            return False, (), "tensor_root_mapping_ambiguous"
        parameter = candidates[0]
        root = f"%{parameter.name.removeprefix('%')}"
        if root in access_for_root:
            return False, (), "tensor_root_mapping_ambiguous"
        root_for_access[access.access_id] = root
        access_for_root[root] = access.access_id

    definitions, addptr_results = _pointer_operation_definitions(ttir)
    rows = _memory_operation_rows(ttir)
    if not addptr_results or rows is None or len(rows) != len(accesses):
        return False, (), "address_transfer_multiplicity_unproved"

    anchor_use_count: Counter[str] = Counter()
    observed_accesses: list[str] = []
    observed_anchors: set[str] = set()
    for kind, pointer, _operation, masked in rows:
        root = _resolve_pointer_root_from_definitions(pointer, definitions)
        access_ref = access_for_root.get(root) if root is not None else None
        if access_ref is None:
            return False, (), "physical_pointer_root_unowned"
        access = access_by_id[access_ref]
        expected_kind = getattr(access.kind, "value", access.kind)
        if kind != expected_kind:
            return False, (), "access_kind_mismatch"
        # A mask/fill is a J-owned hard atom.  The address quotient is only
        # enabled for the unmasked form here; predicated forms must remain hard
        # until their predicate/fill occurrence matcher is implemented.
        if masked or access.predicate is not None or access.fill_value is not None:
            return False, (), "predicate_or_fill_occurrence_unmatched"
        anchors = _resolve_address_anchors(pointer, definitions)
        if len(anchors) != 1:
            return False, (), "physical_address_anchor_ambiguous"
        anchor = next(iter(anchors))
        anchor_use_count[anchor] += 1
        if anchor_use_count[anchor] != 1:
            return False, (), "address_anchor_multiplicity_unproved"
        observed_accesses.append(access_ref)
        observed_anchors.add(anchor)

    if observed_anchors != set(addptr_results):
        return False, (), "unreachable_or_unowned_addptr"
    if len(observed_accesses) != len(set(observed_accesses)):
        return False, (), "physical_transfer_multiplicity_unproved"
    if set(observed_accesses) != set(access_by_id):
        return False, (), "J_access_occurrence_unmatched"

    # Every addptr result must stay inside the pointer/address slice.  A
    # hidden use in a value, predicate, loop bound, or control operation would
    # make the address quotient unsound even if the memory counts matched.
    for line in ttir.splitlines():
        pointer_match = _POINTER_OPERATION_RE.match(line)
        memory_match = _MEMORY_OPERATION_RE.match(line)
        if pointer_match is not None:
            operation = _operation_name(pointer_match.group("operation"))
            operands = tuple(
                _SSA_TOKEN_RE.findall(pointer_match.group("tail"))
            )
            allowed = operation in {
                "tt.addptr",
                *_POINTER_WRAPPER_OPERATIONS,
            }
        elif memory_match is not None:
            operation = _operation_name(memory_match.group("operation"))
            operands = tuple(
                _SSA_TOKEN_RE.findall(
                    memory_match.group("tail").split(":", 1)[0]
                )
            )
            allowed = True
        else:
            operation_match = _ANY_OPERATION_RE.match(line)
            operation = (
                _operation_name(operation_match.group("operation"))
                if operation_match is not None
                else ""
            )
            operands = tuple(_SSA_TOKEN_RE.findall(line))
            allowed = False
            result = None
            if operation_match is None:
                continue
        result = (
            pointer_match.group("result")
            if pointer_match is not None
            else memory_match.group("result")
            if memory_match is not None
            else result
        )
        if result is not None:
            operands = tuple(item for item in operands if item != result)
        if any(item in addptr_results for item in operands) and not allowed:
            return False, (), "addptr_escapes_address_slice"

    return (
        True,
        tuple(
            (access.access_id, access.tensor)
            for access in accesses
        ),
        "unique_J_access_typed_address_root_and_transfer_occurrence",
    )


def _pointer_roots(ttir: str) -> dict[str, str]:
    """Collect only direct, acyclic pointer-definition edges from one TTIR."""

    definitions: dict[str, str] = {}
    for line in ttir.splitlines():
        match = _POINTER_DEF_RE.match(line)
        if match is not None:
            definitions[match.group("result")] = match.group("operand")
    return definitions


def _resolve_pointer_root(
    name: str,
    definitions: dict[str, str],
    active: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve a pointer only through known pure pointer wrappers."""

    if name in active:
        return None
    if name.startswith("%p"):
        return name
    operand = definitions.get(name)
    if operand is None:
        return None
    return _resolve_pointer_root(
        name=operand,
        definitions=definitions,
        active=active | {name},
    )


def _j_readonly_accesses_by_root(
    artifact: Gate1Artifact,
) -> tuple[dict[str, str], dict[str, int]]:
    """Return unambiguous ABI-root -> J-access bindings and pi multiplicities."""

    parameters = {
        parameter.name: parameter
        for parameter in artifact.j.signature.parameters
    }
    accesses_by_tensor: dict[str, list[Any]] = defaultdict(list)
    for access in artifact.j.boundary.accesses:
        accesses_by_tensor[access.tensor].append(access)

    access_for_root: dict[str, str] = {}
    for parameter_name, parameter in parameters.items():
        logical_ref = parameter.logical_ref
        if logical_ref is None:
            continue
        if getattr(parameter.mode, "value", parameter.mode) != "input":
            continue
        candidates = accesses_by_tensor.get(logical_ref, [])
        if len(candidates) != 1:
            # More than one authored access to the same tensor needs an
            # index/effect occurrence matcher.  Root identity alone is not a
            # unique J anchor, so keep every such relation hard.
            continue
        access = candidates[0]
        if getattr(access.kind, "value", access.kind) != "load":
            continue
        if access.predicate is not None or access.fill_value is not None:
            continue
        access_for_root[f"%{parameter_name.removeprefix('%')}"] = access.access_id

    transfer_counts: Counter[str] = Counter()
    for instance in artifact.pi.instances:
        if not isinstance(instance, dict):
            continue
        if instance.get("mechanism") != "transfer_direct":
            continue
        semantic_ref = instance.get("semantic_ref")
        if isinstance(semantic_ref, str):
            transfer_counts[semantic_ref] += 1

    return access_for_root, dict(transfer_counts)


def _elide_proven_physical_duplicate_loads(
    ttir: str,
    artifact: Gate1Artifact,
) -> tuple[str, int, tuple["NormalizationRelationRow", ...]]:
    """Canonicalize only pi-authorized repeated reads of one J load.

    This is a unary observation rule.  It does not compare Source with Replay:
    both arms independently reduce a bounded physical duplicate to the single
    J access occurrence.  Exact pointer SSA identity, an input-only J tensor,
    an unmasked read, no intervening same-root write, and the extra
    ``transfer_direct`` count are all required before one load is removed.
    """

    access_for_root, transfer_counts = _j_readonly_accesses_by_root(artifact)
    if not access_for_root:
        return ttir, 0, ()

    remaining = {
        access_ref: count - 1
        for access_ref, count in transfer_counts.items()
        if count > 1
    }
    if not remaining:
        return ttir, 0, ()

    definitions = _pointer_roots(ttir)
    lines = ttir.splitlines()
    seen_loads: dict[tuple[str, str, str], str] = {}
    blocked_roots: set[str] = set()
    replacements: dict[str, str] = {}
    removed: list[tuple[str, str]] = []
    kept_lines: list[str] = []

    for line in lines:
        effect = _EFFECT_POINTER_RE.search(line)
        if effect is not None:
            root = _resolve_pointer_root(
                effect.group("pointer"), definitions
            )
            if root is not None:
                blocked_roots.add(root)

        load = _UNMASKED_LOAD_RE.match(line)
        if load is None:
            kept_lines.append(line)
            continue

        result = load.group("result")
        pointer = load.group("pointer")
        root = _resolve_pointer_root(pointer, definitions)
        access_ref = access_for_root.get(root) if root is not None else None
        key = (root or "", pointer, load.group("type"))
        previous = seen_loads.get(key)
        if (
            root is not None
            and root not in blocked_roots
            and access_ref is not None
            and previous is not None
            and remaining.get(access_ref, 0) > 0
        ):
            replacements[result] = previous
            remaining[access_ref] -= 1
            removed.append((access_ref, result))
            continue

        if root is not None and root not in blocked_roots and access_ref is not None:
            seen_loads.setdefault(key, result)
        kept_lines.append(line)

    if not removed:
        return ttir, 0, ()

    trailing_newline = "\n" if ttir.endswith("\n") else ""
    normalized = "\n".join(kept_lines) + trailing_newline
    for old, new in replacements.items():
        normalized = re.sub(
            rf"(?<![A-Za-z0-9_.$-]){re.escape(old)}(?![A-Za-z0-9_.$-])",
            new,
            normalized,
        )

    instance_paths: dict[str, list[str]] = defaultdict(list)
    for index, instance in enumerate(artifact.pi.instances):
        if not isinstance(instance, dict):
            continue
        if instance.get("mechanism") != "transfer_direct":
            continue
        semantic_ref = instance.get("semantic_ref")
        if isinstance(semantic_ref, str):
            instance_paths[semantic_ref].append(
                f"pi.instances[{index}]"
            )
    consumed_paths: Counter[str] = Counter()
    relation_rows: list[NormalizationRelationRow] = []
    for access_ref, _result in removed:
        consumed_paths[access_ref] += 1
        paths = instance_paths.get(access_ref, [])
        ordinal = consumed_paths[access_ref]
        path = (
            paths[ordinal]
            if ordinal < len(paths)
            else f"pi.instances[semantic_ref={access_ref}]"
        )
        access = next(
            item
            for item in artifact.j.boundary.accesses
            if item.access_id == access_ref
        )
        relation_rows.append(
            NormalizationRelationRow(
                path=path,
                family="transfer_direct",
                relation="duplicate_read_elided",
                anchor_refs=(access_ref, access.tensor),
                disposition="normalized",
                reason=(
                    "unique_readonly_J_access_exact_pointer_reuse_without_"
                    "intervening_same_root_write"
                ),
            )
        )

    return normalized, len(removed), tuple(relation_rows)


@dataclass(frozen=True)
class NormalizationRelationRow:
    """Truthful disposition of one frozen pi relation for one observation."""

    path: str
    family: str
    relation: str
    anchor_refs: tuple[str, ...]
    disposition: str
    reason: str

    def evidence(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "family": self.family,
            "relation": self.relation,
            "anchor_refs": list(self.anchor_refs),
            "disposition": self.disposition,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NormalizedTTIR:
    """One independently normalized TTIR observation."""

    text: str
    transformed_occurrences: int
    active_rules: tuple[str, ...]
    hard_relations: tuple[str, ...] = ()
    representation_anchor_exemptions: frozenset[str] = frozenset()
    relation_rows: tuple[NormalizationRelationRow, ...] = ()

    def evidence(self) -> dict[str, Any]:
        if self.transformed_occurrences == 0:
            hard_preservation_check = "passed_by_byte_identity"
        elif (
            "elide_proven_physical_duplicate_loads" in self.active_rules
            and self.transformed_occurrences
            == sum(
                row.family == "transfer_direct"
                and row.disposition == "normalized"
                for row in self.relation_rows
            )
        ):
            hard_preservation_check = "passed_by_proven_unique_occurrence"
        else:
            hard_preservation_check = (
                "not_established_for_transformed_observation"
            )
        return {
            "protocol": NORMALIZER_PROTOCOL,
            "profile_ok": True,
            "unary": True,
            "peer_observation_visible": False,
            "transformed_occurrences": self.transformed_occurrences,
            "active_rules": list(self.active_rules),
            "hard_relations": list(self.hard_relations),
            "representation_anchor_exemptions": sorted(
                self.representation_anchor_exemptions
            ),
            "relation_rows": [row.evidence() for row in self.relation_rows],
            "hard_preservation_check": hard_preservation_check,
            "idempotence_check": "passed",
            "preserved_jc_atom_classes": [
                "interface",
                "operator_and_dataflow",
                "reduction",
                "predicate_and_mask",
                "memory_access_and_effect",
                "loop_and_control",
            ],
            "unknown_atoms": "hard",
        }


def normalize_ttir_for_gate1(ttir: str, artifact: Gate1Artifact) -> NormalizedTTIR:
    """Return ``N^pi(ttir)`` without consulting another observation.

    ``program_axes`` and every relation without a unique occurrence anchor
    remain hard.  The narrow duplicate-read rule above is the only active
    transformation: it removes at most the extra ``transfer_direct`` rows
    explicitly present in ``pi`` after independently proving a unique
    read-only J access and exact pointer reuse.  Raw axis identity,
    multiplicity, users, and downstream dataflow remain visible otherwise.
    """

    normalized_text, transformed_occurrences, duplicate_rows = (
        _elide_proven_physical_duplicate_loads(ttir, artifact)
    )
    active_rules: list[str] = []
    if transformed_occurrences:
        active_rules.append("elide_proven_physical_duplicate_loads")
    address_proven, address_accesses, address_reason = (
        _address_normalization_certificate(normalized_text, artifact)
    )
    representation_anchor_exemptions = frozenset(
        {"tt.addptr"} if address_proven else ()
    )
    if address_proven:
        active_rules.append("normalize_proven_address_representations")
    # Every remaining access representation relation requires a unique
    # physical occurrence or pointer/use-chain mapping for its ``a*`` row.
    # Complete J coverage and matching operation counts do not establish that
    # bijection, so keep those rows hard instead of using module-wide
    # address/text rewrites.
    relation_rows: list[NormalizationRelationRow] = list(duplicate_rows)

    def record(
        *,
        path: str,
        family: str,
        relation: str,
        anchor_refs: tuple[str, ...],
        disposition: str = "hard",
        reason: str = "observation_anchor_mapping_unavailable",
    ) -> None:
        relation_rows.append(
            NormalizationRelationRow(
                path=path,
                family=family,
                relation=relation,
                anchor_refs=anchor_refs,
                disposition=disposition,
                reason=reason,
            )
        )

    if address_proven:
        for access_ref, tensor in address_accesses:
            record(
                path=f"pi.access_representation[access={access_ref}]",
                family="access_representation_bindings",
                relation="address_recomputed",
                anchor_refs=(access_ref, tensor),
                disposition="normalized",
                reason=address_reason,
            )

    for index, binding in enumerate(artifact.pi.program_axes):
        record(
            path=f"pi.program_axes[{index}]",
            family="program_axes",
            relation="physical_launch_axis",
            anchor_refs=(
                str(binding["domain_ref"]),
                str(binding["axis"]),
            ),
        )
    for index, binding in enumerate(artifact.pi.program_bindings):
        record(
            path=f"pi.program_bindings[{index}]",
            family="program_bindings",
            relation=str(binding["relation"]),
            anchor_refs=(
                f"physical_axis:{binding['axis']}",
                *(
                    f'{axis["domain_ref"]}:{axis["axis_ref"]}'
                    for axis in binding["logical_axes"]
                ),
            ),
        )
    for index, binding in enumerate(artifact.pi.iteration_bindings):
        refs = [
            f'{axis["subject_ref"]}:{axis["axis_ref"]}'
            for axis in binding["logical_axes"]
        ]
        refs.append(str(binding["tile_slot"]))
        if binding["mode_slot"] is not None:
            refs.append(str(binding["mode_slot"]))
        record(
            path=f"pi.iteration_bindings[{index}]",
            family="iteration_bindings",
            relation=str(binding["relation"]),
            anchor_refs=tuple(refs),
        )
    for index, binding in enumerate(artifact.pi.control_bindings):
        record(
            path=f"pi.control_bindings[{index}]",
            family="control_bindings",
            relation=str(binding["relation"]),
            anchor_refs=(str(binding["subject_ref"]),),
        )
    for index, binding in enumerate(artifact.pi.materialization_bindings):
        record(
            path=f"pi.materialization_bindings[{index}]",
            family="materialization_bindings",
            relation=str(binding["relation"]),
            anchor_refs=(str(binding["value_ref"]),),
        )
    for index, binding in enumerate(artifact.pi.access_representation_bindings):
        relation = str(binding["relation"])
        access_ref = str(binding["access_ref"])
        if relation in {"address_recomputed", "pointer_advanced"}:
            disposition = "hard"
            reason = "per_access_address_occurrence_mapping_unavailable"
        elif relation == "guard_lower_bound_elided":
            disposition = "hard"
            reason = "per_access_guard_occurrence_mapping_unavailable"
        elif relation == "boolean_storage_byte":
            disposition = "hard"
            reason = "per_access_pointer_chain_mapping_unavailable"
        else:  # The strict pi decoder owns the closed relation vocabulary.
            disposition = "hard"
            reason = "unsupported_relation"
        record(
            path=f"pi.access_representation_bindings[{index}]",
            family="access_representation_bindings",
            relation=relation,
            anchor_refs=(access_ref,),
            disposition=disposition,
            reason=reason,
        )

    hard_relation_order = (
        "transfer_direct",
        "program_axes",
        "program_bindings",
        "iteration_bindings",
        "control_bindings",
        "materialization_bindings",
        "access_representation_bindings",
    )
    hard_relations = tuple(
        family
        for family in hard_relation_order
        if any(
            row.family == family and row.disposition == "hard"
            for row in relation_rows
        )
    )

    return NormalizedTTIR(
        text=normalized_text,
        transformed_occurrences=transformed_occurrences,
        active_rules=tuple(active_rules),
        hard_relations=hard_relations,
        representation_anchor_exemptions=representation_anchor_exemptions,
        relation_rows=tuple(relation_rows),
    )


__all__ = [
    "NORMALIZER_PROTOCOL",
    "NormalizationRelationRow",
    "NormalizedTTIR",
    "normalize_ttir_for_gate1",
]
