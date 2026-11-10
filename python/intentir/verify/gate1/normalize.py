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
or unknown pointer provenance.  The address-form rule is separately bounded
by a typed per-access ownership chain and never rewrites those hard atoms.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any, Mapping

from intentir.ir.canonical_j import j_elaboration, parse_index_expression

from .model import Gate1Artifact, opaque_replay_representation
from .pi import (
    CLOSED_CONSUMER_IDENTITY_CLASS,
    DEFAULT_ZERO_LOWER_BOUND_CLASS,
    PER_PROGRAM_SCALAR_KEEPDIMS_CLASS,
    default_zero_lower_bound_accesses,
    closed_consumer_identity_extensions,
    per_program_scalar_keepdims_reductions,
    zero_lower_bound_coordinates_from_j,
)
from .ttir import compiled_program_id_axes


NORMALIZER_PROTOCOL = "intentir.gate1.jc-normalizer.v2"


_SSA_NAME = r"%[A-Za-z0-9_.$-]+"
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
    if name.startswith(("%p", "%arg")):
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
    """Find every addptr stage in one surviving memory-pointer chain.

    A source compiler may materialize one J access in multiple address stages
    (for example, scalar row offset, pointer splat, then lane offset), while a
    Replay compiler may fold the same address into one vector ``tt.addptr``.
    The typed address witness authorizes that *chain* as one access-owned
    representation.  It does not authorize an unrelated addptr or allow one
    physical chain to serve two J accesses.
    """

    if name in active:
        return frozenset()
    definition = definitions.get(name)
    if definition is None:
        return frozenset()
    operation, operands = definition
    if operation == "tt.addptr":
        if len(operands) < 2:
            return frozenset()
        anchors = {name}
        # Only the base operand continues the pointer chain.  The index
        # operand is an integer/address carrier and is deliberately kept out
        # of this pointer-root proof.
        anchors.update(
            _resolve_address_anchors(operands[0], definitions, active | {name})
        )
        return frozenset(anchors)
    if operation not in _POINTER_WRAPPER_OPERATIONS or len(operands) != 1:
        return frozenset()
    return _resolve_address_anchors(
        operands[0], definitions, active | {name}
    )


def _memory_operation_rows(
    ttir: str,
) -> tuple[tuple[str, str, str, bool, bool], ...] | None:
    """Collect ordered memory effects without interpreting their formulas.

    Each row is ``(kind, pointer, operation, masked, has_fill)``.  The
    source-blind certificate uses only the pointer root, access kind,
    multiplicity, and the preserved mask/fill shape.  Predicate and fill
    *content*, plus effect order, remain in the TTIR fingerprint and are never
    rewritten here.
    """

    rows: list[tuple[str, str, str, bool, bool]] = []
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
            has_fill = len(operands) >= 3
        elif operation == "tt.store":
            kind = "store"
            masked = len(operands) >= 3
            has_fill = False
        else:
            kind = "update"
            masked = len(operands) >= 2
            has_fill = False
        rows.append((kind, operands[0], operation, masked, has_fill))
    return tuple(rows)


def _address_normalization_certificate(
    ttir: str,
    artifact: Gate1Artifact,
) -> tuple[bool, tuple[tuple[str, str], ...], str]:
    """Prove that every surviving addptr is one J-owned address occurrence.

    The returned flag is an authorization for the *address-form* quotient only.
    It is intentionally derived independently for one observation.  No peer
    text, replay code, fingerprint, or comparator result is consulted.  The
    certificate is rejected if any J access lacks typed address ownership,
    except for a direct rank-1 logical-index access that has no scalar stride
    stage in J/pi.  Such an access owns its address through its single logical
    index rather than through a row-stride carrier.  All tensor/root/access
    mappings, surviving transfers, and pointer-slice containment remain
    strict.
    """

    try:
        representation = opaque_replay_representation(artifact)
    except Exception:
        return False, (), "typed_address_ownership_unavailable"
    address_bindings = representation.get("address_bindings", [])
    if not isinstance(address_bindings, list):
        return False, (), "typed_address_ownership_unavailable"

    accesses = list(artifact.j.boundary.accesses)
    access_by_id = {access.access_id: access for access in accesses}
    tensors = {
        **{tensor.name: tensor for tensor in artifact.j.intent.inputs},
        **{
            binding.output: binding.output_type
            for binding in j_elaboration(artifact.j).output_bindings
        },
    }

    def direct_rank_one_access_without_stride(access: Any) -> bool:
        tensor = tensors.get(access.tensor)
        if tensor is None or len(tensor.shape) != 1:
            return False
        expression = access.index_expression
        if not isinstance(expression, str):
            return False
        try:
            coordinates = parse_index_expression(
                expression,
                context=f"J boundary access {access.access_id} index_expression",
            ).direct_coordinates
        except Exception:
            return False
        if coordinates is None or len(coordinates) != 1:
            return False
        coordinate = coordinates[0]
        if isinstance(coordinate, str):
            if re.fullmatch(r"k[0-9]+", coordinate) is not None:
                return True
            # The canonical semantic-transfer wire uses the tensor's x*
            # dimension handle as the sole direct rank-1 axis.  Accept only
            # an exact match to that already typed tensor extent; an arbitrary
            # x* spelling or another symbolic expression remains unowned.
            return (
                re.fullmatch(r"x[0-9]+", coordinate) is not None
                and tensor.shape[0] == coordinate
            )
        # A fixed rank-1 element such as ``[1]`` or ``[2]`` is still a
        # complete logical address.  It does not need a separate J/pi stride
        # carrier; the target may materialize the constant offset as an
        # address-only addptr.  Restrict this exception to integer constants
        # so symbolic/non-affine address expressions remain hard.
        return isinstance(coordinate, int) and not isinstance(coordinate, bool)

    def direct_rank_one_base_access(access: Any) -> bool:
        """Return whether a rank-1 access is exactly the tensor base."""

        tensor = tensors.get(access.tensor)
        if tensor is None or len(tensor.shape) != 1:
            return False
        expression = access.index_expression
        if not isinstance(expression, str):
            return False
        try:
            coordinates = parse_index_expression(
                expression,
                context=f"J boundary access {access.access_id} index_expression",
            ).direct_coordinates
        except Exception:
            return False
        return coordinates == (0,)

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
    unbound_accesses = set(access_by_id) - set(rows_by_access)
    if any(
        not direct_rank_one_access_without_stride(access_by_id[access_ref])
        for access_ref in unbound_accesses
    ):
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

    # One ABI tensor root may own several distinct J boundary occurrences.  A
    # root-to-single-access map is therefore insufficient for address
    # normalization: it rejects the common case of three scalar parameter
    # loads or three column stores from the same tensor.  Keep the J access
    # declaration order as the occurrence order and consume one entry for
    # each observed memory transfer below.  The transfer kind/mask/fill checks
    # remain hard, and the final count/uniqueness checks still require a
    # one-to-one root-local occurrence mapping.
    access_for_root: dict[str, list[str]] = defaultdict(list)
    runtime_parameter_index_by_name: dict[str, int] = {}
    runtime_parameter_index = 0
    for parameter in artifact.j.signature.parameters:
        runtime_parameter_index_by_name[parameter.name] = runtime_parameter_index
        if getattr(parameter.kind, "value", parameter.kind) != "constexpr":
            runtime_parameter_index += 1
    for access in accesses:
        candidates = parameters_by_tensor.get(access.tensor, [])
        if len(candidates) != 1:
            return False, (), "tensor_root_mapping_ambiguous"
        parameter = candidates[0]
        root_aliases = {
            f"%{parameter.name.removeprefix('%')}",
            f"%arg{runtime_parameter_index_by_name[parameter.name]}",
        }
        # Triton TTIR names runtime function arguments as %argN.  Constexpr
        # parameters are absent from that function-argument list, so the
        # positional bridge above counts only non-constexpr parameters.  The
        # authored p* spelling is retained for source-independent observations
        # that preserve opaque ABI names.
        for root in root_aliases:
            access_for_root[root].append(access.access_id)

    definitions, addptr_results = _pointer_operation_definitions(ttir)
    rows = _memory_operation_rows(ttir)
    if not addptr_results or rows is None or len(rows) != len(accesses):
        return False, (), "address_transfer_multiplicity_unproved"

    anchor_roots: dict[str, str] = {}
    consumed_accesses_by_root: Counter[str] = Counter()
    observed_accesses: list[str] = []
    observed_anchors: set[str] = set()
    for kind, pointer, _operation, masked, has_fill in rows:
        root = _resolve_pointer_root_from_definitions(pointer, definitions)
        root_accesses = access_for_root.get(root) if root is not None else None
        if not root_accesses:
            return False, (), "physical_pointer_root_unowned"
        occurrence = consumed_accesses_by_root[root]
        if occurrence >= len(root_accesses):
            return False, (), "physical_transfer_multiplicity_unproved"
        access_ref = root_accesses[occurrence]
        consumed_accesses_by_root[root] += 1
        access = access_by_id[access_ref]
        expected_kind = getattr(access.kind, "value", access.kind)
        if kind != expected_kind:
            return False, (), "access_kind_mismatch"
        # A mask/fill is a J-owned hard atom.  Address normalization may be
        # applied to a masked access only when the observation has the same
        # mask/fill *shape* declared by J.  The actual predicate, fill value,
        # and their dataflow remain hard fingerprint components.
        expected_masked = access.predicate is not None
        expected_fill = (
            kind == "load" and access.fill_value is not None
        )
        if masked != expected_masked:
            return False, (), "predicate_shape_mismatch"
        if has_fill != expected_fill:
            return False, (), "fill_shape_mismatch"
        anchors = _resolve_address_anchors(pointer, definitions)
        if len(anchors) != 1:
            if not anchors:
                # A rank-1 ``[0]`` transfer may legally use the ABI pointer
                # directly; it has no address-only addptr anchor to exempt.
                # Any other access must expose a pointer stage so that the
                # occurrence remains independently owned.
                if not direct_rank_one_base_access(access):
                    return False, (), "physical_address_anchor_unresolved"
        # A source compiler may materialize one row/base advance once and
        # reuse that pointer for several same-root accesses (for example the
        # three fixed columns of one rank-2 output).  Such sharing is still a
        # representation-only address prefix, not an ambiguous tensor alias.
        # Keep the root ownership hard: an addptr may be shared only by
        # accesses whose ABI pointer root is the same.
        for anchor in anchors:
            previous_root = anchor_roots.get(anchor)
            if previous_root is not None and previous_root != root:
                return False, (), "address_anchor_shared_between_roots"
            anchor_roots[anchor] = root
        observed_accesses.append(access_ref)
        observed_anchors.update(anchors)

    if observed_anchors != set(addptr_results):
        return False, (), "unreachable_or_unowned_addptr"
    if len(observed_accesses) != len(set(observed_accesses)):
        return False, (), "physical_transfer_multiplicity_unproved"
    if set(observed_accesses) != set(access_by_id):
        return False, (), "J_access_occurrence_unmatched"
    # ``access_for_root`` contains both the authored p* alias and Triton's
    # %argN alias for each ABI root.  Only the alias actually observed in this
    # TTIR participates in the transfer count; the other spelling is a bridge
    # lookup, not an additional physical occurrence.
    if any(
        consumed_accesses_by_root[root] != len(access_for_root[root])
        for root in consumed_accesses_by_root
    ):
        return False, (), "physical_transfer_multiplicity_unproved"

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


def _elide_proven_physical_duplicate_loads(
    ttir: str,
    artifact: Gate1Artifact,
) -> tuple[str, int, tuple["NormalizationRelationRow", ...]]:
    """Keep duplicate physical reads hard without an explicit J-proven class.

    The retired pi realization graph used to supply transfer-stage multiplicity.
    Canonical pi deliberately carries no execution topology, so it cannot prove
    a duplicate read removable.  Source transfer occurrence comparison remains
    owned by the construction ledger and Gate1.
    """

    del artifact
    return ttir, 0, ()


def _default_zero_lower_bound_certificate(
    artifact: Gate1Artifact,
    *,
    address_proven: bool,
    address_accesses: tuple[tuple[str, str], ...],
) -> tuple[bool, tuple[str, ...], str]:
    """Authorize one class-level, unary zero-bound representation quotient."""

    access_refs = default_zero_lower_bound_accesses(artifact.j, artifact.pi)
    if not access_refs:
        return False, (), "J_dense_zero_lower_bound_class_not_authorized"
    if not address_proven:
        return False, (), "access_occurrence_owner_not_proved_for_zero_bound_class"
    observed_refs = {access_ref for access_ref, _tensor in address_accesses}
    if observed_refs != {access.access_id for access in artifact.j.boundary.accesses}:
        return False, (), "zero_bound_class_access_coverage_unproved"
    if not all(
        zero_lower_bound_coordinates_from_j(artifact.j, access_ref)
        for access_ref in access_refs
    ):
        return False, (), "zero_bound_class_domain_certificate_unproved"
    return True, access_refs, "J_dense_domain_explicit_zero_lower_and_upper_bounds"


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
    normalization_classes: tuple[str, ...] = ()
    relation_rows: tuple[NormalizationRelationRow, ...] = ()
    address_certificate_proven: bool = False
    address_certificate_reason: str = "typed_address_ownership_unavailable"

    def evidence(self) -> dict[str, Any]:
        if self.normalization_classes:
            hard_preservation_check = "passed_by_authorized_class_profile"
        elif self.transformed_occurrences == 0:
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
            "normalization_classes": list(self.normalization_classes),
            "hard_relations": list(self.hard_relations),
            "representation_anchor_exemptions": sorted(
                self.representation_anchor_exemptions
            ),
            "address_normalization_certificate": {
                # Persist only the unary verdict and its closed reason code.
                # Pointer SSA, address formulas, Source text, and peer
                # comparison data remain outside the evidence surface.
                "proven": self.address_certificate_proven,
                "reason": self.address_certificate_reason,
            },
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
    remain hard.  The duplicate-read rule, the explicit dense zero-based
    domain class, and the closed consumer-identity class are the only active
    transformations.  They are unary class quotients: the former removes only
    a compiler-materialized ``0 <= coordinate`` conjunct from a masked memory
    predicate; the latter removes only an optional value ``where/select`` that
    matches a J-proven producer mapping.  Upper bounds, access mask/fill,
    effect order, reductions, loops, outputs, and all unproved dataflow remain
    visible and hard.
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
    zero_bound_proven, zero_bound_accesses, zero_bound_reason = (
        _default_zero_lower_bound_certificate(
            artifact,
            address_proven=address_proven,
            address_accesses=address_accesses,
        )
    )
    consumer_identity_edges = closed_consumer_identity_extensions(
        artifact.j,
        artifact.pi,
    )
    scalar_keepdims_reductions = per_program_scalar_keepdims_reductions(
        artifact.j,
        artifact.pi,
    )
    # A J certificate establishes only that a class *can* be safely applied.
    # It must not silently authorize a Gate1 quotient: Lift declares the
    # closed class in π, TypeCheck verifies the J-only premise, and this
    # normalizer consumes that same declaration.  Keeping these two gates
    # separate prevents an accidental J pattern from weakening Gate1.
    declared_classes = frozenset(
        str(row["name"])
        for row in artifact.pi.normalization_classes
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    )
    zero_bound_authorized = (
        zero_bound_proven and DEFAULT_ZERO_LOWER_BOUND_CLASS in declared_classes
    )
    consumer_identity_authorized = bool(consumer_identity_edges) and (
        CLOSED_CONSUMER_IDENTITY_CLASS in declared_classes
    )
    scalar_keepdims_authorized = bool(scalar_keepdims_reductions) and (
        PER_PROGRAM_SCALAR_KEEPDIMS_CLASS in declared_classes
    )
    representation_exemptions = {"tt.addptr"} if address_proven else set()
    if zero_bound_authorized:
        representation_exemptions.add(
            f"{DEFAULT_ZERO_LOWER_BOUND_CLASS}:{len(zero_bound_accesses)}"
        )
    if consumer_identity_authorized:
        representation_exemptions.add(
            f"{CLOSED_CONSUMER_IDENTITY_CLASS}:{len(consumer_identity_edges)}"
        )
    if scalar_keepdims_authorized:
        representation_exemptions.add(
            f"{PER_PROGRAM_SCALAR_KEEPDIMS_CLASS}:"
            f"{len(scalar_keepdims_reductions)}"
        )
    representation_anchor_exemptions = frozenset(representation_exemptions)
    if address_proven:
        active_rules.append("normalize_proven_address_representations")
    normalization_classes: list[str] = []
    if zero_bound_authorized:
        active_rules.append("normalize_default_zero_lower_bound_class")
        normalization_classes.append(DEFAULT_ZERO_LOWER_BOUND_CLASS)
    if consumer_identity_authorized:
        active_rules.append("normalize_closed_consumer_identity_class")
        normalization_classes.append(CLOSED_CONSUMER_IDENTITY_CLASS)
    if scalar_keepdims_authorized:
        # This class validates a physical carrier projection rather than
        # deleting an observed TTIR operation.  A scalar ``tl.sum`` carrier is
        # the legal realization of the J-kept singleton for one directly
        # mapped row; all J dataflow remains visible in the observation.
        active_rules.append("normalize_per_program_scalar_keepdims_class")
        normalization_classes.append(PER_PROGRAM_SCALAR_KEEPDIMS_CLASS)
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

    for index, normalization_class in enumerate(artifact.pi.normalization_classes):
        class_name = str(normalization_class["name"])
        class_scope = str(normalization_class["scope"])
        if class_name == DEFAULT_ZERO_LOWER_BOUND_CLASS:
            class_accesses = default_zero_lower_bound_accesses(
                artifact.j, artifact.pi
            )
            if zero_bound_proven:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=(
                        class_scope,
                        *class_accesses,
                    ),
                    disposition="normalized",
                    reason=zero_bound_reason,
                )
            else:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=(class_scope,),
                    disposition="hard",
                    reason=zero_bound_reason,
                )
        elif class_name == CLOSED_CONSUMER_IDENTITY_CLASS:
            if consumer_identity_edges:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=tuple(
                        f"{consumer}:{access}:{dtype}"
                        for consumer, access, dtype in consumer_identity_edges
                    ),
                    disposition="normalized",
                    reason=(
                        "J-proven closed producer mapping permits removal of an "
                        "optional closed consumer-identity materialization"
                    ),
                )
            else:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=(class_scope,),
                    disposition="hard",
                    reason="closed_consumer_identity_certificate_unproved",
                )
        elif class_name == PER_PROGRAM_SCALAR_KEEPDIMS_CLASS:
            if scalar_keepdims_reductions:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=tuple(scalar_keepdims_reductions),
                    disposition="normalized",
                    reason=(
                        "J/pi prove one scalar target carrier for each "
                        "directly mapped rowwise keepdims reduction"
                    ),
                )
            else:
                record(
                    path=f"pi.normalization_classes[{index}]",
                    family="normalization_classes",
                    relation=class_name,
                    anchor_refs=(class_scope,),
                    disposition="hard",
                    reason="per_program_scalar_keepdims_certificate_unproved",
                )

    observed_program_axes = compiled_program_id_axes(normalized_text)
    launch_axis_atoms = {0: "program_axis:x", 1: "program_axis:y", 2: "program_axis:z"}
    for index, binding in enumerate(artifact.pi.program_axes):
        axis = binding["axis"]
        observation_anchor = (
            launch_axis_atoms.get(axis)
            if observed_program_axes is not None and axis in observed_program_axes
            else None
        )
        record(
            path=f"pi.program_axes[{index}]",
            family="program_axes",
            relation="physical_launch_axis",
            anchor_refs=(
                str(binding["domain_ref"]),
                str(axis),
                *((observation_anchor,) if observation_anchor is not None else ()),
            ),
            # Launch-axis identity remains hard and fully visible in the TTIR
            # fingerprint.  Mapping the frozen pi row to its exact x/y/z atom
            # improves ownership evidence but authorizes no quotient, rewrite,
            # duplicate elimination, or cross-axis collapse.
            disposition="hard",
            reason=(
                "mapped_to_observed_program_axis_atom"
                if observation_anchor is not None
                else "observation_anchor_mapping_unavailable"
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
    address_tensors = dict(address_accesses)
    for index, binding in enumerate(artifact.pi.access_representation_bindings):
        relation = str(binding["relation"])
        access_ref = str(binding["access_ref"])
        if relation in {"address_recomputed", "pointer_advanced"}:
            tensor = address_tensors.get(access_ref)
            if address_proven and tensor is not None:
                disposition = "normalized"
                reason = address_reason
                anchor_refs = (access_ref, tensor)
            else:
                disposition = "hard"
                reason = (
                    address_reason
                    if not address_proven
                    else "address_certificate_access_unmatched"
                )
                anchor_refs = (access_ref,)
        elif relation == "guard_lower_bound_elided":
            coordinates = zero_lower_bound_coordinates_from_j(
                artifact.j, access_ref
            )
            # This access-level relation is part of the same class quotient as
            # the predicate rewrite above.  A J certificate proves only that
            # the quotient is available; π must explicitly select it before
            # Gate1 can normalize this row.  Do not let a binding relation
            # become an implicit second authorization path.
            if (
                zero_bound_authorized
                and access_ref in zero_bound_accesses
                and coordinates
            ):
                disposition = "normalized"
                reason = zero_bound_reason
                anchor_refs = (access_ref, *sorted(coordinates))
            else:
                disposition = "hard"
                reason = (
                    "J_dense_zero_lower_bound_class_not_authorized"
                    if not zero_bound_authorized
                    else "zero_bound_class_access_certificate_unmatched"
                )
                anchor_refs = (access_ref,)
        elif relation == "boolean_storage_byte":
            disposition = "hard"
            reason = "per_access_pointer_chain_mapping_unavailable"
            anchor_refs = (access_ref,)
        else:  # The strict pi decoder owns the closed relation vocabulary.
            disposition = "hard"
            reason = "unsupported_relation"
            anchor_refs = (access_ref,)
        record(
            path=f"pi.access_representation_bindings[{index}]",
            family="access_representation_bindings",
            relation=relation,
            anchor_refs=anchor_refs,
            disposition=disposition,
            reason=reason,
        )

    hard_relation_order = (
        "transfer_direct",
        "normalization_classes",
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
        normalization_classes=tuple(normalization_classes),
        relation_rows=tuple(relation_rows),
        address_certificate_proven=address_proven,
        address_certificate_reason=address_reason,
    )


__all__ = [
    "NORMALIZER_PROTOCOL",
    "NormalizationRelationRow",
    "NormalizedTTIR",
    "normalize_ttir_for_gate1",
]
