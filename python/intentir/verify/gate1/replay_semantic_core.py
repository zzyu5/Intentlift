"""Deterministic, parameterized ``J -> S_J`` view for source-blind Replay.

``S_J`` is a read-only compiler input, not a target-source skeleton.  This module
canonicalizes normal J, retains symbolic dimensions and J anchors, and presents the
typed semantic program without consulting C, pi, Source, Harness values, a target,
or Gate evidence.  A masked episode is materialized from its masked J bytes rather
than from the normal semantic archive.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_construction import (
    CanonicalHandleAllocator,
    canonical_operation_records,
    canonical_output_records,
)
from intentir.ir.canonical_j import elaborate_j, j_elaboration

from .model import (
    ImplementationConfig,
    SemanticArchive,
    _nested_root_owned_field_paths,
)


REPLAY_SEMANTIC_PROGRAM_PROTOCOL = "intentir.replay-semantic-program.v3"
REPLAY_SEMANTIC_QUERY_PROTOCOL = "intentir.replay-semantic-query.v3"
REPLAY_COMPILER_INDEX_PROTOCOL = "intentir.replay-compiler-index.v1"


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _contains_mask(value: Any) -> bool:
    if value == "<masked>":
        return True
    if isinstance(value, Mapping):
        return any(_contains_mask(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mask(item) for item in value)
    return False


def _reject_nested_root_owned_fields(j: Mapping[str, Any]) -> None:
    """Keep the source-blind J entrance aligned with the Gate1 envelope.

    Normal J is parsed through ``SemanticArchive`` below, but masked J is
    intentionally kept as an opaque mapping and therefore bypasses that
    parser.  The bypass must not reintroduce the old shape where C/pi were
    nested under ``j`` or ``j.intent`` and later interpreted as part of J.
    """

    nested_root_owned = _nested_root_owned_field_paths(j, path="Replay J")
    if nested_root_owned:
        raise ValueError(
            f"{nested_root_owned[0]} is a root-owned field nested under J; "
            "root.c and root.pi are siblings of root.j"
        )


def _intent_function(j: Mapping[str, Any]) -> Mapping[str, Any]:
    _reject_nested_root_owned_fields(j)
    intent = _object(j.get("intent"), "J.intent")
    return _object(
        intent.get("intent_function", intent),
        "J.intent.intent_function",
    )


def _node_stream(
    j: Mapping[str, Any],
    *,
    operation_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten J from the shared operation record stream.

    The raw walk remains a diagnostic fallback for masked/malformed input only.
    Normal J and all typed Replay views pass the record stream built by
    canonical J elaboration, so result/type/region fields cannot be re-guessed
    by this consumer.
    """

    if operation_records is not None:
        records: list[dict[str, Any]] = []
        for record in operation_records:
            wire = record.get("wire")
            if not isinstance(wire, Mapping):
                raise ValueError("operation record is missing its complete wire")
            header = {
                str(key): copy.deepcopy(value)
                for key, value in wire.items()
                if key not in {"body", "positive", "negative", "id"}
            }
            arms: dict[str, Any] = {}
            for arm_name in ("positive", "negative"):
                arm = wire.get(arm_name)
                if arm is None:
                    continue
                arm_object = _object(
                    arm,
                    f"J.intent.{record['path']}.{arm_name}",
                )
                arms[arm_name] = {
                    str(key): copy.deepcopy(value)
                    for key, value in arm_object.items()
                    if key != "body"
                }
            if arms:
                header["arms"] = arms
            records.append(
                {
                    "id": str(record["id"]),
                    "path": str(record["path"]),
                    "parent_node": record.get("parent_id"),
                    "lexical_role": record.get("lexical_role", "function"),
                    "depth": record.get("depth", 0),
                    "region_scope_id": record.get("region_scope_id"),
                    "is_final": record.get("is_final", False),
                    "operand_types": copy.deepcopy(
                        record.get("operand_types", {})
                    ),
                    "result_types": copy.deepcopy(
                        record.get("result_types", {})
                    ),
                    **header,
                }
            )
        return records

    function = _intent_function(j)
    node_ids = _object(j.get("node_ids", {}), "J.node_ids")
    path_to_node = {str(path): str(node) for node, path in node_ids.items()}
    records: list[dict[str, Any]] = []

    def visit(
        body: Sequence[Any],
        *,
        prefix: str,
        parent_node: str | None,
        lexical_role: str,
        depth: int,
    ) -> None:
        for index, raw in enumerate(body):
            operation = _object(raw, f"J.intent.{prefix}[{index}]")
            path = f"{prefix}[{index}]"
            operation_id = operation.get("id")
            if not isinstance(operation_id, str):
                operation_id = path_to_node.get(path)
            if not isinstance(operation_id, str) and operation.get("op") == "<masked>":
                result = operation.get("result")
                operation_id = result if isinstance(result, str) else None
            if not isinstance(operation_id, str):
                operation_id = f"node@{path}"

            header = {
                str(key): copy.deepcopy(value)
                for key, value in operation.items()
                if key not in {"body", "positive", "negative", "id"}
            }
            arms: dict[str, Any] = {}
            for arm_name in ("positive", "negative"):
                arm = operation.get(arm_name)
                if arm is None:
                    continue
                arm_object = _object(arm, f"J.intent.{path}.{arm_name}")
                arms[arm_name] = {
                    str(key): copy.deepcopy(value)
                    for key, value in arm_object.items()
                    if key != "body"
                }
            if arms:
                header["arms"] = arms
            records.append(
                {
                    "id": operation_id,
                    "path": path,
                    "parent_node": parent_node,
                    "lexical_role": lexical_role,
                    "depth": depth,
                    **header,
                }
            )

            nested = operation.get("body", [])
            if nested:
                visit(
                    _array(nested, f"J.intent.{path}.body"),
                    prefix=f"{path}.body",
                    parent_node=operation_id,
                    lexical_role="body",
                    depth=depth + 1,
                )
            for arm_name in ("positive", "negative"):
                arm = operation.get(arm_name)
                if arm is None:
                    continue
                arm_object = _object(arm, f"J.intent.{path}.{arm_name}")
                visit(
                    _array(arm_object.get("body", []), f"J.intent.{path}.{arm_name}.body"),
                    prefix=f"{path}.{arm_name}.body",
                    parent_node=operation_id,
                    lexical_role=arm_name,
                    depth=depth + 1,
                )

    visit(
        _array(function.get("body", []), "J.intent.body"),
        prefix="body",
        parent_node=None,
        lexical_role="function",
        depth=0,
    )
    return records


def _value_types(
    j: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    *,
    canonical_types: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if canonical_types is not None:
        return [
            {"value": name, **copy.deepcopy(dict(value_type))}
            for name, value_type in canonical_types.items()
        ]

    function = _intent_function(j)
    values: dict[str, dict[str, Any]] = {}
    for role in ("inputs", "outputs"):
        for raw in _array(function.get(role, []), f"J.intent.{role}"):
            tensor = _object(raw, f"J.intent.{role}[]")
            name = tensor.get("name")
            if isinstance(name, str):
                values[name] = {
                    "dtype": tensor.get("dtype"),
                    "shape": copy.deepcopy(tensor.get("shape")),
                    "io_role": role[:-1],
                }
    for node in nodes:
        produced = node.get("result")
        names = [produced] if isinstance(produced, str) else produced if isinstance(produced, list) else []
        node_type = node.get("type", {"dtype": "<masked>", "shape": ["<masked>"]})
        for name in names:
            if isinstance(name, str):
                values[name] = copy.deepcopy(node_type) if isinstance(node_type, Mapping) else {"type": copy.deepcopy(node_type)}
    return [{"value": name, **value} for name, value in values.items()]


def _boundary_node_refs(
    boundary: Mapping[str, Any], node_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in node_ids}
    for relation, value in boundary.items():
        if not isinstance(value, list):
            continue
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                continue
            refs = {
                item
                for key, item in raw.items()
                if (key == "node_ref" or key.endswith("_node_ref"))
                and isinstance(item, str)
                and item in node_ids
            }
            for ref in sorted(refs):
                result[ref].append({"relation": relation, "index": index})
    return {name: rows for name, rows in result.items() if rows}


def _expression_refs(value: Any) -> tuple[str, ...]:
    """Return first-occurrence refs from one already parsed ContractExpr JSON.

    This is a presentation-only walk over J-authored expressions.  It neither
    solves a domain nor assigns a physical program axis.
    """

    refs: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("op") == "ref" and isinstance(item.get("value"), str):
                ref = str(item["value"])
                if ref not in refs:
                    refs.append(ref)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(refs)


def _boundary_ownership_view(
    boundary: Mapping[str, Any],
    *,
    structured_boundary: Mapping[str, Any] | None,
    output_bindings: Sequence[Any],
    known_value_refs: frozenset[str] | None,
) -> dict[str, Any]:
    """Re-index exact J boundary facts for whole-program Replay compilation.

    The raw boundary remains present in ``S_J.boundary``.  This view only joins
    an authored access with its authored effect and parsed expression, so a
    Replay owner need not rediscover those cross references in a large JSON
    object.  The shared-axis receipts are plain set intersections of authored
    index refs; they are not a schedule, launch mapping, or new J relation.
    """

    parsed_accesses: dict[str, Mapping[str, Any]] = {}
    parsed_domains: dict[str, Mapping[str, Any]] = {}
    if isinstance(structured_boundary, Mapping):
        raw_domains = structured_boundary.get("domains", {})
        if isinstance(raw_domains, Mapping):
            parsed_domains = {
                str(name): value
                for name, value in raw_domains.items()
                if isinstance(value, Mapping)
            }
        for raw in structured_boundary.get("accesses", []):
            if isinstance(raw, Mapping) and isinstance(raw.get("access_id"), str):
                parsed_accesses[str(raw["access_id"])] = raw

    effects_by_access: dict[str, list[dict[str, Any]]] = {}
    orphan_effects: list[dict[str, Any]] = []
    for raw in boundary.get("effects", []):
        if not isinstance(raw, Mapping):
            continue
        effect = copy.deepcopy(dict(raw))
        access_ref = effect.get("access_ref")
        if isinstance(access_ref, str):
            effects_by_access.setdefault(access_ref, []).append(effect)
        else:
            orphan_effects.append(effect)

    def logical_index_refs(expression: Any) -> tuple[str, ...]:
        refs = _expression_refs(expression)
        if known_value_refs is None:
            # MaskJ cannot be elaborated into a complete value table.  Retain the
            # historical source spelling only for that non-semantic audit view.
            return tuple(
                ref
                for ref in refs
                if ref.startswith("k") and ref[1:].isdigit()
            )
        # Normal Replay first elaborates canonical J.  Every authored tensor,
        # scalar semantic value and node result is therefore in value_types;
        # dimensions remain x*.  ContractExpr refs outside both sets are exactly
        # the bound logical index variables, regardless of opaque spelling.  This
        # avoids assuming that the Replay-facing alpha-renamer preserves k*.
        return tuple(
            ref
            for ref in refs
            if ref not in known_value_refs
            and not (ref.startswith("x") and ref[1:].isdigit())
        )

    domain_axis_refs = {
        name: set(logical_index_refs(expression))
        for name, expression in parsed_domains.items()
    }
    accesses: list[dict[str, Any]] = []
    axes_by_target: dict[str, list[set[str]]] = {}
    write_axis_sets: list[set[str]] = []
    for raw in boundary.get("accesses", []):
        if not isinstance(raw, Mapping):
            continue
        access_id = raw.get("access_id")
        parsed = (
            parsed_accesses.get(str(access_id), {})
            if isinstance(access_id, str)
            else {}
        )
        parsed_axes = copy.deepcopy(parsed.get("axes", raw.get("index_expression")))
        axis_refs = list(logical_index_refs(parsed_axes))
        target = str(raw.get("tensor", "<missing>"))
        axis_set = set(axis_refs)
        covering_domains = {
            name: refs
            for name, refs in domain_axis_refs.items()
            if refs and refs.issubset(axis_set)
        }
        # A domain constrains this access exactly when all of the domain's authored
        # logical index refs occur in the authored access index.  Keep every such
        # domain: overlapping predicates are conjunctive J facts, not alternatives
        # from which fixed code may choose.  The previous containment direction
        # incorrectly made ordinary multi-axis accesses lose their one-axis domain
        # bounds and forced Replay to guess those bounds from target conventions.
        applicable_domains = sorted(covering_domains)
        axes_by_target.setdefault(target, []).append(axis_set)
        if raw.get("kind") == "store":
            write_axis_sets.append(axis_set)
        access = {
            "access_id": copy.deepcopy(access_id),
            "kind": copy.deepcopy(raw.get("kind")),
            "tensor": copy.deepcopy(raw.get("tensor")),
            "node_ref": copy.deepcopy(raw.get("node_ref")),
            # The accepted J already stores the canonical wire spelling.  Keep
            # that compact spelling in the first-screen view; exact parsed ASTs
            # remain available in ``canonical_boundary`` on demand.
            "index": copy.deepcopy(raw.get("index_expression")),
            "axis_refs": axis_refs,
            "domain_candidates": applicable_domains,
            "predicate": copy.deepcopy(raw.get("predicate")),
            "effects": copy.deepcopy(
                effects_by_access.get(str(access_id), [])
                if isinstance(access_id, str)
                else []
            ),
        }
        if "fill_value" in raw:
            access["fill_value"] = copy.deepcopy(raw.get("fill_value"))
        accesses.append(access)

    shared_by_target = {
        target: sorted(set.intersection(*sets)) if sets else []
        for target, sets in axes_by_target.items()
    }
    shared_writes = (
        sorted(set.intersection(*write_axis_sets)) if write_axis_sets else []
    )
    write_domain_sets = [
        domain_axis_refs[name]
        for access in accesses
        if access.get("kind") == "store"
        for name in access.get("domain_candidates", [])
        if name in domain_axis_refs
    ]
    domain_dependency_subsets = [
        {
            "subset_domain": subset_name,
            "superset_domain": superset_name,
            "shared_refs": sorted(subset_refs),
            "additional_refs": sorted(superset_refs - subset_refs),
        }
        for subset_name, subset_refs in sorted(domain_axis_refs.items())
        for superset_name, superset_refs in sorted(domain_axis_refs.items())
        if (
            subset_name != superset_name
            and subset_refs
            and subset_refs < superset_refs
        )
    ]
    return {
        "authority": (
            "deterministic re-indexing of exact J boundary/domain/effect/output "
            "facts; shared refs are set intersections only, not target mapping"
        ),
        "domains": copy.deepcopy(boundary.get("domains", {})),
        "domain_axis_refs": {
            name: sorted(refs) for name, refs in sorted(domain_axis_refs.items())
        },
        # This is only a syntactic dependency-set relation between authored
        # ContractExprs.  It does not claim logical set containment and does not
        # choose which domain becomes a program axis, loop, lane, or guard.
        "domain_dependency_subsets": domain_dependency_subsets,
        "access_effects": accesses,
        "shared_axis_refs_by_tensor": shared_by_target,
        "shared_write_axis_refs": shared_writes,
        "shared_write_domain_axis_refs": (
            sorted(set.intersection(*write_domain_sets))
            if write_domain_sets
            else []
        ),
        "output_bindings": copy.deepcopy(list(output_bindings)),
        "result_relations": copy.deepcopy(boundary.get("result_relations", [])),
        "orphan_effects": orphan_effects,
    }


def _value_list(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "(" + ", ".join(str(item) for item in value) + ")"
    return "_"


def _render_node_ir(
    node: Mapping[str, Any], value_types: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Render one exact J node as compact typed SSA, not target source."""

    depth = node.get("depth", 0)
    depth = depth if isinstance(depth, int) and depth >= 0 else 0
    indent = "  " + "  " * depth
    node_id = str(node.get("id", "<anonymous>"))
    operation = str(node.get("op", "<missing>"))
    result = _value_list(node.get("result"))
    operands = node.get("operands", [])
    if isinstance(operands, list):
        rendered_operands: list[str] = []
        for item in operands:
            item_name = str(item)
            item_type = value_types.get(item_name)
            rendered_operands.append(
                item_name
                if item_type is None
                else f"{item_name}:{_compact(item_type)}"
            )
        operand_text = ", ".join(rendered_operands)
    else:
        operand_text = _compact(operands)
    value_type = node.get("type")
    type_text = " : " + _compact(value_type) if value_type is not None else ""
    attrs = node.get("attrs")
    attrs_text = " attrs=" + _compact(attrs) if attrs is not None else ""
    location = (
        f" @J.{node.get('path')} role={node.get('lexical_role')}"
        + (f" parent={node.get('parent_node')}" if node.get("parent_node") is not None else "")
        + (
            f" region={node.get('region_scope_id')}"
            if node.get("region_scope_id") is not None
            else ""
        )
    )
    special = {"carry", "scan", "trip", "yield", "scope_id"}
    ordinary = {
        "id",
        "path",
        "parent_node",
        "lexical_role",
        "depth",
        "region_scope_id",
        "is_final",
        "op",
        "operands",
        "result",
        "type",
        "attrs",
        *special,
    }
    extra = {key: copy.deepcopy(value) for key, value in node.items() if key not in ordinary}
    lines = [
        f"{indent}{node_id} {result}{type_text} = {operation}({operand_text})"
        + attrs_text
        + (" meta=" + _compact(extra) if extra else "")
        + location
    ]
    if "scope_id" in node:
        lines.append(f"{indent}  scope {_compact(node.get('scope_id'))}")
    if "trip" in node:
        lines.append(f"{indent}  trip {_compact(node.get('trip'))}")
    scans = node.get("scan")
    for scan in scans if isinstance(scans, list) else []:
        lines.append(f"{indent}  scan {_compact(scan)}")
    carries = node.get("carry")
    for carry in carries if isinstance(carries, list) else []:
        if not isinstance(carry, Mapping):
            lines.append(f"{indent}  carry {_compact(carry)}")
            continue
        lines.append(
            f"{indent}  carry {carry.get('state_id')} state={carry.get('body_argument')}"
            f" init={carry.get('init_value')} next={carry.get('next_value')}"
            f" result={carry.get('result_value')} type={_compact(carry.get('type'))}"
        )
    if "yield" in node:
        lines.append(f"{indent}  yield {_compact(node.get('yield'))}")
    return lines


def _render_text(value: Mapping[str, Any]) -> str:
    lines = [
        f"semantic_program {value['protocol']}",
        f"status {value['status']}",
        "output_binding_authority "
        + str(value.get("output_binding_authority", "<unspecified>")),
        "scope_binding_authority "
        + str(value.get("scope_binding_authority", "<unspecified>")),
        "authority read_only_J_semantics; not_target_source; no_C; no_pi; no_Harness_valuation",
        "",
        "signature:",
    ]
    signature = _object(value.get("signature", {}), "S_J.signature")
    for parameter in _array(signature.get("parameters", []), "S_J.signature.parameters"):
        lines.append("  parameter " + _compact(parameter))

    ownership = _object(
        value.get("boundary_ownership", {}), "S_J.boundary_ownership"
    )
    lines.append("")
    lines.append("boundary_ownership:")
    lines.append("  authority " + str(ownership.get("authority", "")))
    for name, expression in _object(
        ownership.get("domains", {}), "S_J.boundary_ownership.domains"
    ).items():
        lines.append(f"  domain {name} = " + _compact(expression))
    for name, refs in _object(
        ownership.get("domain_axis_refs", {}),
        "S_J.boundary_ownership.domain_axis_refs",
    ).items():
        lines.append(f"  domain_axes {name} = " + _compact(refs))
    for relation in _array(
        ownership.get("domain_dependency_subsets", []),
        "S_J.boundary_ownership.domain_dependency_subsets",
    ):
        lines.append("  domain_dependency_subset " + _compact(relation))
    lines.append(
        "  shared_write_axis_refs "
        + _compact(ownership.get("shared_write_axis_refs", []))
    )
    lines.append(
        "  shared_write_domain_axis_refs "
        + _compact(ownership.get("shared_write_domain_axis_refs", []))
    )
    for target, refs in _object(
        ownership.get("shared_axis_refs_by_tensor", {}),
        "S_J.boundary_ownership.shared_axis_refs_by_tensor",
    ).items():
        lines.append(f"  tensor_axes {target} = " + _compact(refs))
    for access in _array(
        ownership.get("access_effects", []),
        "S_J.boundary_ownership.access_effects",
    ):
        lines.append("  access_effect " + _compact(access))
    for binding in _array(
        ownership.get("output_bindings", []),
        "S_J.boundary_ownership.output_bindings",
    ):
        lines.append("  output " + _compact(binding))
    for relation in _array(
        ownership.get("result_relations", []),
        "S_J.boundary_ownership.result_relations",
    ):
        lines.append("  result_relation " + _compact(relation))
    for effect in _array(
        ownership.get("orphan_effects", []),
        "S_J.boundary_ownership.orphan_effects",
    ):
        lines.append("  effect " + _compact(effect))

    value_types = {
        str(item["value"]): {
            key: copy.deepcopy(field)
            for key, field in item.items()
            if key != "value"
        }
        for item in _array(value.get("value_types", []), "S_J.value_types")
        if isinstance(item, Mapping) and isinstance(item.get("value"), str)
    }
    function = _object(value.get("function", {}), "S_J.function")
    lines.append("")
    lines.append(f"function {function.get('name', '<masked>')}:")
    for role in ("inputs", "outputs"):
        for tensor in _array(function.get(role, []), f"S_J.function.{role}"):
            lines.append(f"  {role[:-1]} " + _compact(tensor))
    for node in _array(value.get("nodes", []), "S_J.nodes"):
        node_object = _object(node, "S_J.nodes[]")
        lines.extend(_render_node_ir(node_object, value_types))
    for binding in _array(value.get("output_bindings", []), "S_J.output_bindings"):
        lines.append("  return " + _compact(binding))

    boundary = _object(value.get("boundary", {}), "S_J.boundary")
    lines.append("")
    lines.append("remaining_boundary_relations:")
    for name, rows in boundary.items():
        if name in {"domains", "accesses", "effects", "result_relations"}:
            continue
        if isinstance(rows, Mapping):
            for ref, item in rows.items():
                lines.append(f"  {name} {ref} = " + _compact(item))
        elif isinstance(rows, list):
            for index, item in enumerate(rows):
                lines.append(f"  {name}[{index}] " + _compact(item))
        else:
            lines.append(f"  {name} " + _compact(rows))

    for section in (
        "dimension_constraints",
        "value_dimension_relations",
        "dimension_relations",
        "scopes",
        "storage",
    ):
        lines.append("")
        lines.append(section + ":")
        rows = value.get(section, [])
        if isinstance(rows, list):
            lines.extend("  " + _compact(item) for item in rows)
        else:
            lines.append("  " + _compact(rows))
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ReplaySemanticProgram:
    value: dict[str, Any]
    text: str

    @property
    def masked(self) -> bool:
        return self.value.get("status") == "masked"

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)


@dataclass(frozen=True)
class ReplayCompilerIndex:
    """Compact cross-reference over the already-authorized Replay inputs.

    This contains no fact beyond ``S_J`` and the already accepted
    Source/Harness-bound ``C``. It places existing domain dependencies and
    the physical C value ledger next to each other so a coding agent need not
    reconstruct those joins repeatedly; it never creates a target C.  C slots
    intentionally have no logical J attachment: π is the sole owner of a
    tile-slot to logical-axis association.
    """

    value: dict[str, Any]
    text: str

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)


def materialize_replay_compiler_index(
    semantic_program: Mapping[str, Any],
    c: Mapping[str, Any] | None,
) -> ReplayCompilerIndex:
    """Materialize a source-neutral ``S_J + source-bound C`` index."""

    if semantic_program.get("protocol") != REPLAY_SEMANTIC_PROGRAM_PROTOCOL:
        raise ValueError("unsupported Replay semantic-program protocol")
    ownership = _object(
        semantic_program.get("boundary_ownership", {}),
        "S_J.boundary_ownership",
    )
    domain_axis_refs = _object(
        ownership.get("domain_axis_refs", {}),
        "S_J.boundary_ownership.domain_axis_refs",
    )
    if c is None:
        slots: list[Any] = []
        values: Mapping[str, Any] = {}
    else:
        config = ImplementationConfig.from_json(copy.deepcopy(dict(c)))
        if not config.slots:
            raise ValueError("empty C must be omitted from Replay")
        expected_names = list(
            CanonicalHandleAllocator.canonical_names("c", len(config.slots))
        )
        if [slot.name for slot in config.slots] != expected_names:
            raise ValueError("Replay C slots must use c0, c1, ... declaration order")
        config.validate_complete_values(path="Replay C")
        slots = [slot.to_json() for slot in config.slots]
        values = config.values
    c_slots: list[dict[str, Any]] = []
    for raw in slots:
        slot = _object(raw, "C.slots[]")
        name = slot.get("name")
        assert isinstance(name, str)
        c_slots.append(
            {
                "slot": copy.deepcopy(dict(slot)),
                "value": copy.deepcopy(values[name]),
            }
        )
    value = {
        "protocol": REPLAY_COMPILER_INDEX_PROTOCOL,
        "domain_axis_refs": copy.deepcopy(domain_axis_refs),
        "tensor_axis_refs": copy.deepcopy(
            ownership.get("shared_axis_refs_by_tensor", {})
        ),
        "domain_dependency_subsets": copy.deepcopy(
            ownership.get("domain_dependency_subsets", [])
        ),
        "shared_write_axis_refs": copy.deepcopy(
            ownership.get("shared_write_axis_refs", [])
        ),
        "shared_write_domain_axis_refs": copy.deepcopy(
            ownership.get("shared_write_domain_axis_refs", [])
        ),
        "anonymous_c_slots": c_slots,
        "authority": (
            "deterministic cross-reference of exact S_J and optional complete C "
            "only; dependency subsets are syntactic and no program mapping, loop, "
            "lane, guard, target API, Source, pi, or Gate fact is inferred"
        ),
    }
    lines = [
        f"compiler_index {REPLAY_COMPILER_INDEX_PROTOCOL}",
        "authority exact_S_J_plus_optional_C_cross_reference; no_source; no_pi",
    ]
    for name, refs in value["domain_axis_refs"].items():
        lines.append(f"domain_axes {name} = " + _compact(refs))
    for name, refs in value["tensor_axis_refs"].items():
        lines.append(f"tensor_axes {name} = " + _compact(refs))
    for relation in value["domain_dependency_subsets"]:
        lines.append("domain_dependency_subset " + _compact(relation))
    lines.append(
        "shared_write_axis_refs " + _compact(value["shared_write_axis_refs"])
    )
    lines.append(
        "shared_write_domain_axis_refs "
        + _compact(value["shared_write_domain_axis_refs"])
    )
    for slot in c_slots:
        lines.append("anonymous_c_slot " + _compact(slot))
    return ReplayCompilerIndex(value, "\n".join(lines) + "\n")


def materialize_replay_semantics(j_value: Mapping[str, Any]) -> ReplaySemanticProgram:
    """Materialize one normal or MaskJ source-blind J view without valuation."""

    masked = _contains_mask(j_value)
    canonical_types: dict[str, Any] | None = None
    structured_boundary: dict[str, Any] | None = None
    operation_records: Sequence[Mapping[str, Any]] | None = None
    if masked:
        j = copy.deepcopy(dict(j_value))
        operation_records = canonical_operation_records(_intent_function(j))
    else:
        canonical = elaborate_j(SemanticArchive.from_json(copy.deepcopy(dict(j_value))))
        j = canonical.to_json()
        elaboration = j_elaboration(canonical)
        operation_records = elaboration.operation_records
        canonical_types = {
            name: value_type.to_json()
            for name, value_type in elaboration.value_types
        }
        structured_boundary = elaboration.structured_boundary_json()

    function = _intent_function(j)
    nodes = _node_stream(j, operation_records=operation_records)
    if structured_boundary is not None:
        # Normal Replay consumes the same boundary construction projection as
        # Gate1.  Only the explicitly masked diagnostic path below may read a
        # raw boundary fallback.
        boundary = copy.deepcopy(dict(structured_boundary["boundary"]))
    else:
        boundary = copy.deepcopy(dict(_object(j.get("boundary", {}), "J.boundary")))
    node_ids = {str(node["id"]) for node in nodes}
    output_binding_authority = "canonical_construction"
    if structured_boundary is not None:
        output_bindings = copy.deepcopy(structured_boundary["output_bindings"])
    else:
        # A masked episode may still retain a complete output/return/mapping
        # surface.  In that case use the same raw-wire construction record as
        # Lift ingress and Gate1; do not rebuild ABI order from returns alone.
        raw_output_records = canonical_output_records(
            function,
            boundary,
            operation_records,
            signature=_object(j.get("signature", {}), "J.signature"),
        )
        if raw_output_records is not None:
            output_bindings = [
                {
                    "output": record["output"],
                    "output_type": copy.deepcopy(record["output_type"]),
                    "value": record["return_value"],
                    "node_ref": record["node_ref"],
                    "abi_index": record["abi_index"],
                    "abi_parameter_index": record["abi_parameter_index"],
                    "return_index": record["return_index"],
                    "mapping_index": record["mapping_index"],
                }
                for record in raw_output_records
            ]
        else:
            # The opaque MaskJ view intentionally masks fields such as return
            # values/types.  It remains useful for diagnostics, but its raw
            # fallback is not an acceptance authority and must never be used
            # to prove a normal Replay/Gate1 output contract.
            output_binding_authority = "masked_raw_diagnostic_fallback"
            raw_returns = function.get("returns")
            if not isinstance(raw_returns, list):
                raise ValueError(
                    "masked J.intent.intent_function.returns must be a complete array"
                )
            output_indexes = {
                item.get("name"): index
                for index, item in enumerate(
                    _array(function.get("outputs", []), "J.intent.outputs")
                )
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
            output_bindings = []
            for return_index, raw_return in enumerate(raw_returns):
                item = _object(
                    raw_return,
                    f"J.intent.intent_function.returns[{return_index}]",
                )
                output_bindings.append(
                    {
                        **copy.deepcopy(dict(item)),
                        "abi_index": output_indexes.get(item.get("output")),
                        "return_index": return_index,
                        "mapping_index": None,
                    }
                )
    boundary_ownership = _boundary_ownership_view(
        boundary,
        structured_boundary=structured_boundary,
        output_bindings=output_bindings,
        known_value_refs=(
            frozenset(canonical_types)
            if canonical_types is not None
            else None
        ),
    )
    if structured_boundary is not None:
        # Normal Replay must consume the scope table emitted by the same
        # canonical construction transaction as operations, value types, and
        # outputs.  Reading J.scopes here would create a second scope
        # authority and let the semantic view drift from Gate1's typed view.
        scopes = copy.deepcopy(structured_boundary["scopes"])
        scope_binding_authority = "canonical_construction"
    else:
        # A masked episode has no complete typed elaboration.  Preserve its
        # raw scopes only as a diagnostic projection; it cannot establish
        # acceptance or repair a missing scope record.
        scopes = copy.deepcopy(j.get("scopes", []))
        scope_binding_authority = "masked_raw_diagnostic_projection"
    value: dict[str, Any] = {
        "protocol": REPLAY_SEMANTIC_PROGRAM_PROTOCOL,
        "status": "masked" if masked else "normal",
        "signature": copy.deepcopy(j.get("signature", {})),
        "function": {
            key: copy.deepcopy(function.get(key, [] if key != "name" else "<masked>"))
            for key in ("name", "inputs", "outputs")
        },
        "nodes": nodes,
        "value_types": _value_types(
            j,
            nodes,
            canonical_types=canonical_types,
        ),
        "boundary": boundary,
        "boundary_ownership": boundary_ownership,
        "boundary_refs_by_node": _boundary_node_refs(boundary, node_ids),
        "output_bindings": output_bindings,
        "output_binding_authority": output_binding_authority,
        "scope_binding_authority": scope_binding_authority,
        "node_ids": copy.deepcopy(j.get("node_ids", {})),
        "scopes": scopes,
        "storage": copy.deepcopy(j.get("storage", [])),
        "dimension_constraints": copy.deepcopy(j.get("dimension_constraints", [])),
        "value_dimension_relations": copy.deepcopy(j.get("value_dimension_relations", [])),
        "dimension_relations": copy.deepcopy(j.get("dimension_relations", [])),
        "canonical_boundary": structured_boundary,
        "authority": (
            "deterministic read-only parameterized view of this episode's J only; "
            "no C, pi, Source, Harness valuation, target choice, Gate, generation, "
            "repair, or certification"
        ),
    }
    return ReplaySemanticProgram(value, _render_text(value))


def query_replay_semantics(
    semantic_program: Mapping[str, Any], arguments: Sequence[str]
) -> dict[str, Any]:
    """Return a bounded exact view of one already-materialized ``S_J``."""

    if semantic_program.get("protocol") != REPLAY_SEMANTIC_PROGRAM_PROTOCOL:
        raise ValueError("unsupported Replay semantic-program protocol")
    nodes = _array(semantic_program.get("nodes", []), "S_J.nodes")
    boundary = _object(semantic_program.get("boundary", {}), "S_J.boundary")
    boundary_ownership = _object(
        semantic_program.get("boundary_ownership", {}),
        "S_J.boundary_ownership",
    )
    value_types = _array(semantic_program.get("value_types", []), "S_J.value_types")
    if list(arguments) == ["overview"]:
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "overview",
            "semantic_status": semantic_program.get("status"),
            "node_count": len(nodes),
            "value_count": len(value_types),
            "boundary_counts": {
                name: len(value) if isinstance(value, (list, Mapping)) else 1
                for name, value in boundary.items()
            },
            "outputs": semantic_program.get("output_bindings", []),
            "shared_write_axis_refs": boundary_ownership.get(
                "shared_write_axis_refs", []
            ),
            "shared_write_domain_axis_refs": boundary_ownership.get(
                "shared_write_domain_axis_refs", []
            ),
            "next": "ownership | node REF | value REF | boundary NAME START COUNT",
        }
    if list(arguments) == ["ownership"]:
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "boundary_ownership",
            "boundary_ownership": copy.deepcopy(dict(boundary_ownership)),
        }
    if len(arguments) == 2 and arguments[0] == "node":
        ref = arguments[1]
        matches = [item for item in nodes if isinstance(item, Mapping) and item.get("id") == ref]
        if len(matches) != 1:
            raise KeyError(f"unknown S_J node {ref!r}")
        relations = semantic_program.get("boundary_refs_by_node", {}).get(ref, [])
        expanded: list[dict[str, Any]] = []
        for relation in relations if isinstance(relations, list) else []:
            if not isinstance(relation, Mapping):
                continue
            name = relation.get("relation")
            index = relation.get("index")
            rows = boundary.get(name) if isinstance(name, str) else None
            if isinstance(rows, list) and isinstance(index, int) and 0 <= index < len(rows):
                expanded.append({"relation": name, "index": index, "value": rows[index]})
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "node",
            "node": matches[0],
            "boundary_relations": expanded,
        }
    if len(arguments) == 2 and arguments[0] == "value":
        ref = arguments[1]
        matches = [item for item in value_types if isinstance(item, Mapping) and item.get("value") == ref]
        if len(matches) != 1:
            raise KeyError(f"unknown S_J value {ref!r}")
        consumers = [
            item.get("id")
            for item in nodes
            if isinstance(item, Mapping)
            and ref in (item.get("operands") if isinstance(item.get("operands"), list) else [])
        ]
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "value",
            "value": matches[0],
            "consumers": consumers,
        }
    if len(arguments) == 4 and arguments[0] == "boundary":
        name = arguments[1]
        start = int(arguments[2])
        count = int(arguments[3])
        if start < 0 or count < 1 or count > 24:
            raise ValueError("boundary START/COUNT is outside the closed bound")
        rows = boundary.get(name)
        if isinstance(rows, Mapping):
            values = [{"id": key, "value": value} for key, value in rows.items()]
        elif isinstance(rows, list):
            values = list(rows)
        else:
            raise KeyError(f"unknown or scalar S_J boundary relation {name!r}")
        items = values[start : start + count]
        return {
            "protocol": REPLAY_SEMANTIC_QUERY_PROTOCOL,
            "status": "pass",
            "view": "boundary",
            "relation": name,
            "total": len(values),
            "start": start,
            "items": items,
            "next": (
                f"boundary {name} {start + len(items)} {count}"
                if start + len(items) < len(values)
                else None
            ),
        }
    raise ValueError(
        "usage: overview | ownership | node REF | value REF | "
        "boundary NAME START COUNT"
    )


__all__ = [
    "REPLAY_COMPILER_INDEX_PROTOCOL",
    "REPLAY_SEMANTIC_PROGRAM_PROTOCOL",
    "REPLAY_SEMANTIC_QUERY_PROTOCOL",
    "ReplayCompilerIndex",
    "ReplaySemanticProgram",
    "materialize_replay_compiler_index",
    "materialize_replay_semantics",
    "query_replay_semantics",
]
