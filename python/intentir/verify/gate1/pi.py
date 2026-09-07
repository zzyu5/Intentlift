"""Typed realization witness for the current J workflow.

Fresh Lift candidates author a compact algebra of direct J/C-bound realization
relations.  BindSource then materializes the complete realization graph that
Replay and Gate consume.  Neither form contains source/replay operation
identities or a target recipe.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from intentir.ir.intent_op_contract import is_scalar_literal_or_value_reference


class RealizationWitnessError(ValueError):
    """Raised when pi contains malformed or semantic-bearing data."""


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

NORMALIZATION_WITNESS_SCHEMA = "intentir.pi.normalization-witness.v1"

PROGRAM_BINDING_RELATIONS: tuple[str, ...] = (
    "direct",
    "product_linearized",
    "guarded_union",
)

ITERATION_BINDING_RELATIONS: tuple[str, ...] = (
    "tiled_domain",
    "tiled_recurrence",
)

CONTROL_BINDING_RELATIONS: tuple[str, ...] = (
    "terminal_domain_guard",
    "terminal_effect_guard",
)

MATERIALIZATION_BINDING_RELATIONS: tuple[str, ...] = (
    "scratch_roundtrip",
    "recompute",
)

ACCESS_REPRESENTATION_BINDING_RELATIONS: tuple[str, ...] = (
    "address_recomputed",
    "boolean_storage_byte",
    "guard_lower_bound_elided",
    "pointer_advanced",
)

REALIZATION_MECHANISMS: tuple[str, ...] = (
    "realize",
    "address_base",
    "address_lane",
    "transfer_direct",
    "transfer_async",
    "update_atomic",
    "update_privatized",
    "synchronize",
)

ITERATION_BASES: tuple[str, ...] = (
    "extent_step",
    "tile_ordinal",
    "rounded_extent_step",
)

PHYSICAL_ITERATION_DIRECTIONS: tuple[str, ...] = ("forward", "reverse")

ROUTE_EXTENT_ROUNDINGS: tuple[str, ...] = ("floor", "ceil")

REPRESENTATION_REFINEMENTS: tuple[str, ...] = (
    "identity_materialization",
    "reduction_linearization",
    "signed_widen",
    "unsigned_widen",
    "numeric_cast",
    "accumulation_dtype",
    "reassociation",
    "native_atomic",
    "privatized_update",
)


def _record(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
    allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RealizationWitnessError(f"{path} must be an object")
    allowed = required if allowed is None else allowed
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise RealizationWitnessError(f"{path} has forbidden fields: {unknown}")
    if missing:
        raise RealizationWitnessError(f"{path} is missing fields: {missing}")
    return value


def _wire_object(value: Any, path: str) -> dict[str, Any]:
    """Read one JSON object without importing the Gate1 model layer.

    The public normalization validator is intentionally owned by this module.
    It therefore accepts either the persisted mapping or a typed object with a
    ``to_json`` method, but it must not depend on ``model.py`` to discover the
    declaration namespaces it is validating.
    """

    if not isinstance(value, Mapping):
        to_json = getattr(value, "to_json", None)
        if not callable(to_json):
            raise RealizationWitnessError(f"{path} must be an object")
        try:
            value = to_json()
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise RealizationWitnessError(
                f"{path} could not be converted to a JSON object: {exc}"
            ) from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RealizationWitnessError(f"{path} must be an object")
    return dict(value)


def _wire_array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RealizationWitnessError(f"{path} must be an array")
    return value


def _wire_handle(
    value: Any,
    path: str,
    pattern: str,
    description: str,
) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise RealizationWitnessError(f"{path} must be {description}")
    return value


def _wire_handle_set(
    values: Any,
    *,
    path: str,
    pattern: str,
    fields: tuple[str, ...] = ("name", "id"),
) -> set[str]:
    result: set[str] = set()
    for index, value in enumerate(_wire_array(values, path)):
        item = _wire_object(value, f"{path}[{index}]")
        handle = next(
            (item.get(field) for field in fields if item.get(field) is not None),
            None,
        )
        if not isinstance(handle, str) or re.fullmatch(pattern, handle) is None:
            raise RealizationWitnessError(
                f"{path}[{index}] must declare a handle matching {pattern}"
            )
        if handle in result:
            raise RealizationWitnessError(f"{path} contains duplicate handle {handle!r}")
        result.add(handle)
    return result


def _wire_operation_results(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (
            _wire_handle(value, path, r"v[0-9]+", "an SSA v* result handle"),
        )
    if isinstance(value, list):
        return tuple(
            _wire_handle(
                item,
                f"{path}[{index}]",
                r"v[0-9]+",
                "an SSA v* result handle",
            )
            for index, item in enumerate(value)
        )
    raise RealizationWitnessError(f"{path} must be a string, string array, or null")


def _wire_walk_operations(
    operations: Any,
    *,
    prefix: str,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for index, raw in enumerate(_wire_array(operations, prefix)):
        path = f"{prefix}[{index}]"
        operation = _wire_object(raw, path)
        rows.append((path, operation))
        body = operation.get("body")
        if body is not None:
            rows.extend(_wire_walk_operations(body, prefix=f"{path}.body"))
        if operation.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if arm is None:
                    continue
                arm_object = _wire_object(arm, f"{path}.{polarity}")
                arm_body = arm_object.get("body")
                if arm_body is not None:
                    rows.extend(
                        _wire_walk_operations(
                            arm_body,
                            prefix=f"{path}.{polarity}.body",
                        )
                    )
    return tuple(rows)


def _wire_declared_j_names(j: Any) -> dict[str, set[str] | dict[str, Any]]:
    """Collect only J-owned namespaces needed by normalization bindings."""

    data = _wire_object(j, "j")
    boundary = _wire_object(data.get("boundary"), "j.boundary")
    domains_object = _wire_object(boundary.get("domains"), "j.boundary.domains")
    domains = set(domains_object)
    if any(re.fullmatch(r"d[0-9]+", item) is None for item in domains):
        raise RealizationWitnessError(
            "j.boundary.domains must contain only declaration-order d* handles"
        )

    access_refs: set[str] = set()
    for index, raw in enumerate(
        _wire_array(boundary.get("accesses"), "j.boundary.accesses")
    ):
        item = _wire_object(raw, f"j.boundary.accesses[{index}]")
        access_ref = item.get("access_id")
        if not isinstance(access_ref, str) or re.fullmatch(r"a[0-9]+", access_ref) is None:
            raise RealizationWitnessError(
                f"j.boundary.accesses[{index}].access_id must be an a* handle"
            )
        if access_ref in access_refs:
            raise RealizationWitnessError("J access handles must be unique")
        access_refs.add(access_ref)

    effect_refs: set[str] = set()
    for index, raw in enumerate(
        _wire_array(boundary.get("effects"), "j.boundary.effects")
    ):
        item = _wire_object(raw, f"j.boundary.effects[{index}]")
        effect_ref = item.get("effect_id", f"e{index}")
        if not isinstance(effect_ref, str) or re.fullmatch(r"e[0-9]+", effect_ref) is None:
            raise RealizationWitnessError(
                f"j.boundary.effects[{index}].effect_id must be an e* handle"
            )
        if effect_ref in effect_refs:
            raise RealizationWitnessError("J effect handles must be unique")
        effect_refs.add(effect_ref)

    scopes = _wire_handle_set(
        boundary.get("scopes", data.get("scopes", [])),
        path="j.scopes",
        pattern=r"q[0-9]+",
        fields=("scope_id",),
    )
    storage = _wire_handle_set(
        data.get("storage", []),
        path="j.storage",
        pattern=r"(?:b|s)[0-9]+",
        fields=("storage_id",),
    )
    authorizations = _wire_handle_set(
        boundary.get("realization_authorizations", []),
        path="j.boundary.realization_authorizations",
        pattern=r"u[0-9]+",
        fields=("auth_id",),
    )

    node_ids = _wire_object(data.get("node_ids"), "j.node_ids")
    nodes: set[str] = set()
    for node_ref, node_path in node_ids.items():
        if re.fullmatch(r"n[0-9]+", node_ref) is None or not isinstance(node_path, str):
            raise RealizationWitnessError(
                "j.node_ids must map n* handles to operation paths"
            )
        if node_ref in nodes:
            raise RealizationWitnessError("J node handles must be unique")
        nodes.add(node_ref)

    intent = _wire_object(data.get("intent"), "j.intent")
    intent_function = _wire_object(
        intent.get("intent_function"), "j.intent.intent_function"
    )
    value_refs: set[str] = set()
    tensor_refs: set[str] = set()
    states: set[str] = set()
    dimensions: set[str] = set()
    for field_name in ("inputs", "outputs"):
        for index, raw in enumerate(
            _wire_array(intent_function.get(field_name), f"j.intent.intent_function.{field_name}")
        ):
            item = _wire_object(
                raw, f"j.intent.intent_function.{field_name}[{index}]"
            )
            name = item.get("name")
            shape = item.get("shape")
            name = _wire_handle(
                name,
                f"j.intent.intent_function.{field_name}[{index}].name",
                r"t[0-9]+",
                "a logical tensor t* handle",
            )
            if not isinstance(shape, list):
                raise RealizationWitnessError(
                    f"j.intent.intent_function.{field_name}[{index}].shape must be an array"
                )
            tensor_refs.add(name)
            value_refs.add(name)
            for extent_index, extent in enumerate(shape):
                if isinstance(extent, str) and re.fullmatch(r"x[0-9]+", extent):
                    dimensions.add(extent)
                elif isinstance(extent, str) and re.fullmatch(r"p[0-9]+", extent):
                    raise RealizationWitnessError(
                        f"j.intent.intent_function.{field_name}[{index}].shape"
                        f"[{extent_index}] cannot use physical parameter {extent!r}"
                    )

    operation_rows = _wire_walk_operations(
        intent_function.get("body"),
        prefix="body",
    )
    operations_by_path = {path: operation for path, operation in operation_rows}
    operations_by_node: dict[str, dict[str, Any]] = {}
    for node_ref, node_path in node_ids.items():
        operation = operations_by_path.get(node_path)
        if operation is None:
            raise RealizationWitnessError(
                f"j.node_ids.{node_ref} does not resolve to a J operation"
            )
        operations_by_node[node_ref] = operation
        value_refs.update(
            _wire_operation_results(
                operation.get("result"),
                f"j.node_ids.{node_ref}.result",
            )
        )
        carries = operation.get("carry")
        if carries is None:
            continue
        for carry_index, raw_carry in enumerate(_wire_array(carries, f"j.node_ids.{node_ref}.carry")):
            carry = _wire_object(
                raw_carry,
                f"j.node_ids.{node_ref}.carry[{carry_index}]",
            )
            carry_path = f"j.node_ids.{node_ref}.carry[{carry_index}]"
            required_fields = {
                "state_id",
                "init_value",
                "body_argument",
                "next_value",
                "result_value",
                "type",
            }
            missing_fields = sorted(required_fields - set(carry))
            if missing_fields:
                raise RealizationWitnessError(
                    f"{carry_path} is missing complete loop-carry fields: "
                    + ", ".join(missing_fields)
                )
            state_id = _wire_handle(
                carry["state_id"],
                f"{carry_path}.state_id",
                r"s[0-9]+",
                "an s* loop-state handle",
            )
            if state_id in states:
                raise RealizationWitnessError(
                    f"{carry_path}.state_id duplicates loop state {state_id!r}"
                )
            states.add(state_id)
            if not isinstance(carry["type"], dict):
                raise RealizationWitnessError(f"{carry_path}.type must be an object")
            init_value = carry["init_value"]
            if not is_scalar_literal_or_value_reference(init_value):
                raise RealizationWitnessError(
                    f"{carry_path}.init_value must be a scalar literal or t*/v* "
                    "value reference; physical p* parameters are not J values"
                )
            if isinstance(init_value, str) and re.fullmatch(
                r"(?:t|v)[0-9]+", init_value
            ):
                value_refs.add(init_value)
            for field_name in ("body_argument", "next_value", "result_value"):
                ref = _wire_handle(
                    carry[field_name],
                    f"{carry_path}.{field_name}",
                    r"v[0-9]+",
                    "an SSA v* handle",
                )
                value_refs.add(ref)

    return {
        "domains": domains,
        "nodes": nodes,
        "operations_by_node": operations_by_node,
        "accesses": access_refs,
        "effects": effect_refs,
        "tensors": tensor_refs,
        "values": value_refs,
        "states": states,
        "dimensions": dimensions,
        "scopes": scopes,
        "storage": storage,
        "authorizations": authorizations,
    }


def _wire_axis_ref(value: Any, path: str) -> int | str:
    if isinstance(value, bool):
        raise RealizationWitnessError(f"{path} must be a non-negative integer or x*")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"x[0-9]+", value):
        return value
    raise RealizationWitnessError(f"{path} must be a non-negative integer or x*")


def _wire_unique_relation_rows(
    rows: list[Any],
    *,
    path: str,
    field: str,
    relations: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for index, raw in enumerate(rows):
        item_path = f"{path}[{index}]"
        item = _record(
            raw,
            path=item_path,
            required=frozenset({"relation", field}),
        )
        relation = _name(item["relation"], f"{item_path}.relation")
        if relation not in relations:
            raise RealizationWitnessError(
                f"{item_path}.relation is not a closed relation"
            )
        ref = _opaque_anchor_ref(item[field], f"{item_path}.{field}")
        key = (relation, ref)
        if key in seen:
            raise RealizationWitnessError(f"{path} rows must be unique")
        seen.add(key)
        result.append(key)
    return tuple(result)


def validate_normalization_bindings_against_j(
    j: Any,
    c: Any,
    pi: Any,
) -> None:
    """Validate the public J/C ownership boundary of active normalization π.

    This API deliberately has no dependency on :mod:`gate1.model` or
    :mod:`verify.typecheck`.  It accepts either JSON mappings or typed objects,
    verifies that every authored relation names a real J/C declaration, and
    raises :class:`RealizationWitnessError` when the relation cannot be proved
    from those declarations.  The deeper canonical axis/predicate checks remain
    in TypeCheck; this function prevents malformed or cross-owner references
    from reaching that private checker.
    """

    j_data = _wire_object(j, "j")
    c_data = _wire_object(c, "c")
    pi_data = _wire_object(pi, "pi")
    schema = pi_data.get("schema")
    if schema is None:
        return
    if schema != NORMALIZATION_WITNESS_SCHEMA:
        raise RealizationWitnessError(f"unsupported pi subwire schema {schema!r}")

    declared = _wire_declared_j_names(j_data)
    domains = declared["domains"]
    nodes = declared["nodes"]
    operations_by_node = declared["operations_by_node"]
    accesses = declared["accesses"]
    effects = declared["effects"]
    tensors = declared["tensors"]
    values = declared["values"]
    scopes = declared["scopes"]
    storage = declared["storage"]
    authorizations = declared["authorizations"]

    raw_slots = _wire_array(c_data.get("slots"), "c.slots")
    slots: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_slots):
        path = f"c.slots[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"name", "kind", "role"}),
            allowed=frozenset(
                {
                    "name",
                    "kind",
                    "role",
                    "value_type",
                    "required",
                    "subject_ref",
                    "axis_ref",
                }
            ),
        )
        name = _name(item["name"], f"{path}.name")
        if re.fullmatch(r"c[0-9]+", name) is None:
            raise RealizationWitnessError(f"{path}.name must be a C slot c*")
        if name in slots:
            raise RealizationWitnessError("C slot names must be unique")
        kind = _name(item["kind"], f"{path}.kind")
        if kind not in {"constexpr", "compiler"}:
            raise RealizationWitnessError(f"{path}.kind is not a closed C slot kind")
        role = _name(item["role"], f"{path}.role")
        subject_ref = item.get("subject_ref")
        axis_ref = item.get("axis_ref")
        if subject_ref is not None:
            subject_ref = _name(subject_ref, f"{path}.subject_ref")
            known_subjects = (
                set(tensors)
                | set(nodes)
                | set(domains)
                | set(accesses)
                | set(effects)
                | set(scopes)
                | set(storage)
                | set(authorizations)
            )
            if subject_ref not in known_subjects:
                raise RealizationWitnessError(
                    f"{path}.subject_ref is not a declared J scheduling subject"
                )
        if axis_ref is not None:
            if subject_ref is None:
                raise RealizationWitnessError(f"{path}.axis_ref requires subject_ref")
            axis_ref = _wire_axis_ref(axis_ref, f"{path}.axis_ref")
        if kind == "compiler" and (subject_ref is not None or axis_ref is not None):
            raise RealizationWitnessError(
                f"{path} compiler slots cannot attach to a J subject or axis"
            )
        slots[name] = {
            "kind": kind,
            "role": role,
            "subject_ref": subject_ref,
            "axis_ref": axis_ref,
        }

    def rows(name: str) -> list[Any]:
        raw = pi_data.get(name, [])
        return _wire_array(raw, f"pi.{name}")

    program_axes = rows("program_axes")
    bound_domains: set[str] = set()
    for index, raw in enumerate(program_axes):
        path = f"pi.program_axes[{index}]"
        item = _record(raw, path=path, required=frozenset({"domain_ref", "axis"}))
        domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
        if domain_ref not in domains:
            raise RealizationWitnessError(
                f"{path}.domain_ref is not a declared J boundary domain"
            )
        axis = item["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
            raise RealizationWitnessError(f"{path}.axis must be an integer in [0, 2]")
        if domain_ref in bound_domains:
            raise RealizationWitnessError("pi program-axis domain bindings must be unique")
        bound_domains.add(domain_ref)

    program_bindings = rows("program_bindings")
    claimed_program_axes: set[tuple[str, int | str]] = set()
    claimed_physical_axes: set[int] = set()
    physical_axis_by_domain: dict[str, int] = {}
    for index, raw in enumerate(program_bindings):
        path = f"pi.program_bindings[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"axis", "relation", "logical_axes"}),
        )
        physical_axis = item["axis"]
        if (
            isinstance(physical_axis, bool)
            or not isinstance(physical_axis, int)
            or not 0 <= physical_axis <= 2
        ):
            raise RealizationWitnessError(f"{path}.axis must be an integer in [0, 2]")
        if physical_axis in claimed_physical_axes:
            raise RealizationWitnessError(
                "pi program bindings must bind each physical launch axis once"
            )
        claimed_physical_axes.add(physical_axis)
        relation = _name(item["relation"], f"{path}.relation")
        if relation not in PROGRAM_BINDING_RELATIONS:
            raise RealizationWitnessError(f"{path}.relation is not a closed program relation")
        axes = _wire_array(item["logical_axes"], f"{path}.logical_axes")
        if not axes:
            raise RealizationWitnessError(f"{path}.logical_axes must not be empty")
        if relation == "direct" and len(axes) != 1:
            raise RealizationWitnessError(f"{path}.direct requires one logical axis")
        if relation != "direct" and len(axes) < 2:
            raise RealizationWitnessError(
                f"{path}.{relation} requires at least two logical axes"
            )
        local: set[tuple[str, int | str]] = set()
        for axis_index, raw_axis in enumerate(axes):
            axis_path = f"{path}.logical_axes[{axis_index}]"
            axis = _record(
                raw_axis,
                path=axis_path,
                required=frozenset({"domain_ref", "axis_ref"}),
            )
            domain_ref = _name(axis["domain_ref"], f"{axis_path}.domain_ref")
            if domain_ref not in domains:
                raise RealizationWitnessError(
                    f"{axis_path}.domain_ref is not a declared J domain"
                )
            axis_ref = _wire_axis_ref(axis["axis_ref"], f"{axis_path}.axis_ref")
            key = (domain_ref, axis_ref)
            if key in local or key in claimed_program_axes:
                raise RealizationWitnessError(
                    "pi program bindings must not claim one logical J axis twice"
                )
            previous_physical_axis = physical_axis_by_domain.setdefault(
                domain_ref, physical_axis
            )
            if previous_physical_axis != physical_axis:
                raise RealizationWitnessError(
                    "pi program bindings must map one J domain to one physical "
                    "launch axis"
                )
            local.add(key)
            claimed_program_axes.add(key)
    if program_axes and program_bindings:
        raise RealizationWitnessError(
            "pi.program_axes and pi.program_bindings are alternative complete "
            "program-coordinate mappings and are mutually exclusive"
        )

    iteration_bindings = rows("iteration_bindings")
    claimed_iteration_axes: set[tuple[str, int | str]] = set()
    for index, raw in enumerate(iteration_bindings):
        path = f"pi.iteration_bindings[{index}]"
        item = _record(
            raw,
            path=path,
            required=frozenset({"relation", "logical_axes", "tile_slot", "mode_slot"}),
        )
        relation = _name(item["relation"], f"{path}.relation")
        if relation not in ITERATION_BINDING_RELATIONS:
            raise RealizationWitnessError(f"{path}.relation is not a closed iteration relation")
        axes = _wire_array(item["logical_axes"], f"{path}.logical_axes")
        if not axes:
            raise RealizationWitnessError(f"{path}.logical_axes must not be empty")
        valid_axis_keys: set[tuple[str, int | str]] = set()
        for axis_index, raw_axis in enumerate(axes):
            axis_path = f"{path}.logical_axes[{axis_index}]"
            axis = _record(
                raw_axis,
                path=axis_path,
                required=frozenset({"subject_ref", "axis_ref"}),
            )
            subject_ref = _name(axis["subject_ref"], f"{axis_path}.subject_ref")
            if subject_ref not in domains and subject_ref not in nodes:
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref is not a declared J domain or node"
                )
            if relation == "tiled_domain" and subject_ref not in domains:
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref for tiled_domain must be a J domain"
                )
            if relation == "tiled_recurrence" and (
                subject_ref not in nodes
                or operations_by_node[subject_ref].get("op") != "loop"
            ):
                raise RealizationWitnessError(
                    f"{axis_path}.subject_ref for tiled_recurrence must be a J loop"
                )
            axis_ref = _wire_axis_ref(axis["axis_ref"], f"{axis_path}.axis_ref")
            key = (subject_ref, axis_ref)
            if key in valid_axis_keys or key in claimed_iteration_axes:
                raise RealizationWitnessError(
                    "pi iteration bindings must not claim one logical J axis twice"
                )
            valid_axis_keys.add(key)
            claimed_iteration_axes.add(key)

        tile_slot_ref = _name(item["tile_slot"], f"{path}.tile_slot")
        if tile_slot_ref not in slots:
            raise RealizationWitnessError(
                f"{path}.tile_slot is not declared by C"
            )
        tile_slot = slots[tile_slot_ref]
        if tile_slot["role"] not in {"tile_extent", "reduction_tile_extent"}:
            raise RealizationWitnessError(
                f"{path}.tile_slot must have role tile_extent or reduction_tile_extent"
            )
        tile_attachment = (tile_slot["subject_ref"], tile_slot["axis_ref"])
        if tile_attachment not in valid_axis_keys:
            raise RealizationWitnessError(
                f"{path}.tile_slot must attach to one of its authored logical axes"
            )

        mode_slot_ref = item["mode_slot"]
        if mode_slot_ref is not None:
            mode_slot_ref = _name(mode_slot_ref, f"{path}.mode_slot")
            if mode_slot_ref not in slots:
                raise RealizationWitnessError(
                    f"{path}.mode_slot is not declared by C"
                )
            mode_slot = slots[mode_slot_ref]
            mode_attachment = (mode_slot["subject_ref"], mode_slot["axis_ref"])
            if mode_slot["subject_ref"] is not None and mode_attachment not in valid_axis_keys:
                raise RealizationWitnessError(
                    f"{path}.mode_slot must attach to one of its authored logical axes"
                )

    _wire_unique_relation_rows(
        rows("control_bindings"),
        path="pi.control_bindings",
        field="subject_ref",
        relations=CONTROL_BINDING_RELATIONS,
    )
    for index, raw in enumerate(rows("control_bindings")):
        path = f"pi.control_bindings[{index}]"
        item = _record(raw, path=path, required=frozenset({"relation", "subject_ref"}))
        relation = item["relation"]
        subject_ref = item["subject_ref"]
        allowed = domains if relation == "terminal_domain_guard" else nodes | effects
        if subject_ref not in allowed:
            raise RealizationWitnessError(
                f"{path}.subject_ref is not a declared J anchor for {relation}"
            )

    _wire_unique_relation_rows(
        rows("materialization_bindings"),
        path="pi.materialization_bindings",
        field="value_ref",
        relations=MATERIALIZATION_BINDING_RELATIONS,
    )
    for index, raw in enumerate(rows("materialization_bindings")):
        path = f"pi.materialization_bindings[{index}]"
        item = _record(raw, path=path, required=frozenset({"relation", "value_ref"}))
        value_ref = item["value_ref"]
        if value_ref not in tensors | values | nodes:
            raise RealizationWitnessError(
                f"{path}.value_ref is not a J value-producing anchor"
            )

    _wire_unique_relation_rows(
        rows("access_representation_bindings"),
        path="pi.access_representation_bindings",
        field="access_ref",
        relations=ACCESS_REPRESENTATION_BINDING_RELATIONS,
    )
    for index, raw in enumerate(rows("access_representation_bindings")):
        path = f"pi.access_representation_bindings[{index}]"
        item = _record(raw, path=path, required=frozenset({"relation", "access_ref"}))
        if item["access_ref"] not in accesses:
            raise RealizationWitnessError(
                f"{path}.access_ref is not a declared J boundary access"
            )


def _name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealizationWitnessError(f"{path} must be a non-empty string")
    return value


def _opaque_anchor_ref(value: Any, path: str) -> str:
    result = _name(value, path)
    if re.fullmatch(r"(?:t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+", result) is None:
        raise RealizationWitnessError(
            f"{path} must be an anonymous declaration-order handle"
        )
    return result


def _declared_instance_ref(
    value: Any,
    *,
    path: str,
    known_instances: set[str],
) -> str:
    result = _name(value, path)
    if re.fullmatch(r"i[0-9]+", result) is None:
        raise RealizationWitnessError(
            f"{path} must reference a declared pi instance id i*; got {result!r}"
        )
    if result not in known_instances:
        raise RealizationWitnessError(
            f"{path} references undeclared pi instance {result!r}"
        )
    return result


def _declared_instance_refs(
    value: Any,
    *,
    path: str,
    known_instances: set[str],
) -> list[str]:
    if not isinstance(value, list):
        raise RealizationWitnessError(f"{path} must be an array")
    result = [
        _declared_instance_ref(
            item,
            path=f"{path}[{index}]",
            known_instances=known_instances,
        )
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise RealizationWitnessError(f"{path} must not contain duplicates")
    return result


def render_realization_witness_wire_grammar() -> str:
    """Return the single model-visible grammar for pi graph references."""

    return (
        "Pi has exactly two reference roles. instances[k].semantic_ref is an "
        "anchor into a declared anonymous J/C handle; J.signature handles are "
        "ordinary J handles and there is no interface namespace. It is not a pi "
        "graph node. instances[k].id declares the only pi graph-node namespace "
        "i0,i1,... in declaration order. Every routes source/target, order "
        "before/after, and sharing source/consumer must equal one already "
        "declared instance id i*. Edges never use raw t*/n*/a*/d*/e*/q*/b*/"
        "s*/v*/p*/c*/x* handles and never declare instances implicitly. To "
        "represent a p0 dependency on access a0, declare i0 anchored by p0 and "
        "i1 anchored by a0, then route i0 to i1; routing p0 to i1 is invalid. "
        "Multiple instances may share one semantic_ref when the source has "
        "multiple compiler-visible stages. Route slots are distinct, "
        "non-negative, target-local opaque input ordinals. An instance region is "
        "exact lexical realization scope, not an optimization hint or consumer "
        "membership. The witness states that Source emitted every named "
        "compiler-visible stage inside that exact region: a loop-region instance "
        "stays lexically inside its loop even when its operands are loop invariant, "
        "and a parent-region instance stays outside. This is certification input to "
        "a fixed Gate1 normalizer, not an instruction to Replay. A relation remains "
        "hard unless the active profile has a sound unary matcher. program_axes is "
        "the direct domain-to-launch-axis form. program_bindings is the complete "
        "relation form: every row carries its one physical launch axis 0, 1, or 2 "
        "together with direct, product_linearized, or guarded_union logical axes. "
        "The two forms are mutually exclusive, and a relation row is unique by its "
        "physical axis while every named J domain stays on one physical axis. "
        "Distinct logical consumer domains may share one physical axis through one "
        "relation row; repeated instances of the same d* reuse that coordinate. "
        "An empty-slot route from a program-region d* realization into a branch "
        "region makes that region an inside-domain physical realization. The "
        "condition remains the J domain; pi carries no predicate, grid extent, "
        "Source spelling, or frontend API. program_linearizations gives the "
        "outer-to-inner J dimension handles of one mixed-radix decode of that "
        "coordinate; its denotation is the unique row-major quotient/remainder "
        "projection and it carries no formula. "
        "route_extent_transforms qualifies one existing route slot: "
        "a required positive-integer C extent reaches its attached J domain as "
        "floor_div/ceil_div by a literal divisor greater than one. It is "
        "representation-only and creates no J fact or Source/API spelling. Each iterations "
        "entry relates one loop region to one instance anchored by its J loop "
        "and one instance anchored by the matching C tile slot. Its closed "
        "induction_basis records only whether the physical induction variable "
        "is measured in logical extent units, rounded extent units, or tile "
        "ordinals. The global "
        "source-neutral denotation is closed: extent_step means lower zero, "
        "upper equal to the J loop trip extent, and step equal to the matching "
        "C tile; rounded_extent_step means lower zero, upper equal to "
        "ceil_div(J loop trip extent, matching C tile) multiplied by that C "
        "tile, and step equal to the matching C tile; "
        "tile_ordinal means lower zero, upper equal to ceil_div of that "
        "J trip extent by that matching C tile, and unit step. Every frontend "
        "maps this same relation to its public constructs. The wire entry never "
        "carries a bound expression, value, operator, source fragment, or "
        "frontend API. Each declaration-ordered loop_carriers entry has exactly "
        "region, loop_instance, init, state, next, and result instance refs. "
        "Its global temporal denotation is state[0]=init, "
        "state[k+1]=next[k], result=state[T], including T=0. The four roles are "
        "distinct occurrences even when they anchor the same J semantic value; "
        "ordinary routes close init -> state -> next. The temporal "
        "next-to-state edge and loop-exit result edge are not ordinary DAG "
        "routes and never carry an expression or frontend spelling. Each "
        "physical_iterations entry has exactly region, body_instance, "
        "tile_instance, induction_basis, and direction. body_instance anchors "
        "an existing non-loop J reduce/contraction or boundary access/domain with "
        "a closed dependent axis; tile_instance anchors one required positive C "
        "tile attached to that same semantic subject and axis. forward uses the "
        "corresponding zero-origin basis order; reverse "
        "visits that same closed basis in descending order. It never adds a J "
        "loop or carries bounds/formulas. Each physical_loop_carriers entry has "
        "exactly region, init, state, next, and result, with the same temporal "
        "denotation as loop_carriers, but its region is indexed by one "
        "physical_iterations relation rather than a structured J loop. The four "
        "roles are distinct physical occurrences even when they anchor the same J "
        "semantic value; ordinary routes close init -> state -> next, while the "
        "next-to-state temporal edge and loop-exit result remain denoted by this "
        "carrier relation. For all "
        "physical_loop_carriers entries sharing one region, declaration order is "
        "the common Source carrier-tuple ordinal across init/state/next/result. It is "
        "not event execution order and does not replace "
        "explicit pi.order edges."
        " terminal_controls has one of two exact row shapes. An effect row has "
        "terminal_instance and continuation_instance: distinct realize instances "
        "on sibling J effect_if arm scopes. A program-domain row has domain_ref and "
        "continuation_region: that existing branch region executes inside the same "
        "J domain and its complement terminates the invocation. Neither form "
        "carries a predicate, polarity, return, Source spelling, or API. Each "
        "reduction_linearization "
        "refinement has null auth_ref and refines one realize instance whose sole "
        "slot-0 route enters a J reduce realize instance. It denotes full-axis "
        "linearization before that consumer without carrying an axis, operator, "
        "value, or frontend API."
    )


def render_realization_mechanism_grammar() -> str:
    """Return the shared, non-executable meaning of pi mechanism tags."""

    return (
        "Pi mechanism is one closed source-neutral event class. realize is a "
        "generic representation-only stage when no more specific class applies. "
        "One realize instance anchored directly to a J.signature handle or C slot "
        "may be discharged by that parameter binding and its direct routed use; "
        "it does not require an identity assignment. Multiple instances anchored "
        "to the same handle still denote distinct routed use-sites/stages. "
        "address_base is an independently compiler-visible, program-uniform "
        "pointer/base or integer base-address contribution before lane merge; "
        "Lift uses it only when that separate source stage exists. address_lane "
        "is the compiler-visible stage that forms the lane-varying contribution "
        "or final address. "
        "When routes contain address_base -> address_lane, the witness records two "
        "distinct Source stages in that order rather than one flat vector offset. "
        "Pointer-operation count is determined only "
        "by source-neutral carrier routing and the carrier kind declared in "
        "J.signature. If a pointer/tensor carrier directly enters address_base, "
        "that base is a pointer/address operation and its route to address_lane "
        "requires a second pointer/address operation for the lane contribution. "
        "If the pointer/tensor carrier bypasses address_base and enters "
        "address_lane or the final address stage, while address_base receives only "
        "integer, domain, or C contributions, the Source witness denotes an integer "
        "base/lane merge before exactly one pointer/address operation. The "
        "address_base -> address_lane edge alone never decides between these cases. "
        "transfer_direct and "
        "transfer_async classify "
        "data movement after address formation; update_atomic and "
        "update_privatized classify physical update realization; synchronize "
        "classifies synchronization. Region placement='program' only scopes an "
        "instance to one program; it does not assert that a program/grid "
        "coordinate is read. A compiler-visible program/grid coordinate is a "
        "separate realization stage and requires its own realize instance and "
        "routes. Its source-neutral encoding is a realize instance anchored to "
        "the corresponding J domain in a program region whose scalar dependency "
        "path enters address_base before transfer; a domain/C dependency path "
        "entering address_lane is lane-varying instead. This base-versus-lane "
        "distinction is a Source-realization witness only. "
        "Program-axis identity follows the closed pi.program_axes direct mapping "
        "or pi.program_bindings relation row for the J domain semantic_ref, not "
        "instance multiplicity: repeated "
        "program-coordinate instances on one d* remain distinct stages of the "
        "same bound axis. Lift must not infer an axis from declaration order, "
        "grid extent, instance multiplicity, or C. "
        "Integer coordinate arithmetic is a realize stage; address_base and "
        "address_lane classify where its program-uniform and lane-varying paths "
        "enter address formation rather than relabeling every arithmetic op. If "
        "Source visibly forms a separable program-uniform integer base and a lane "
        "offset, then merges them before its first pointer application, the program "
        "path still enters address_base and routes to address_lane while the pointer "
        "carrier bypasses address_base. The witness records both integer stages and "
        "one pointer/address operation at the lane/final stage. Only "
        "a Source with no separable program-uniform contribution uses a lone "
        "address_lane stage. "
        "Lift must not invent program_id, a grid offset, or an affine "
        "address stage from region placement, grid convention, or C alone. "
        "Address classes never carry an index/offset "
        "formula, constant, integer width, cast, dtype, shape, vector length, "
        "predicate, fill, frontend API, or logical access semantics."
    )


def _require_acyclic(
    nodes: set[str], edges: list[tuple[str, str]], path: str
) -> None:
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise RealizationWitnessError(f"{path} must be acyclic")


@dataclass
class RealizationWitness:
    """Typed compact-or-materialized realization witness.

    Lift authors the compact normalization subwire.  BindSource owns the
    subsequent materialization into the full graph required by Replay and Gate.
    """

    wire_schema: str | None = None
    regions: list[dict[str, JsonValue]] = field(default_factory=list)
    instances: list[dict[str, JsonValue]] = field(default_factory=list)
    routes: list[dict[str, JsonValue]] = field(default_factory=list)
    program_axes: list[dict[str, JsonValue]] = field(default_factory=list)
    program_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    iteration_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    control_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    materialization_bindings: list[dict[str, JsonValue]] = field(default_factory=list)
    access_representation_bindings: list[dict[str, JsonValue]] = field(
        default_factory=list
    )
    program_linearizations: list[dict[str, JsonValue]] = field(default_factory=list)
    route_extent_transforms: list[dict[str, JsonValue]] = field(default_factory=list)
    order: list[dict[str, JsonValue]] = field(default_factory=list)
    sharing: list[dict[str, JsonValue]] = field(default_factory=list)
    refinements: list[dict[str, JsonValue]] = field(default_factory=list)
    iterations: list[dict[str, JsonValue]] = field(default_factory=list)
    loop_carriers: list[dict[str, JsonValue]] = field(default_factory=list)
    physical_iterations: list[dict[str, JsonValue]] = field(default_factory=list)
    physical_loop_carriers: list[dict[str, JsonValue]] = field(default_factory=list)
    terminal_controls: list[dict[str, JsonValue]] = field(default_factory=list)

    PLACEMENTS: ClassVar[frozenset[str]] = frozenset(
        {"root", "program", "loop", "branch"}
    )
    MECHANISMS: ClassVar[frozenset[str]] = frozenset(REALIZATION_MECHANISMS)
    ITERATION_BASES: ClassVar[frozenset[str]] = frozenset(ITERATION_BASES)
    PHYSICAL_ITERATION_DIRECTIONS: ClassVar[frozenset[str]] = frozenset(
        PHYSICAL_ITERATION_DIRECTIONS
    )
    ROUTE_EXTENT_ROUNDINGS: ClassVar[frozenset[str]] = frozenset(
        ROUTE_EXTENT_ROUNDINGS
    )

    def __post_init__(self) -> None:
        if self.wire_schema not in {None, NORMALIZATION_WITNESS_SCHEMA}:
            raise RealizationWitnessError(
                f"unsupported pi subwire schema {self.wire_schema!r}"
            )
        if self.wire_schema == NORMALIZATION_WITNESS_SCHEMA:
            full_graph_fields = {
                "program_linearizations": self.program_linearizations,
                "regions": self.regions,
                "instances": self.instances,
                "routes": self.routes,
                "route_extent_transforms": self.route_extent_transforms,
                "order": self.order,
                "sharing": self.sharing,
                "refinements": self.refinements,
                "iterations": self.iterations,
                "loop_carriers": self.loop_carriers,
                "physical_iterations": self.physical_iterations,
                "physical_loop_carriers": self.physical_loop_carriers,
                "terminal_controls": self.terminal_controls,
            }
            present = sorted(
                name for name, rows in full_graph_fields.items() if rows
            )
            if present:
                raise RealizationWitnessError(
                    "normalization-witness pi may not contain the full "
                    "materialized realization graph; "
                    f"found unsupported fields {present}"
                )
            self._check_normalization_bindings()
            return
        checked_regions: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.regions):
            item = _record(
                raw,
                path=f"pi.regions[{index}]",
                required=frozenset({"id", "parent", "placement"}),
            )
            region_id = _name(item["id"], f"pi.regions[{index}].id")
            if region_id != f"r{index}":
                raise RealizationWitnessError(
                    "pi region ids must be declaration-order handles r0, r1, ..."
                )
            parent = item["parent"]
            if parent is not None:
                parent = _name(parent, f"pi.regions[{index}].parent")
            placement = _name(item["placement"], f"pi.regions[{index}].placement")
            if placement not in self.PLACEMENTS:
                raise RealizationWitnessError(f"unsupported pi placement {placement!r}")
            checked_regions.append(
                {"id": region_id, "parent": parent, "placement": placement}
            )
        region_ids = [item["id"] for item in checked_regions]
        if len(region_ids) != len(set(region_ids)):
            raise RealizationWitnessError("pi region ids must be unique")
        known_regions = set(region_ids)
        for item in checked_regions:
            if item["parent"] is not None and item["parent"] not in known_regions:
                raise RealizationWitnessError("pi region parent is unknown")
            if item["parent"] == item["id"]:
                raise RealizationWitnessError("pi region cannot parent itself")
        _require_acyclic(
            known_regions,
            [
                (item["parent"], item["id"])
                for item in checked_regions
                if item["parent"] is not None
            ],
            "pi region tree",
        )

        checked_instances: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.instances):
            item = _record(
                raw,
                path=f"pi.instances[{index}]",
                required=frozenset({"id", "semantic_ref", "region", "mechanism"}),
            )
            instance_id = _name(item["id"], f"pi.instances[{index}].id")
            if instance_id != f"i{index}":
                raise RealizationWitnessError(
                    "pi instance ids must be declaration-order handles i0, i1, ..."
                )
            semantic_ref = _opaque_anchor_ref(
                item["semantic_ref"], f"pi.instances[{index}].semantic_ref"
            )
            region = _name(item["region"], f"pi.instances[{index}].region")
            mechanism = _name(item["mechanism"], f"pi.instances[{index}].mechanism")
            if region not in known_regions:
                raise RealizationWitnessError("pi instance region is unknown")
            if mechanism not in self.MECHANISMS:
                raise RealizationWitnessError(
                    f"pi mechanism {mechanism!r} is not source-neutral"
                )
            checked_instances.append(
                {
                    "id": instance_id,
                    "semantic_ref": semantic_ref,
                    "region": region,
                    "mechanism": mechanism,
                }
            )
        instance_ids = [item["id"] for item in checked_instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise RealizationWitnessError("pi instance ids must be unique")
        known_instances = set(instance_ids)

        checked_program_axes: list[dict[str, JsonValue]] = []
        bound_domains: set[str] = set()
        for index, raw in enumerate(self.program_axes):
            path = f"pi.program_axes[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"domain_ref", "axis"}),
            )
            domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
            if re.fullmatch(r"d[0-9]+", domain_ref) is None:
                raise RealizationWitnessError(
                    f"{path}.domain_ref must be a J boundary domain handle d*"
                )
            axis = item["axis"]
            if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
                raise RealizationWitnessError(
                    f"{path}.axis must be an integer launch axis in [0, 2]"
                )
            if domain_ref in bound_domains:
                raise RealizationWitnessError(
                    "pi program-axis domain bindings must be unique"
                )
            bound_domains.add(domain_ref)
            checked_program_axes.append({"domain_ref": domain_ref, "axis": axis})

        checked_program_linearizations: list[dict[str, JsonValue]] = []
        linearized_domains: set[str] = set()
        for index, raw in enumerate(self.program_linearizations):
            path = f"pi.program_linearizations[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"domain_ref", "dimension_refs"}),
            )
            domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
            if re.fullmatch(r"d[0-9]+", domain_ref) is None:
                raise RealizationWitnessError(
                    f"{path}.domain_ref must be a J boundary domain handle d*"
                )
            raw_dimensions = item["dimension_refs"]
            if not isinstance(raw_dimensions, list) or not raw_dimensions:
                raise RealizationWitnessError(
                    f"{path}.dimension_refs must be a non-empty array"
                )
            dimension_refs: list[str] = []
            for ordinal, raw_dimension in enumerate(raw_dimensions):
                dimension_ref = _name(
                    raw_dimension, f"{path}.dimension_refs[{ordinal}]"
                )
                if re.fullmatch(r"x[0-9]+", dimension_ref) is None:
                    raise RealizationWitnessError(
                        f"{path}.dimension_refs must contain only J dimensions x*"
                    )
                dimension_refs.append(dimension_ref)
            if len(dimension_refs) != len(set(dimension_refs)):
                raise RealizationWitnessError(
                    "pi program-linearization dimensions must be unique"
                )
            if domain_ref in linearized_domains:
                raise RealizationWitnessError(
                    "pi program linearizations must be unique by domain_ref"
                )
            linearized_domains.add(domain_ref)
            checked_program_linearizations.append(
                {"domain_ref": domain_ref, "dimension_refs": dimension_refs}
            )

        checked_terminal_controls: list[dict[str, JsonValue]] = []
        terminal_control_instances: set[str] = set()
        terminal_control_domains: set[str] = set()
        terminal_continuation_regions: set[str] = set()
        for index, raw in enumerate(self.terminal_controls):
            path = f"pi.terminal_controls[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(),
                allowed=frozenset(
                    {
                        "terminal_instance",
                        "continuation_instance",
                        "domain_ref",
                        "continuation_region",
                    }
                ),
            )
            fields = set(item)
            if fields == {"domain_ref", "continuation_region"}:
                domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
                if re.fullmatch(r"d[0-9]+", domain_ref) is None:
                    raise RealizationWitnessError(
                        f"{path}.domain_ref must be a J boundary domain handle d*"
                    )
                continuation_region = _name(
                    item["continuation_region"], f"{path}.continuation_region"
                )
                if continuation_region not in known_regions:
                    raise RealizationWitnessError(
                        "pi program-domain terminal names an unknown continuation region"
                    )
                if domain_ref in terminal_control_domains:
                    raise RealizationWitnessError(
                        "one J domain may have at most one terminal complement"
                    )
                if continuation_region in terminal_continuation_regions:
                    raise RealizationWitnessError(
                        "one continuation region may have at most one terminal complement"
                    )
                terminal_control_domains.add(domain_ref)
                terminal_continuation_regions.add(continuation_region)
                checked_terminal_controls.append(
                    {
                        "domain_ref": domain_ref,
                        "continuation_region": continuation_region,
                    }
                )
                continue
            if fields != {"terminal_instance", "continuation_instance"}:
                raise RealizationWitnessError(
                    f"{path} must be exactly an effect-arm pair or a "
                    "program-domain continuation pair"
                )
            terminal_instance = _declared_instance_ref(
                item["terminal_instance"],
                path=f"{path}.terminal_instance",
                known_instances=known_instances,
            )
            continuation_instance = _declared_instance_ref(
                item["continuation_instance"],
                path=f"{path}.continuation_instance",
                known_instances=known_instances,
            )
            if terminal_instance == continuation_instance:
                raise RealizationWitnessError(
                    "pi terminal control requires distinct terminal and "
                    "continuation instances"
                )
            if (
                terminal_instance in terminal_control_instances
                or continuation_instance in terminal_control_instances
            ):
                raise RealizationWitnessError(
                    "pi terminal control arm instances must be globally unique"
                )
            terminal_control_instances.update(
                {terminal_instance, continuation_instance}
            )
            checked_terminal_controls.append(
                {
                    "terminal_instance": terminal_instance,
                    "continuation_instance": continuation_instance,
                }
            )

        checked_iterations: list[dict[str, JsonValue]] = []
        iteration_regions: set[str] = set()
        for index, raw in enumerate(self.iterations):
            item = _record(
                raw,
                path=f"pi.iterations[{index}]",
                required=frozenset(
                    {
                        "region",
                        "loop_instance",
                        "tile_instance",
                        "induction_basis",
                    }
                ),
            )
            region = _name(item["region"], f"pi.iterations[{index}].region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi iteration relation names an unknown region"
                )
            if region in iteration_regions:
                raise RealizationWitnessError(
                    "pi iteration regions may have at most one relation"
                )
            iteration_regions.add(region)
            loop_instance = _declared_instance_ref(
                item["loop_instance"],
                path=f"pi.iterations[{index}].loop_instance",
                known_instances=known_instances,
            )
            tile_instance = _declared_instance_ref(
                item["tile_instance"],
                path=f"pi.iterations[{index}].tile_instance",
                known_instances=known_instances,
            )
            if loop_instance == tile_instance:
                raise RealizationWitnessError(
                    "pi iteration loop and tile instances must be distinct"
                )
            induction_basis = _name(
                item["induction_basis"],
                f"pi.iterations[{index}].induction_basis",
            )
            if induction_basis not in self.ITERATION_BASES:
                raise RealizationWitnessError(
                    f"unsupported pi induction basis {induction_basis!r}"
                )
            checked_iterations.append(
                {
                    "region": region,
                    "loop_instance": loop_instance,
                    "tile_instance": tile_instance,
                    "induction_basis": induction_basis,
                }
            )

        checked_physical_iterations: list[dict[str, JsonValue]] = []
        physical_iteration_regions: set[str] = set()
        for index, raw in enumerate(self.physical_iterations):
            path = f"pi.physical_iterations[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {
                        "region",
                        "body_instance",
                        "tile_instance",
                        "induction_basis",
                        "direction",
                    }
                ),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi physical iteration relation names an unknown region"
                )
            if region in physical_iteration_regions:
                raise RealizationWitnessError(
                    "pi physical iteration regions may have at most one relation"
                )
            if region in iteration_regions:
                raise RealizationWitnessError(
                    "one pi region cannot carry both structured and physical-only "
                    "iteration relations"
                )
            physical_iteration_regions.add(region)
            body_instance = _declared_instance_ref(
                item["body_instance"],
                path=f"{path}.body_instance",
                known_instances=known_instances,
            )
            tile_instance = _declared_instance_ref(
                item["tile_instance"],
                path=f"{path}.tile_instance",
                known_instances=known_instances,
            )
            if body_instance == tile_instance:
                raise RealizationWitnessError(
                    "pi physical iteration body and tile instances must be distinct"
                )
            induction_basis = _name(
                item["induction_basis"], f"{path}.induction_basis"
            )
            if induction_basis not in self.ITERATION_BASES:
                raise RealizationWitnessError(
                    f"unsupported pi physical induction basis {induction_basis!r}"
                )
            direction = _name(item["direction"], f"{path}.direction")
            if direction not in self.PHYSICAL_ITERATION_DIRECTIONS:
                raise RealizationWitnessError(
                    f"unsupported pi physical iteration direction {direction!r}"
                )
            checked_physical_iterations.append(
                {
                    "region": region,
                    "body_instance": body_instance,
                    "tile_instance": tile_instance,
                    "induction_basis": induction_basis,
                    "direction": direction,
                }
            )

        checked_loop_carriers: list[dict[str, JsonValue]] = []
        carrier_states: set[str] = set()
        carrier_results: set[str] = set()
        for index, raw in enumerate(self.loop_carriers):
            path = f"pi.loop_carriers[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {
                        "region",
                        "loop_instance",
                        "init",
                        "state",
                        "next",
                        "result",
                    }
                ),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in known_regions:
                raise RealizationWitnessError(
                    "pi loop carrier relation names an unknown region"
                )
            refs = {
                key: _declared_instance_ref(
                    item[key],
                    path=f"{path}.{key}",
                    known_instances=known_instances,
                )
                for key in ("loop_instance", "init", "state", "next", "result")
            }
            if refs["init"] in {refs["state"], refs["next"], refs["result"]}:
                raise RealizationWitnessError(
                    "pi loop carrier init must be a distinct instance"
                )
            if refs["result"] in {refs["state"], refs["next"]}:
                raise RealizationWitnessError(
                    "pi loop carrier result must be a distinct instance"
                )
            if refs["state"] == refs["next"]:
                raise RealizationWitnessError(
                    "pi loop carrier state and next must be distinct temporal instances"
                )
            if refs["state"] in carrier_states:
                raise RealizationWitnessError(
                    "pi loop carrier state slots must be unique"
                )
            if refs["result"] in carrier_results:
                raise RealizationWitnessError(
                    "pi loop carrier result slots must be unique"
                )
            carrier_states.add(refs["state"])
            carrier_results.add(refs["result"])
            checked_loop_carriers.append({"region": region, **refs})

        checked_physical_loop_carriers: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.physical_loop_carriers):
            path = f"pi.physical_loop_carriers[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"region", "init", "state", "next", "result"}),
            )
            region = _name(item["region"], f"{path}.region")
            if region not in physical_iteration_regions:
                raise RealizationWitnessError(
                    "pi physical loop carrier requires one declared physical "
                    "iteration region"
                )
            refs = {
                key: _declared_instance_ref(
                    item[key],
                    path=f"{path}.{key}",
                    known_instances=known_instances,
                )
                for key in ("init", "state", "next", "result")
            }
            if refs["init"] in {refs["state"], refs["next"], refs["result"]}:
                raise RealizationWitnessError(
                    "pi physical loop carrier init must be a distinct instance"
                )
            if refs["result"] in {refs["state"], refs["next"]}:
                raise RealizationWitnessError(
                    "pi physical loop carrier result must be a distinct instance"
                )
            if refs["state"] == refs["next"]:
                raise RealizationWitnessError(
                    "pi physical loop carrier state and next must be distinct temporal instances"
                )
            if refs["state"] in carrier_states:
                raise RealizationWitnessError(
                    "pi physical loop carrier state slots must be globally unique"
                )
            if refs["result"] in carrier_results:
                raise RealizationWitnessError(
                    "pi physical loop carrier result slots must be globally unique"
                )
            carrier_states.add(refs["state"])
            carrier_results.add(refs["result"])
            checked_physical_loop_carriers.append({"region": region, **refs})

        checked_routes: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.routes):
            item = _record(
                raw,
                path=f"pi.routes[{index}]",
                required=frozenset({"source", "target", "slots"}),
            )
            source = _declared_instance_ref(
                item["source"],
                path=f"pi.routes[{index}].source",
                known_instances=known_instances,
            )
            target = _declared_instance_ref(
                item["target"],
                path=f"pi.routes[{index}].target",
                known_instances=known_instances,
            )
            slots = item["slots"]
            if not isinstance(slots, list):
                raise RealizationWitnessError("pi route slots must be an array")
            checked_slots: list[int] = []
            for slot in slots:
                if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                    raise RealizationWitnessError("pi route slots must be non-negative integers")
                checked_slots.append(slot)
            if len(checked_slots) != len(set(checked_slots)):
                raise RealizationWitnessError("pi route slots must be unique")
            checked_routes.append(
                {"source": source, "target": target, "slots": checked_slots}
            )
        route_keys = [
            (item["source"], item["target"], tuple(item["slots"]))
            for item in checked_routes
        ]
        if len(route_keys) != len(set(route_keys)):
            raise RealizationWitnessError("pi routes must be unique")

        checked_route_extent_transforms: list[dict[str, JsonValue]] = []
        transformed_route_slots: set[tuple[str, str, int]] = set()
        for index, raw in enumerate(self.route_extent_transforms):
            path = f"pi.route_extent_transforms[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {"source", "target", "slot", "divisor", "rounding"}
                ),
            )
            source = _declared_instance_ref(
                item["source"],
                path=f"{path}.source",
                known_instances=known_instances,
            )
            target = _declared_instance_ref(
                item["target"],
                path=f"{path}.target",
                known_instances=known_instances,
            )
            slot = item["slot"]
            if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
                raise RealizationWitnessError(
                    f"{path}.slot must be a non-negative integer"
                )
            divisor = item["divisor"]
            if (
                isinstance(divisor, bool)
                or not isinstance(divisor, int)
                or divisor <= 1
            ):
                raise RealizationWitnessError(
                    f"{path}.divisor must be an integer greater than one"
                )
            rounding = _name(item["rounding"], f"{path}.rounding")
            if rounding not in self.ROUTE_EXTENT_ROUNDINGS:
                raise RealizationWitnessError(
                    f"unsupported pi route extent rounding {rounding!r}"
                )
            route_slot = (source, target, slot)
            if not any(
                route["source"] == source
                and route["target"] == target
                and slot in route["slots"]
                for route in checked_routes
            ):
                raise RealizationWitnessError(
                    f"{path} must qualify one slot of an existing pi route"
                )
            if route_slot in transformed_route_slots:
                raise RealizationWitnessError(
                    "pi route extent transforms must be unique per route slot"
                )
            transformed_route_slots.add(route_slot)
            checked_route_extent_transforms.append(
                {
                    "source": source,
                    "target": target,
                    "slot": slot,
                    "divisor": divisor,
                    "rounding": rounding,
                }
            )

        checked_order: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.order):
            item = _record(
                raw,
                path=f"pi.order[{index}]",
                required=frozenset({"before", "after"}),
            )
            before = _declared_instance_ref(
                item["before"],
                path=f"pi.order[{index}].before",
                known_instances=known_instances,
            )
            after = _declared_instance_ref(
                item["after"],
                path=f"pi.order[{index}].after",
                known_instances=known_instances,
            )
            if before == after:
                raise RealizationWitnessError("pi order cannot contain a self edge")
            checked_order.append({"before": before, "after": after})
        pairs = [(item["before"], item["after"]) for item in checked_order]
        if len(pairs) != len(set(pairs)):
            raise RealizationWitnessError("pi order edges must be unique")
        _require_acyclic(known_instances, pairs, "pi order")

        checked_sharing: list[dict[str, JsonValue]] = []
        for index, raw in enumerate(self.sharing):
            item = _record(
                raw,
                path=f"pi.sharing[{index}]",
                required=frozenset({"source", "consumers"}),
            )
            source = _declared_instance_ref(
                item["source"],
                path=f"pi.sharing[{index}].source",
                known_instances=known_instances,
            )
            consumers = _declared_instance_refs(
                item["consumers"],
                path=f"pi.sharing[{index}].consumers",
                known_instances=known_instances,
            )
            if not consumers:
                raise RealizationWitnessError("pi sharing consumers must not be empty")
            checked_sharing.append({"source": source, "consumers": consumers})
        sharing_sources = [item["source"] for item in checked_sharing]
        if len(sharing_sources) != len(set(sharing_sources)):
            raise RealizationWitnessError("pi sharing source must be unique")

        checked_refinements: list[dict[str, JsonValue]] = []
        refined_instances: set[str] = set()
        for index, raw in enumerate(self.refinements):
            item = _record(
                raw,
                path=f"pi.refinements[{index}]",
                required=frozenset({"instance", "kind", "auth_ref"}),
            )
            instance = _declared_instance_ref(
                item["instance"],
                path=f"pi.refinements[{index}].instance",
                known_instances=known_instances,
            )
            if instance in refined_instances:
                raise RealizationWitnessError(
                    "pi refinements may refine each instance at most once"
                )
            refined_instances.add(instance)
            kind = _name(item["kind"], f"pi.refinements[{index}].kind")
            if kind not in REPRESENTATION_REFINEMENTS:
                raise RealizationWitnessError(
                    f"unsupported pi representation refinement {kind!r}"
                )
            auth_ref = item["auth_ref"]
            if auth_ref is not None:
                auth_ref = _name(
                    auth_ref, f"pi.refinements[{index}].auth_ref"
                )
                if re.fullmatch(r"u[0-9]+", auth_ref) is None:
                    raise RealizationWitnessError(
                        "pi refinement auth_ref must be a J authorization handle u*"
                    )
            checked_refinements.append(
                {"instance": instance, "kind": kind, "auth_ref": auth_ref}
            )

        self.regions = checked_regions
        self.instances = checked_instances
        self.routes = checked_routes
        self.program_axes = checked_program_axes
        self.program_linearizations = checked_program_linearizations
        self.route_extent_transforms = checked_route_extent_transforms
        self.order = checked_order
        self.sharing = checked_sharing
        self.refinements = checked_refinements
        self.iterations = checked_iterations
        self.loop_carriers = checked_loop_carriers
        self.physical_iterations = checked_physical_iterations
        self.physical_loop_carriers = checked_physical_loop_carriers
        self.terminal_controls = checked_terminal_controls

    def _check_normalization_bindings(self) -> None:
        """Validate only the closed active subwire, without deriving relations."""

        checked_program_axes: list[dict[str, JsonValue]] = []
        bound_domains: set[str] = set()
        for index, raw in enumerate(self.program_axes):
            path = f"pi.program_axes[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"domain_ref", "axis"}),
            )
            domain_ref = _name(item["domain_ref"], f"{path}.domain_ref")
            if re.fullmatch(r"d[0-9]+", domain_ref) is None:
                raise RealizationWitnessError(
                    f"{path}.domain_ref must be a J boundary domain handle d*"
                )
            axis = item["axis"]
            if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
                raise RealizationWitnessError(
                    f"{path}.axis must be an integer launch axis in [0, 2]"
                )
            if domain_ref in bound_domains:
                raise RealizationWitnessError(
                    "pi program-axis domain bindings must be unique"
                )
            bound_domains.add(domain_ref)
            checked_program_axes.append({"domain_ref": domain_ref, "axis": axis})

        def axis_ref(value: Any, path: str) -> int | str:
            if isinstance(value, bool):
                raise RealizationWitnessError(
                    f"{path} must be a non-negative integer or J dimension x*"
                )
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and re.fullmatch(r"x[0-9]+", value):
                return value
            raise RealizationWitnessError(
                f"{path} must be a non-negative integer or J dimension x*"
            )

        def logical_axes(
            value: Any,
            *,
            path: str,
            subject_field: str,
            subject_pattern: str,
        ) -> list[dict[str, JsonValue]]:
            if not isinstance(value, list) or not value:
                raise RealizationWitnessError(f"{path} must be a non-empty array")
            result: list[dict[str, JsonValue]] = []
            seen: set[tuple[str, int | str]] = set()
            for ordinal, raw_axis in enumerate(value):
                axis_path = f"{path}[{ordinal}]"
                item = _record(
                    raw_axis,
                    path=axis_path,
                    required=frozenset({subject_field, "axis_ref"}),
                )
                subject = _name(item[subject_field], f"{axis_path}.{subject_field}")
                if re.fullmatch(subject_pattern, subject) is None:
                    raise RealizationWitnessError(
                        f"{axis_path}.{subject_field} has an unsupported J handle"
                    )
                logical_axis = axis_ref(item["axis_ref"], f"{axis_path}.axis_ref")
                key = (subject, logical_axis)
                if key in seen:
                    raise RealizationWitnessError(
                        f"{path} must not repeat a logical J axis"
                    )
                seen.add(key)
                result.append({subject_field: subject, "axis_ref": logical_axis})
            return result

        checked_program_bindings: list[dict[str, JsonValue]] = []
        used_program_axes: set[tuple[str, int | str]] = set()
        used_physical_axes: set[int] = set()
        physical_axis_by_domain: dict[str, int] = {}
        for index, raw in enumerate(self.program_bindings):
            path = f"pi.program_bindings[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset({"axis", "relation", "logical_axes"}),
            )
            physical_axis = item["axis"]
            if (
                isinstance(physical_axis, bool)
                or not isinstance(physical_axis, int)
                or not 0 <= physical_axis <= 2
            ):
                raise RealizationWitnessError(
                    f"{path}.axis must be an integer launch axis in [0, 2]"
                )
            if physical_axis in used_physical_axes:
                raise RealizationWitnessError(
                    "pi program bindings must bind each physical launch axis once"
                )
            used_physical_axes.add(physical_axis)
            relation = _name(item["relation"], f"{path}.relation")
            if relation not in PROGRAM_BINDING_RELATIONS:
                raise RealizationWitnessError(
                    f"{path}.relation is not a closed program binding relation"
                )
            axes = logical_axes(
                item["logical_axes"],
                path=f"{path}.logical_axes",
                subject_field="domain_ref",
                subject_pattern=r"d[0-9]+",
            )
            if relation == "direct" and len(axes) != 1:
                raise RealizationWitnessError(
                    f"{path}.logical_axes must contain exactly one axis for direct"
                )
            if relation != "direct" and len(axes) < 2:
                raise RealizationWitnessError(
                    f"{path}.logical_axes must contain at least two axes for {relation}"
                )
            keys = {
                (str(axis["domain_ref"]), axis["axis_ref"])
                for axis in axes
            }
            if used_program_axes & keys:
                raise RealizationWitnessError(
                    "pi program bindings must not claim one logical J axis twice"
                )
            for domain_ref, _axis_ref in keys:
                previous_physical_axis = physical_axis_by_domain.setdefault(
                    domain_ref, physical_axis
                )
                if previous_physical_axis != physical_axis:
                    raise RealizationWitnessError(
                        "pi program bindings must map one J domain to one physical "
                        "launch axis"
                    )
            used_program_axes.update(keys)
            checked_program_bindings.append(
                {
                    "axis": physical_axis,
                    "relation": relation,
                    "logical_axes": axes,
                }
            )
        if checked_program_axes and checked_program_bindings:
            raise RealizationWitnessError(
                "pi.program_axes and pi.program_bindings are alternative complete "
                "program-coordinate mappings and are mutually exclusive"
            )

        checked_iteration_bindings: list[dict[str, JsonValue]] = []
        used_iteration_axes: set[tuple[str, int | str]] = set()
        for index, raw in enumerate(self.iteration_bindings):
            path = f"pi.iteration_bindings[{index}]"
            item = _record(
                raw,
                path=path,
                required=frozenset(
                    {"relation", "logical_axes", "tile_slot", "mode_slot"}
                ),
            )
            relation = _name(item["relation"], f"{path}.relation")
            if relation not in ITERATION_BINDING_RELATIONS:
                raise RealizationWitnessError(
                    f"{path}.relation is not a closed iteration relation"
                )
            axes = logical_axes(
                item["logical_axes"],
                path=f"{path}.logical_axes",
                subject_field="subject_ref",
                subject_pattern=r"(?:d|n)[0-9]+",
            )
            keys = {
                (str(axis["subject_ref"]), axis["axis_ref"])
                for axis in axes
            }
            if used_iteration_axes & keys:
                raise RealizationWitnessError(
                    "pi iteration bindings must not claim one logical J axis twice"
                )
            used_iteration_axes.update(keys)
            tile_slot = _name(item["tile_slot"], f"{path}.tile_slot")
            if re.fullmatch(r"c[0-9]+", tile_slot) is None:
                raise RealizationWitnessError(f"{path}.tile_slot must be a C slot c*")
            mode_slot = item["mode_slot"]
            if mode_slot is not None:
                mode_slot = _name(mode_slot, f"{path}.mode_slot")
                if re.fullmatch(r"c[0-9]+", mode_slot) is None:
                    raise RealizationWitnessError(
                        f"{path}.mode_slot must be null or a C slot c*"
                    )
            checked_iteration_bindings.append(
                {
                    "relation": relation,
                    "logical_axes": axes,
                    "tile_slot": tile_slot,
                    "mode_slot": mode_slot,
                }
            )

        def closed_anchor_rows(
            rows: list[dict[str, JsonValue]],
            *,
            field: str,
            relations: tuple[str, ...],
            path_name: str,
        ) -> list[dict[str, JsonValue]]:
            result: list[dict[str, JsonValue]] = []
            seen: set[tuple[str, str]] = set()
            for index, raw in enumerate(rows):
                path = f"pi.{path_name}[{index}]"
                item = _record(
                    raw,
                    path=path,
                    required=frozenset({"relation", field}),
                )
                relation = _name(item["relation"], f"{path}.relation")
                if relation not in relations:
                    raise RealizationWitnessError(
                        f"{path}.relation is not supported by this closed relation family"
                    )
                anchor = _opaque_anchor_ref(item[field], f"{path}.{field}")
                key = (relation, anchor)
                if key in seen:
                    raise RealizationWitnessError(f"{path_name} rows must be unique")
                seen.add(key)
                result.append({"relation": relation, field: anchor})
            return result

        checked_control_bindings = closed_anchor_rows(
            self.control_bindings,
            field="subject_ref",
            relations=CONTROL_BINDING_RELATIONS,
            path_name="control_bindings",
        )
        checked_materialization_bindings = closed_anchor_rows(
            self.materialization_bindings,
            field="value_ref",
            relations=MATERIALIZATION_BINDING_RELATIONS,
            path_name="materialization_bindings",
        )
        checked_access_bindings = closed_anchor_rows(
            self.access_representation_bindings,
            field="access_ref",
            relations=ACCESS_REPRESENTATION_BINDING_RELATIONS,
            path_name="access_representation_bindings",
        )

        self.program_axes = checked_program_axes
        self.program_bindings = checked_program_bindings
        self.iteration_bindings = checked_iteration_bindings
        self.control_bindings = checked_control_bindings
        self.materialization_bindings = checked_materialization_bindings
        self.access_representation_bindings = checked_access_bindings

    @classmethod
    def from_json(cls, value: Any) -> "RealizationWitness":
        if isinstance(value, dict) and "schema" in value:
            data = _record(
                value,
                path="pi",
                required=frozenset({"schema", "program_axes"}),
                allowed=frozenset(
                    {
                        "schema",
                        "program_axes",
                        "program_bindings",
                        "iteration_bindings",
                        "control_bindings",
                        "materialization_bindings",
                        "access_representation_bindings",
                    }
                ),
            )
            schema = _name(data["schema"], "pi.schema")
            if schema != NORMALIZATION_WITNESS_SCHEMA:
                raise RealizationWitnessError(
                    f"unsupported pi subwire schema {schema!r}"
                )
            if not isinstance(data["program_axes"], list):
                raise RealizationWitnessError("pi.program_axes must be an array")
            for key in (
                "program_bindings",
                "iteration_bindings",
                "control_bindings",
                "materialization_bindings",
                "access_representation_bindings",
            ):
                if key in data and not isinstance(data[key], list):
                    raise RealizationWitnessError(f"pi.{key} must be an array")
            return cls(
                wire_schema=schema,
                program_axes=copy.deepcopy(data["program_axes"]),
                program_bindings=copy.deepcopy(data.get("program_bindings", [])),
                iteration_bindings=copy.deepcopy(data.get("iteration_bindings", [])),
                control_bindings=copy.deepcopy(data.get("control_bindings", [])),
                materialization_bindings=copy.deepcopy(
                    data.get("materialization_bindings", [])
                ),
                access_representation_bindings=copy.deepcopy(
                    data.get("access_representation_bindings", [])
                ),
            )
        data = _record(
            value,
            path="pi",
            required=frozenset(
                {
                    "regions",
                    "instances",
                    "routes",
                    "order",
                    "sharing",
                    "refinements",
                }
            ),
            allowed=frozenset(
                {
                    "regions",
                    "instances",
                    "routes",
                    "program_axes",
                    "program_linearizations",
                    "route_extent_transforms",
                    "order",
                    "sharing",
                    "refinements",
                    "iterations",
                    "loop_carriers",
                    "physical_iterations",
                    "physical_loop_carriers",
                    "terminal_controls",
                }
            ),
        )
        for key in (
            "regions",
            "instances",
            "routes",
            "program_axes",
            "program_linearizations",
            "route_extent_transforms",
            "order",
            "sharing",
            "refinements",
            "iterations",
            "loop_carriers",
            "physical_iterations",
            "physical_loop_carriers",
            "terminal_controls",
        ):
            if key in data and not isinstance(data[key], list):
                raise RealizationWitnessError(f"pi.{key} must be an array")
        return cls(
            regions=copy.deepcopy(data["regions"]),
            instances=copy.deepcopy(data["instances"]),
            routes=copy.deepcopy(data["routes"]),
            program_axes=copy.deepcopy(data.get("program_axes", [])),
            program_linearizations=copy.deepcopy(
                data.get("program_linearizations", [])
            ),
            route_extent_transforms=copy.deepcopy(
                data.get("route_extent_transforms", [])
            ),
            order=copy.deepcopy(data["order"]),
            sharing=copy.deepcopy(data["sharing"]),
            refinements=copy.deepcopy(data["refinements"]),
            iterations=copy.deepcopy(data.get("iterations", [])),
            loop_carriers=copy.deepcopy(data.get("loop_carriers", [])),
            physical_iterations=copy.deepcopy(data.get("physical_iterations", [])),
            physical_loop_carriers=copy.deepcopy(
                data.get("physical_loop_carriers", [])
            ),
            terminal_controls=copy.deepcopy(data.get("terminal_controls", [])),
        )

    def to_json(
        self,
        *,
        include_iterations: bool = True,
        include_loop_carriers: bool = True,
        include_physical_time: bool = True,
        include_terminal_controls: bool = True,
        include_route_extent_transforms: bool = True,
        include_program_axes: bool = True,
        include_program_coordinate_maps: bool = True,
    ) -> dict[str, Any]:
        if self.wire_schema == NORMALIZATION_WITNESS_SCHEMA:
            result = {
                "schema": NORMALIZATION_WITNESS_SCHEMA,
                "program_axes": copy.deepcopy(self.program_axes),
            }
            for name in (
                "program_bindings",
                "iteration_bindings",
                "control_bindings",
                "materialization_bindings",
                "access_representation_bindings",
            ):
                rows = getattr(self, name)
                if rows:
                    result[name] = copy.deepcopy(rows)
            return result
        result = {
            "regions": copy.deepcopy(self.regions),
            "instances": copy.deepcopy(self.instances),
            "routes": copy.deepcopy(self.routes),
            "order": copy.deepcopy(self.order),
            "sharing": copy.deepcopy(self.sharing),
            "refinements": copy.deepcopy(self.refinements),
        }
        if include_route_extent_transforms:
            result["route_extent_transforms"] = copy.deepcopy(
                self.route_extent_transforms
            )
        if include_program_axes:
            result["program_axes"] = copy.deepcopy(self.program_axes)
        if include_program_coordinate_maps:
            result["program_linearizations"] = copy.deepcopy(
                self.program_linearizations
            )
        if include_iterations:
            result["iterations"] = copy.deepcopy(self.iterations)
        if include_loop_carriers:
            result["loop_carriers"] = copy.deepcopy(self.loop_carriers)
        if include_physical_time:
            result["physical_iterations"] = copy.deepcopy(self.physical_iterations)
            result["physical_loop_carriers"] = copy.deepcopy(
                self.physical_loop_carriers
            )
        if include_terminal_controls:
            result["terminal_controls"] = copy.deepcopy(self.terminal_controls)
        return result


__all__ = [
    "NORMALIZATION_WITNESS_SCHEMA",
    "PROGRAM_BINDING_RELATIONS",
    "ITERATION_BINDING_RELATIONS",
    "CONTROL_BINDING_RELATIONS",
    "MATERIALIZATION_BINDING_RELATIONS",
    "ACCESS_REPRESENTATION_BINDING_RELATIONS",
    "RealizationWitness",
    "RealizationWitnessError",
    "validate_normalization_bindings_against_j",
    "REALIZATION_MECHANISMS",
    "ITERATION_BASES",
    "PHYSICAL_ITERATION_DIRECTIONS",
    "ROUTE_EXTENT_ROUNDINGS",
    "REPRESENTATION_REFINEMENTS",
    "render_realization_mechanism_grammar",
    "render_realization_witness_wire_grammar",
]
