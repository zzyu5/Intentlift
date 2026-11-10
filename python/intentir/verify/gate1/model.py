"""Strict, JSON-serializable data model for the SCRC Gate1 pipeline.

Gate1 deliberately separates the information owned by the semantic archive
(``J``), realization witness (``pi``), pure schedule (``C``), and harness-only
diagnostics.
The classes in this module are intentionally dependency-free dataclasses.  A
strict parser is used at every persisted boundary: unknown keys, malformed
types, non-finite JSON numbers, and duplicate identifiers are rejected rather
than silently ignored.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Mapping, Sequence, TypeVar

from intentir.ir.intent_op_contract import (
    IntentOpContractError,
    canonicalize_dtype,
    is_non_boolean_integer_dtype,
    is_scalar_literal,
    validate_const_literal,
    validate_intent_op_structure,
)
from intentir.ir.canonical_construction import (
    CANONICAL_ARTIFACT_ROOT_FIELDS,
    CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS,
    CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS,
    CANONICAL_BOUNDARY_WIRE_FIELDS,
    CANONICAL_C_WIRE_FIELDS,
    CANONICAL_J_WIRE_FIELDS,
    CanonicalHandleAllocator,
    SOURCE_OWNER_TOKEN_PATTERN,
    canonical_archive_records,
    canonical_operation_records,
)
from intentir.ir.canonical_j import j_elaboration
from intentir.ir.dimension_relation import (
    DIMENSION_INT64_MAX,
    DIMENSION_RELATION_OPS,
    declared_dimension_handles,
)
from intentir.ir.dim_expr import dimension_expression_references
from intentir.ir.model import (
    IntentFunction,
    ValueType,
)
from intentir.verify.gate1.pi import (
    NORMALIZATION_WITNESS_SCHEMA,
    canonical_program_coordinate_projection,
    RealizationWitness,
    RealizationWitnessError,
)


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


# One declaration for the closed proof-relevant Lift envelope.  Source,
# harness coordinates and provider receipts live outside this object.  Keeping
# the declaration here makes it difficult for a new caller to accidentally
# smuggle a fourth proof object into Replay or a Gate verdict.
# Compatibility names used by existing Gate1 callers.  The wire authority is
# owned by canonical_construction; keep these aliases so old imports do not
# become a second declaration.
GATE1_ARTIFACT_MODEL_FIELDS = CANONICAL_ARTIFACT_ROOT_FIELDS
GATE1_ARTIFACT_PRIVATE_FIELDS: tuple[str, ...] = ()
SEMANTIC_ARCHIVE_WIRE_FIELDS = CANONICAL_J_WIRE_FIELDS
# Fields that appeared in the retired nested-J wire shape.  ``intent_function``
# remains valid under ``j.intent``; these siblings never do.  Keep this list
# diagnostic-only: old payloads must not be auto-migrated into the canonical J.
LEGACY_NESTED_J_FIELDS = frozenset(CANONICAL_J_WIRE_FIELDS) - {"intent"}
# The canonical construction module owns the boundary collection set.  Gate1
# exposes the same name for compatibility with its model/schema callers, but
# it must not maintain a second list of boundary fields.
BOUNDARY_WIRE_FIELDS = CANONICAL_BOUNDARY_WIRE_FIELDS
BOUNDARY_ACCESS_WIRE_FIELDS = CANONICAL_BOUNDARY_ACCESS_WIRE_FIELDS
BOUNDARY_ACCESS_REQUIRED_FIELDS = CANONICAL_BOUNDARY_ACCESS_REQUIRED_FIELDS


def render_gate1_artifact_envelope_grammar() -> str:
    """Render the closed ownership boundary of a model-visible Lift result."""

    return (
        "The response root is one closed object with exactly these fields: "
        f"{', '.join(GATE1_ARTIFACT_MODEL_FIELDS)}. The j object is one closed "
        "object with exactly these fields: "
        f"{', '.join(SEMANTIC_ARCHIVE_WIRE_FIELDS)}. Logical callable facts are "
        "declared only by j.signature. The root c object is the single "
        "implementation-selection object and pi is indexed by j/c. Do not emit "
        "interface, harness, location, source, provenance, binding-map or any "
        "other root sibling. Every J archive field, including boundary, "
        "signature, node_ids, scopes, storage, dimension_constraints, "
        "value_dimension_relations, and dimension_relations, must be nested "
        "directly under root.j. An object that contains root.j.intent together "
        "with root.boundary or root.signature is a malformed flattened legacy "
        "shape. Fields such as root.j.intent.boundary or root.j.intent.scopes "
        "are also retired nested-J fields and are rejected without migration. "
        "root.c and root.pi must be present as siblings of root.j; they "
        "must never be nested under j or j.intent."
    )


def render_boundary_access_wire_grammar() -> str:
    """Render structural access rules and source-aware J ownership together."""

    return (
        "The J boundary is one closed object with exactly these fields: "
        f"{', '.join(BOUNDARY_WIRE_FIELDS)}. Every load, store, or update "
        "access requires an ordered logical index_expression, including scalar "
        "accesses. The bounded ContractExpr elaborator maps k0/[k0] and "
        "k1,k0/[k1,k0] to the same structured axes before any consumer runs. "
        "predicate, "
        "fill_value, and logical domains belong only "
        "to J when present in source semantics. Preserve every source logical "
        "domain, access predicate, and masked-load fill even when the current "
        "bound invocation makes a mask tautological. A fill_value is legal only "
        "on a load with a non-null predicate; an unconditional access has neither "
        "predicate nor fill_value, and consumers reject an orphan fill rather "
        "than silently erasing it. Every access has exactly "
        "one EffectSpec whose kind, target, node_ref (including null), and "
        "access_ref exactly match that access. Do not move these facts to pi, "
        "omit the matching effect, or invent them when the source has none. "
        "boundary.broadcasts is only an implicit operand-broadcast relation for "
        "a map_binary or select node; never attach one to an explicit "
        "broadcast_in_dim or another operation. "
        "For a predicated pure-functional boundary, the observable store "
        "producer carries an explicit result type equal to the J output. A "
        "same-handle inout return needs no output_mapping. Prefer a complete "
        "returns list in output ABI order; output_mappings is sparse and may "
        "also name an output only when its value exactly agrees with returns. "
        "Do not omit a returns entry merely because an output_mapping exists. "
        "consumer_extensions is the sole authority for a physical "
        "outside-domain contribution to one sum/max/min reduce or scan operand. "
        "It references the predicate-bearing access instead of copying or "
        "promoting that predicate into Layer-A SSA. access_guard_relations is "
        "the J-owned relation between one boundary access and one "
        "explicit Layer-A guard value; Lift must author every field and fixed "
        "code never infers or inserts the relation."
    )


class SchemaError(ValueError):
    """Raised when a Gate1 JSON object does not match its strict schema."""

    def __init__(
        self,
        message: str,
        *,
        issues: list["_SchemaIssue"] | tuple["_SchemaIssue", ...] = (),
    ) -> None:
        super().__init__(message)
        self.issues = tuple(issues)

    @classmethod
    def from_issues(
        cls, issues: list["_SchemaIssue"] | tuple["_SchemaIssue", ...]
    ) -> "SchemaError":
        if not issues:
            raise ValueError("SchemaError.from_issues requires at least one issue")
        return cls(issues[0].detail, issues=issues)


class SemanticArchiveCrossContractError(SchemaError):
    """Content-free classification for a strict J cross-contract failure.

    ``SemanticArchive`` still executes every underlying validator unchanged.
    This subtype only preserves which fixed constructor boundary rejected the
    archive, so an ingress receipt can distinguish structural contracts without
    serializing an authored row, handle, value, or validator message.
    """

    _CONTRACTS = frozenset(
        {
            "node_path",
            "dimension",
            "signature",
            "typed_boundary",
            "typed_boundary_operation",
            "typed_boundary_index_bounds",
            "typed_boundary_index_bounds_binding",
            "typed_boundary_index_bounds_coverage",
            "typed_boundary_updates",
            "typed_boundary_result_relations",
            "access_guard",
            "scope_storage",
        }
    )

    def __init__(self, contract: str) -> None:
        if contract not in self._CONTRACTS:
            raise ValueError("unknown SemanticArchive cross-contract")
        self.contract = contract
        super().__init__("SemanticArchive cross-contract rejected")


@dataclass(frozen=True)
class _SchemaIssue:
    """One independently decidable failure from the strict Gate1 decoder."""

    path: str
    detail: str


T = TypeVar("T", bound="StrictJSONMixin")


class StrictJSONMixin:
    """Convenience methods shared by strict Gate1 JSON dataclasses."""

    @classmethod
    def from_json(cls: type[T], value: dict[str, Any]) -> T:  # pragma: no cover - interface
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_json_str(cls: type[T], value: str) -> T:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid JSON: {exc}") from exc
        return cls.from_json(_object(parsed, cls.__name__))

    def to_json_str(self, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_json(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{path} object keys must be strings")
    return value


def _strict_object(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    result = _object(value, path)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise SchemaError(f"{path} has unknown key(s): {', '.join(unknown)}")
    missing = sorted(required - set(result))
    if missing:
        raise SchemaError(f"{path} is missing required key(s): {', '.join(missing)}")
    return result


def _nested_root_owned_field_paths(
    value: Any,
    *,
    path: str,
) -> tuple[str, ...]:
    """Find root-owned ``c``/``pi`` keys anywhere inside one J subtree.

    ``c`` and ``pi`` are envelope siblings of ``j``. Checking only the first
    few known containers is insufficient because a malformed candidate can
    hide either key inside an operation, a conditional arm, a loop carry, or
    an open JSON ``attrs`` object. Keep the traversal here so Gate1, Replay,
    and the typed semantic archive share one ownership boundary.
    """

    found: list[str] = []

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_path = (
                    f"{item_path}.{key}"
                    if isinstance(key, str)
                    else f"{item_path}[{key!r}]"
                )
                if key in {"c", "pi"}:
                    found.append(key_path)
                visit(child, key_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")

    visit(value, path)
    return tuple(found)


def gate1_artifact_wire_diagnostics(
    value: Any,
    *,
    path: str = "Gate1Artifact",
) -> tuple[dict[str, Any], ...]:
    """Return deterministic diagnostics for the root ``J/C/pi`` envelope.

    The JSON Schema and the typed decoder both reject this shape, but they used
    to report different secondary failures for the same malformed response. In
    particular, a provider response with ``j.intent`` plus root-level J fields
    was reported as a long list of missing/unknown fields, hiding the one
    actionable error: the J archive was flattened and root ``c``/``pi`` were
    absent. Keep this check structural only. It never moves fields or repairs a
    candidate; it gives CandidateCheck and Decode one shared ownership diagnosis.
    """

    if not isinstance(value, dict):
        return (
            {
                "code": "artifact_wire_root",
                "path": path,
                "message": "root must be one JSON object",
                "expected_root": list(GATE1_ARTIFACT_MODEL_FIELDS),
                "actual_root": type(value).__name__,
            },
        )

    expected = set(GATE1_ARTIFACT_MODEL_FIELDS)
    actual = set(value)
    # ``json.loads`` always produces string keys, but this helper is also used
    # directly by local callers before ``_object`` runs. Sort by a textual
    # projection so malformed non-string Python keys are diagnosed as a wire
    # error instead of leaking a Python ``TypeError`` from ``sorted``.
    stable_actual_keys = sorted(actual, key=lambda item: str(item))
    diagnostics: list[dict[str, Any]] = []

    j_candidate = value.get("j")
    if isinstance(j_candidate, dict):
        nested_intent = j_candidate.get("intent")
        if isinstance(nested_intent, dict):
            legacy_fields = sorted(
                set(nested_intent).intersection(LEGACY_NESTED_J_FIELDS)
            )
            if legacy_fields:
                diagnostics.append(
                    {
                        "code": "artifact_wire_legacy_nested_j",
                        "path": f"{path}.j.intent.{legacy_fields[0]}",
                        "message": (
                            "legacy nested J fields are not canonical; move no "
                            "fields automatically and emit one root.j archive "
                            "with c/pi as root siblings"
                        ),
                        "legacy_nested_fields": legacy_fields,
                        "actual_root": stable_actual_keys,
                    }
                )
                return tuple(diagnostics)
        nested_root_owned = _nested_root_owned_field_paths(
            j_candidate,
            path=f"{path}.j",
        )
        if nested_root_owned:
            diagnostics.append(
                {
                    "code": "artifact_wire_nested_owner",
                    "path": nested_root_owned[0],
                    "message": (
                        "root-owned c/pi fields must be siblings of root.j; "
                        "they must not be nested under J or j.intent"
                    ),
                    "actual_root": stable_actual_keys,
                }
            )
            return tuple(diagnostics)

    flattened_j_fields = sorted(
        (actual - expected) & set(SEMANTIC_ARCHIVE_WIRE_FIELDS)
    )
    missing_root_fields = sorted(expected - actual)
    if flattened_j_fields:
        diagnostics.append(
            {
                "code": "artifact_wire_flattened_j",
                "path": path,
                "message": (
                    "J archive fields are flattened at the artifact root; move "
                    "them under root.j and provide root.c/root.pi as siblings"
                ),
                "flattened_j_fields": flattened_j_fields,
                "missing_root_fields": missing_root_fields,
                "expected_root": list(GATE1_ARTIFACT_MODEL_FIELDS),
                "actual_root": stable_actual_keys,
            }
        )
        return tuple(diagnostics)

    if actual != expected:
        diagnostics.append(
            {
                "code": "artifact_wire_root",
                "path": path,
                "message": (
                    "artifact root must contain exactly schema, j, c, and pi; "
                    "J archive fields belong under root.j"
                ),
                "missing_root_fields": missing_root_fields,
                "extra_root_fields": sorted(
                    actual - expected, key=lambda item: str(item)
                ),
                "expected_root": list(GATE1_ARTIFACT_MODEL_FIELDS),
                "actual_root": stable_actual_keys,
            }
        )
        return tuple(diagnostics)

    for component in ("j", "c", "pi"):
        if not isinstance(value.get(component), dict):
            diagnostics.append(
                {
                    "code": "artifact_wire_component",
                    "path": f"{path}.{component}",
                    "message": f"root.{component} must be a JSON object",
                    "expected_root": list(GATE1_ARTIFACT_MODEL_FIELDS),
                    "actual_root": stable_actual_keys,
                }
            )
    return tuple(diagnostics)


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{path} must be a string")
    if nonempty and not value:
        raise SchemaError(f"{path} must not be empty")
    return value


def _handle(
    value: Any,
    path: str,
    pattern: str,
    description: str,
) -> str:
    result = _string(value, path)
    if re.fullmatch(pattern, result) is None:
        raise SchemaError(f"{path} must be {description}")
    return result


def _handle_list(
    value: Any,
    path: str,
    pattern: str,
    description: str,
    *,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    result = [
        _handle(item, f"{path}[{index}]", pattern, description)
        for index, item in enumerate(value)
    ]
    if unique and len(result) != len(set(result)):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise SchemaError(f"{path} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    return result


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    return result


def _string_list(value: Any, path: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def _dtype_key(value: str) -> str:
    key = value.removeprefix("torch.").lower()
    aliases = {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float64": "f64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "index": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "ui8": "u8",
        "ui16": "u16",
        "ui32": "u32",
        "bool": "i1",
    }
    return aliases.get(key, key)


def _json_value(value: Any, path: str) -> JsonValue:
    """Validate and defensively copy a proper RFC-8259 JSON value."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SchemaError(f"{path} object keys must be strings")
        return {
            key: _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise SchemaError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(_string(value, path))
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise SchemaError(f"{path} must be one of: {allowed}") from exc


def _unique(items: list[Any], key: str, path: str) -> None:
    values = [getattr(item, key) for item in items]
    if len(values) != len(set(values)):
        raise SchemaError(f"{path} contains duplicate {key} values")


def _strict_intent(value: Any, path: str = "SemanticArchive.intent") -> IntentFunction:
    """Strictly validate IntentFunction, including closed semantic loops.

    Operation-specific ``attrs`` and ``type`` remain extensible while the
    compositional IntentOp core is being audited.  Layer B is never accepted at
    op scope, and destination/writeback contracts never use a per-op ``out``
    shortcut.  Loop state carries cross-region SSA semantics, so SCRC requires
    a closed, typed ``init -> body_argument -> next_value -> result_value``
    contract and checks its positional ``result``/``yield`` witnesses.
    """

    nested_root_owned = _nested_root_owned_field_paths(value, path=path)
    if nested_root_owned:
        raise SchemaError(
            f"{nested_root_owned[0]} is a root-owned field nested under J; "
            "root.c and root.pi must be siblings of root.j"
        )
    root = _object(value, path)
    if "intent_function" in root:
        root = _strict_object(
            root,
            path=path,
            allowed={"intent_function"},
            required={"intent_function"},
        )
        fn = _object(root["intent_function"], f"{path}.intent_function")
    else:
        fn = root
    fn = _strict_object(
        fn,
        path=f"{path}.intent_function",
        allowed={"name", "inputs", "outputs", "returns", "body"},
        required={"name", "inputs", "outputs", "returns", "body"},
    )
    _string(fn["name"], f"{path}.intent_function.name")

    def validate_tensor(item: Any, tensor_path: str) -> None:
        tensor = _strict_object(
            item,
            path=tensor_path,
            allowed={"name", "dtype", "shape"},
            required={"name", "dtype", "shape"},
        )
        tensor_name = _string(tensor["name"], f"{tensor_path}.name")
        if re.fullmatch(r"t[0-9]+", tensor_name) is None:
            raise SchemaError(
                f"{tensor_path}.name must be a logical tensor t* handle; "
                "physical p* parameters belong only to J.signature.parameters"
            )
        try:
            canonicalize_dtype(
                tensor["dtype"], context=f"{tensor_path}.dtype"
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc
        shape = tensor["shape"]
        if not isinstance(shape, list):
            raise SchemaError(f"{tensor_path}.shape must be an array")
        for index, dim in enumerate(shape):
            if isinstance(dim, str):
                _string(dim, f"{tensor_path}.shape[{index}]")
                if re.search(
                    r"(?<![A-Za-z0-9_])p[0-9]+(?![A-Za-z0-9_])", dim
                ):
                    raise SchemaError(
                        f"{tensor_path}.shape[{index}] cannot reference "
                        "physical parameter p* in a logical shape"
                    )
            else:
                _integer(dim, f"{tensor_path}.shape[{index}]", minimum=0)

    tensor_contracts: dict[str, tuple[str, list[int | str]]] = {}
    tensor_collections: dict[str, str] = {}
    for collection in ("inputs", "outputs"):
        tensors = fn.get(collection, [])
        if not isinstance(tensors, list):
            raise SchemaError(f"{path}.intent_function.{collection} must be an array")
        for index, tensor in enumerate(tensors):
            tensor_path = f"{path}.intent_function.{collection}[{index}]"
            validate_tensor(tensor, tensor_path)
            tensor_object = _object(tensor, tensor_path)
            name = _string(tensor_object["name"], f"{tensor_path}.name")
            try:
                tensor_dtype = canonicalize_dtype(
                    tensor_object["dtype"],
                    context=f"{tensor_path}.dtype",
                )
            except (TypeError, ValueError) as exc:
                raise SchemaError(str(exc)) from exc
            contract = (
                tensor_dtype,
                list(tensor_object["shape"]),
            )
            previous = tensor_contracts.get(name)
            if previous is not None:
                if tensor_collections[name] == collection:
                    raise SchemaError(
                        f"{path}.intent_function has duplicate {collection} "
                        f"tensor {name!r}"
                    )
                if previous != contract:
                    raise SchemaError(
                        f"{path}.intent_function inout tensor {name!r} must "
                        "have identical input/output dtype and shape"
                    )
                continue
            tensor_contracts[name] = contract
            tensor_collections[name] = collection

    # Keep the authored tensor portion of the value environment available
    # while validating later operations.  ``tensor_contracts`` only covers
    # the function boundary; derived values such as ``ragged_outer`` are
    # checked as soon as their explicit operation type is encountered.
    authored_value_contracts = dict(tensor_contracts)
    authored_value_types: dict[str, ValueType] = {
        name: ValueType.tensor(dtype, shape)
        for name, (dtype, shape) in tensor_contracts.items()
    }

    semantic_dimensions = {
        reference
        for _dtype, shape in tensor_contracts.values()
        for dim in shape
        for reference in dimension_expression_references(dim)
        if reference != "?"
    }
    declared_node_ids: set[str] = set()
    branch_local_definitions: set[str] = set()

    def is_literal_reference(item: Any) -> bool:
        return is_scalar_literal(item)

    def require_tensor_handle(value: Any, value_path: str) -> str:
        result = _string(value, value_path)
        if re.fullmatch(r"t[0-9]+", result) is None:
            raise SchemaError(
                f"{value_path} must be a logical tensor t* handle; "
                "physical p* parameters cannot be used as J values"
            )
        return result

    def require_value_handle(value: Any, value_path: str) -> str:
        result = _string(value, value_path)
        if re.fullmatch(r"(?:t|v)[0-9]+", result) is None:
            raise SchemaError(
                f"{value_path} must reference a logical t* or SSA v* value; "
                "physical p* parameters cannot be used in J"
            )
        return result

    def require_ssa_handle(value: Any, value_path: str) -> str:
        result = _string(value, value_path)
        if re.fullmatch(r"v[0-9]+", result) is None:
            raise SchemaError(
                f"{value_path} must be an SSA v* handle; logical tensor t* "
                "and physical parameter p* handles are not operation results"
            )
        return result

    def require_dimension_handle(value: Any, value_path: str) -> str:
        result = _string(value, value_path)
        if re.fullmatch(r"x[0-9]+", result) is None:
            raise SchemaError(
                f"{value_path} must be a symbolic dimension x* handle"
            )
        return result

    def result_names(op: dict[str, Any], op_path: str) -> list[str]:
        result = op.get("result")
        if result is None:
            return []
        if isinstance(result, str):
            return [require_ssa_handle(result, f"{op_path}.result")]
        values = _string_list(result, f"{op_path}.result", unique=True)
        return [
            require_ssa_handle(item, f"{op_path}.result[{index}]")
            for index, item in enumerate(values)
        ]

    def validate_value_type(item: Any, type_path: str) -> ValueType:
        if isinstance(item, dict) and item.get("kind") in {"record", "ragged"}:
            try:
                return ValueType.from_json(item, context=type_path)
            except (TypeError, ValueError) as exc:
                raise SchemaError(str(exc)) from exc
        value_type = _strict_object(
            item,
            path=type_path,
            allowed={"dtype", "shape"},
            required={"dtype", "shape"},
        )
        try:
            dtype = canonicalize_dtype(
                value_type["dtype"], context=f"{type_path}.dtype"
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc
        shape = value_type["shape"]
        if not isinstance(shape, list):
            raise SchemaError(f"{type_path}.shape must be an array")
        normalized_shape: list[int | str] = []
        for index, dim in enumerate(shape):
            if isinstance(dim, str):
                normalized_shape.append(_string(dim, f"{type_path}.shape[{index}]"))
            else:
                normalized_shape.append(
                    _integer(dim, f"{type_path}.shape[{index}]", minimum=0)
                )
        return ValueType.tensor(dtype, normalized_shape)

    def validate_record_type(item: Any, type_path: str) -> None:
        if not isinstance(item, dict) or item.get("kind") != "record":
            raise SchemaError(f"{type_path} must be a record value type")
        try:
            ValueType.from_json(item, context=type_path)
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc

    def validate_loop_type(item: Any, type_path: str) -> ValueType:
        return validate_value_type(item, type_path)

    def validate_sequence(
        items: Any,
        sequence_path: str,
        visible: set[str],
        *,
        strict_operands: bool,
    ) -> tuple[set[str], set[str]]:
        if not isinstance(items, list):
            raise SchemaError(f"{sequence_path} must be an array")
        scope_visible = set(visible)
        definitions: set[str] = set()
        for index, child in enumerate(items):
            child_path = f"{sequence_path}[{index}]"
            produced = validate_op(
                child,
                child_path,
                scope_visible,
                strict_operands=strict_operands,
            )
            duplicates = sorted(set(produced) & scope_visible)
            if duplicates:
                raise SchemaError(
                    f"{child_path}.result redefines visible value(s): "
                    + ", ".join(duplicates)
                )
            scope_visible.update(produced)
            definitions.update(produced)
        return scope_visible, definitions

    def validate_op(
        item: Any,
        op_path: str,
        visible: set[str],
        *,
        strict_operands: bool,
    ) -> list[str]:
        try:
            validate_intent_op_structure(item, path=op_path, recursive=False)
        except IntentOpContractError as exc:
            raise SchemaError(str(exc)) from exc
        op = _object(item, op_path)
        op_name = _string(op["op"], f"{op_path}.op")
        if "id" in op:
            node_id = _string(op["id"], f"{op_path}.id")
            if re.fullmatch(r"n[0-9]+", node_id) is None:
                raise SchemaError(
                    f"{op_path}.id must be a stable J node n* handle"
                )
            if node_id in declared_node_ids:
                raise SchemaError(f"{op_path}.id duplicates node id {node_id!r}")
            declared_node_ids.add(node_id)
        produced = result_names(op, op_path)

        def remember_authored_tensor_types() -> None:
            declared = op.get("type")
            if not isinstance(declared, dict):
                return
            if len(produced) != 1:
                return
            value_type = validate_value_type(
                declared, f"{op_path}.type"
            )
            authored_value_types[produced[0]] = value_type
            if value_type.is_tensor:
                authored_value_contracts[produced[0]] = (
                    value_type.dtype,
                    list(value_type.shape),
                )

        if "operands" in op:
            operands = _string_list(op["operands"], f"{op_path}.operands")
            for operand_index, operand in enumerate(operands):
                require_value_handle(
                    operand, f"{op_path}.operands[{operand_index}]"
                )
            if strict_operands:
                unknown_operands = sorted(
                    operand
                    for operand in operands
                    if operand not in visible and not is_literal_reference(operand)
                )
                if unknown_operands:
                    raise SchemaError(
                        f"{op_path}.operands reference value(s) outside the current "
                        f"scope: {', '.join(unknown_operands)}; "
                        "define each value before its first use in the same scope "
                        "or use an existing input/capture, and do not retain a "
                        "stale v* handle from an earlier draft"
                    )
        for open_key in ("attrs", "type"):
            if open_key in op and op[open_key] is not None:
                _json_value(op[open_key], f"{op_path}.{open_key}")
        if (
            isinstance(op.get("type"), dict)
            and op["type"].get("kind") in {"record", "ragged"}
        ):
            validate_value_type(op["type"], f"{op_path}.type")

        if op_name in {"effect_if", "value_if"}:
            predicate = require_value_handle(
                op["predicate"], f"{op_path}.predicate"
            )
            if predicate not in visible:
                raise SchemaError(
                    f"{op_path}.predicate references non-visible value "
                    f"{predicate!r}"
                )
            arm_definitions: dict[str, set[str]] = {}
            # Each arm is a sibling lexical region.  Start both arms from the
            # same parent environment and discard every arm-local type/contract
            # after validation; otherwise visiting positive first leaks its
            # temporary producers into negative (and back into the parent), so
            # type-dependent checks become traversal-order dependent.
            parent_value_types = dict(authored_value_types)
            parent_value_contracts = dict(authored_value_contracts)
            try:
                for polarity in ("positive", "negative"):
                    authored_value_types.clear()
                    authored_value_types.update(parent_value_types)
                    authored_value_contracts.clear()
                    authored_value_contracts.update(parent_value_contracts)
                    arm_path = f"{op_path}.{polarity}"
                    arm = _strict_object(
                        op[polarity],
                        path=arm_path,
                        allowed=(
                            {"scope_id", "body", "yield_value"}
                            if op_name == "value_if"
                            else {"scope_id", "body"}
                        ),
                        required=(
                            {"scope_id", "body", "yield_value"}
                            if op_name == "value_if"
                            else {"scope_id", "body"}
                        ),
                    )
                    _handle(
                        arm["scope_id"],
                        f"{arm_path}.scope_id",
                        r"q[0-9]+",
                        "a q* portable-scope handle",
                    )
                    arm_visible_after, definitions = validate_sequence(
                        arm["body"],
                        f"{arm_path}.body",
                        set(visible),
                        strict_operands=True,
                    )
                    if op_name == "value_if":
                        yield_value = require_value_handle(
                            arm["yield_value"], f"{arm_path}.yield_value"
                        )
                        if yield_value not in arm_visible_after:
                            raise SchemaError(
                                f"{arm_path}.yield_value references value outside "
                                f"the authored arm: {yield_value!r}"
                            )
                    arm_definitions[polarity] = definitions
                    branch_local_definitions.update(definitions)
            finally:
                authored_value_types.clear()
                authored_value_types.update(parent_value_types)
                authored_value_contracts.clear()
                authored_value_contracts.update(parent_value_contracts)
            duplicates = sorted(
                arm_definitions["positive"] & arm_definitions["negative"]
            )
            if duplicates:
                raise SchemaError(
                    f"{op_path} arms define duplicate branch-local value(s): "
                    + ", ".join(duplicates)
                )
            # Arm-local SSA is deliberately not added to the parent-visible
            # set.  A value_if exports exactly its enclosing result; effect_if
            # remains resultless.
            remember_authored_tensor_types()
            return produced if op_name == "value_if" else []

        if op_name != "loop":
            remember_authored_tensor_types()
            return produced

        loop_scope = _handle(
            op["scope_id"],
            f"{op_path}.scope_id",
            r"q[0-9]+",
            "a q* portable-scope handle",
        )

        trip = _object(op["trip"], f"{op_path}.trip")
        dynamic_trip = "axis" not in trip
        if not dynamic_trip:
            trip = _strict_object(
                trip,
                path=f"{op_path}.trip",
                allowed={"axis"},
                required={"axis"},
            )
            trip_axis = require_dimension_handle(
                trip["axis"], f"{op_path}.trip.axis"
            )
            if semantic_dimensions and trip_axis not in semantic_dimensions:
                raise SchemaError(
                    f"{op_path}.trip.axis {trip_axis!r} is not a J tensor dimension"
                )
        else:
            trip = _strict_object(
                trip,
                path=f"{op_path}.trip",
                allowed={"upper_bound", "index"},
                required={"upper_bound", "index"},
            )
            upper_bound = require_value_handle(
                trip["upper_bound"], f"{op_path}.trip.upper_bound"
            )
            trip_index = require_ssa_handle(
                trip["index"], f"{op_path}.trip.index"
            )
            if upper_bound not in visible:
                raise SchemaError(
                    f"{op_path}.trip.upper_bound references non-visible value "
                    f"{upper_bound!r}"
                )
            if trip_index in visible:
                raise SchemaError(
                    f"{op_path}.trip.index must be local to the loop scope"
                )
            # The dynamic-loop ABI is intentionally narrower than MLIR's
            # general integer type family: i1/bool is a predicate, not a
            # meaningful trip or induction carrier.  Check any bound whose
            # tensor type is already visible here; canonical J elaboration
            # repeats the same rule after producer inference for derived
            # values (for example ragged_outer).
            bound_type = authored_value_types.get(upper_bound)
            if bound_type is not None:
                if (
                    not bound_type.is_tensor
                    or list(bound_type.shape) != []
                    or not is_non_boolean_integer_dtype(
                        _dtype_key(bound_type.dtype or "")
                    )
                ):
                    raise SchemaError(
                        f"{op_path}.trip.upper_bound must be a rank-0 "
                        "non-boolean integer tensor"
                    )

        scans = op["scan"]
        if not isinstance(scans, list):
            raise SchemaError(f"{op_path}.scan must be an array")
        scan_names: list[str] = []
        for index, scan_item in enumerate(scans):
            scan_path = f"{op_path}.scan[{index}]"
            scan = _strict_object(
                scan_item,
                path=scan_path,
                allowed={"name", "from", "axis"},
                required={"name", "from", "axis"},
            )
            scan_name = require_ssa_handle(scan["name"], f"{scan_path}.name")
            source = require_value_handle(scan["from"], f"{scan_path}.from")
            axis = _integer(scan["axis"], f"{scan_path}.axis", minimum=0)
            if source not in visible:
                raise SchemaError(f"{scan_path}.from references non-visible value {source!r}")
            if scan_name in visible:
                raise SchemaError(f"{scan_path}.name shadows outer value {scan_name!r}")
            source_type = authored_value_types.get(source)
            if source_type is not None:
                if not source_type.is_tensor:
                    raise SchemaError(
                        f"{scan_path}.from must reference a tensor value"
                    )
                source_shape = list(source_type.shape)
                if axis >= len(source_shape):
                    raise SchemaError(f"{scan_path}.axis is outside source tensor rank")
                if not dynamic_trip and source_shape[axis] != trip_axis:
                    raise SchemaError(
                        f"{scan_path}.axis does not select semantic trip dimension "
                        f"{trip_axis!r}"
                    )
            scan_names.append(scan_name)
        if len(scan_names) != len(set(scan_names)):
            raise SchemaError(f"{op_path}.scan names must be unique")

        carries = op["carry"]
        if not isinstance(carries, list) or not carries:
            raise SchemaError(f"{op_path}.carry must be a non-empty array")
        states: list[dict[str, Any]] = []
        for index, carry_item in enumerate(carries):
            carry_path = f"{op_path}.carry[{index}]"
            carry = _strict_object(
                carry_item,
                path=carry_path,
                allowed={
                    "state_id", "init_value", "body_argument", "next_value",
                    "result_value", "type",
                },
                required={
                    "state_id", "init_value", "body_argument", "next_value",
                    "result_value", "type",
                },
            )
            state_id = _handle(
                carry["state_id"],
                f"{carry_path}.state_id",
                r"s[0-9]+",
                "an s* loop-state handle",
            )
            body_argument = require_ssa_handle(
                carry["body_argument"], f"{carry_path}.body_argument"
            )
            next_value = require_ssa_handle(
                carry["next_value"], f"{carry_path}.next_value"
            )
            result_value = require_ssa_handle(
                carry["result_value"], f"{carry_path}.result_value"
            )
            init_value = _json_value(carry["init_value"], f"{carry_path}.init_value")
            carry_type = validate_loop_type(carry["type"], f"{carry_path}.type")
            if init_value is None:
                raise SchemaError(
                    f"{carry_path}.init_value must be an authored scalar literal "
                    "or a visible value reference"
                )
            if is_literal_reference(init_value):
                if carry_type.kind != "tensor" or carry_type.dtype is None:
                    raise SchemaError(
                        f"{carry_path}.init_value scalar literals require a "
                        "tensor carry type"
                    )
                try:
                    validate_const_literal(
                        {
                            "value": init_value,
                            "dtype": carry_type.dtype,
                            "shape": list(carry_type.shape),
                        },
                        result_type=carry_type.to_json(),
                        context=f"{carry_path}.init_value",
                    )
                except IntentOpContractError as exc:
                    raise SchemaError(str(exc)) from exc
            else:
                if not isinstance(init_value, str):
                    raise SchemaError(
                        f"{carry_path}.init_value must be a scalar literal "
                        "or a visible value reference"
                    )
                if init_value not in visible:
                    raise SchemaError(
                        f"{carry_path}.init_value references non-visible "
                        f"value {init_value!r}"
                    )
            if isinstance(init_value, str) and not is_literal_reference(init_value):
                init_type = authored_value_types.get(init_value)
                if init_type is not None and init_type != carry_type:
                    raise SchemaError(
                        f"{carry_path}.type must match referenced init_value type"
                    )
            if body_argument in visible or body_argument in scan_names:
                raise SchemaError(
                    f"{carry_path}.body_argument must be local to loop scope {loop_scope!r}"
                )
            if result_value == body_argument:
                raise SchemaError(
                    f"{carry_path}.result_value must be distinct from body_argument"
                )
            states.append(
                {
                    "state_id": state_id,
                    "body_argument": body_argument,
                    "next_value": next_value,
                    "result_value": result_value,
                    "type": carry_type,
                }
            )

        for key in ("state_id", "body_argument", "next_value", "result_value"):
            values = [state[key] for state in states]
            if len(values) != len(set(values)):
                raise SchemaError(f"{op_path}.carry {key} values must be unique")

        role_values = [
            value
            for state in states
            for value in (
                state["body_argument"],
                state["next_value"],
                state["result_value"],
            )
        ]
        if len(role_values) != len(set(role_values)):
            raise SchemaError(
                f"{op_path}.carry body_argument/next_value/result_value "
                "handles must be distinct across all states"
            )

        expected_results = [state["result_value"] for state in states]
        expected_yields = [state["next_value"] for state in states]
        actual_yields = _string_list(op["yield"], f"{op_path}.yield", unique=True)
        if produced != expected_results:
            raise SchemaError(
                f"{op_path}.result must exactly match carry.result_value order"
            )
        if actual_yields != expected_yields:
            raise SchemaError(
                f"{op_path}.yield must exactly match carry.next_value order"
            )

        body_visible = set(visible) | set(scan_names) | {
            state["body_argument"] for state in states
        }
        if dynamic_trip:
            if scan_names:
                raise SchemaError(
                    f"{op_path}.scan must be empty for a dynamic value trip"
                )
            if trip_index in scan_names or trip_index in {
                state["body_argument"] for state in states
            }:
                raise SchemaError(
                    f"{op_path}.trip.index must be distinct from scan and carry bindings"
                )
            body_visible.add(trip_index)

        local_non_carry_bindings = set(scan_names)
        if dynamic_trip:
            local_non_carry_bindings.add(trip_index)
        role_export_collisions = sorted(
            {
                name
                for state in states
                for name in (state["next_value"], state["result_value"])
                if name in local_non_carry_bindings
            }
        )
        if role_export_collisions:
            raise SchemaError(
                f"{op_path}.carry next_value/result_value must be distinct "
                "from loop-local scan/induction bindings: "
                + ", ".join(role_export_collisions)
            )

        outer_value_types = dict(authored_value_types)
        outer_value_contracts = dict(authored_value_contracts)
        for scan_name in scan_names:
            source = next(
                scan_item["from"]
                for scan_item in scans
                if scan_item["name"] == scan_name
            )
            source_type = outer_value_types.get(source)
            if source_type is not None:
                authored_value_types[scan_name] = source_type
                if source_type.is_tensor:
                    authored_value_contracts[scan_name] = (
                        source_type.dtype,
                        list(source_type.shape),
                    )
        for state in states:
            authored_value_types[state["body_argument"]] = state["type"]
            if state["type"].is_tensor:
                authored_value_contracts[state["body_argument"]] = (
                    state["type"].dtype,
                    list(state["type"].shape),
                )
        if dynamic_trip:
            upper_type = outer_value_types.get(upper_bound)
            if upper_type is not None:
                authored_value_types[trip_index] = upper_type
                if upper_type.is_tensor:
                    authored_value_contracts[trip_index] = (
                        upper_type.dtype,
                        list(upper_type.shape),
                    )
        try:
            _body_visible_after, body_definitions = validate_sequence(
                op["body"], f"{op_path}.body", body_visible,
                strict_operands=True,
            )
        finally:
            authored_value_types.clear()
            authored_value_types.update(outer_value_types)
            authored_value_contracts.clear()
            authored_value_contracts.update(outer_value_contracts)
        missing_next = sorted(set(expected_yields) - body_definitions)
        if missing_next:
            raise SchemaError(
                f"{op_path}.carry next_value must be defined in the loop body: "
                + ", ".join(missing_next)
            )
        body_result_collisions = sorted(
            set(expected_results) & body_definitions
        )
        if body_result_collisions:
            raise SchemaError(
                f"{op_path}.carry result_value must be owned by the loop, "
                "not defined by its body: "
                + ", ".join(body_result_collisions)
            )

        def produced_type(value_name: str) -> ValueType | None:
            """Return only a directly exported loop-body producer type.

            A value created inside a conditional arm is not visible at the
            loop body boundary.  Searching nested ``body`` arrays here used to
            turn such a branch-local value into a legal loop yield.
            """

            for child_index, child in enumerate(op["body"]):
                child_results = child.get("result")
                names = (
                    [child_results]
                    if isinstance(child_results, str)
                    else list(child_results or [])
                )
                if value_name not in names:
                    continue
                if child.get("op") == "loop":
                    result_index = names.index(value_name)
                    carry_items = child.get("carry", [])
                    if result_index < len(carry_items):
                        return validate_loop_type(
                            carry_items[result_index].get("type"),
                            f"{op_path}.body[{child_index}].carry"
                            f"[{result_index}].type",
                        )
                if "type" not in child or child["type"] is None:
                    return None
                return validate_value_type(
                    child["type"], f"{op_path}.body[{child_index}].type"
                )
            return None

        for state_index, state in enumerate(states):
            next_type = produced_type(state["next_value"])
            if next_type is None:
                raise SchemaError(
                    f"{op_path}.carry next_value {state['next_value']!r} "
                    "requires an explicit producer type"
                )
            if next_type != state["type"]:
                raise SchemaError(
                    f"{op_path}.carry[{state_index}].next_value "
                    f"{state['next_value']!r} direct producer type "
                    f"{next_type.to_json()} differs from declared carry type "
                    f"{state['type'].to_json()}; state type must exactly "
                    "match next/yield type"
                )
        for result_name, state in zip(produced, states, strict=True):
            state_type = state["type"]
            authored_value_types[result_name] = state_type
            if state_type.kind == "tensor":
                authored_value_contracts[result_name] = (
                    state_type.dtype,
                    list(state_type.shape),
                )
        return produced

    body = fn.get("body", [])
    visible_after, _definitions = validate_sequence(
        body,
        f"{path}.intent_function.body",
        {
            _string(tensor["name"], f"{path}.intent_function.inputs[{index}].name")
            for index, tensor in enumerate(fn.get("inputs", []))
        },
        strict_operands=True,
    )
    output_names = [
        require_tensor_handle(
            tensor["name"],
            f"{path}.intent_function.outputs[{index}].name",
        )
        for index, tensor in enumerate(fn.get("outputs", []))
    ]
    escaped_outputs = sorted(set(output_names) & branch_local_definitions)
    if escaped_outputs:
        raise SchemaError(
            "conditional arm-local value(s) cannot implicitly define function "
            "outputs: " + ", ".join(escaped_outputs)
        )
    returns = fn["returns"]
    if not isinstance(returns, list):
        raise SchemaError(f"{path}.intent_function.returns must be an array")
    returned_outputs: list[str] = []
    for index, item in enumerate(returns):
        return_path = f"{path}.intent_function.returns[{index}]"
        binding = _strict_object(
            item,
            path=return_path,
            allowed={"output", "value"},
            required={"output", "value"},
        )
        output = require_tensor_handle(binding["output"], f"{return_path}.output")
        value = require_value_handle(binding["value"], f"{return_path}.value")
        returned_outputs.append(output)
        if value not in visible_after:
            raise SchemaError(
                f"{return_path}.value references unknown Layer-A value {value!r}; "
                "use an existing input or operation result visible after the "
                "body, and make output_mapping (when present) name the same value"
            )
    if returned_outputs != output_names:
        raise SchemaError(
            "IntentFunction.returns must cover outputs once and in ABI order"
        )

    normalized = {"intent_function": copy.deepcopy(fn)}
    return IntentFunction.from_json(normalized)


class Frontend(str, Enum):
    TRITON = "triton"
    CUTE = "cute"
    TILELANG = "tilelang"
    CUDA = "cuda"
    OTHER = "other"


class ParameterKind(str, Enum):
    TENSOR = "tensor"
    POINTER = "pointer"
    SCALAR = "scalar"
    OPAQUE = "opaque"
    CONSTEXPR = "constexpr"


class ParameterMode(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    VALUE = "value"


class ScheduleSlotKind(str, Enum):
    CONSTEXPR = "constexpr"
    COMPILER = "compiler"


class AccessKind(str, Enum):
    LOAD = "load"
    STORE = "store"
    UPDATE = "update"


class AccessGuardPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    UPDATE = "update"


class UpdateCombiner(str, Enum):
    OVERWRITE = "overwrite"
    ADD = "add"


class DuplicatePolicy(str, Enum):
    FORBID = "forbid"
    COMBINE = "combine"
    ONE_OF = "one_of"


class UpdateOrder(str, Enum):
    INPUT_ORDER = "input_order"
    UNORDERED = "unordered"


class ResultRelationKind(str, Enum):
    EXACT = "exact"
    TOLERANCE = "tolerance"
    ONE_OF_UPDATES = "one_of_updates"


class RefinementKind(str, Enum):
    """Closed J/B authorization classes for sensitive realization choices."""

    SIGNED_WIDEN = "signed_widen"
    UNSIGNED_WIDEN = "unsigned_widen"
    NUMERIC_CAST = "numeric_cast"
    ACCUMULATION_DTYPE = "accumulation_dtype"
    REASSOCIATION = "reassociation"
    NATIVE_ATOMIC = "native_atomic"
    PRIVATIZED_UPDATE = "privatized_update"


class AliasKind(str, Enum):
    MUST_ALIAS = "must_alias"
    MAY_ALIAS = "may_alias"
    NO_ALIAS = "no_alias"


class ScopeKind(str, Enum):
    FUNCTION = "function"
    PROGRAM_INSTANCE = "program_instance"
    LOOP_REGION = "loop_region"
    BRANCH_REGION = "branch_region"


class StorageMutability(str, Enum):
    IMMUTABLE = "immutable"
    WRITE_ONCE = "write_once"
    MUTABLE = "mutable"


class SemanticEventKind(str, Enum):
    SCOPE_ENTRY = "scope_entry"
    SCOPE_EXIT = "scope_exit"
    NODE = "node"
    ACCESS = "access"
    STATE = "state"
    EFFECT = "effect"


class ProvenanceRole(str, Enum):
    LIFTER = "lifter"
    REWRITER = "rewriter"
    INJECTOR = "injector"
    COMPILER = "compiler"
    COMPARATOR = "comparator"
    DIAGNOSTICIAN = "diagnostician"


class Gate1Status(str, Enum):
    COMPILE_FAILED = "compile_failed"
    NO_CLOSURE = "no_closure"
    REPLAY_CLOSED = "replay_closed"


class JNecessityStatus(str, Enum):
    PROVED = "proved"
    NOT_PROVED = "not_proved"
    INCONCLUSIVE = "inconclusive"


@dataclass
class BoundaryAccess(StrictJSONMixin):
    access_id: str
    kind: AccessKind
    tensor: str
    predicate: str | None = None
    fill_value: JsonValue = None
    node_ref: str | None = None
    index_expression: str | None = None
    semantic_access_authorization: str | None = None
    semantic_owner_authorization: str | None = None
    semantic_guard_authorization: str | None = None

    _KEYS: ClassVar[set[str]] = set(BOUNDARY_ACCESS_WIRE_FIELDS)

    def __post_init__(self) -> None:
        self.access_id = _handle(
            self.access_id,
            "BoundaryAccess.access_id",
            r"a[0-9]+",
            "an a* access handle",
        )
        self.kind = _enum(AccessKind, self.kind, "BoundaryAccess.kind")
        self.tensor = _handle(
            self.tensor,
            "BoundaryAccess.tensor",
            r"t[0-9]+",
            "a logical tensor t* handle",
        )
        if self.predicate is not None:
            self.predicate = _string(self.predicate, "BoundaryAccess.predicate")
        self.fill_value = _json_value(self.fill_value, "BoundaryAccess.fill_value")
        if self.node_ref is not None:
            self.node_ref = _handle(
                self.node_ref,
                "BoundaryAccess.node_ref",
                r"n[0-9]+",
                "an n* node handle",
            )
        if self.index_expression is not None:
            self.index_expression = _string(
                self.index_expression, "BoundaryAccess.index_expression"
            )
        for value, field_name in (
            (self.semantic_access_authorization, "semantic_access_authorization"),
            (self.semantic_owner_authorization, "semantic_owner_authorization"),
            (self.semantic_guard_authorization, "semantic_guard_authorization"),
        ):
            if value is not None and (
                not isinstance(value, str)
                or re.fullmatch(
                    SOURCE_OWNER_TOKEN_PATTERN
                    if field_name == "semantic_owner_authorization"
                    else r"so_[0-9a-f]{32}", value
                ) is None
            ):
                raise SchemaError(
                    f"BoundaryAccess.{field_name} must be an opaque Source authorization token"
                )
        if self.semantic_owner_authorization is not None and self.node_ref is None:
            raise SchemaError("BoundaryAccess owner authorization requires its exact node_ref")
        if self.kind in {AccessKind.LOAD, AccessKind.STORE, AccessKind.UPDATE}:
            if self.index_expression is None:
                raise SchemaError(
                    f"BoundaryAccess {self.access_id!r} requires index_expression"
                )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BoundaryAccess":
        data = _strict_object(
            value,
            path="BoundaryAccess",
            allowed=cls._KEYS,
            required=set(BOUNDARY_ACCESS_REQUIRED_FIELDS),
        )
        return cls(
            access_id=data["access_id"],
            kind=data["kind"],
            tensor=data["tensor"],
            predicate=data.get("predicate"),
            fill_value=data.get("fill_value"),
            node_ref=data.get("node_ref"),
            index_expression=data.get("index_expression"),
            semantic_access_authorization=data.get("semantic_access_authorization"),
            semantic_owner_authorization=data.get("semantic_owner_authorization"),
            semantic_guard_authorization=data.get("semantic_guard_authorization"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "access_id": self.access_id,
            "kind": self.kind.value,
            "tensor": self.tensor,
        }
        if self.predicate is not None:
            result["predicate"] = self.predicate
        if self.fill_value is not None:
            result["fill_value"] = copy.deepcopy(self.fill_value)
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.index_expression is not None:
            result["index_expression"] = self.index_expression
        if self.semantic_access_authorization is not None:
            result["semantic_access_authorization"] = self.semantic_access_authorization
        if self.semantic_owner_authorization is not None:
            result["semantic_owner_authorization"] = self.semantic_owner_authorization
        if self.semantic_guard_authorization is not None:
            result["semantic_guard_authorization"] = self.semantic_guard_authorization
        return result


@dataclass
class BroadcastSpec(StrictJSONMixin):
    node_ref: str
    operand: str
    axes: list[int | str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_ref = _handle(
            self.node_ref,
            "BroadcastSpec.node_ref",
            r"n[0-9]+",
            "an n* node handle",
        )
        self.operand = _handle(
            self.operand,
            "BroadcastSpec.operand",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        checked: list[int | str] = []
        for index, axis in enumerate(self.axes):
            if isinstance(axis, bool) or not isinstance(axis, (int, str)):
                raise SchemaError(f"BroadcastSpec.axes[{index}] must be an integer or string")
            checked.append(axis)
        self.axes = checked

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "BroadcastSpec":
        data = _strict_object(
            value, path="BroadcastSpec", allowed={"node_ref", "operand", "axes"},
            required={"node_ref", "operand", "axes"},
        )
        if not isinstance(data["axes"], list):
            raise SchemaError("BroadcastSpec.axes must be an array")
        return cls(data["node_ref"], data["operand"], list(data["axes"]))

    def to_json(self) -> dict[str, Any]:
        return {"node_ref": self.node_ref, "operand": self.operand, "axes": list(self.axes)}


@dataclass
class AccessGuardRelation(StrictJSONMixin):
    """One explicit J relation from a boundary access to a Layer-A guard."""

    access_ref: str
    value_ref: str
    axes: list[int]
    polarity: AccessGuardPolarity

    def __post_init__(self) -> None:
        self.access_ref = _handle(
            self.access_ref,
            "AccessGuardRelation.access_ref",
            r"a[0-9]+",
            "an a* access handle",
        )
        self.value_ref = _handle(
            self.value_ref,
            "AccessGuardRelation.value_ref",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        if not isinstance(self.axes, list):
            raise SchemaError("AccessGuardRelation.axes must be an array")
        self.axes = [
            _integer(
                axis,
                f"AccessGuardRelation.axes[{index}]",
                minimum=0,
            )
            for index, axis in enumerate(self.axes)
        ]
        self.polarity = _enum(
            AccessGuardPolarity,
            self.polarity,
            "AccessGuardRelation.polarity",
        )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "AccessGuardRelation":
        data = _strict_object(
            value,
            path="AccessGuardRelation",
            allowed={"access_ref", "value_ref", "axes", "polarity"},
            required={"access_ref", "value_ref", "axes", "polarity"},
        )
        if not isinstance(data["axes"], list):
            raise SchemaError("AccessGuardRelation.axes must be an array")
        return cls(
            access_ref=data["access_ref"],
            value_ref=data["value_ref"],
            axes=list(data["axes"]),
            polarity=data["polarity"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "access_ref": self.access_ref,
            "value_ref": self.value_ref,
            "axes": list(self.axes),
            "polarity": self.polarity.value,
        }


@dataclass
class AliasSpec(StrictJSONMixin):
    left: str
    right: str
    kind: AliasKind = AliasKind.MAY_ALIAS

    def __post_init__(self) -> None:
        self.left = _handle(
            self.left,
            "AliasSpec.left",
            r"(?:t|v|s)[0-9]+",
            "a logical value/storage t*/v*/s* handle",
        )
        self.right = _handle(
            self.right,
            "AliasSpec.right",
            r"(?:t|v|s)[0-9]+",
            "a logical value/storage t*/v*/s* handle",
        )
        if self.left == self.right:
            raise SchemaError(
                "AliasSpec endpoints must name two distinct logical handles"
            )
        self.kind = _enum(AliasKind, self.kind, "AliasSpec.kind")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "AliasSpec":
        data = _strict_object(
            value, path="AliasSpec", allowed={"left", "right", "kind"},
            required={"left", "right"},
        )
        return cls(data["left"], data["right"], data.get("kind", AliasKind.MAY_ALIAS.value))

    def to_json(self) -> dict[str, Any]:
        return {"left": self.left, "right": self.right, "kind": self.kind.value}


@dataclass
class EffectSpec(StrictJSONMixin):
    kind: EffectKind
    target: str
    node_ref: str | None = None
    effect_id: str = ""
    access_ref: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(EffectKind, self.kind, "EffectSpec.kind")
        self.target = _handle(
            self.target,
            "EffectSpec.target",
            r"(?:t|v|s)[0-9]+",
            "a logical value/storage t*/v*/s* handle",
        )
        if self.node_ref is not None:
            self.node_ref = _handle(
                self.node_ref,
                "EffectSpec.node_ref",
                r"n[0-9]+",
                "an n* node handle",
            )
        self.effect_id = _handle(
            self.effect_id,
            "EffectSpec.effect_id",
            r"e[0-9]+",
            "an e* effect handle",
        )
        if self.access_ref is not None:
            self.access_ref = _handle(
                self.access_ref,
                "EffectSpec.access_ref",
                r"a[0-9]+",
                "an a* access handle",
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "EffectSpec":
        data = _strict_object(
            value,
            path="EffectSpec",
            allowed={
                "effect_id", "kind", "target", "node_ref", "access_ref",
            },
            required={"effect_id", "kind", "target"},
        )
        return cls(
            data["kind"], data["target"], data.get("node_ref"),
            data["effect_id"], data.get("access_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result = {
            "effect_id": self.effect_id,
            "kind": self.kind.value,
            "target": self.target,
        }
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        if self.access_ref is not None:
            result["access_ref"] = self.access_ref
        return result


@dataclass
class OutputMapping(StrictJSONMixin):
    output: str
    value: str
    node_ref: str | None = None

    def __post_init__(self) -> None:
        self.output = _handle(
            self.output,
            "OutputMapping.output",
            r"t[0-9]+",
            "a logical tensor t* handle",
        )
        self.value = _handle(
            self.value,
            "OutputMapping.value",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        if self.node_ref is not None:
            self.node_ref = _handle(
                self.node_ref,
                "OutputMapping.node_ref",
                r"n[0-9]+",
                "an n* node handle",
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "OutputMapping":
        data = _strict_object(
            value, path="OutputMapping", allowed={"output", "value", "node_ref"},
            required={"output", "value"},
        )
        return cls(data["output"], data["value"], data.get("node_ref"))

    def to_json(self) -> dict[str, Any]:
        result = {"output": self.output, "value": self.value}
        if self.node_ref is not None:
            result["node_ref"] = self.node_ref
        return result


@dataclass
class ReductionBoundary(StrictJSONMixin):
    """Algebra/local-axis boundary contract for a reduce or inclusive scan."""

    node_ref: str
    tile_axis: int | str
    identity: JsonValue

    def __post_init__(self) -> None:
        self.node_ref = _handle(
            self.node_ref,
            "ReductionBoundary.node_ref",
            r"n[0-9]+",
            "an n* node handle",
        )
        if isinstance(self.tile_axis, bool) or not isinstance(
            self.tile_axis, (int, str)
        ):
            raise SchemaError(
                "ReductionBoundary.tile_axis must be an integer or string"
            )
        if isinstance(self.tile_axis, int) and self.tile_axis < 0:
            raise SchemaError("ReductionBoundary.tile_axis must be non-negative")
        if isinstance(self.tile_axis, str):
            self.tile_axis = _string(self.tile_axis, "ReductionBoundary.tile_axis")
        self.identity = _json_value(self.identity, "ReductionBoundary.identity")
        if self.identity is None:
            raise SchemaError("ReductionBoundary.identity must be explicit")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ReductionBoundary":
        data = _strict_object(
            value,
            path="ReductionBoundary",
            allowed={"node_ref", "tile_axis", "identity"},
            required={"node_ref", "tile_axis", "identity"},
        )
        return cls(
            node_ref=data["node_ref"],
            tile_axis=data["tile_axis"],
            identity=data["identity"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "node_ref": self.node_ref,
            "tile_axis": self.tile_axis,
            "identity": copy.deepcopy(self.identity),
        }


@dataclass
class TypedOutsideValue(StrictJSONMixin):
    """One explicitly typed scalar outside a consumer's logical domain."""

    dtype: str
    value: JsonValue

    def __post_init__(self) -> None:
        try:
            self.dtype = canonicalize_dtype(
                self.dtype,
                context="TypedOutsideValue.dtype",
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc
        self.value = _json_value(self.value, "TypedOutsideValue.value")
        if self.value is None or isinstance(self.value, (list, dict)):
            raise SchemaError("TypedOutsideValue.value must be a JSON scalar")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TypedOutsideValue":
        data = _strict_object(
            value,
            path="TypedOutsideValue",
            allowed={"dtype", "value"},
            required={"dtype", "value"},
        )
        return cls(data["dtype"], data["value"])

    def to_json(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "value": copy.deepcopy(self.value)}


@dataclass
class ConsumerExtension(StrictJSONMixin):
    """Layer-B outside-domain contract for one reduce/scan operand.

    The predicate remains owned by the referenced ``BoundaryAccess``.  This
    relation states only which typed value the named consumer sees outside
    that predicate on one frontend-local physical tile axis.
    """

    consumer_ref: str
    operand_index: int
    access_ref: str
    tile_axis: int
    outside: TypedOutsideValue
    identity_law: str

    def __post_init__(self) -> None:
        self.consumer_ref = _handle(
            self.consumer_ref,
            "ConsumerExtension.consumer_ref",
            r"n[0-9]+",
            "an n* node handle",
        )
        self.operand_index = _integer(
            self.operand_index,
            "ConsumerExtension.operand_index",
            minimum=0,
        )
        self.access_ref = _handle(
            self.access_ref,
            "ConsumerExtension.access_ref",
            r"a[0-9]+",
            "an a* access handle",
        )
        self.tile_axis = _integer(
            self.tile_axis, "ConsumerExtension.tile_axis", minimum=0
        )
        if not isinstance(self.outside, TypedOutsideValue):
            self.outside = TypedOutsideValue.from_json(self.outside)
        self.identity_law = _string(
            self.identity_law, "ConsumerExtension.identity_law"
        )
        if self.identity_law != "two_sided":
            raise SchemaError(
                "ConsumerExtension.identity_law must be 'two_sided'"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ConsumerExtension":
        data = _strict_object(
            value,
            path="ConsumerExtension",
            allowed={
                "consumer_ref",
                "operand_index",
                "access_ref",
                "tile_axis",
                "outside",
                "identity_law",
            },
            required={
                "consumer_ref",
                "operand_index",
                "access_ref",
                "tile_axis",
                "outside",
                "identity_law",
            },
        )
        if not isinstance(data["outside"], dict):
            raise SchemaError("ConsumerExtension.outside must be an object")
        return cls(
            consumer_ref=data["consumer_ref"],
            operand_index=data["operand_index"],
            access_ref=data["access_ref"],
            tile_axis=data["tile_axis"],
            outside=TypedOutsideValue.from_json(data["outside"]),
            identity_law=data["identity_law"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "consumer_ref": self.consumer_ref,
            "operand_index": self.operand_index,
            "access_ref": self.access_ref,
            "tile_axis": self.tile_axis,
            "outside": self.outside.to_json(),
            "identity_law": self.identity_law,
        }


@dataclass
class IndexBoundsSpec(StrictJSONMixin):
    """Typed precondition for one data-dependent index operation.

    Its fixed meaning is ``0 <= indices < dim(data, axis)`` for every active
    index.  There is deliberately no boolean ``proven`` flag or proof text.
    """

    node_ref: str
    data: str
    indices: str
    axis: int

    def __post_init__(self) -> None:
        self.node_ref = _handle(
            self.node_ref,
            "IndexBoundsSpec.node_ref",
            r"n[0-9]+",
            "an n* node handle",
        )
        self.data = _handle(
            self.data,
            "IndexBoundsSpec.data",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        self.indices = _handle(
            self.indices,
            "IndexBoundsSpec.indices",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        self.axis = _integer(self.axis, "IndexBoundsSpec.axis", minimum=0)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "IndexBoundsSpec":
        data = _strict_object(
            value,
            path="IndexBoundsSpec",
            allowed={"node_ref", "data", "indices", "axis"},
            required={"node_ref", "data", "indices", "axis"},
        )
        return cls(data["node_ref"], data["data"], data["indices"], data["axis"])

    def to_json(self) -> dict[str, Any]:
        return {
            "node_ref": self.node_ref,
            "data": self.data,
            "indices": self.indices,
            "axis": self.axis,
        }


@dataclass
class LogicalUpdateSpec(StrictJSONMixin):
    """Source-neutral logical update, including complete compare-exchange roles."""

    node_ref: str
    base: str
    indices: str
    values: str
    axis: int
    combiner: UpdateCombiner
    duplicate_policy: DuplicatePolicy
    order: UpdateOrder
    compare: str | None = None
    observed: str | None = None

    def __post_init__(self) -> None:
        self.node_ref = _handle(
            self.node_ref,
            "LogicalUpdateSpec.node_ref",
            r"n[0-9]+",
            "an n* node handle",
        )
        self.base = _handle(
            self.base,
            "LogicalUpdateSpec.base",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        self.indices = _handle(
            self.indices,
            "LogicalUpdateSpec.indices",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        self.values = _handle(
            self.values,
            "LogicalUpdateSpec.values",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
        )
        if self.compare is not None:
            self.compare = _handle(
                self.compare,
                "LogicalUpdateSpec.compare",
                r"(?:t|v)[0-9]+",
                "a logical value t*/v* handle",
            )
        if self.observed is not None:
            self.observed = _handle(
                self.observed,
                "LogicalUpdateSpec.observed",
                r"(?:t|v)[0-9]+",
                "a logical value t*/v* handle",
            )
        self.axis = _integer(self.axis, "LogicalUpdateSpec.axis", minimum=0)
        self.combiner = _enum(
            UpdateCombiner, self.combiner, "LogicalUpdateSpec.combiner"
        )
        self.duplicate_policy = _enum(
            DuplicatePolicy,
            self.duplicate_policy,
            "LogicalUpdateSpec.duplicate_policy",
        )
        self.order = _enum(UpdateOrder, self.order, "LogicalUpdateSpec.order")
        if (
            self.combiner is UpdateCombiner.OVERWRITE
            and self.duplicate_policy is DuplicatePolicy.COMBINE
        ):
            raise SchemaError(
                "overwrite update cannot use duplicate_policy='combine'"
            )
        if (
            self.combiner is UpdateCombiner.ADD
            and self.duplicate_policy is DuplicatePolicy.ONE_OF
        ):
            raise SchemaError("add update cannot use duplicate_policy='one_of'")
        if (
            self.duplicate_policy is DuplicatePolicy.ONE_OF
            and self.order is not UpdateOrder.UNORDERED
        ):
            raise SchemaError("one_of update requires order='unordered'")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LogicalUpdateSpec":
        data = _strict_object(
            value,
            path="LogicalUpdateSpec",
            allowed={
                "node_ref", "base", "indices", "values", "axis",
                "combiner", "duplicate_policy", "order", "compare", "observed",
            },
            required={
                "node_ref", "base", "indices", "values", "axis",
                "combiner", "duplicate_policy", "order",
            },
        )
        return cls(
            node_ref=data["node_ref"],
            base=data["base"],
            indices=data["indices"],
            values=data["values"],
            axis=data["axis"],
            combiner=data["combiner"],
            duplicate_policy=data["duplicate_policy"],
            order=data["order"],
            compare=data.get("compare"),
            observed=data.get("observed"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_ref": self.node_ref,
            "base": self.base,
            "indices": self.indices,
            "values": self.values,
            "axis": self.axis,
            "combiner": self.combiner.value,
            "duplicate_policy": self.duplicate_policy.value,
            "order": self.order.value,
        }
        if self.compare is not None:
            result["compare"] = self.compare
        if self.observed is not None:
            result["observed"] = self.observed
        return result


@dataclass
class ResultRelationSpec(StrictJSONMixin):
    """J-owned comparison class for one declared logical output.

    Numeric thresholds are deliberately absent: the external Gate2 policy Q
    owns them. J may request a trusted tolerance class, but it cannot choose
    its own passing threshold.
    """

    output: str
    kind: ResultRelationKind
    update_node_ref: str | None = None

    def __post_init__(self) -> None:
        self.output = _handle(
            self.output,
            "ResultRelationSpec.output",
            r"t[0-9]+",
            "a logical tensor t* handle",
        )
        self.kind = _enum(
            ResultRelationKind, self.kind, "ResultRelationSpec.kind"
        )
        if self.kind is ResultRelationKind.ONE_OF_UPDATES:
            if self.update_node_ref is None:
                raise SchemaError(
                    "one_of_updates result relation requires update_node_ref"
                )
            self.update_node_ref = _handle(
                self.update_node_ref,
                "ResultRelationSpec.update_node_ref",
                r"n[0-9]+",
                "an n* node handle",
            )
        else:
            if self.update_node_ref is not None:
                raise SchemaError(
                    f"{self.kind.value} result relation cannot reference an update"
                )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ResultRelationSpec":
        data = _strict_object(
            value,
            path="ResultRelationSpec",
            allowed={"output", "kind", "update_node_ref"},
            required={"output", "kind"},
        )
        return cls(
            output=data["output"],
            kind=data["kind"],
            update_node_ref=data.get("update_node_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "output": self.output,
            "kind": self.kind.value,
        }
        if self.kind is ResultRelationKind.ONE_OF_UPDATES:
            result["update_node_ref"] = self.update_node_ref
        return result


@dataclass
class RealizationAuthorization(StrictJSONMixin):
    """A J/B-owned permission referenced by a π representation refinement."""

    auth_id: str
    kind: RefinementKind
    subject_ref: str
    source_dtype: str | None = None
    target_dtype: str | None = None

    def __post_init__(self) -> None:
        self.auth_id = _string(self.auth_id, "RealizationAuthorization.auth_id")
        if re.fullmatch(r"u[0-9]+", self.auth_id) is None:
            raise SchemaError("realization authorization ids must be u0, u1, ...")
        self.kind = _enum(
            RefinementKind, self.kind, "RealizationAuthorization.kind"
        )
        self.subject_ref = _string(
            self.subject_ref, "RealizationAuthorization.subject_ref"
        )
        if self.source_dtype is not None:
            self.source_dtype = _string(
                self.source_dtype, "RealizationAuthorization.source_dtype"
            )
        if self.target_dtype is not None:
            self.target_dtype = _string(
                self.target_dtype, "RealizationAuthorization.target_dtype"
            )
        typed = {self.source_dtype, self.target_dtype}
        if None in typed and len(typed) > 1:
            raise SchemaError(
                "realization authorization source/target dtype must appear together"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RealizationAuthorization":
        data = _strict_object(
            value,
            path="RealizationAuthorization",
            allowed={
                "auth_id", "kind", "subject_ref", "source_dtype",
                "target_dtype",
            },
            required={"auth_id", "kind", "subject_ref"},
        )
        return cls(
            data["auth_id"],
            data["kind"],
            data["subject_ref"],
            data.get("source_dtype"),
            data.get("target_dtype"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "auth_id": self.auth_id,
            "kind": self.kind.value,
            "subject_ref": self.subject_ref,
        }
        if self.source_dtype is not None:
            result["source_dtype"] = self.source_dtype
            result["target_dtype"] = self.target_dtype
        return result


@dataclass
class Boundary(StrictJSONMixin):
    """Closed Layer-B schema owned by ``J``."""

    domains: dict[str, str] = field(default_factory=dict)
    accesses: list[BoundaryAccess] = field(default_factory=list)
    broadcasts: list[BroadcastSpec] = field(default_factory=list)
    aliases: list[AliasSpec] = field(default_factory=list)
    effects: list[EffectSpec] = field(default_factory=list)
    output_mappings: list[OutputMapping] = field(default_factory=list)
    reductions: list[ReductionBoundary] = field(default_factory=list)
    index_bounds: list[IndexBoundsSpec] = field(default_factory=list)
    updates: list[LogicalUpdateSpec] = field(default_factory=list)
    result_relations: list[ResultRelationSpec] = field(default_factory=list)
    realization_authorizations: list[RealizationAuthorization] = field(
        default_factory=list
    )
    consumer_extensions: list[ConsumerExtension] = field(default_factory=list)
    access_guard_relations: list[AccessGuardRelation] = field(
        default_factory=list
    )

    _KEYS: ClassVar[set[str]] = set(BOUNDARY_WIRE_FIELDS)

    def __post_init__(self) -> None:
        if not isinstance(self.domains, dict):
            raise SchemaError("Boundary.domains must be an object")
        normalized_domains = {
            _string(key, "Boundary.domains key"): _string(
                value, f"Boundary.domains.{key}"
            )
            for key, value in self.domains.items()
        }
        # As with node_ids, JSON object order cannot encode declaration
        # semantics.  Domain handles are already explicit, so retain every
        # authored key/expression pair and canonicalize only their map order.
        self.domains = dict(
            sorted(
                normalized_domains.items(),
                key=lambda item: (
                    int(match.group(1))
                    if (match := re.fullmatch(r"d([0-9]+)", item[0]))
                    else math.inf,
                    item[0],
                ),
            )
        )
        self.accesses = [
            item if isinstance(item, BoundaryAccess) else BoundaryAccess.from_json(item)
            for item in self.accesses
        ]
        self.broadcasts = [
            item if isinstance(item, BroadcastSpec) else BroadcastSpec.from_json(item)
            for item in self.broadcasts
        ]
        self.aliases = [
            item if isinstance(item, AliasSpec) else AliasSpec.from_json(item)
            for item in self.aliases
        ]
        seen_alias_pairs: set[tuple[str, str]] = set()
        for item in self.aliases:
            pair = tuple(sorted((item.left, item.right)))
            if pair in seen_alias_pairs:
                raise SchemaError(
                    "Boundary.aliases may declare each unordered endpoint pair "
                    "only once"
                )
            seen_alias_pairs.add(pair)
        self.effects = [
            item if isinstance(item, EffectSpec) else EffectSpec.from_json(item)
            for item in self.effects
        ]
        # Effects participate in scopes, predicates, source-transfer order,
        # and relift continuity.  ``EffectSpec`` therefore requires an
        # explicit e* identity; it must never allocate one while parsing.
        self.output_mappings = [
            item if isinstance(item, OutputMapping) else OutputMapping.from_json(item)
            for item in self.output_mappings
        ]
        self.reductions = [
            item if isinstance(item, ReductionBoundary) else ReductionBoundary.from_json(item)
            for item in self.reductions
        ]
        self.index_bounds = [
            item if isinstance(item, IndexBoundsSpec) else IndexBoundsSpec.from_json(item)
            for item in self.index_bounds
        ]
        self.updates = [
            item if isinstance(item, LogicalUpdateSpec) else LogicalUpdateSpec.from_json(item)
            for item in self.updates
        ]
        self.result_relations = [
            item if isinstance(item, ResultRelationSpec) else ResultRelationSpec.from_json(item)
            for item in self.result_relations
        ]
        self.realization_authorizations = [
            item
            if isinstance(item, RealizationAuthorization)
            else RealizationAuthorization.from_json(item)
            for item in self.realization_authorizations
        ]
        self.consumer_extensions = [
            item
            if isinstance(item, ConsumerExtension)
            else ConsumerExtension.from_json(item)
            for item in self.consumer_extensions
        ]
        self.access_guard_relations = [
            item
            if isinstance(item, AccessGuardRelation)
            else AccessGuardRelation.from_json(item)
            for item in self.access_guard_relations
        ]
        _unique(self.accesses, "access_id", "Boundary.accesses")
        _unique(self.effects, "effect_id", "Boundary.effects")
        _unique(self.output_mappings, "output", "Boundary.output_mappings")
        access_ids = {item.access_id for item in self.accesses}
        unknown_effect_accesses = sorted(
            item.access_ref
            for item in self.effects
            if item.access_ref is not None and item.access_ref not in access_ids
        )
        if unknown_effect_accesses:
            raise SchemaError(
                "Boundary.effects access_ref references unknown access(es): "
                + ", ".join(unknown_effect_accesses)
            )
        effect_access_refs = [
            item.access_ref for item in self.effects if item.access_ref is not None
        ]
        if len(effect_access_refs) != len(set(effect_access_refs)):
            raise SchemaError("Boundary.effects access_ref values must be unique")
        _unique(self.index_bounds, "node_ref", "Boundary.index_bounds")
        _unique(self.updates, "node_ref", "Boundary.updates")
        _unique(self.result_relations, "output", "Boundary.result_relations")
        _unique(
            self.realization_authorizations,
            "auth_id",
            "Boundary.realization_authorizations",
        )
        extension_keys = [
            (item.consumer_ref, item.operand_index, item.access_ref)
            for item in self.consumer_extensions
        ]
        if len(extension_keys) != len(set(extension_keys)):
            raise SchemaError(
                "Boundary.consumer_extensions consumer/operand/access keys "
                "must be unique"
            )
        unknown_extension_accesses = sorted(
            item.access_ref
            for item in self.consumer_extensions
            if item.access_ref not in access_ids
        )
        if unknown_extension_accesses:
            raise SchemaError(
                "Boundary.consumer_extensions references unknown access(es): "
                + ", ".join(unknown_extension_accesses)
            )
        unknown_guard_accesses = sorted(
            item.access_ref
            for item in self.access_guard_relations
            if item.access_ref not in access_ids
        )
        if unknown_guard_accesses:
            raise SchemaError(
                "Boundary.access_guard_relations references unknown "
                "access(es): " + ", ".join(unknown_guard_accesses)
            )
        if [item.auth_id for item in self.realization_authorizations] != [
            *CanonicalHandleAllocator.canonical_names(
                "u", len(self.realization_authorizations)
            )
        ]:
            raise SchemaError(
                "realization authorization ids must be declaration ordered"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Boundary":
        data = _strict_object(
            value,
            path="Boundary",
            allowed=cls._KEYS,
            required=set(BOUNDARY_WIRE_FIELDS),
        )
        for key in cls._KEYS - {"domains"}:
            if not isinstance(data[key], list):
                raise SchemaError(f"Boundary.{key} must be an array")
        domains = data["domains"]
        if not isinstance(domains, dict):
            raise SchemaError("Boundary.domains must be an object")
        return cls(
            domains=dict(domains),
            accesses=[BoundaryAccess.from_json(item) for item in data["accesses"]],
            broadcasts=[BroadcastSpec.from_json(item) for item in data["broadcasts"]],
            aliases=[AliasSpec.from_json(item) for item in data["aliases"]],
            effects=[EffectSpec.from_json(item) for item in data["effects"]],
            output_mappings=[OutputMapping.from_json(item) for item in data["output_mappings"]],
            reductions=[ReductionBoundary.from_json(item) for item in data["reductions"]],
            index_bounds=[IndexBoundsSpec.from_json(item) for item in data["index_bounds"]],
            updates=[LogicalUpdateSpec.from_json(item) for item in data["updates"]],
            result_relations=[ResultRelationSpec.from_json(item) for item in data["result_relations"]],
            realization_authorizations=[
                RealizationAuthorization.from_json(item)
                for item in data["realization_authorizations"]
            ],
            consumer_extensions=[
                ConsumerExtension.from_json(item)
                for item in data["consumer_extensions"]
            ],
            access_guard_relations=[
                AccessGuardRelation.from_json(item)
                for item in data["access_guard_relations"]
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "domains": dict(self.domains),
            "accesses": [item.to_json() for item in self.accesses],
            "broadcasts": [item.to_json() for item in self.broadcasts],
            "aliases": [item.to_json() for item in self.aliases],
            "effects": [item.to_json() for item in self.effects],
            "output_mappings": [item.to_json() for item in self.output_mappings],
            "reductions": [item.to_json() for item in self.reductions],
            "index_bounds": [item.to_json() for item in self.index_bounds],
            "updates": [item.to_json() for item in self.updates],
            "result_relations": [item.to_json() for item in self.result_relations],
            "realization_authorizations": [
                item.to_json() for item in self.realization_authorizations
            ],
            "consumer_extensions": [
                item.to_json() for item in self.consumer_extensions
            ],
            "access_guard_relations": [
                item.to_json() for item in self.access_guard_relations
            ],
        }


@dataclass
class PortableScope(StrictJSONMixin):
    """Frontend-neutral dominance/capture/effect scope in semantic J.

    The record intentionally contains no Python/Triton AST spelling, source
    line, helper name, or frontend API.  Its only job is to make visibility and
    lifetime boundaries explicit enough for deterministic composition.
    """

    scope_id: str
    kind: ScopeKind
    parent_scope: str | None = None
    captures: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scope_id = _handle(
            self.scope_id,
            "PortableScope.scope_id",
            r"q[0-9]+",
            "a q* portable-scope handle",
        )
        self.kind = _enum(ScopeKind, self.kind, "PortableScope.kind")
        if self.parent_scope is not None:
            self.parent_scope = _handle(
                self.parent_scope,
                "PortableScope.parent_scope",
                r"q[0-9]+",
                "a q* portable-scope handle",
            )
        self.captures = _handle_list(
            self.captures,
            "PortableScope.captures",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
            unique=True,
        )
        self.definitions = _handle_list(
            self.definitions,
            "PortableScope.definitions",
            r"(?:t|v)[0-9]+",
            "a logical value t*/v* handle",
            unique=True,
        )
        self.effects = _handle_list(
            self.effects,
            "PortableScope.effects",
            r"e[0-9]+",
            "an e* effect handle",
            unique=True,
        )
        overlap = sorted(set(self.captures) & set(self.definitions))
        if overlap:
            raise SchemaError(
                "PortableScope captures and definitions overlap: "
                + ", ".join(overlap)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "PortableScope":
        data = _strict_object(
            value,
            path="PortableScope",
            allowed={
                "scope_id", "kind", "parent_scope", "captures",
                "definitions", "effects",
            },
            required={
                "scope_id", "kind", "parent_scope", "captures",
                "definitions", "effects",
            },
        )
        return cls(
            scope_id=data["scope_id"],
            kind=data["kind"],
            parent_scope=data["parent_scope"],
            captures=_string_list(data["captures"], "PortableScope.captures", unique=True),
            definitions=_string_list(
                data["definitions"], "PortableScope.definitions", unique=True
            ),
            effects=_string_list(data["effects"], "PortableScope.effects", unique=True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "kind": self.kind.value,
            "parent_scope": self.parent_scope,
            "captures": list(self.captures),
            "definitions": list(self.definitions),
            "effects": list(self.effects),
        }


@dataclass
class SemanticEventRef(StrictJSONMixin):
    """Namespaced semantic event used by storage lifetime endpoints."""

    kind: SemanticEventKind
    ref: str | None = None

    def __post_init__(self) -> None:
        self.kind = _enum(SemanticEventKind, self.kind, "SemanticEventRef.kind")
        if self.ref is None:
            raise SchemaError(
                f"SemanticEventRef kind {self.kind.value!r} requires ref"
            )
        patterns = {
            SemanticEventKind.SCOPE_ENTRY: (
                r"q[0-9]+", "a q* portable-scope handle"
            ),
            SemanticEventKind.SCOPE_EXIT: (
                r"q[0-9]+", "a q* portable-scope handle"
            ),
            SemanticEventKind.NODE: (r"n[0-9]+", "an n* node handle"),
            SemanticEventKind.ACCESS: (r"a[0-9]+", "an a* access handle"),
            SemanticEventKind.STATE: (r"s[0-9]+", "an s* loop-state handle"),
            SemanticEventKind.EFFECT: (r"e[0-9]+", "an e* effect handle"),
        }
        pattern, description = patterns[self.kind]
        self.ref = _handle(self.ref, "SemanticEventRef.ref", pattern, description)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SemanticEventRef":
        data = _strict_object(
            value,
            path="SemanticEventRef",
            allowed={"kind", "ref"},
            required={"kind", "ref"},
        )
        return cls(kind=data["kind"], ref=data.get("ref"))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "ref": self.ref}


@dataclass
class StorageObject(StrictJSONMixin):
    """One semantic storage identity and all J values that denote it."""

    storage_id: str
    members: list[str]
    scope_id: str
    mutability: StorageMutability
    init_event: SemanticEventRef
    last_use_event: SemanticEventRef
    escapes: bool
    crosses_iteration: bool

    def __post_init__(self) -> None:
        self.storage_id = _handle(
            self.storage_id,
            "StorageObject.storage_id",
            r"(?:b|s)[0-9]+",
            "a b*/s* storage handle",
        )
        self.members = _handle_list(
            self.members,
            "StorageObject.members",
            r"(?:t|v|s)[0-9]+",
            "a logical value/storage t*/v*/s* handle",
            unique=True,
        )
        if not self.members:
            raise SchemaError("StorageObject.members must not be empty")
        self.scope_id = _handle(
            self.scope_id,
            "StorageObject.scope_id",
            r"q[0-9]+",
            "a q* portable-scope handle",
        )
        self.mutability = _enum(
            StorageMutability, self.mutability, "StorageObject.mutability"
        )
        if not isinstance(self.init_event, SemanticEventRef):
            self.init_event = SemanticEventRef.from_json(self.init_event)
        if not isinstance(self.last_use_event, SemanticEventRef):
            self.last_use_event = SemanticEventRef.from_json(self.last_use_event)
        self.escapes = _boolean(self.escapes, "StorageObject.escapes")
        self.crosses_iteration = _boolean(
            self.crosses_iteration, "StorageObject.crosses_iteration"
        )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "StorageObject":
        data = _strict_object(
            value,
            path="StorageObject",
            allowed={
                "storage_id", "members", "scope_id", "mutability",
                "init_event", "last_use_event", "escapes",
                "crosses_iteration",
            },
            required={
                "storage_id", "members", "scope_id", "mutability",
                "init_event", "last_use_event", "escapes",
                "crosses_iteration",
            },
        )
        return cls(
            storage_id=data["storage_id"],
            members=_string_list(data["members"], "StorageObject.members", unique=True),
            scope_id=data["scope_id"],
            mutability=data["mutability"],
            init_event=SemanticEventRef.from_json(data["init_event"]),
            last_use_event=SemanticEventRef.from_json(data["last_use_event"]),
            escapes=data["escapes"],
            crosses_iteration=data["crosses_iteration"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "members": list(self.members),
            "scope_id": self.scope_id,
            "mutability": self.mutability.value,
            "init_event": self.init_event.to_json(),
            "last_use_event": self.last_use_event.to_json(),
            "escapes": self.escapes,
            "crosses_iteration": self.crosses_iteration,
        }


def _validate_typed_boundary_relations(archive: "SemanticArchive") -> None:
    """Classify one strict typed-Boundary validation entry without relaxing it."""

    phase = ["typed_boundary_operation"]
    try:
        _validate_typed_boundary_relations_impl(archive, phase)
    except SemanticArchiveCrossContractError:
        raise
    except SchemaError as exc:
        raise SemanticArchiveCrossContractError(phase[0]) from exc


def _validate_typed_boundary_relations_impl(
    archive: "SemanticArchive",
    phase: list[str],
) -> None:
    """Bind every typed bounds/update/result relation to this exact J graph."""

    construction_records = canonical_archive_records(archive)
    if construction_records is None:
        raise SchemaError(
            "typed boundary relations require canonical operation records"
        )
    # Build the typed operation tree exactly once for this cross-contract
    # check.  ``typed_operation_rows`` deserializes the canonical operation
    # records, so independently calling ``typed_operation_by_ref`` and
    # ``typed_operation_by_path`` produces equal-but-distinct IntentOp
    # instances.  Index-bound coverage is deliberately an identity relation:
    # it must prove that a bound and a gather name the *same* operation in the
    # one final graph.  Constructing the path and node-reference views from
    # one row set preserves that strict relation instead of making every
    # otherwise valid gather appear uncovered.
    typed_rows = construction_records.typed_operation_rows
    if typed_rows is None:
        raise SchemaError(
            "typed boundary relations require canonical typed operations"
        )
    by_path = dict(typed_rows)
    operation_by_path = construction_records.operation_by_path
    if len(by_path) != len(typed_rows) or set(by_path) != set(operation_by_path):
        raise SchemaError(
            "typed boundary relations require canonical operation paths"
        )
    nodes = dict(by_path)
    for node_ref, path in archive.node_ids.items():
        if not isinstance(node_ref, str) or not isinstance(path, str):
            raise SchemaError(
                "typed boundary relations require canonical node references"
            )
        operation = by_path.get(path)
        record = operation_by_path.get(path)
        if (
            operation is None
            or not isinstance(record, Mapping)
            or record.get("id") != node_ref
            or node_ref in nodes
        ):
            raise SchemaError(
                "typed boundary relations require canonical node references"
            )
        nodes[node_ref] = operation

    tensors = {
        item.name: item
        for item in [*archive.intent.inputs, *archive.intent.outputs]
    }

    phase[0] = "typed_boundary_index_bounds_binding"
    bounds_by_node: dict[str, IndexBoundsSpec] = {}
    bounded_operations: set[int] = set()
    for item in archive.boundary.index_bounds:
        op = nodes.get(item.node_ref)
        if op is None:
            raise SchemaError(
                f"IndexBoundsSpec references unknown node {item.node_ref!r}"
            )
        if op.op not in {"gather", "scatter"}:
            raise SchemaError(
                "IndexBoundsSpec must target a gather or scatter node"
            )
        if len(op.operands) < 2:
            raise SchemaError("indexed node does not have data and indices operands")
        if item.data != op.operands[0] or item.indices != op.operands[1]:
            raise SchemaError(
                "IndexBoundsSpec data/indices do not match the target node"
            )
        op_axis = op.attrs.get("axis")
        if isinstance(op_axis, bool) or not isinstance(op_axis, int):
            raise SchemaError("indexed node axis must be a canonical integer")
        if item.axis != op_axis:
            raise SchemaError("IndexBoundsSpec axis does not match the target node")
        tensor = tensors.get(item.data)
        if tensor is not None and item.axis >= len(tensor.shape):
            raise SchemaError("IndexBoundsSpec axis is outside the data rank")
        if id(op) in bounded_operations:
            raise SchemaError(
                "Boundary.index_bounds contains multiple aliases for one node"
            )
        bounded_operations.add(id(op))
        bounds_by_node[item.node_ref] = item

    phase[0] = "typed_boundary_index_bounds_coverage"
    gather_operations = {
        id(op) for op in by_path.values() if op.op == "gather"
    }
    if gather_operations - bounded_operations:
        raise SchemaError(
            "canonical J gather requires exactly one typed IndexBoundsSpec "
            "matching the gather node's data, indices, and axis"
        )

    phase[0] = "typed_boundary_updates"
    updates_by_node: dict[str, LogicalUpdateSpec] = {}
    updated_operations: set[int] = set()
    scatter_refs: set[str] = set()
    for node_ref, op in nodes.items():
        if op.op == "scatter":
            scatter_refs.add(node_ref)
    # A node can be named by both path and opaque id; canonicalize through the
    # relation refs rather than treating those aliases as duplicate nodes.
    scatter_ops = {id(nodes[ref]): nodes[ref] for ref in scatter_refs}

    for item in archive.boundary.updates:
        op = nodes.get(item.node_ref)
        if op is None:
            raise SchemaError(
                f"LogicalUpdateSpec references unknown node {item.node_ref!r}"
            )
        if op.op != "scatter":
            raise SchemaError("LogicalUpdateSpec must target a scatter node")
        if op.attrs.get("mode") != "along_axis":
            raise SchemaError(
                "LogicalUpdateSpec currently requires scatter mode='along_axis'"
            )
        is_cas = op.attrs.get("atomic") == "cas"
        if is_cas:
            if (
                len(op.operands) != 4
                or not isinstance(op.result, list)
                or len(op.result) != 2
                or item.compare != op.operands[2]
                or item.observed != op.result[1]
            ):
                raise SchemaError(
                    "compare-exchange update requires data, indices, compare, "
                    "replacement, updated, and observed bindings"
                )
            expected_operands = [
                item.base,
                item.indices,
                item.compare,
                item.values,
            ]
        else:
            if item.compare is not None or item.observed is not None:
                raise SchemaError(
                    "compare and observed update bindings require atomic='cas'"
                )
            if len(op.operands) != 3:
                raise SchemaError("scatter update requires base, indices, and values")
            expected_operands = [item.base, item.indices, item.values]
        if expected_operands != list(op.operands):
            raise SchemaError(
                "LogicalUpdateSpec operands do not match the target scatter"
            )
        op_axis = op.attrs.get("axis")
        if isinstance(op_axis, bool) or not isinstance(op_axis, int):
            raise SchemaError("scatter axis must be a canonical integer")
        if item.axis != op_axis:
            raise SchemaError("LogicalUpdateSpec axis does not match scatter")
        expected_combiner = {
            "none": UpdateCombiner.OVERWRITE,
            "add": UpdateCombiner.ADD,
        }.get(op.attrs.get("reduce"))
        if expected_combiner is None or item.combiner is not expected_combiner:
            raise SchemaError(
                "LogicalUpdateSpec combiner does not match scatter reduce"
            )
        if id(op) in updated_operations:
            raise SchemaError(
                "Boundary.updates contains multiple aliases for one node"
            )
        updated_operations.add(id(op))
        bound = bounds_by_node.get(item.node_ref)
        if bound is None:
            # Accept a path/id alias for the same operation, but never infer a
            # bound from an unrelated indexed node.
            matching = [
                candidate
                for ref, candidate in bounds_by_node.items()
                if nodes.get(ref) is op
            ]
            if len(matching) != 1:
                raise SchemaError(
                    "LogicalUpdateSpec requires one matching IndexBoundsSpec"
                )
        updates_by_node[item.node_ref] = item

    covered_scatter_ops = {
        id(nodes[item.node_ref]) for item in archive.boundary.updates
    }
    missing_scatter = [
        op for key, op in scatter_ops.items() if key not in covered_scatter_ops
    ]
    if missing_scatter:
        raise SchemaError(
            "canonical J scatter requires a typed LogicalUpdateSpec"
        )

    phase[0] = "typed_boundary_result_relations"
    # Output ABI, sparse mapping, return value, and producer ownership are one
    # canonical record.  Gate1 consumes that record instead of reconstructing
    # return_values/mapping_values independently from the typed archive.
    output_records = construction_records.output_records
    if not output_records:
        raise SchemaError("canonical J output construction record is unavailable")
    output_names = [item.name for item in archive.intent.outputs]
    outputs = set(output_names)
    if [record["output"] for record in output_records] != output_names:
        raise SchemaError(
            "canonical output records must preserve output ABI declaration order"
        )
    observable_values = {
        record["output"]: record["return_value"]
        for record in output_records
    }
    relation_by_output = {
        item.output: item for item in archive.boundary.result_relations
    }
    for item in archive.boundary.result_relations:
        if item.output not in outputs:
            raise SchemaError(
                f"ResultRelationSpec references unknown output {item.output!r}"
            )
        if item.kind is not ResultRelationKind.ONE_OF_UPDATES:
            continue
        update = updates_by_node.get(item.update_node_ref or "")
        update_op = nodes.get(item.update_node_ref or "")
        if update is None:
            # Resolve the alternate path/id spelling for the same update node.
            referenced = nodes.get(item.update_node_ref or "")
            matches = [
                candidate
                for ref, candidate in updates_by_node.items()
                if nodes.get(ref) is referenced
            ]
            if len(matches) == 1:
                update = matches[0]
                update_op = referenced
        if update is None or update_op is None:
            raise SchemaError(
                "one_of_updates relation references an unknown logical update"
            )
        if not (
            update.combiner is UpdateCombiner.OVERWRITE
            and update.duplicate_policy is DuplicatePolicy.ONE_OF
        ):
            raise SchemaError(
                "one_of_updates relation requires overwrite + one_of update"
            )
        if not isinstance(update_op.result, str) or (
            observable_values[item.output] != update_op.result
        ):
            raise SchemaError(
                "one_of_updates currently requires the update result to be "
                "returned directly"
            )

    if archive.boundary.updates:
        missing_relations = sorted(outputs - set(relation_by_output))
        if missing_relations:
            raise SchemaError(
                "canonical J with logical updates requires an explicit result "
                "relation for every output: " + ", ".join(missing_relations)
            )
        for item in archive.boundary.updates:
            op = nodes[item.node_ref]
            updated_result = (
                op.result[0]
                if op.attrs.get("atomic") == "cas"
                and isinstance(op.result, list)
                else op.result
            )
            direct_outputs = {
                output
                for output, value in observable_values.items()
                if value == updated_result
            }
            if item.duplicate_policy is DuplicatePolicy.ONE_OF:
                matching = [
                    relation_by_output[output]
                    for output in direct_outputs
                    if relation_by_output[output].kind
                    is ResultRelationKind.ONE_OF_UPDATES
                    and nodes.get(relation_by_output[output].update_node_ref or "")
                    is op
                ]
                if len(matching) != 1:
                    raise SchemaError(
                        "one_of logical update requires exactly one matching "
                        "one_of_updates output relation"
                    )
            elif any(
                relation.kind is ResultRelationKind.ONE_OF_UPDATES
                and nodes.get(relation.update_node_ref or "") is op
                for relation in archive.boundary.result_relations
            ):
                raise SchemaError(
                    "deterministic/combine update cannot use one_of_updates relation"
                )


def _validate_semantic_scopes_and_storage(archive: "SemanticArchive") -> None:
    """Cross-check explicit portable scopes and semantic storage against J."""

    if not archive.scopes:
        raise SchemaError(
            "SemanticArchive.scopes must contain the function scope"
        )

    scope_by_id = {scope.scope_id: scope for scope in archive.scopes}
    _unique(archive.scopes, "scope_id", "SemanticArchive.scopes")
    roots = [
        scope
        for scope in archive.scopes
        if scope.kind is ScopeKind.FUNCTION and scope.parent_scope is None
    ]
    if len(roots) != 1:
        raise SchemaError(
            "SemanticArchive.scopes must contain exactly one parentless function scope"
        )
    root = roots[0]
    for scope in archive.scopes:
        if scope is root:
            continue
        if scope.parent_scope is None:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} requires parent_scope"
            )
        if scope.parent_scope not in scope_by_id:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} references unknown parent_scope"
            )

    for scope in archive.scopes:
        seen: set[str] = set()
        current: PortableScope | None = scope
        while current is not None:
            if current.scope_id in seen:
                raise SchemaError("SemanticArchive.scopes contains a parent cycle")
            seen.add(current.scope_id)
            current = (
                scope_by_id[current.parent_scope]
                if current.parent_scope is not None
                else None
            )
        if root.scope_id not in seen:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} is disconnected from function root"
            )

    all_program_scopes = [
        scope
        for scope in archive.scopes
        if scope.kind is ScopeKind.PROGRAM_INSTANCE
    ]
    nested_program_scopes = [
        scope
        for scope in all_program_scopes
        if scope.parent_scope != root.scope_id
    ]
    if nested_program_scopes:
        raise SchemaError(
            "program_instance scope must be the unique direct child of the "
            "function scope"
        )
    program_scopes = [
        scope
        for scope in all_program_scopes
        if scope.parent_scope == root.scope_id
    ]
    if len(program_scopes) > 1:
        raise SchemaError(
            "SemanticArchive supports at most one top-level program_instance scope"
        )
    execution_scope = program_scopes[0] if program_scopes else root
    construction_records = canonical_archive_records(archive)
    if construction_records is None:
        raise SchemaError(
            "SemanticArchive has no complete canonical construction record"
        )
    operation_record_by_path = construction_records.operation_by_path
    typed_rows = construction_records.typed_operation_rows
    if typed_rows is None:
        raise SchemaError(
            "canonical operation records cannot materialize typed operations"
        )
    typed_operations = dict(typed_rows)
    if set(operation_record_by_path) != set(typed_operations):
        raise SchemaError(
            "canonical operation record paths disagree with typed IntentOp paths"
        )
    for path, operation in typed_operations.items():
        record = operation_record_by_path[path]
        typed_wire = operation._structural_payload()
        typed_results = (
            (operation.result,)
            if isinstance(operation.result, str)
            else tuple(operation.result)
            if isinstance(operation.result, list)
            else ()
        )
        if (
            record["id"] != typed_wire.get("id")
            or record["op"] != operation.op
            or tuple(record["operands"]) != tuple(operation.operands)
            or tuple(record["results"]) != typed_results
            or record["type"] != typed_wire.get("type")
            or record["attrs"] != typed_wire.get("attrs")
        ):
            raise SchemaError(
                f"canonical operation record disagrees with typed operation {path!r}"
            )
    shared_scope_records = construction_records.scope_records
    shared_scope_by_id = {
        record["scope_id"]: record for record in shared_scope_records
    }
    reverse_node_ids = {path: node_id for node_id, path in archive.node_ids.items()}
    path_scope: dict[str, str] = {}
    event_order: dict[tuple[SemanticEventKind, str | None], int] = {
        (SemanticEventKind.SCOPE_ENTRY, root.scope_id): -1,
    }
    event_scope: dict[tuple[SemanticEventKind, str | None], str] = {
        (SemanticEventKind.SCOPE_ENTRY, root.scope_id): root.scope_id,
    }
    if execution_scope is not root:
        event_order[(SemanticEventKind.SCOPE_ENTRY, execution_scope.scope_id)] = -1
        event_scope[
            (SemanticEventKind.SCOPE_ENTRY, execution_scope.scope_id)
        ] = execution_scope.scope_id
    value_events: dict[str, set[tuple[SemanticEventKind, str | None]]] = {}
    valid_values = {tensor.name for tensor in archive.intent.inputs}
    valid_values.update(tensor.name for tensor in archive.intent.outputs)
    state_records: list[tuple[Any, str, str]] = []
    referenced_loop_scopes: set[str] = set()
    referenced_branch_scopes: set[str] = set()
    order = 0

    # The construction table is already in canonical lexical preorder.  Use
    # it for both operation events and region ownership; recursively walking
    # archive.intent.body here would create a second path/SSA interpretation.
    operation_orders: dict[str, int] = {}
    region_entries: dict[str, int] = {}
    for path, op in typed_rows:
        record = operation_record_by_path.get(path)
        if record is None:
            raise SchemaError(
                f"canonical operation record is missing for typed path {path!r}"
            )
        owner_scope = record.get("region_scope_id") or execution_scope.scope_id
        if owner_scope not in scope_by_id:
            raise SchemaError(
                f"canonical operation {path!r} references unknown scope "
                f"{owner_scope!r}"
            )
        path_scope[path] = owner_scope
        path_event = (SemanticEventKind.NODE, path)
        operation_orders[path] = order
        event_order[path_event] = order
        event_scope[path_event] = owner_scope
        node_id = record.get("id")
        authored_node_id = reverse_node_ids.get(path)
        if authored_node_id != node_id:
            raise SchemaError(
                f"canonical operation node id disagrees at {path!r}"
            )
        node_event = (SemanticEventKind.NODE, node_id)
        event_order[node_event] = order
        event_scope[node_event] = owner_scope
        order += 1

        produced = list(record["results"])
        valid_values.update(produced)
        event_ref = node_event
        trip_values = (
            [op.trip.upper_bound]
            if op.op == "loop"
            and op.trip is not None
            and op.trip.is_dynamic
            else []
        )
        for value in [*produced, *op.operands, *trip_values]:
            value_events.setdefault(value, set()).add(event_ref)

        if op.op in {"effect_if", "value_if"}:
            if op.predicate is None or op.positive is None or op.negative is None:
                raise SchemaError(f"conditional {path!r} lacks complete arms")
            value_events.setdefault(op.predicate, set()).add(event_ref)
            arms = (("positive", op.positive), ("negative", op.negative))
            arm_scopes = [arm.scope_id for _polarity, arm in arms]
            if len(set(arm_scopes)) != 2:
                raise SchemaError(
                    f"conditional {path!r} positive/negative scopes must be distinct"
                )
            for polarity, arm in arms:
                branch_scope_id = arm.scope_id
                branch_scope = scope_by_id.get(branch_scope_id)
                if branch_scope is None:
                    raise SchemaError(
                        f"conditional {path!r} {polarity} arm references "
                        "unknown portable scope"
                    )
                if branch_scope_id in referenced_branch_scopes:
                    raise SchemaError(
                        f"portable branch scope {branch_scope_id!r} is "
                        "referenced more than once"
                    )
                referenced_branch_scopes.add(branch_scope_id)
                if branch_scope.kind is not ScopeKind.BRANCH_REGION:
                    raise SchemaError(
                        f"conditional {path!r} {polarity} scope must have "
                        "kind branch_region"
                    )
                if branch_scope.parent_scope != owner_scope:
                    raise SchemaError(
                        f"conditional {path!r} arm scopes must share parent "
                        f"{owner_scope!r}"
                    )
                region_entries[branch_scope_id] = operation_orders[path]
                branch_entry = (SemanticEventKind.SCOPE_ENTRY, branch_scope_id)
                event_order[branch_entry] = operation_orders[path]
                event_scope[branch_entry] = branch_scope_id
                if op.op == "value_if":
                    value_events.setdefault(arm.yield_value, set()).add(event_ref)
            continue

        if op.op != "loop":
            if op.scope_id is not None:
                raise SchemaError(
                    f"non-loop operation {path!r} must not declare scope_id"
                )
            continue

        loop_scope_id = op.scope_id
        loop_scope = scope_by_id.get(loop_scope_id)
        if loop_scope is None:
            raise SchemaError(
                f"semantic loop {path!r} references unknown portable scope"
            )
        if loop_scope_id in referenced_loop_scopes:
            raise SchemaError(
                f"portable loop scope {loop_scope_id!r} is referenced more than once"
            )
        referenced_loop_scopes.add(loop_scope_id)
        if loop_scope.kind is not ScopeKind.LOOP_REGION:
            raise SchemaError(
                f"semantic loop {path!r} scope must have kind loop_region"
            )
        if loop_scope.parent_scope != owner_scope:
            raise SchemaError(
                f"semantic loop {path!r} scope parent must be {owner_scope!r}"
            )
        region_entries[loop_scope_id] = operation_orders[path]
        loop_entry = (SemanticEventKind.SCOPE_ENTRY, loop_scope_id)
        event_order[loop_entry] = operation_orders[path]
        event_scope[loop_entry] = loop_scope_id
        loop_definitions = {scan.name for scan in op.scan}
        loop_definitions.update(state.body_argument for state in op.carry)
        if op.trip is not None and op.trip.is_dynamic:
            loop_definitions.add(op.trip.index or "")
        valid_values.update(loop_definitions)
        for scan in op.scan:
            value_events.setdefault(scan.source, set()).add(event_ref)
        for state in op.carry:
            valid_values.add(state.state_id)
            state_records.append((state, loop_scope_id, node_id))
            state_event = (SemanticEventKind.STATE, state.state_id)
            event_order[state_event] = operation_orders[path]
            event_scope[state_event] = loop_scope_id
            for value in (
                state.state_id, state.body_argument, state.next_value,
                state.result_value,
            ):
                value_events.setdefault(value, set()).add(state_event)
            if (
                isinstance(state.init_value, str)
                and not _looks_like_semantic_literal(state.init_value)
            ):
                value_events.setdefault(state.init_value, set()).add(state_event)

    # Region exits occur after the last operation in that region or any of its
    # descendants.  This preserves the old event ordering while deriving the
    # boundaries from canonical region ownership rather than a second walk.
    for scope_id, entry_order in region_entries.items():
        scope_ids = {scope_id}
        changed = True
        while changed:
            changed = False
            for candidate in scope_by_id.values():
                if candidate.parent_scope in scope_ids and candidate.scope_id not in scope_ids:
                    scope_ids.add(candidate.scope_id)
                    changed = True
        operation_end = max(
            (
                operation_orders[path]
                for path, owner in path_scope.items()
                if owner in scope_ids
            ),
            default=entry_order,
        )
        exit_event = (SemanticEventKind.SCOPE_EXIT, scope_id)
        event_order[exit_event] = operation_end + 1
        event_scope[exit_event] = scope_id
    declared_loop_scopes = {
        scope.scope_id
        for scope in archive.scopes
        if scope.kind is ScopeKind.LOOP_REGION
    }
    if referenced_loop_scopes != declared_loop_scopes:
        raise SchemaError(
            "portable loop scopes must be in one-to-one correspondence with J loops"
        )
    declared_branch_scopes = {
        scope.scope_id
        for scope in archive.scopes
        if scope.kind is ScopeKind.BRANCH_REGION
    }
    if referenced_branch_scopes != declared_branch_scopes:
        raise SchemaError(
            "branch_region scopes require a first-class IntentIR branch-region "
            "op; portable branch scopes must be in one-to-one correspondence "
            "with effect_if/value_if arms"
        )
    event_order[(SemanticEventKind.SCOPE_EXIT, execution_scope.scope_id)] = order + 1
    event_scope[
        (SemanticEventKind.SCOPE_EXIT, execution_scope.scope_id)
    ] = execution_scope.scope_id
    if execution_scope is not root:
        event_order[(SemanticEventKind.SCOPE_EXIT, root.scope_id)] = order + 2
        event_scope[(SemanticEventKind.SCOPE_EXIT, root.scope_id)] = root.scope_id
    valid_effect_refs: set[str] = set()
    declared_effect_owner: dict[str, str] = {}
    for scope in archive.scopes:
        for effect_ref in scope.effects:
            previous = declared_effect_owner.get(effect_ref)
            if previous is not None and previous != scope.scope_id:
                raise SchemaError(
                    f"effect {effect_ref!r} belongs to multiple portable scopes"
                )
            declared_effect_owner[effect_ref] = scope.scope_id
    for effect in archive.boundary.effects:
        assert effect.effect_id is not None
        token = effect.effect_id
        valid_effect_refs.add(token)
        if effect.node_ref is None:
            # A node-less effect is owned by exactly the authored scope that
            # lists it.  This is required for direct effects in an otherwise
            # empty branch arm and never infers an arm from access spelling.
            owner = declared_effect_owner.get(token, execution_scope.scope_id)
            effect_order = event_order.get(
                (SemanticEventKind.SCOPE_ENTRY, owner), order
            )
        else:
            effect_path = archive.node_ids.get(effect.node_ref, effect.node_ref)
            owner = path_scope.get(effect_path, execution_scope.scope_id)
            effect_order = event_order.get(
                (SemanticEventKind.NODE, effect.node_ref),
                event_order.get((SemanticEventKind.NODE, effect_path), order),
            )
        effect_event = (SemanticEventKind.EFFECT, token)
        event_order[effect_event] = effect_order
        event_scope[effect_event] = owner
        value_events.setdefault(effect.target, set()).add(effect_event)

    access_effect_kind = {
        AccessKind.LOAD: EffectKind.READ,
        AccessKind.STORE: EffectKind.WRITE,
        AccessKind.UPDATE: EffectKind.UPDATE,
    }
    for access in archive.boundary.accesses:
        matches = [
            effect
            for effect in archive.boundary.effects
            if effect.kind is access_effect_kind[access.kind]
            and effect.target == access.tensor
            and effect.node_ref == access.node_ref
            and effect.access_ref == access.access_id
        ]
        if len(matches) != 1:
            raise SchemaError(
                f"boundary access {access.access_id!r} requires exactly one "
                "matching EffectSpec"
            )
        matching_effect = matches[0]
        assert matching_effect.effect_id is not None
        effect_event = (
            SemanticEventKind.EFFECT,
            matching_effect.effect_id,
        )
        owner = event_scope[effect_event]
        access_order = event_order[effect_event]
        access_event = (SemanticEventKind.ACCESS, access.access_id)
        event_order[access_event] = access_order
        event_scope[access_event] = owner
        value_events.setdefault(access.tensor, set()).add(access_event)

    defined_owner: dict[str, str] = {}
    all_declared_values = set(valid_values)
    for scope in archive.scopes:
        shared_scope = shared_scope_by_id[scope.scope_id]
        expected_defs = set(shared_scope["definitions"])
        declared_defs = set(scope.definitions)
        missing_defs = sorted(expected_defs - declared_defs)
        extra_defs = sorted(declared_defs - expected_defs)
        if missing_defs or extra_defs:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} definitions must exactly "
                f"match derived J values; missing={missing_defs}, extra={extra_defs}; "
                "for a function scope list every logical input t* and every "
                "top-level operation result exactly once, not output names unless "
                "they are actual inputs or results"
            )
        unknown_defs = sorted(declared_defs - all_declared_values)
        if unknown_defs:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown definition(s): "
                + ", ".join(unknown_defs)
            )
        for definition in scope.definitions:
            previous = defined_owner.get(definition)
            if previous is not None and previous != scope.scope_id:
                raise SchemaError(
                    f"value {definition!r} is defined in multiple portable scopes"
                )
            defined_owner[definition] = scope.scope_id
        expected_captures = set(shared_scope["captures"])
        declared_captures = set(scope.captures)
        missing_captures = sorted(expected_captures - declared_captures)
        extra_captures = sorted(declared_captures - expected_captures)
        if missing_captures or extra_captures:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} captures must exactly "
                f"match derived outer uses; missing={missing_captures}, "
                f"extra={extra_captures}"
            )
        unknown_captures = sorted(declared_captures - all_declared_values)
        if unknown_captures:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown capture(s): "
                + ", ".join(unknown_captures)
            )
        expected_effects = set(shared_scope["effects"])
        declared_effects = set(scope.effects)
        missing_effects = sorted(expected_effects - declared_effects)
        extra_effects = sorted(declared_effects - expected_effects)
        if missing_effects or extra_effects:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} effects must exactly "
                f"match Boundary effects; missing={missing_effects}, "
                f"extra={extra_effects}"
            )
        if list(scope.effects) != list(shared_scope["effects"]):
            raise SchemaError(
                f"portable scope {scope.scope_id!r} effects must preserve the canonical Boundary effect order"
            )
        unknown_effects = sorted(declared_effects - valid_effect_refs)
        if unknown_effects:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} has unknown effect(s): "
                + ", ".join(unknown_effects)
            )

    for scope in archive.scopes:
        ancestor = scope.parent_scope
        ancestor_visible: set[str] = set()
        while ancestor is not None:
            parent = scope_by_id[ancestor]
            ancestor_visible.update(parent.definitions)
            ancestor_visible.update(parent.captures)
            ancestor = parent.parent_scope
        invisible = sorted(set(scope.captures) - ancestor_visible)
        if invisible:
            raise SchemaError(
                f"portable scope {scope.scope_id!r} captures value(s) not visible "
                f"from an ancestor: {', '.join(invisible)}"
            )

    _unique(archive.storage, "storage_id", "SemanticArchive.storage")
    member_owner: dict[str, StorageObject] = {}
    for item in archive.storage:
        for member in item.members:
            previous = member_owner.get(member)
            if previous is not None and previous is not item:
                raise SchemaError(
                    f"semantic storage member {member!r} belongs to multiple "
                    "StorageObject identities"
                )
            member_owner[member] = item

    node_refs = set(archive.node_ids) | set(archive.node_ids.values())
    access_refs = {access.access_id for access in archive.boundary.accesses}
    state_refs = {state.state_id for state, _scope, _event in state_records}
    effect_refs = {
        effect.effect_id
        for effect in archive.boundary.effects
        if effect.effect_id is not None
    }
    event_domains = {
        SemanticEventKind.SCOPE_ENTRY: set(scope_by_id),
        SemanticEventKind.SCOPE_EXIT: set(scope_by_id),
        SemanticEventKind.NODE: node_refs,
        SemanticEventKind.ACCESS: access_refs,
        SemanticEventKind.STATE: state_refs,
        SemanticEventKind.EFFECT: effect_refs,
    }

    def event_key(event: SemanticEventRef, storage_id: str) -> tuple[SemanticEventKind, str | None]:
        if event.ref not in event_domains[event.kind]:
            raise SchemaError(
                f"storage {storage_id!r} {event.kind.value} event references "
                f"unknown ref {event.ref!r}"
            )
        key = (event.kind, event.ref)
        if key not in event_order:
            raise SchemaError(
                f"storage {storage_id!r} event has no deterministic order"
            )
        return key

    def is_descendant_or_same(scope_id: str, ancestor_id: str) -> bool:
        current: str | None = scope_id
        while current is not None:
            if current == ancestor_id:
                return True
            current = scope_by_id[current].parent_scope
        return False

    for item in archive.storage:
        if item.scope_id not in scope_by_id:
            raise SchemaError(
                f"storage {item.storage_id!r} references unknown scope_id"
            )
        unknown_members = sorted(set(item.members) - all_declared_values)
        if unknown_members:
            raise SchemaError(
                f"storage {item.storage_id!r} has unknown member(s): "
                + ", ".join(unknown_members)
            )
        init_key = event_key(item.init_event, item.storage_id)
        last_key = event_key(item.last_use_event, item.storage_id)
        init_order = event_order[init_key]
        last_order = event_order[last_key]
        if last_order < init_order:
            raise SchemaError(
                f"storage {item.storage_id!r} last_use_event precedes init_event"
            )
        scope = scope_by_id[item.scope_id]
        init_owner = event_scope.get(init_key)
        last_owner = event_scope.get(last_key)
        if init_owner is not None and not (
            is_descendant_or_same(init_owner, item.scope_id)
            or is_descendant_or_same(item.scope_id, init_owner)
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} init_event is outside its scope lineage"
            )
        if last_owner is not None and not is_descendant_or_same(
            last_owner, item.scope_id
        ):
            if not (
                item.escapes
                and is_descendant_or_same(item.scope_id, last_owner)
            ):
                raise SchemaError(
                    f"storage {item.storage_id!r} last_use_event is outside its "
                    "scope lineage"
                )
        if item.crosses_iteration and scope.kind is not ScopeKind.LOOP_REGION:
            raise SchemaError(
                f"storage {item.storage_id!r} crosses_iteration outside a loop scope"
            )

        observed = set().union(
            *(value_events.get(member, set()) for member in item.members)
        )
        if (
            init_key not in observed
            and item.init_event.kind is not SemanticEventKind.SCOPE_ENTRY
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} init_event is unrelated to its members"
            )
        if (
            last_key not in observed
            and item.last_use_event.kind is not SemanticEventKind.SCOPE_EXIT
        ):
            raise SchemaError(
                f"storage {item.storage_id!r} last_use_event is unrelated to its members"
            )
        for observed_event in sorted(
            observed,
            key=lambda event: (event_order[event], event),
        ):
            observed_order = event_order[observed_event]
            if not (init_order <= observed_order <= last_order):
                offending_members = sorted(
                    member
                    for member in item.members
                    if observed_event in value_events.get(member, set())
                )
                raise SchemaError(
                    f"storage {item.storage_id!r} member(s) "
                    f"{offending_members!r} use "
                    f"{observed_event[0]} event {observed_event[1]!r} outside "
                    "declared lifetime "
                    f"[{item.init_event.kind.value}:{item.init_event.ref!r}, "
                    f"{item.last_use_event.kind.value}:"
                    f"{item.last_use_event.ref!r}]"
                )
            owner_scope = event_scope.get(observed_event)
            if (
                owner_scope is not None
                and not is_descendant_or_same(owner_scope, item.scope_id)
                and not item.escapes
            ):
                raise SchemaError(
                    f"storage {item.storage_id!r} escapes its scope without "
                    "escapes=true"
                )
        if observed:
            first_observed = min(event_order[key] for key in observed)
            last_observed = max(event_order[key] for key in observed)
            if init_order > first_observed:
                raise SchemaError(
                    f"storage {item.storage_id!r} is used before init_event"
                )
            if item.last_use_event.kind is not SemanticEventKind.SCOPE_EXIT:
                if last_order != last_observed:
                    raise SchemaError(
                        f"storage {item.storage_id!r} last_use_event is not its "
                        "last semantic use"
                    )
            elif not item.escapes:
                raise SchemaError(
                    f"storage {item.storage_id!r} uses scope_exit lifetime without "
                    "escapes=true"
                )

        write_events: set[tuple[str, str]] = set()
        for access in archive.boundary.accesses:
            if access.tensor not in item.members or access.kind not in {
                AccessKind.STORE, AccessKind.UPDATE,
            }:
                continue
            write_events.add(("access", access.access_id))
        for effect in archive.boundary.effects:
            if effect.target not in item.members or effect.kind not in {
                EffectKind.WRITE, EffectKind.READ_WRITE, EffectKind.UPDATE,
            }:
                continue
            assert effect.effect_id is not None
            if effect.access_ref is not None:
                write_events.add(("access", effect.access_ref))
            else:
                write_events.add(("effect", effect.effect_id))
        if item.mutability is StorageMutability.IMMUTABLE and write_events:
            raise SchemaError(
                f"immutable storage {item.storage_id!r} has a write event"
            )
        if item.mutability is StorageMutability.WRITE_ONCE and len(write_events) != 1:
            raise SchemaError(
                f"write_once storage {item.storage_id!r} requires exactly one "
                "semantic write event"
            )

    for state, scope_id, _loop_event in state_records:
        item = member_owner.get(state.state_id)
        if item is None:
            raise SchemaError(
                f"semantic loop state {state.state_id!r} requires a StorageObject"
            )
        required_members = {
            state.state_id, state.body_argument, state.next_value,
            state.result_value,
        }
        if (
            isinstance(state.init_value, str)
            and not _looks_like_semantic_literal(state.init_value)
        ):
            required_members.add(state.init_value)
        if not required_members.issubset(item.members):
            missing_members = sorted(required_members - set(item.members))
            raise SchemaError(
                f"semantic loop state {state.state_id!r} storage "
                f"{item.storage_id!r} is missing required member(s) "
                f"{missing_members!r}; the complete carry ownership is "
                f"{sorted(required_members)!r}"
            )
        if item.scope_id != scope_id or not item.crosses_iteration or not item.escapes:
            raise SchemaError(
                f"semantic loop state {state.state_id!r} storage "
                f"{item.storage_id!r} requires scope_id={scope_id!r}, "
                "crosses_iteration=true, escapes=true; authored "
                f"scope_id={item.scope_id!r}, "
                f"crosses_iteration={item.crosses_iteration!r}, "
                f"escapes={item.escapes!r}"
            )

    for alias in archive.boundary.aliases:
        left = member_owner.get(alias.left)
        right = member_owner.get(alias.right)
        if left is None or right is None:
            raise SchemaError(
                "Boundary alias endpoints require explicit semantic storage"
            )
        same_storage = left is right
        if alias.kind is AliasKind.MUST_ALIAS and not same_storage:
            raise SchemaError(
                "must_alias endpoints must be members of the same StorageObject"
            )
        if alias.kind is AliasKind.NO_ALIAS and same_storage:
            raise SchemaError(
                "no_alias endpoints must use different StorageObject identities"
            )


def _looks_like_semantic_literal(value: str) -> bool:
    return is_scalar_literal(value)


@dataclass(frozen=True)
class ValueDimensionRelation(StrictJSONMixin):
    """One J-authored equality between a rank-0 value and a shape dimension."""

    value_ref: str
    dimension_ref: str
    relation: str = "eq"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value_ref", _string(self.value_ref, "ValueDimensionRelation.value_ref")
        )
        object.__setattr__(
            self,
            "dimension_ref",
            _string(self.dimension_ref, "ValueDimensionRelation.dimension_ref"),
        )
        object.__setattr__(
            self, "relation", _string(self.relation, "ValueDimensionRelation.relation")
        )
        if re.fullmatch(r"t[0-9]+", self.value_ref) is None:
            raise SchemaError(
                "ValueDimensionRelation.value_ref must be a logical rank-0 "
                "tensor t* handle; physical p* and SSA v* handles are invalid"
            )
        if re.fullmatch(r"x[0-9]+", self.dimension_ref) is None:
            raise SchemaError(
                "ValueDimensionRelation.dimension_ref must be a symbolic "
                "dimension x* handle"
            )
        if self.relation != "eq":
            raise SchemaError("ValueDimensionRelation.relation must be 'eq'")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ValueDimensionRelation":
        data = _strict_object(
            value,
            path="ValueDimensionRelation",
            allowed={"value_ref", "dimension_ref", "relation"},
            required={"value_ref", "dimension_ref", "relation"},
        )
        return cls(data["value_ref"], data["dimension_ref"], data["relation"])

    def to_json(self) -> dict[str, str]:
        return {
            "value_ref": self.value_ref,
            "dimension_ref": self.dimension_ref,
            "relation": self.relation,
        }


_DIMENSION_RELATION_OPS = DIMENSION_RELATION_OPS


@dataclass(frozen=True)
class DimensionRelationExpr(StrictJSONMixin):
    """One closed J-authored expression over a declared logical dimension."""

    op: str
    arg: str
    value: int | None = None
    rhs: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _string(self.op, "DimensionRelationExpr.op"))
        object.__setattr__(
            self, "arg", _string(self.arg, "DimensionRelationExpr.arg")
        )
        if self.op not in _DIMENSION_RELATION_OPS:
            raise SchemaError("DimensionRelationExpr.op is not in the closed grammar")
        if re.fullmatch(r"x[0-9]+", self.arg) is None:
            raise SchemaError("DimensionRelationExpr.arg must be a dimension handle x*")
        if self.rhs is not None:
            object.__setattr__(
                self, "rhs", _string(self.rhs, "DimensionRelationExpr.rhs")
            )
            if re.fullmatch(r"x[0-9]+", self.rhs) is None:
                raise SchemaError("DimensionRelationExpr.rhs must be a dimension handle x*")
        if self.op == "identity":
            if self.value is not None or self.rhs is not None:
                raise SchemaError("identity dimension expression forbids value/rhs")
            return
        if self.op == "mul_dims":
            if self.rhs is None or self.value is not None:
                raise SchemaError("mul_dims dimension expression requires rhs and forbids value")
            return
        if self.rhs is not None:
            raise SchemaError(f"{self.op} dimension expression forbids rhs")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or self.value <= 0
            or self.value > DIMENSION_INT64_MAX
        ):
            raise SchemaError(
                f"{self.op} dimension expression requires a positive signed-int64 value"
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "DimensionRelationExpr":
        data = _strict_object(
            value,
            path="DimensionRelationExpr",
            allowed={"op", "arg", "value", "rhs"},
            required={"op", "arg"},
        )
        return cls(data["op"], data["arg"], data.get("value"), data.get("rhs"))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "arg": self.arg}
        if self.value is not None:
            result["value"] = self.value
        if self.rhs is not None:
            result["rhs"] = self.rhs
        return result


@dataclass(frozen=True)
class DimensionRelation(StrictJSONMixin):
    """One J-owned definition of a logical dimension from another dimension."""

    result: str
    expr: DimensionRelationExpr

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", _string(self.result, "DimensionRelation.result"))
        if re.fullmatch(r"x[0-9]+", self.result) is None:
            raise SchemaError("DimensionRelation.result must be a dimension handle x*")
        if not isinstance(self.expr, DimensionRelationExpr):
            object.__setattr__(
                self, "expr", DimensionRelationExpr.from_json(self.expr)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "DimensionRelation":
        data = _strict_object(
            value,
            path="DimensionRelation",
            allowed={"result", "expr"},
            required={"result", "expr"},
        )
        return cls(data["result"], DimensionRelationExpr.from_json(data["expr"]))

    def to_json(self) -> dict[str, Any]:
        return {"result": self.result, "expr": self.expr.to_json()}


def _validate_access_guard_reference_closure(
    archive: "SemanticArchive",
) -> None:
    """Require access-guard endpoints to name real declarations in this J.

    Function outputs and carry ``next_value`` fields are references, not value
    declarations by themselves.  In particular, accepting either one here
    would let an output-only name or a misspelled carry update cross the wire
    merely because another field repeated the same spelling.  The ordinary
    Layer-A validator separately proves that loop results and next values are
    backed by actual operation results.
    """

    construction_records = canonical_archive_records(archive)
    typed_rows = (
        construction_records.typed_operation_rows
        if construction_records is not None
        else None
    )
    if typed_rows is None:
        raise SchemaError(
            "access-guard validation requires canonical operation records"
        )
    declared_values = {item.name for item in archive.intent.inputs}
    for _path, operation in typed_rows:
        if isinstance(operation.result, str):
            declared_values.add(operation.result)
        elif isinstance(operation.result, list):
            declared_values.update(operation.result)
        declared_values.update(scan.name for scan in operation.scan)
        declared_values.update(state.body_argument for state in operation.carry)
    unknown_values = sorted(
        {
            relation.value_ref
            for relation in archive.boundary.access_guard_relations
            if relation.value_ref not in declared_values
        }
    )
    if unknown_values:
        raise SchemaError(
            "Boundary.access_guard_relations references unknown Layer-A "
            "value(s): " + ", ".join(unknown_values)
        )


@dataclass
class SemanticArchive(StrictJSONMixin):
    """The semantic archive ``J``: existing IntentFunction plus strict Layer B."""

    kernel_id: str
    intent: IntentFunction
    node_ids: dict[str, str]
    boundary: Boundary
    signature: "LogicalSignature"
    scopes: list[PortableScope] = field(default_factory=list)
    storage: list[StorageObject] = field(default_factory=list)
    dimension_constraints: list[str] = field(default_factory=list)
    value_dimension_relations: list[ValueDimensionRelation] = field(
        default_factory=list
    )
    dimension_relations: list[DimensionRelation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kernel_id = _string(self.kernel_id, "SemanticArchive.kernel_id")
        if not isinstance(self.intent, IntentFunction):
            self.intent = _strict_intent(self.intent)
        else:
            # Reparse to enforce the closed persisted representation even for a
            # directly constructed IntentFunction.
            self.intent = _strict_intent(self.intent.to_json())
        if not isinstance(self.node_ids, dict):
            raise SchemaError("SemanticArchive.node_ids must be an object")
        try:
            self.node_ids = {
                _string(node_id, "SemanticArchive.node_ids key"): _string(path, f"SemanticArchive.node_ids.{node_id}")
                for node_id, path in self.node_ids.items()
            }
            if len(set(self.node_ids.values())) != len(self.node_ids):
                raise SchemaError("SemanticArchive.node_ids paths must be unique")
            canonical_path = re.compile(
                r"^body\[\d+\]"
                r"(?:(?:\.body|\.positive\.body|\.negative\.body)\[\d+\])*$"
            )
            operation_records = canonical_operation_records(self.intent.to_json())
            if operation_records is None:
                raise SchemaError(
                    "SemanticArchive requires a complete canonical operation table"
                )
            by_path = {
                record["path"]: record for record in operation_records
            }
            for node_id, node_path in self.node_ids.items():
                if canonical_path.fullmatch(node_path) is None:
                    raise SchemaError(
                        f"SemanticArchive.node_ids.{node_id} must be a canonical "
                        "body/loop/conditional-arm JSON path"
                    )
                if node_path not in by_path:
                    raise SchemaError(
                        f"SemanticArchive.node_ids.{node_id} path is out of range"
                    )
        except SchemaError as exc:
            raise SemanticArchiveCrossContractError("node_path") from exc
        # JSON object member order is not semantic.  Canonicalize the node map
        # by the authored body-path traversal before any declaration-order
        # checks or opaque projections consume it.  This preserves every LLM-
        # authored id/path pair and merely removes serializer-dependent key
        # ordering (for example n0,n1,n10,...,n2 from sort_keys=True).
        path_order = {path: index for index, path in enumerate(by_path)}
        self.node_ids = dict(
            sorted(
                self.node_ids.items(),
                key=lambda item: path_order[item[1]],
            )
        )
        if not isinstance(self.boundary, Boundary):
            self.boundary = Boundary.from_json(self.boundary)
        if not isinstance(self.signature, LogicalSignature):
            self.signature = LogicalSignature.from_json(self.signature)
        self.scopes = [
            item if isinstance(item, PortableScope) else PortableScope.from_json(item)
            for item in self.scopes
        ]
        self.storage = [
            item if isinstance(item, StorageObject) else StorageObject.from_json(item)
            for item in self.storage
        ]
        if not isinstance(self.dimension_constraints, list):
            raise SchemaError(
                "SemanticArchive.dimension_constraints must be an array"
            )
        self.dimension_constraints = [
            _string(
                expression,
                f"SemanticArchive.dimension_constraints[{index}]",
            )
            for index, expression in enumerate(self.dimension_constraints)
        ]
        if not isinstance(self.value_dimension_relations, list):
            raise SchemaError(
                "SemanticArchive.value_dimension_relations must be an array"
            )
        self.value_dimension_relations = [
            item
            if isinstance(item, ValueDimensionRelation)
            else ValueDimensionRelation.from_json(item)
            for item in self.value_dimension_relations
        ]
        if not isinstance(self.dimension_relations, list):
            raise SchemaError("SemanticArchive.dimension_relations must be an array")
        self.dimension_relations = [
            item
            if isinstance(item, DimensionRelation)
            else DimensionRelation.from_json(item)
            for item in self.dimension_relations
        ]
        # Relation declarations are part of J's closed semantic boundary.  Do
        # this at construction time as well as in TypeCheck/materialization so
        # direct model construction cannot carry a dangling x* reference into
        # a later consumer.
        try:
            from intentir.ir.dimension_relation import validate_dimension_relations
            from intentir.ir.value_dimension import (
                validate_value_dimension_relations,
            )

            validate_dimension_relations(self)
            validate_value_dimension_relations(self)
        except (TypeError, ValueError) as exc:
            raise SemanticArchiveCrossContractError("dimension") from exc
        try:
            _validate_logical_signature(self)
        except SchemaError as exc:
            raise SemanticArchiveCrossContractError("signature") from exc
        try:
            _validate_typed_boundary_relations(self)
        except SemanticArchiveCrossContractError:
            raise
        except SchemaError as exc:
            raise SemanticArchiveCrossContractError("typed_boundary") from exc
        try:
            _validate_access_guard_reference_closure(self)
        except SchemaError as exc:
            raise SemanticArchiveCrossContractError("access_guard") from exc
        try:
            _validate_semantic_scopes_and_storage(self)
        except SchemaError as exc:
            raise SemanticArchiveCrossContractError("scope_storage") from exc

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SemanticArchive":
        nested_root_owned = _nested_root_owned_field_paths(
            value,
            path="SemanticArchive",
        )
        if nested_root_owned:
            raise SchemaError(
                f"{nested_root_owned[0]} is a root-owned field nested under J; "
                "root.c and root.pi must be siblings of root.j"
            )
        data = _strict_object(
            value,
            path="SemanticArchive",
            allowed=set(SEMANTIC_ARCHIVE_WIRE_FIELDS),
            required=set(SEMANTIC_ARCHIVE_WIRE_FIELDS),
        )
        node_ids = _object(data["node_ids"], "SemanticArchive.node_ids")
        scopes = data["scopes"]
        storage = data["storage"]
        if not isinstance(scopes, list):
            raise SchemaError("SemanticArchive.scopes must be an array")
        if not isinstance(storage, list):
            raise SchemaError("SemanticArchive.storage must be an array")
        return cls(
            kernel_id=data["kernel_id"],
            intent=_strict_intent(data["intent"]),
            node_ids=dict(node_ids),
            boundary=Boundary.from_json(data["boundary"]),
            signature=LogicalSignature.from_json(data["signature"]),
            scopes=[PortableScope.from_json(item) for item in scopes],
            storage=[StorageObject.from_json(item) for item in storage],
            dimension_constraints=data["dimension_constraints"],
            value_dimension_relations=[
                ValueDimensionRelation.from_json(item)
                for item in data["value_dimension_relations"]
            ],
            dimension_relations=[
                DimensionRelation.from_json(item)
                for item in data["dimension_relations"]
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "intent": self.intent.to_json(),
            "node_ids": dict(self.node_ids),
            "boundary": self.boundary.to_json(),
            "signature": self.signature.to_json(),
            "scopes": [item.to_json() for item in self.scopes],
            "storage": [item.to_json() for item in self.storage],
            "dimension_constraints": list(self.dimension_constraints),
            "value_dimension_relations": [
                item.to_json() for item in self.value_dimension_relations
            ],
            "dimension_relations": [
                item.to_json() for item in self.dimension_relations
            ],
        }


@dataclass
class SignatureParameter(StrictJSONMixin):
    """One J-owned logical callable handle.

    ``name`` is a canonical J handle, never a source parameter spelling.
    ``logical_ref`` links the handle to a J tensor/value/dimension when one
    exists. Runtime representation plumbing and Harness-owned compile-time
    specialization carriers may omit the link; neither becomes an operand of
    Layer A or a fourth production interface.
    """

    name: str
    kind: ParameterKind
    mode: ParameterMode = ParameterMode.VALUE
    dtype: str | None = None
    rank: int | None = None
    alias_of: str | None = None
    logical_ref: str | None = None

    def __post_init__(self) -> None:
        self.name = _string(self.name, "SignatureParameter.name")
        if re.fullmatch(r"p[0-9]+", self.name) is None:
            raise SchemaError("J.signature parameter names must be p0, p1, ...")
        self.kind = _enum(ParameterKind, self.kind, "SignatureParameter.kind")
        self.mode = _enum(ParameterMode, self.mode, "SignatureParameter.mode")
        if self.dtype is not None:
            self.dtype = _string(self.dtype, "SignatureParameter.dtype")
        if self.rank is not None:
            self.rank = _integer(self.rank, "SignatureParameter.rank", minimum=0)
        if self.alias_of is not None:
            self.alias_of = _string(self.alias_of, "SignatureParameter.alias_of")
        if self.logical_ref is not None:
            self.logical_ref = _string(
                self.logical_ref, "SignatureParameter.logical_ref"
            )
            if re.fullmatch(r"(?:t|x)[0-9]+", self.logical_ref) is None:
                raise SchemaError(
                    "SignatureParameter.logical_ref must be a logical tensor "
                    "t* or dimension x* handle; physical p* and operation "
                    "SSA v* handles are not valid signature bindings"
                )
        # A signature field without a logical_ref may be a physical source or
        # Harness constexpr carrier (for example a backend token), so its
        # spelling is intentionally not forced through the logical dtype set.
        # Once the parameter is tied to J, however, its dtype is part of the
        # closed logical ABI and must use the same canonical set as IntentOp
        # tensors.  Tensor/pointer parameters are always J-owned by contract.
        if self.dtype is not None and (
            self.kind in {ParameterKind.TENSOR, ParameterKind.POINTER}
            or self.logical_ref is not None
        ):
            try:
                self.dtype = canonicalize_dtype(
                    self.dtype, context="SignatureParameter.dtype"
                )
            except (TypeError, ValueError) as exc:
                raise SchemaError(str(exc)) from exc
        issues: list[_SchemaIssue] = []
        if self.kind is ParameterKind.OPAQUE:
            detail = (
                "opaque signature parameters expose only a handle and public dtype"
            )
            for field_name, invalid in (
                ("mode", self.mode is not ParameterMode.VALUE),
                ("dtype", self.dtype is None),
                ("rank", self.rank is not None),
                ("alias_of", self.alias_of is not None),
                ("logical_ref", self.logical_ref is not None),
            ):
                if invalid:
                    issues.append(
                        _SchemaIssue(f"SignatureParameter.{field_name}", detail)
                    )
        if self.kind in {ParameterKind.TENSOR, ParameterKind.POINTER}:
            detail = (
                "tensor/pointer signature parameters require dtype, rank, "
                "and a J logical_ref"
            )
            for field_name, field_value in (
                ("dtype", self.dtype),
                ("rank", self.rank),
                ("logical_ref", self.logical_ref),
            ):
                if field_value is None:
                    issues.append(
                        _SchemaIssue(f"SignatureParameter.{field_name}", detail)
                    )
        if self.kind in {ParameterKind.SCALAR, ParameterKind.CONSTEXPR}:
            if self.dtype is None:
                issues.append(
                    _SchemaIssue(
                        "SignatureParameter.dtype",
                        "scalar/constexpr signature parameters require dtype",
                    )
                )
            if self.logical_ref is None and not (
                self.mode is ParameterMode.VALUE
                and self.rank == 0
                and self.alias_of is None
            ):
                issues.append(
                    _SchemaIssue(
                        "SignatureParameter.logical_ref",
                        "a scalar/constexpr without J logical_ref must be an "
                        "explicit rank-0, mode=value, non-alias carrier; a "
                        "constexpr carrier is bound only by the fixed Harness "
                        "specialization and never becomes Layer-A data",
                    )
                )
            elif (
                isinstance(self.logical_ref, str)
                and re.fullmatch(r"t[0-9]+", self.logical_ref)
                and self.rank != 0
            ):
                issues.append(
                    _SchemaIssue(
                        "SignatureParameter.rank",
                        "a scalar/constexpr logical t* binding must have rank=0",
                    )
                )
        if issues:
            raise SchemaError.from_issues(issues)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SignatureParameter":
        data = _strict_object(
            value,
            path="SignatureParameter",
            allowed={
                "name", "kind", "mode", "dtype", "rank", "alias_of",
                "logical_ref",
            },
            required={"name", "kind"},
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            mode=data.get("mode", ParameterMode.VALUE.value),
            dtype=data.get("dtype"),
            rank=data.get("rank"),
            alias_of=data.get("alias_of"),
            logical_ref=data.get("logical_ref"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "mode": self.mode.value,
        }
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.rank is not None:
            result["rank"] = self.rank
        if self.alias_of is not None:
            result["alias_of"] = self.alias_of
        if self.logical_ref is not None:
            result["logical_ref"] = self.logical_ref
        return result


@dataclass
class LogicalSignature(StrictJSONMixin):
    """The canonical callable signature owned by J.

    Declaration order is the canonical Replay order for logical handles.  C
    slots are declared by C and appended by the binding adapter; source-side
    positions remain private Harness coordinates.
    """

    parameters: list[SignatureParameter] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parameters = [
            item
            if isinstance(item, SignatureParameter)
            else SignatureParameter.from_json(item)
            for item in self.parameters
        ]
        _unique(self.parameters, "name", "LogicalSignature.parameters")
        expected = list(
            CanonicalHandleAllocator.canonical_names("p", len(self.parameters))
        )
        names = {item.name for item in self.parameters}
        if [item.name for item in self.parameters] != expected:
            raise SchemaError(
                "J.signature parameter ids must be declaration-order handles "
                "p0, p1, ..."
            )
        self_aliases = sorted(
            item.name
            for item in self.parameters
            if item.alias_of == item.name
        )
        if self_aliases:
            raise SchemaError(
                "J.signature alias_of cannot self-reference parameter(s): "
                + ", ".join(self_aliases)
            )
        unknown_aliases = sorted(
            item.alias_of for item in self.parameters
            if item.alias_of is not None and item.alias_of not in names
        )
        if unknown_aliases:
            raise SchemaError(
                "J.signature alias_of references unknown parameters: "
                + ", ".join(unknown_aliases)
            )
        alias_graph = {
            item.name: item.alias_of
            for item in self.parameters
            if item.alias_of is not None
        }
        for start in alias_graph:
            path: list[str] = []
            current: str | None = start
            while current is not None and current in alias_graph:
                if current in path:
                    cycle = path[path.index(current):] + [current]
                    raise SchemaError(
                        "J.signature alias_of contains a cycle: "
                        + " -> ".join(cycle)
                    )
                path.append(current)
                current = alias_graph[current]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LogicalSignature":
        data = _strict_object(
            value,
            path="LogicalSignature",
            allowed={"parameters"},
            required={"parameters"},
        )
        if not isinstance(data["parameters"], list):
            raise SchemaError("LogicalSignature.parameters must be an array")
        parameters: list[SignatureParameter] = []
        issues: list[_SchemaIssue] = []
        for index, item in enumerate(data["parameters"]):
            try:
                parameters.append(SignatureParameter.from_json(item))
            except SchemaError as exc:
                if exc.issues:
                    for issue in exc.issues:
                        issues.append(
                            _SchemaIssue(
                                f"LogicalSignature.parameters[{index}]"
                                + issue.path.removeprefix("SignatureParameter"),
                                issue.detail,
                            )
                        )
                else:
                    issues.append(
                        _SchemaIssue(
                            f"LogicalSignature.parameters[{index}]",
                            str(exc),
                        )
                    )
        if issues:
            raise SchemaError.from_issues(issues)
        return cls(parameters=parameters)

    def to_json(self) -> dict[str, Any]:
        return {"parameters": [item.to_json() for item in self.parameters]}


def _validate_logical_signature(archive: SemanticArchive) -> None:
    """Check that J.signature is a total, typed view of J's logical boundary."""

    tensors = {
        tensor.name: tensor
        for tensor in [*archive.intent.inputs, *archive.intent.outputs]
    }
    dimensions = declared_dimension_handles(archive)
    input_refs = {tensor.name for tensor in archive.intent.inputs}
    output_refs = {tensor.name for tensor in archive.intent.outputs}
    seen_logical: dict[str, str] = {}
    parameters_by_name = {
        parameter.name: parameter for parameter in archive.signature.parameters
    }
    declared_alias_kinds = {
        frozenset((alias.left, alias.right)): alias.kind
        for alias in archive.boundary.aliases
    }
    issues: list[_SchemaIssue] = []
    for index, parameter in enumerate(archive.signature.parameters):
        base = f"SemanticArchive.signature.parameters[{index}]"
        ref = parameter.logical_ref
        if ref is None:
            continue
        if re.fullmatch(r"(?:t|x)[0-9]+", ref) is None:
            issues.append(
                _SchemaIssue(
                    f"{base}.logical_ref",
                    f"J.signature parameter {parameter.name!r} logical_ref "
                    f"{ref!r} must be a logical tensor t* or dimension x*; "
                    "physical p* and operation SSA v* handles are not valid "
                    "signature bindings",
                )
            )
            continue
        if ref not in tensors and ref not in dimensions:
            issues.append(
                _SchemaIssue(
                    f"{base}.logical_ref",
                    f"J.signature parameter {parameter.name!r} references unknown "
                    f"J logical handle {ref!r}; logical_ref must name a declared "
                    "J input/output tensor or symbolic dimension, never a body "
                    "SSA value",
                )
            )
            continue
        previous_parameter = seen_logical.get(ref)
        if previous_parameter is not None:
            issues.append(
                _SchemaIssue(
                    f"{base}.logical_ref",
                    f"J logical handle {ref!r} is already bound by signature "
                    f"parameter {previous_parameter!r}; one logical t*/x* "
                    "handle must have one physical binding. Use one "
                    "mode=inout parameter for an input/output identity, and "
                    "use distinct logical handles plus an explicit Boundary "
                    "must_alias for physical aliases",
                )
            )
        else:
            seen_logical[ref] = parameter.name
        if parameter.alias_of is not None:
            target = parameters_by_name.get(parameter.alias_of)
            target_ref = target.logical_ref if target is not None else None
            logical_pair = (
                frozenset((ref, target_ref))
                if target_ref is not None
                else None
            )
            if (
                logical_pair is None
                or len(logical_pair) != 2
                or declared_alias_kinds.get(logical_pair)
                is not AliasKind.MUST_ALIAS
            ):
                issues.append(
                    _SchemaIssue(
                        f"{base}.alias_of",
                        "a physical alias must bind two distinct logical "
                        "t* handles joined by one Boundary alias with "
                        "kind='must_alias'",
                    )
                )
        if ref in tensors:
            tensor = tensors[ref]
            if parameter.dtype != tensor.dtype:
                issues.append(
                    _SchemaIssue(
                        f"{base}.dtype",
                        f"J.signature parameter {parameter.name!r} dtype "
                        f"{parameter.dtype!r} differs from logical tensor {ref!r} "
                        f"dtype {tensor.dtype!r}",
                    )
                )
            if parameter.rank != len(tensor.shape):
                issues.append(
                    _SchemaIssue(
                        f"{base}.rank",
                        f"J.signature parameter {parameter.name!r} rank "
                        f"{parameter.rank!r} differs from logical tensor {ref!r} "
                        f"rank {len(tensor.shape)!r}",
                    )
                )
            if (
                parameter.kind in {ParameterKind.SCALAR, ParameterKind.CONSTEXPR}
                and re.fullmatch(r"t[0-9]+", ref)
            ):
                if ref not in input_refs:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.logical_ref",
                            "a scalar/constexpr logical t* binding must name a "
                            "J input rank-0 value",
                        )
                    )
                if tensor.shape != []:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.logical_ref",
                            "a scalar/constexpr logical t* binding must name a "
                            "rank-0 J tensor with shape=[]",
                        )
                    )
                if parameter.rank != 0:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.rank",
                            "a scalar/constexpr logical t* binding must have "
                            "rank=0",
                        )
                    )
            if parameter.mode is ParameterMode.INPUT:
                if ref not in input_refs:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.mode",
                            "input signature parameter must bind a J input",
                        )
                    )
                elif ref in output_refs:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.mode",
                            "a logical tensor declared in both J inputs and J "
                            "outputs must use signature mode=inout, never input",
                        )
                    )
            if parameter.mode is ParameterMode.OUTPUT:
                if ref not in output_refs:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.mode",
                            "output signature parameter must bind a J output",
                        )
                    )
                elif ref in input_refs:
                    issues.append(
                        _SchemaIssue(
                            f"{base}.mode",
                            "a logical tensor declared in both J inputs and J "
                            "outputs must use signature mode=inout, never output",
                        )
                    )
            if parameter.mode is ParameterMode.INOUT and not (
                ref in input_refs and ref in output_refs
            ):
                issues.append(
                    _SchemaIssue(
                        f"{base}.mode",
                        "inout signature parameter must bind the same logical "
                        "tensor declared in both J inputs and J outputs",
                    )
                )
        elif parameter.kind not in {
            ParameterKind.SCALAR,
            ParameterKind.CONSTEXPR,
        }:
            issues.append(
                _SchemaIssue(
                    f"{base}.kind",
                    "only scalar/constexpr signature parameters may bind a J "
                    "dimension",
                )
            )
    missing = sorted((input_refs | output_refs) - set(seen_logical))
    if missing:
        issues.append(
            _SchemaIssue(
                "SemanticArchive.signature.parameters",
                "J.signature does not bind logical tensor(s): " + ", ".join(missing),
            )
        )
    if issues:
        raise SchemaError.from_issues(issues)


@dataclass
class ScheduleSlot(StrictJSONMixin):
    name: str
    kind: ScheduleSlotKind
    role: str
    value_type: str = "int"
    required: bool = True

    def __post_init__(self) -> None:
        self.name = _handle(
            self.name,
            "ScheduleSlot.name",
            r"c[0-9]+",
            "a C slot c* handle",
        )
        self.kind = _enum(ScheduleSlotKind, self.kind, "ScheduleSlot.kind")
        self.role = _string(self.role, "ScheduleSlot.role")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.role) is None:
            raise SchemaError("ScheduleSlot.role must use lower_snake_case")
        forbidden_role_terms = {
            "api",
            "combiner",
            "conflict",
            "family",
            "fill",
            "identity",
            "index_map",
            "mask",
            "operator",
            "predicate",
            "realization",
            "rule",
            "semantic",
            "source",
            "strategy",
        }
        if any(term in self.role for term in forbidden_role_terms):
            raise SchemaError("ScheduleSlot.role must remain pure scheduling metadata")
        self.value_type = _string(self.value_type, "ScheduleSlot.value_type")
        self.required = _boolean(self.required, "ScheduleSlot.required")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ScheduleSlot":
        data = _strict_object(
            value,
            path="ScheduleSlot",
            allowed={
                "name", "kind", "role", "value_type", "required",
            },
            required={"name", "kind", "role"},
        )
        return cls(
            name=data["name"],
            kind=data["kind"],
            role=data["role"],
            value_type=data.get("value_type", "int"),
            required=data.get("required", True),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "role": self.role,
            "value_type": self.value_type,
            "required": self.required,
        }
        return result


def implementation_value_matches(value_type: str, value: Any) -> bool:
    """Return whether one C value is a closed scalar of its declared type."""

    normalized = value_type.casefold()
    if normalized in {"bool", "i1"}:
        return isinstance(value, bool)
    if normalized in {"int", "integer"} or normalized.startswith(("i", "u")):
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"float", "number"} or normalized.startswith("f"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized in {"str", "string", "layout", "enum"}:
        # C may select one closed implementation token, but it must not become
        # a source-code or free-text side channel.  Rich layout expressions
        # need their own closed schema rather than being smuggled through a
        # string-valued slot.
        return isinstance(value, str) and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.:+/-]*", value
        ) is not None
    return False


@dataclass
class ImplementationConfig(StrictJSONMixin):
    """Canonical C: anonymous selection slots plus Harness-held valuation."""

    slots: list[ScheduleSlot] = field(default_factory=list)
    values: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slots = [
            item if isinstance(item, ScheduleSlot) else ScheduleSlot.from_json(item)
            for item in self.slots
        ]
        _unique(self.slots, "name", "ImplementationConfig.slots")
        if not isinstance(self.values, dict):
            raise SchemaError("ImplementationConfig.values must be an object")
        self.values = {
            _string(key, "ImplementationConfig.values key"): _json_value(
                value, f"ImplementationConfig.values.{key}"
            )
            for key, value in self.values.items()
        }
        known = {slot.name for slot in self.slots}
        unknown = sorted(set(self.values) - known)
        if unknown:
            raise SchemaError(
                "ImplementationConfig.values has no declared slot(s): "
                + ", ".join(unknown)
            )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ImplementationConfig":
        data = _strict_object(
            value,
            path="ImplementationConfig",
            allowed=set(CANONICAL_C_WIRE_FIELDS),
            required=set(CANONICAL_C_WIRE_FIELDS),
        )
        if not isinstance(data["slots"], list):
            raise SchemaError("ImplementationConfig.slots must be an array")
        values = _object(data["values"], "ImplementationConfig.values")
        return cls(
            slots=[ScheduleSlot.from_json(item) for item in data["slots"]],
            values=dict(values),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "slots": [item.to_json() for item in self.slots],
            "values": copy.deepcopy(self.values),
        }

    def ordered_value_items(self) -> tuple[tuple[str, JsonValue], ...]:
        """Return bound C values in the declaration order of ``slots``.

        C slot names are opaque handles.  Their spelling, especially the
        numeric suffix of ``c*`` handles, is not a serialization order.  All
        consumers that need a positional C binding must therefore follow the
        declared slot sequence and never sort the value-object keys.
        """

        return tuple(
            (slot.name, self.values[slot.name])
            for slot in self.slots
            if slot.name in self.values
        )

    def missing_required_values(self) -> list[str]:
        """Return required C slots that do not have a bound value."""

        return [
            slot.name
            for slot in self.slots
            if slot.required and slot.name not in self.values
        ]

    def missing_values(self) -> list[str]:
        """Return every declared slot without an actual implementation value."""

        return [slot.name for slot in self.slots if slot.name not in self.values]

    def mistyped_values(self) -> list[str]:
        """Return bound slots whose values do not match their closed C types."""

        return [
            slot.name
            for slot in self.slots
            if slot.name in self.values
            and not implementation_value_matches(
                slot.value_type, self.values[slot.name]
            )
        ]

    def validate_complete_values(
        self, *, path: str = "ImplementationConfig"
    ) -> None:
        """Fail closed unless every declared active C slot has a valid value."""

        missing = self.missing_values()
        if missing:
            raise SchemaError(
                f"{path} is missing actual value(s) for declared slot(s): "
                + ", ".join(missing)
            )
        mistyped = self.mistyped_values()
        if mistyped:
            raise SchemaError(
                f"{path} has value(s) that do not match their declared type: "
                + ", ".join(mistyped)
            )



@dataclass
class SourceSpan(StrictJSONMixin):
    start_line: int
    end_line: int
    start_column: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        self.start_line = _integer(self.start_line, "SourceSpan.start_line", minimum=1)
        self.end_line = _integer(self.end_line, "SourceSpan.end_line", minimum=self.start_line)
        if self.start_column is not None:
            self.start_column = _integer(self.start_column, "SourceSpan.start_column", minimum=0)
        if self.end_column is not None:
            self.end_column = _integer(self.end_column, "SourceSpan.end_column", minimum=0)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SourceSpan":
        data = _strict_object(
            value,
            path="SourceSpan",
            allowed={"start_line", "end_line", "start_column", "end_column"},
            required={"start_line", "end_line"},
        )
        return cls(
            data["start_line"], data["end_line"],
            data.get("start_column"), data.get("end_column"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"start_line": self.start_line, "end_line": self.end_line}
        if self.start_column is not None:
            result["start_column"] = self.start_column
        if self.end_column is not None:
            result["end_column"] = self.end_column
        return result


@dataclass
class LocationEntry(StrictJSONMixin):
    span: SourceSpan
    node_refs: list[str] = field(default_factory=list)
    access_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan):
            self.span = SourceSpan.from_json(self.span)
        self.node_refs = _string_list(self.node_refs, "LocationEntry.node_refs", unique=True)
        self.access_refs = _string_list(self.access_refs, "LocationEntry.access_refs", unique=True)
        if not self.node_refs and not self.access_refs:
            raise SchemaError("LocationEntry must reference a node or access")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LocationEntry":
        data = _strict_object(
            value,
            path="LocationEntry",
            allowed={"span", "node_refs", "access_refs"},
            required={"span"},
        )
        return cls(
            SourceSpan.from_json(data["span"]),
            _string_list(data.get("node_refs", []), "LocationEntry.node_refs", unique=True),
            _string_list(data.get("access_refs", []), "LocationEntry.access_refs", unique=True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "span": self.span.to_json(),
            "node_refs": list(self.node_refs),
            "access_refs": list(self.access_refs),
        }


@dataclass
class LocationMap(StrictJSONMixin):
    source_name: str
    entries: list[LocationEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_name = _string(self.source_name, "LocationMap.source_name")
        self.entries = [
            item if isinstance(item, LocationEntry) else LocationEntry.from_json(item)
            for item in self.entries
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LocationMap":
        data = _strict_object(
            value,
            path="LocationMap",
            allowed={"source_name", "entries"},
            required={"source_name", "entries"},
        )
        if not isinstance(data["entries"], list):
            raise SchemaError("LocationMap.entries must be an array")
        return cls(data["source_name"], [LocationEntry.from_json(item) for item in data["entries"]])

    def to_json(self) -> dict[str, Any]:
        return {"source_name": self.source_name, "entries": [item.to_json() for item in self.entries]}


@dataclass
class RuntimeArgument(StrictJSONMixin):
    name: str
    value: JsonValue = None
    dtype: str | None = None
    shape: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = _string(self.name, "RuntimeArgument.name")
        self.value = _json_value(self.value, "RuntimeArgument.value")
        if self.dtype is not None:
            self.dtype = _string(self.dtype, "RuntimeArgument.dtype")
        if not isinstance(self.shape, list):
            raise SchemaError("RuntimeArgument.shape must be an array")
        self.shape = [
            _integer(dim, f"RuntimeArgument.shape[{index}]", minimum=0)
            for index, dim in enumerate(self.shape)
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RuntimeArgument":
        data = _strict_object(
            value,
            path="RuntimeArgument",
            allowed={"name", "value", "dtype", "shape"},
            required={"name"},
        )
        shape = data.get("shape", [])
        if not isinstance(shape, list):
            raise SchemaError("RuntimeArgument.shape must be an array")
        return cls(data["name"], data.get("value"), data.get("dtype"), list(shape))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.value is not None:
            result["value"] = copy.deepcopy(self.value)
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.shape:
            result["shape"] = list(self.shape)
        return result


@dataclass
class CompileCase(StrictJSONMixin):
    arguments: list[RuntimeArgument]
    grid: list[int]
    case_id: str = "default"

    def __post_init__(self) -> None:
        self.case_id = _string(self.case_id, "CompileCase.case_id")
        self.arguments = [
            item if isinstance(item, RuntimeArgument) else RuntimeArgument.from_json(item)
            for item in self.arguments
        ]
        _unique(self.arguments, "name", "CompileCase.arguments")
        if not isinstance(self.grid, list):
            raise SchemaError("CompileCase.grid must be an array")
        self.grid = [
            _integer(dim, f"CompileCase.grid[{index}]", minimum=1)
            for index, dim in enumerate(self.grid)
        ]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CompileCase":
        data = _strict_object(
            value,
            path="CompileCase",
            allowed={"case_id", "arguments", "grid"},
            required={"arguments", "grid"},
        )
        if not isinstance(data["arguments"], list):
            raise SchemaError("CompileCase.arguments must be an array")
        if not isinstance(data["grid"], list):
            raise SchemaError("CompileCase.grid must be an array")
        return cls(
            arguments=[RuntimeArgument.from_json(item) for item in data["arguments"]],
            grid=list(data["grid"]),
            case_id=data.get("case_id", "default"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arguments": [item.to_json() for item in self.arguments],
            "grid": list(self.grid),
        }


@dataclass
class ProvenanceRecord(StrictJSONMixin):
    role: ProvenanceRole
    producer: str
    model: str | None = None
    source_digest: str | None = None
    created_at: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role = _enum(ProvenanceRole, self.role, "ProvenanceRecord.role")
        self.producer = _string(self.producer, "ProvenanceRecord.producer")
        for name in ("model", "source_digest", "created_at"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _string(value, f"ProvenanceRecord.{name}"))
        if not isinstance(self.metadata, dict):
            raise SchemaError("ProvenanceRecord.metadata must be an object")
        self.metadata = _json_value(self.metadata, "ProvenanceRecord.metadata")  # type: ignore[assignment]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ProvenanceRecord":
        data = _strict_object(
            value,
            path="ProvenanceRecord",
            allowed={"role", "producer", "model", "source_digest", "created_at", "metadata"},
            required={"role", "producer"},
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SchemaError("ProvenanceRecord.metadata must be an object")
        return cls(
            role=data["role"], producer=data["producer"], model=data.get("model"),
            source_digest=data.get("source_digest"), created_at=data.get("created_at"),
            metadata=dict(metadata),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role.value, "producer": self.producer}
        for name in ("model", "source_digest", "created_at"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.metadata:
            result["metadata"] = copy.deepcopy(self.metadata)
        return result


@dataclass
class Gate1Artifact(StrictJSONMixin):
    """The complete proof-relevant Lift result: exactly (J, C, pi)."""

    SCHEMA: ClassVar[str] = "intentir.scrc.artifact.v12"

    j: SemanticArchive
    c: ImplementationConfig
    pi: RealizationWitness
    wire_schema: str = field(default=SCHEMA, repr=False)

    def __post_init__(self) -> None:
        if self.wire_schema != self.SCHEMA:
            raise SchemaError(
                f"unsupported Gate1 artifact schema {self.wire_schema!r}; "
                f"expected the current schema {self.SCHEMA!r}"
            )
        if not isinstance(self.j, SemanticArchive):
            self.j = SemanticArchive.from_json(self.j)
        if not isinstance(self.c, ImplementationConfig):
            self.c = ImplementationConfig.from_json(self.c)
        if not isinstance(self.pi, RealizationWitness):
            try:
                self.pi = RealizationWitness.from_json(self.pi)
            except RealizationWitnessError as exc:
                raise SchemaError(str(exc)) from exc

        node_refs = set(self.j.node_ids) | set(self.j.node_ids.values())
        boundary_node_refs = {
            *(item.node_ref for item in self.j.boundary.accesses if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.broadcasts),
            *(item.node_ref for item in self.j.boundary.effects if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.output_mappings if item.node_ref is not None),
            *(item.node_ref for item in self.j.boundary.reductions),
            *(item.node_ref for item in self.j.boundary.index_bounds),
            *(item.node_ref for item in self.j.boundary.updates),
            *(item.consumer_ref for item in self.j.boundary.consumer_extensions),
            *(
                item.update_node_ref
                for item in self.j.boundary.result_relations
                if item.update_node_ref is not None
            ),
        }
        unknown_boundary_nodes = sorted(boundary_node_refs - node_refs)
        if unknown_boundary_nodes:
            raise SchemaError(
                "Layer B references unknown J nodes: "
                + ", ".join(unknown_boundary_nodes)
            )
        from intentir.ir.canonical_j import elaborate_j, j_elaboration

        accepted_j = elaborate_j(self.j)
        if accepted_j is not self.j:
            self.j = accepted_j
        construction_records = j_elaboration(self.j).construction_records
        operation_records = construction_records.operation_records
        typed_rows = construction_records.typed_operation_rows
        if typed_rows is None:
            raise SchemaError(
                "Gate1 artifact requires canonical operation records for "
                "consumer extensions"
            )
        operation_by_id = {
            record.get("id"): operation
            for (_path, operation), record in zip(
                typed_rows,
                operation_records,
                strict=True,
            )
            if isinstance(record.get("id"), str)
        }
        for extension in self.j.boundary.consumer_extensions:
            consumer = operation_by_id.get(extension.consumer_ref)
            if consumer is None:
                # The node-id closure above normally catches this; keep the
                # local diagnostic explicit because this is the field that
                # owns the operand ordinal contract.
                raise SchemaError(
                    "Boundary.consumer_extensions references an unknown "
                    f"consumer node {extension.consumer_ref!r}"
                )
            if extension.operand_index >= len(consumer.operands):
                raise SchemaError(
                    "Boundary.consumer_extensions operand_index exceeds the "
                    f"consumer operand list for {extension.consumer_ref!r}"
                )
        missing_values = self.c.missing_required_values()
        if missing_values:
            raise SchemaError(
                "SCRC artifact is missing required C value(s): "
                + ", ".join(missing_values)
            )
    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Gate1Artifact":
        # Diagnose root ownership before the required-key check, so a draft
        # flattening J or nesting c/pi under J reports one actionable envelope
        # violation rather than a list of dependent missing/unknown fields.
        wire_diagnostics = gate1_artifact_wire_diagnostics(value)
        if wire_diagnostics:
            diagnostic = wire_diagnostics[0]
            raise SchemaError(
                f"{diagnostic['path']}: {diagnostic['message']}"
            )
        data = _strict_object(
            value,
            path="Gate1Artifact",
            allowed=set(GATE1_ARTIFACT_MODEL_FIELDS),
            required=set(GATE1_ARTIFACT_MODEL_FIELDS),
        )
        if data["schema"] != cls.SCHEMA:
            raise SchemaError(
                f"unsupported Gate1 artifact schema {data['schema']!r}; "
                f"expected the current schema {cls.SCHEMA!r}"
            )
        pi_value = data["pi"] if isinstance(data["pi"], dict) else None
        if (
            pi_value is None
            or pi_value.get("schema") != NORMALIZATION_WITNESS_SCHEMA
        ):
            raise SchemaError(
                "Gate1Artifact.pi must use the canonical normalization witness "
                "wire; legacy materialized pi graphs are not accepted"
            )
        try:
            pi = RealizationWitness.from_json(data["pi"])
        except RealizationWitnessError as exc:
            raise SchemaError(str(exc)) from exc
        return cls(
            j=SemanticArchive.from_json(data["j"]),
            c=ImplementationConfig.from_json(data["c"]),
            pi=pi,
            wire_schema=data["schema"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "j": self.j.to_json(),
            "c": self.c.to_json(),
            "pi": self.pi.to_json(),
        }

    def rewriter_view(self, *, masked: bool = False) -> dict[str, Any]:
        if masked:
            return _masked_rewriter_view(self)
        return _opaque_normal_rewriter_view(self)


def opaque_replay_abi(
    artifact: Gate1Artifact,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return declaration-order aliases used by both replay packets."""

    parameter_names = {
        parameter.name: parameter.name
        for parameter in artifact.j.signature.parameters
    }
    schedule_names = {
        slot.name: slot.name for slot in artifact.c.slots
    }
    return "scrc_k0", parameter_names, schedule_names


_OPAQUE_EXPRESSION_RESERVED = frozenset(
    {
        "and",
        "or",
        "not",
        "True",
        "False",
        "None",
        "inf",
        "nan",
    }
)


def _intent_result_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _expression_identifiers(value: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end())
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value)
    ]


def _is_expression_function_or_qualified(
    value: str, start: int, end: int
) -> bool:
    before = value[:start].rstrip()
    after = value[end:].lstrip()
    return before.endswith(".") or after.startswith(".") or after.startswith("(")


def _boundary_expression_fields(
    artifact: Gate1Artifact,
) -> list[tuple[str, str]]:
    elaboration = j_elaboration(artifact.j)
    boundary = elaboration.boundary_record
    fields: list[tuple[str, str]] = []
    for index, expression in enumerate(artifact.j.dimension_constraints):
        fields.append((expression, f"j.dimension_constraints[{index}]"))
    domains = boundary.get("domains", {})
    if not isinstance(domains, Mapping):
        raise SchemaError("canonical boundary record has invalid domains")
    for index, expression in enumerate(domains.values()):
        fields.append((expression, f"j.boundary.domains[{index}]"))
    accesses = boundary.get("accesses", [])
    if not isinstance(accesses, list):
        raise SchemaError("canonical boundary record has invalid accesses")
    for index, access in enumerate(accesses):
        if access.get("index_expression") is not None:
            fields.append(
                (
                    access["index_expression"],
                    f"j.boundary.accesses[{index}].index_expression",
                )
            )
        if access.get("predicate") is not None:
            fields.append(
                (access["predicate"], f"j.boundary.accesses[{index}].predicate")
            )
    mappings = boundary.get("output_mappings", [])
    if not isinstance(mappings, list):
        raise SchemaError("canonical boundary record has invalid output mappings")
    for index, mapping in enumerate(mappings):
        fields.append(
            (mapping["value"], f"j.boundary.output_mappings[{index}].value")
        )
    broadcasts = boundary.get("broadcasts", [])
    if not isinstance(broadcasts, list):
        raise SchemaError("canonical boundary record has invalid broadcasts")
    for index, broadcast in enumerate(broadcasts):
        for axis_index, axis in enumerate(broadcast.get("axes", [])):
            if isinstance(axis, str):
                fields.append(
                    (
                        axis,
                        f"j.boundary.broadcasts[{index}].axes[{axis_index}]",
                    )
                )
    reductions = boundary.get("reductions", [])
    if not isinstance(reductions, list):
        raise SchemaError("canonical boundary record has invalid reductions")
    for index, reduction in enumerate(reductions):
        if isinstance(reduction.get("tile_axis"), str):
            fields.append(
                (
                    reduction["tile_axis"],
                    f"j.boundary.reductions[{index}].tile_axis",
                )
            )
    return fields


def _opaque_reference_data(
    artifact: Gate1Artifact,
) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
    """Build declaration-order aliases without exposing the original spelling."""

    from intentir.ir.canonical_j import j_elaboration

    elaboration = j_elaboration(artifact.j)
    construction_records = elaboration.construction_records
    output_bindings = elaboration.output_bindings
    operation_records = construction_records.operation_records
    scope_records = construction_records.scope_records
    boundary_record = elaboration.boundary_record
    boundary_domains = tuple(boundary_record["domains"])
    boundary_accesses = tuple(boundary_record["accesses"])
    boundary_effects = tuple(boundary_record["effects"])
    boundary_authorizations = tuple(
        boundary_record["realization_authorizations"]
    )
    canonical_node_rows = tuple(
        record
        for record in operation_records
        if isinstance(record.get("id"), str)
        and isinstance(record.get("path"), str)
    )
    operation_rows = tuple(
        (record["wire"], record["path"]) for record in operation_records
    )
    expression_fields = _boundary_expression_fields(artifact)
    aliases: dict[str, str] = {}
    owners: dict[str, list[dict[str, str]]] = {}
    alias_order: list[str] = []

    def canonical_handle(prefix: str, index: int) -> str:
        return CanonicalHandleAllocator.canonical_names(prefix, index + 1)[-1]

    def allocate(original: str, stem: str, kind: str, path: str) -> str:
        if original in aliases:
            alias = aliases[original]
        else:
            # This is an opaque *projection* of the accepted construction,
            # not a place to allocate a second namespace.  The old fallback
            # appended ``_1``/``_2`` when a source spelling happened to equal
            # a generated handle.  That made Replay and the masked Gate1
            # prompt observe non-canonical identifiers even though the
            # semantic artifact had already crossed the shared allocator.
            # Canonical handles are intentionally allowed to shadow the
            # hidden spelling; collisions are only possible when the
            # construction has two declarations for one canonical ordinal.
            alias = stem
            if alias in owners:
                raise SchemaError(
                    "canonical opaque projection has duplicate handle "
                    f"{alias!r} at {path}"
                )
            aliases[original] = alias
            owners[alias] = []
            alias_order.append(alias)
        owner = {"kind": kind, "path": path}
        if owner not in owners[alias]:
            owners[alias].append(owner)
        return alias

    tensor_index = 0
    for collection_name, tensors in (
        ("inputs", ((tensor.name, tensor.shape) for tensor in artifact.j.intent.inputs)),
        (
            "outputs",
            ((binding.output, binding.output_type.shape) for binding in output_bindings),
        ),
    ):
        for index, (tensor_name, _tensor_shape) in enumerate(tensors):
            first_declaration = tensor_name not in aliases
            allocate(
                tensor_name,
                f"t{tensor_index}",
                "j_tensor",
                f"j.intent.intent_function.{collection_name}[{index}]",
            )
            if first_declaration:
                tensor_index += 1

    for index, record in enumerate(canonical_node_rows):
        node_id = record["id"]
        path = record["path"]
        alias = allocate(
            node_id,
            canonical_handle("n", index),
            "j_node",
            f"j.intent.{path}",
        )
        aliases[path] = alias

    for index, name in enumerate(boundary_domains):
        allocate(
            name,
            canonical_handle("d", index),
            "boundary_domain",
            f"j.boundary.domains[{index}]",
        )
    for index, access in enumerate(boundary_accesses):
        allocate(
            access["access_id"],
            canonical_handle("a", index),
            "boundary_access",
            f"j.boundary.accesses[{index}]",
        )
    for index, effect in enumerate(boundary_effects):
        allocate(
            effect["effect_id"],
            canonical_handle("e", index),
            "boundary_effect",
            f"j.boundary.effects[{index}]",
        )
    for index, authorization in enumerate(boundary_authorizations):
        allocate(
            authorization["auth_id"],
            canonical_handle("u", index),
            "boundary_realization_authorization",
            f"j.boundary.realization_authorizations[{index}]",
        )

    value_index = 0
    for op, path in operation_rows:
        if isinstance(op.get("scope_id"), str):
            scope_index = next(
                (
                    index
                    for index, scope in enumerate(scope_records)
                    if scope["scope_id"] == op["scope_id"]
                ),
                0,
            )
            allocate(
                op["scope_id"],
                canonical_handle("q", scope_index),
                "j_scope",
                f"j.intent.{path}.scope_id",
            )
        if op.get("op") in {"effect_if", "value_if"}:
            for polarity in ("positive", "negative"):
                arm = op.get(polarity)
                if not isinstance(arm, dict) or not isinstance(
                    arm.get("scope_id"), str
                ):
                    continue
                arm_scope = arm["scope_id"]
                scope_index = next(
                    (
                        index
                    for index, scope in enumerate(scope_records)
                    if scope["scope_id"] == arm_scope
                    ),
                    0,
                )
                allocate(
                    arm_scope,
                    canonical_handle("q", scope_index),
                    "j_scope",
                    f"j.intent.{path}.{polarity}.scope_id",
                )
        for scan_index, scan in enumerate(op.get("scan", [])):
            allocate(
                scan["name"],
                canonical_handle("v", value_index),
                "j_loop_scan_value",
                f"j.intent.{path}.scan[{scan_index}].name",
            )
            value_index += 1
        for state_index, state in enumerate(op.get("carry", [])):
            allocate(
                state["state_id"],
                canonical_handle("s", state_index),
                "j_loop_state",
                f"j.intent.{path}.carry[{state_index}].state_id",
            )
            allocate(
                state["body_argument"],
                canonical_handle("v", value_index),
                "j_loop_body_argument",
                f"j.intent.{path}.carry[{state_index}].body_argument",
            )
            value_index += 1
        for result_index, name in enumerate(_intent_result_names(op.get("result"))):
            allocate(
                name,
                canonical_handle("v", value_index),
                "j_value",
                f"j.intent.{path}.result[{result_index}]",
            )
            value_index += 1

    for index, scope in enumerate(scope_records):
        allocate(
            scope["scope_id"],
            canonical_handle("q", index),
            "j_scope",
            f"j.scopes[{index}]",
        )
    for index, storage in enumerate(artifact.j.storage):
        allocate(
            storage.storage_id,
            canonical_handle("b", index),
            "j_storage",
            f"j.storage[{index}]",
        )

    for index, parameter in enumerate(artifact.j.signature.parameters):
        allocate(
            parameter.name,
            canonical_handle("p", index),
            "j_signature_parameter",
            f"j.signature.parameters[{index}]",
        )

    dimension_index = 0
    boundary_tensors = (
        *(
            (tensor.name, tensor.shape)
            for tensor in artifact.j.intent.inputs
        ),
        *((binding.output, binding.output_type.shape) for binding in output_bindings),
    )
    for tensor_index, (_tensor_name, tensor_shape) in enumerate(boundary_tensors):
        for shape_index, dim in enumerate(tensor_shape):
            for reference in dimension_expression_references(dim):
                if reference in aliases or reference == "?":
                    continue
                allocate(
                    reference,
                    canonical_handle("x", dimension_index),
                    "j_dimension",
                    f"j.intent.tensor_shapes[{tensor_index}][{shape_index}]",
                )
                dimension_index += 1
    for relation_index, relation in enumerate(artifact.j.dimension_relations):
        for field, dim in (
            ("result", relation.result),
            ("arg", relation.expr.arg),
            ("rhs", relation.expr.rhs),
        ):
            if dim is None or dim in aliases:
                continue
            allocate(
                dim,
                canonical_handle("x", dimension_index),
                "j_dimension",
                f"j.dimension_relations[{relation_index}].{field}",
            )
            dimension_index += 1
    for index, slot in enumerate(artifact.c.slots):
        allocate(
            slot.name,
            canonical_handle("c", index),
            "c_slot",
            f"c.slots[{index}]",
        )

    for expression, path in expression_fields:
        for token, start, end in _expression_identifiers(expression):
            if token in aliases or token in _OPAQUE_EXPRESSION_RESERVED:
                continue
            if _is_expression_function_or_qualified(expression, start, end):
                continue
            allocate(
                token,
                canonical_handle("v", value_index),
                "boundary_symbol",
                path,
            )
            value_index += 1

    bindings = [
        {"ref": alias, "owners": copy.deepcopy(owners[alias])}
        for alias in alias_order
    ]
    return aliases, bindings, alias_order


def _rewriter_output_views(
    artifact: Gate1Artifact,
    reference: Callable[[str], str],
    *,
    masked: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project returns and sparse mappings from one canonical output record.

    Rewriter views are transport projections, but they must not become a
    second output constructor.  The accepted J elaboration owns output/return
    value, ABI order, and mapping index; the raw mapping is consulted only for
    its optional wire-preserving ``node_ref``.
    """

    from intentir.ir.canonical_j import j_elaboration

    bindings = j_elaboration(artifact.j).output_bindings
    returns = [
        {
            "output": reference(binding.output),
            "value": "<masked>"
            if masked
            else reference(binding.value),
        }
        for binding in bindings
    ]
    mappings: list[dict[str, Any]] = []
    for binding in bindings:
        index = binding.mapping_index
        if index is None:
            continue
        if not isinstance(index, int) or index < 0:
            raise SchemaError(
                "canonical output record has an invalid mapping index"
            )
        row: dict[str, Any] = {
            "output": reference(binding.output),
            "value": "<masked>"
            if masked
            else reference(binding.value),
        }
        if binding.authored_node_ref is not None:
            row["node_ref"] = reference(binding.authored_node_ref)
        mappings.append(row)
    return returns, mappings


def _opaque_reference(value: str, aliases: dict[str, str]) -> str:
    try:
        return aliases[value]
    except KeyError as exc:
        raise SchemaError(
            f"replay reference {value!r} has no opaque alias"
        ) from exc


def _opaque_expression(value: str, aliases: dict[str, str]) -> str:
    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: aliases.get(match.group(0), match.group(0)),
        value,
    )


def _opaque_json_exact(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, list):
        return [_opaque_json_exact(item, aliases) for item in value]
    if isinstance(value, dict):
        return {
            key: _opaque_json_exact(item, aliases)
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def _opaque_intent_contract_fields(value: Any, aliases: dict[str, str]) -> Any:
    """Rewrite expression-bearing shape/extent fields without changing keys."""

    if isinstance(value, list):
        return [
            _opaque_intent_contract_fields(item, aliases) for item in value
        ]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"shape", "lo", "hi"} and isinstance(item, list):
            result[key] = [
                _opaque_expression(element, aliases)
                if isinstance(element, str) and element != "?"
                else copy.deepcopy(element)
                for element in item
            ]
        else:
            result[key] = _opaque_intent_contract_fields(item, aliases)
    return result


def _opaque_intent_op(
    op: dict[str, Any], aliases: dict[str, str]
) -> dict[str, Any]:
    result = copy.deepcopy(op)
    if "result" in result:
        result["result"] = _opaque_json_exact(result["result"], aliases)
    for key in (
        "operands", "scope_id", "carry", "trip", "scan", "yield",
    ):
        if key in result:
            result[key] = _opaque_json_exact(result[key], aliases)
    for key in ("attrs", "type"):
        if key in result:
            result[key] = _opaque_json_exact(result[key], aliases)
            result[key] = _opaque_intent_contract_fields(result[key], aliases)
    if result.get("op") == "pad" and isinstance(result.get("attrs"), dict):
        for field in ("lo", "hi"):
            values = result["attrs"].get(field)
            if isinstance(values, list):
                result["attrs"][field] = [
                    _opaque_expression(value, aliases)
                    if isinstance(value, str)
                    else value
                    for value in values
                ]
    if "body" in result:
        result["body"] = [
            _opaque_intent_op(item, aliases) for item in result["body"]
        ]
    if result.get("op") in {"effect_if", "value_if"}:
        result["predicate"] = _opaque_reference(
            result["predicate"], aliases
        )
        for polarity in ("positive", "negative"):
            arm = result[polarity]
            arm["scope_id"] = _opaque_reference(arm["scope_id"], aliases)
            arm["body"] = [
                _opaque_intent_op(item, aliases) for item in arm["body"]
            ]
            if "yield_value" in arm:
                arm["yield_value"] = _opaque_reference(
                    arm["yield_value"], aliases
                )
    return result



def _rewriter_scope_views(
    artifact: Gate1Artifact,
    reference: Callable[[str], str],
    *,
    masked: bool,
) -> list[dict[str, Any]]:
    """Project lexical scopes from the canonical allocator output."""

    from intentir.ir.canonical_j import j_elaboration

    result: list[dict[str, Any]] = []
    for scope in j_elaboration(artifact.j).scope_records:
        scope_id = scope.get("scope_id")
        parent_scope = scope.get("parent_scope")
        if not isinstance(scope_id, str) or not (
            parent_scope is None or isinstance(parent_scope, str)
        ):
            raise SchemaError("canonical scope record has invalid ownership")
        result.append(
            {
                "scope_id": reference(scope_id),
                "kind": "<masked>" if masked else scope.get("kind"),
                "parent_scope": (
                    reference(parent_scope) if parent_scope is not None else None
                ),
                "captures": [
                    reference(value) for value in scope.get("captures", [])
                ],
                "definitions": [
                    reference(value) for value in scope.get("definitions", [])
                ],
                "effects": [
                    reference(value) for value in scope.get("effects", [])
                ],
            }
        )
    return result


def _opaque_storage_json(
    storage: StorageObject, aliases: dict[str, str]
) -> dict[str, Any]:
    def event(value: SemanticEventRef) -> dict[str, str]:
        assert value.ref is not None
        return {
            "kind": value.kind.value,
            "ref": _opaque_reference(value.ref, aliases),
        }

    return {
        "storage_id": _opaque_reference(storage.storage_id, aliases),
        "members": [
            _opaque_reference(member, aliases) for member in storage.members
        ],
        "scope_id": _opaque_reference(storage.scope_id, aliases),
        "mutability": storage.mutability.value,
        "init_event": event(storage.init_event),
        "last_use_event": event(storage.last_use_event),
        "escapes": storage.escapes,
        "crosses_iteration": storage.crosses_iteration,
    }


def _opaque_signature_and_c_view(
    artifact: Gate1Artifact, aliases: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project J.signature and complete C without source binding coordinates."""

    signature = artifact.j.signature.to_json()
    for parameter in signature["parameters"]:
        original = parameter["name"]
        parameter["name"] = _opaque_reference(original, aliases)
        if parameter.get("logical_ref") is not None:
            parameter["logical_ref"] = _opaque_reference(
                parameter["logical_ref"], aliases
            )
        if parameter.get("alias_of") is not None:
            parameter["alias_of"] = _opaque_reference(
                parameter["alias_of"], aliases
            )

    c_view = artifact.c.to_json()
    for slot in c_view["slots"]:
        slot["name"] = _opaque_reference(slot["name"], aliases)
    c_view["values"] = {
        _opaque_reference(name, aliases): copy.deepcopy(value)
        for name, value in artifact.c.ordered_value_items()
    }
    return signature, c_view


def opaque_replay_representation(artifact: Gate1Artifact) -> dict[str, Any]:
    """Project closed Source realization facts to the Replay ABI view.

    Replay must not receive the full materialized ``pi`` graph.  This helper
    projects only the typed, source-free facts needed by the target callable
    ABI:

    * an explicit compact access-representation row binds a null-logical-ref
      scalar ABI carrier to one J access/tensor axis;
    * a direct ``program_axes`` row becomes a domain-to-launch-axis binding;
    * a compact ``program_bindings`` row projects each of its named J domains
      to that same explicitly witnessed physical launch axis.

    This projection is deliberately fail-closed: a missing or conflicting
    explicit access binding is not guessed or silently omitted.
    The relation class itself remains a Gate1 normalization witness rather than
    a target recipe; Replay receives only the already-closed domain-to-axis
    projection required to select its callable launch coordinates.
    """

    aliases, _bindings, _order = _opaque_reference_data(artifact)
    elaboration = j_elaboration(artifact.j)
    accesses = elaboration.access_by_id
    access_rows = {
        str(row["access_id"]): row
        for row in elaboration.boundary_rows("accesses")
        if isinstance(row.get("access_id"), str)
    }
    access_order = tuple(item.access_id for item in elaboration.accesses)
    tensors = {
        **{tensor.name: tensor for tensor in artifact.j.intent.inputs},
        **{
            binding.output: binding.output_type
            for binding in elaboration.output_bindings
        },
    }
    parameters = {
        item.name: item for item in artifact.j.signature.parameters
    }

    address_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    access_axes: dict[tuple[str, int], str] = {}
    for binding_index, binding in enumerate(
        artifact.pi.access_representation_bindings
    ):
        relation = binding.get("relation")
        if relation not in {"address_recomputed", "pointer_advanced"}:
            continue
        access_name = binding.get("access_ref")
        carrier_name = binding.get("carrier_ref")
        axis = binding.get("axis_ref")
        access = accesses.get(access_name)
        access_row = access_rows.get(access_name)
        parameter = parameters.get(carrier_name)
        if access is None or access_row is None:
            raise SchemaError(
                "pi.access_representation_bindings["
                f"{binding_index}].access_ref has no J access"
            )
        tensor_name = access_row.get("tensor")
        if not isinstance(tensor_name, str):
            raise SchemaError(
                "canonical J access record has no logical tensor owner"
            )
        if (
            parameter is None
            or parameter.kind is not ParameterKind.SCALAR
            or parameter.logical_ref is not None
        ):
            raise SchemaError(
                "pi.access_representation_bindings["
                f"{binding_index}].carrier_ref must be a null-logical-ref "
                "scalar ABI carrier"
            )
        tensor = tensors.get(tensor_name)
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or axis < 0
            or tensor is None
            or axis >= len(tensor.shape)
        ):
            raise SchemaError(
                "pi.access_representation_bindings["
                f"{binding_index}].axis_ref exceeds the access tensor rank"
            )
        owner_key = (str(access_name), axis)
        previous = access_axes.get(owner_key)
        if previous is not None and previous != carrier_name:
            raise SchemaError(
                "one J access axis has conflicting Replay stride carriers: "
                f"{access_name}[{axis}]"
            )
        access_axes[owner_key] = str(carrier_name)
        row_key = (str(carrier_name), str(access_name), axis)
        address_rows[row_key] = {
            "carrier": _opaque_reference(str(carrier_name), aliases),
            "access_ref": _opaque_reference(str(access_name), aliases),
            "tensor": _opaque_reference(str(tensor_name), aliases),
            "axis": axis,
            "role": "stride",
        }

    program_rows: dict[str, dict[str, Any]] = {}
    domain_names = set(elaboration.boundary_domains)
    def add_program_row(domain_name: Any, axis: Any, *, path: str) -> None:
        if domain_name not in domain_names:
            raise SchemaError(
                f"{path} references an unknown J domain"
            )
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 2:
            raise SchemaError(
                f"{path} has an invalid launch axis"
            )
        existing = program_rows.get(domain_name)
        if existing is not None:
            if existing["axis"] != axis:
                raise SchemaError(
                    f"{path} maps one J domain to multiple launch axes"
                )
            return
        program_rows[domain_name] = {
            "domain_ref": _opaque_reference(domain_name, aliases),
            "axis": axis,
        }

    try:
        program_projection = canonical_program_coordinate_projection(artifact.pi)
    except ValueError as exc:
        # The typed pi decoder normally makes this unreachable.  Keep the
        # replay projection fail-closed if a future typed witness gains a new
        # representation without updating this shared consumer.
        raise SchemaError(str(exc)) from exc
    for row in program_projection.rows:
        for domain_ref, _axis_ref in row.logical_axes:
            add_program_row(
                domain_ref,
                row.physical_axis,
                path=f"pi.{row.family}",
            )

    if not address_rows and not program_rows:
        return {}

    return {
        "address_bindings": [
            copy.deepcopy(address_rows[key])
            for key in sorted(
                address_rows,
                key=lambda key: (
                    access_order.index(key[1]),
                    key[2],
                    (
                        int(key[0][1:])
                        if re.fullmatch(r"p[0-9]+", key[0]) is not None
                        else key[0]
                    ),
                ),
            )
        ],
        "program_bindings": sorted(
            program_rows.values(),
            key=lambda row: (
                row["axis"],
                int(row["domain_ref"][1:])
                if re.fullmatch(r"d[0-9]+", row["domain_ref"]) is not None
                else row["domain_ref"],
            ),
        ),
    }


def _opaque_normal_rewriter_view(artifact: Gate1Artifact) -> dict[str, Any]:
    """Preserve J/B semantics while removing source-derived identifier hints."""

    aliases, _bindings, _ = _opaque_reference_data(artifact)
    elaboration = j_elaboration(artifact.j)
    boundary = Boundary.from_json(copy.deepcopy(dict(elaboration.boundary_record)))
    rewriter_returns, rewriter_mappings = _rewriter_output_views(
        artifact,
        lambda value: _opaque_reference(value, aliases),
        masked=False,
    )
    rewriter_scopes = _rewriter_scope_views(
        artifact,
        lambda value: _opaque_reference(value, aliases),
        masked=False,
    )
    construction_records = elaboration.construction_records
    operation_records = construction_records.operation_records
    top_level_operations = [
        record["wire"]
        for record in operation_records
        if re.fullmatch(r"body\[\d+\]", record["path"])
    ]
    opaque_intent = {
        "intent_function": {
            "name": "scrc_intent0",
            "inputs": [
                {
                    "name": _opaque_reference(tensor.name, aliases),
                    "dtype": tensor.dtype,
                    "shape": [
                        _opaque_expression(dim, aliases)
                        if isinstance(dim, str) and dim != "?"
                        else dim
                        for dim in tensor.shape
                    ],
                }
                for tensor in artifact.j.intent.inputs
            ],
            "outputs": [
                {
                    "name": _opaque_reference(binding.output, aliases),
                    "dtype": binding.output_type.dtype,
                    "shape": [
                        _opaque_expression(dim, aliases)
                        if isinstance(dim, str) and dim != "?"
                        else dim
                        for dim in binding.output_type.shape
                    ],
                }
                for binding in j_elaboration(artifact.j).output_bindings
            ],
            "returns": rewriter_returns,
            "body": [
                _opaque_intent_op(op, aliases) for op in top_level_operations
            ],
        }
    }

    opaque_boundary = {
        "domains": {
            _opaque_reference(name, aliases): _opaque_expression(expression, aliases)
            for name, expression in boundary.domains.items()
        },
        "accesses": [
            {
                "access_id": _opaque_reference(access.access_id, aliases),
                "kind": access.kind.value,
                "tensor": _opaque_reference(access.tensor, aliases),
                "index_expression": (
                    _opaque_expression(access.index_expression, aliases)
                    if access.index_expression is not None
                    else None
                ),
                **(
                    {"predicate": _opaque_expression(access.predicate, aliases)}
                    if access.predicate is not None
                    else {}
                ),
                **(
                    {"fill_value": copy.deepcopy(access.fill_value)}
                    if access.fill_value is not None
                    else {}
                ),
                **(
                    {"node_ref": _opaque_reference(access.node_ref, aliases)}
                    if access.node_ref is not None
                    else {}
                ),
            }
            for access in boundary.accesses
        ],
        "broadcasts": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "operand": _opaque_reference(item.operand, aliases),
                "axes": [
                    _opaque_expression(axis, aliases)
                    if isinstance(axis, str)
                    else axis
                    for axis in item.axes
                ],
            }
            for item in boundary.broadcasts
        ],
        "aliases": [
            {
                "left": _opaque_reference(item.left, aliases),
                "right": _opaque_reference(item.right, aliases),
                "kind": item.kind.value,
            }
            for item in boundary.aliases
        ],
        "effects": [
            {
                "effect_id": _opaque_reference(item.effect_id, aliases),
                "kind": item.kind.value,
                "target": _opaque_reference(item.target, aliases),
                **(
                    {"node_ref": _opaque_reference(item.node_ref, aliases)}
                    if item.node_ref is not None
                    else {}
                ),
                "access_ref": (
                    _opaque_reference(item.access_ref, aliases)
                    if item.access_ref is not None else None
                ),
            }
            for item in boundary.effects
        ],
        "output_mappings": rewriter_mappings,
        "reductions": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "tile_axis": (
                    _opaque_expression(item.tile_axis, aliases)
                    if isinstance(item.tile_axis, str)
                    else item.tile_axis
                ),
                **(
                    {"identity": copy.deepcopy(item.identity)}
                    if item.identity is not None
                    else {}
                ),
            }
            for item in boundary.reductions
        ],
        "index_bounds": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "data": _opaque_reference(item.data, aliases),
                "indices": _opaque_reference(item.indices, aliases),
                "axis": item.axis,
            }
            for item in boundary.index_bounds
        ],
        "updates": [
            {
                "node_ref": _opaque_reference(item.node_ref, aliases),
                "base": _opaque_reference(item.base, aliases),
                "indices": _opaque_reference(item.indices, aliases),
                "values": _opaque_reference(item.values, aliases),
                **(
                    {"compare": _opaque_reference(item.compare, aliases)}
                    if item.compare is not None
                    else {}
                ),
                **(
                    {"observed": _opaque_reference(item.observed, aliases)}
                    if item.observed is not None
                    else {}
                ),
                "axis": item.axis,
                "combiner": item.combiner.value,
                "duplicate_policy": item.duplicate_policy.value,
                "order": item.order.value,
            }
            for item in boundary.updates
        ],
        "result_relations": [
            {
                "output": _opaque_reference(item.output, aliases),
                "kind": item.kind.value,
                **(
                    {
                        "update_node_ref": _opaque_reference(
                            item.update_node_ref, aliases
                        )
                    }
                    if item.update_node_ref is not None
                    else {}
                ),
            }
            for item in boundary.result_relations
        ],
        "realization_authorizations": [
            {
                "auth_id": _opaque_reference(item.auth_id, aliases),
                "kind": item.kind.value,
                "subject_ref": _opaque_reference(item.subject_ref, aliases),
                **(
                    {
                        "source_dtype": item.source_dtype,
                        "target_dtype": item.target_dtype,
                    }
                    if item.source_dtype is not None
                    else {}
                ),
            }
            for item in boundary.realization_authorizations
        ],
        "consumer_extensions": [
            {
                "consumer_ref": _opaque_reference(item.consumer_ref, aliases),
                "operand_index": item.operand_index,
                "access_ref": _opaque_reference(item.access_ref, aliases),
                "tile_axis": item.tile_axis,
                "outside": item.outside.to_json(),
                "identity_law": item.identity_law,
            }
            for item in boundary.consumer_extensions
        ],
        "access_guard_relations": [
            {
                "access_ref": _opaque_reference(item.access_ref, aliases),
                "value_ref": _opaque_reference(item.value_ref, aliases),
                "axes": list(item.axes),
                "polarity": item.polarity.value,
            }
            for item in boundary.access_guard_relations
        ],
    }

    signature, c_view = _opaque_signature_and_c_view(artifact, aliases)

    return {
        "j": {
            "kernel_id": "scrc_j0",
            "intent": opaque_intent,
            "node_ids": {
                _opaque_reference(node_id, aliases): path
                for node_id, path in artifact.j.node_ids.items()
            },
            "boundary": opaque_boundary,
            "signature": signature,
            "scopes": rewriter_scopes,
            "storage": [
                _opaque_storage_json(storage, aliases)
                for storage in artifact.j.storage
            ],
            "dimension_constraints": [
                _opaque_expression(expression, aliases)
                for expression in artifact.j.dimension_constraints
            ],
            "value_dimension_relations": [
                {
                    "value_ref": _opaque_reference(item.value_ref, aliases),
                    "dimension_ref": _opaque_reference(
                        item.dimension_ref, aliases
                    ),
                    "relation": item.relation,
                }
                for item in artifact.j.value_dimension_relations
            ],
            "dimension_relations": [
                {
                    "result": _opaque_reference(item.result, aliases),
                    "expr": {
                        **item.expr.to_json(),
                        "arg": _opaque_reference(item.expr.arg, aliases),
                        **(
                            {"rhs": _opaque_reference(item.expr.rhs, aliases)}
                            if item.expr.rhs is not None
                            else {}
                        ),
                    },
                }
                for item in artifact.j.dimension_relations
            ],
        },
        "c": c_view,
    }


def _rename_identifier_tokens(value: str, aliases: dict[str, str]) -> str:
    """Alpha-rename complete Python-like identifier tokens in an expression."""

    return re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda match: aliases.get(match.group(0), match.group(0)),
        value,
    )


def _masked_rewriter_view(artifact: Gate1Artifact) -> dict[str, Any]:
    """Create a consistently opaque J/pi/interface view for leakage ablation."""

    from intentir.ir.canonical_j import j_elaboration

    elaboration = j_elaboration(artifact.j)
    construction_records = elaboration.construction_records
    scope_records = elaboration.scope_records
    boundary = Boundary.from_json(copy.deepcopy(dict(elaboration.boundary_record)))
    # The masked prompt is diagnostic input, but its aliases must still be a
    # projection of the accepted canonical construction.  In particular,
    # dict insertion order in ``j.node_ids`` is not an operation traversal
    # contract and must not allocate a second n* namespace.
    operation_records = elaboration.operation_records
    canonical_node_ids = tuple(
        record["id"]
        for record in operation_records
        if isinstance(record.get("id"), str)
    )
    access_rows = elaboration.boundary_rows("accesses")
    effect_rows = elaboration.boundary_rows("effects")
    authorization_rows = elaboration.boundary_rows("realization_authorizations")
    if (
        len(canonical_node_ids) != len(artifact.j.node_ids)
        or len(access_rows) != len(boundary.accesses)
        or len(effect_rows) != len(boundary.effects)
        or len(authorization_rows) != len(boundary.realization_authorizations)
    ):
        raise SchemaError(
            "canonical construction and decoded J declarations disagree"
        )

    def canonical_handles(prefix: str, count: int) -> tuple[str, ...]:
        return CanonicalHandleAllocator.canonical_names(prefix, count)

    tensor_names = [tensor.name for tensor in artifact.j.intent.inputs]
    tensor_names.extend(tensor.name for tensor in artifact.j.intent.outputs)
    tensor_map = {
        name: canonical_handles("t", len(dict.fromkeys(tensor_names)))[index]
        for index, name in enumerate(dict.fromkeys(tensor_names))
    }
    node_aliases = canonical_handles("n", len(canonical_node_ids))
    node_map = {
        name: node_aliases[index]
        for index, name in enumerate(canonical_node_ids)
    }
    access_ids = tuple(
        row["access_id"]
        for row in access_rows
        if isinstance(row.get("access_id"), str)
    )
    access_aliases = canonical_handles("a", len(access_ids))
    access_map = {
        access_id: access_aliases[index]
        for index, access_id in enumerate(access_ids)
    }
    domain_names = tuple(boundary.domains)
    domain_aliases = canonical_handles("d", len(domain_names))
    domain_map = {
        name: domain_aliases[index]
        for index, name in enumerate(domain_names)
    }
    effect_ids = tuple(
        row["effect_id"]
        for row in effect_rows
        if isinstance(row.get("effect_id"), str)
    )
    effect_aliases = canonical_handles("e", len(effect_ids))
    effect_map = {
        effect_id: effect_aliases[index]
        for index, effect_id in enumerate(effect_ids)
    }
    authorization_ids = tuple(
        row["auth_id"]
        for row in authorization_rows
        if isinstance(row.get("auth_id"), str)
    )
    authorization_aliases = canonical_handles(
        "u", len(authorization_ids)
    )
    authorization_map = {
        auth_id: authorization_aliases[index]
        for index, auth_id in enumerate(authorization_ids)
    }
    scope_aliases = canonical_handles("q", len(scope_records))
    scope_map = {
        scope["scope_id"]: scope_aliases[index]
        for index, scope in enumerate(scope_records)
    }
    storage_aliases = canonical_handles("b", len(artifact.j.storage))
    storage_map = {
        storage.storage_id: storage_aliases[index]
        for index, storage in enumerate(artifact.j.storage)
    }
    dimension_names = list(
        dict.fromkeys(
            dimension
            for tensor in [*artifact.j.intent.inputs, *artifact.j.intent.outputs]
            for dimension in tensor.shape
            if isinstance(dimension, str) and dimension != "?"
        )
    )
    dimension_map = {
        name: canonical_handles("x", len(dimension_names))[index]
        for index, name in enumerate(dimension_names)
    }

    opaque_function, parameter_map, schedule_map = opaque_replay_abi(artifact)
    abi_map = {**parameter_map, **schedule_map}

    reference_map = {
        **tensor_map, **node_map, **access_map, **domain_map, **effect_map,
        **authorization_map,
        **scope_map, **storage_map, **dimension_map,
    }
    fallback_reference_map: dict[str, str] = {}

    def renamed(value: str) -> str:
        if value in reference_map:
            return reference_map[value]
        # Replace exact identifier components in qualified references while
        # leaving positional syntax (e.g. body[2]) intact.
        parts = re.split(r"([.\[\]/:])", value)
        return "".join(reference_map.get(part, part) for part in parts)

    def opaque_reference(value: str) -> str:
        mapped = renamed(value)
        if mapped != value or value in reference_map:
            return mapped
        if value in abi_map:
            return abi_map[value]
        if value not in fallback_reference_map:
            fallback_reference_map[value] = f"r{len(fallback_reference_map)}"
        return fallback_reference_map[value]

    path_to_node = {
        record["path"]: node_map[record["id"]]
        for record in operation_records
        if isinstance(record.get("path"), str)
        and isinstance(record.get("id"), str)
    }
    top_level_operations = [
        record["wire"]
        for record in operation_records
        if re.fullmatch(r"body\[\d+\]", record["path"])
    ]

    def mask_ops(
        ops: Sequence[Mapping[str, Any]], prefix: str
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, op in enumerate(ops):
            path = f"{prefix}[{index}]"
            node_id = path_to_node.get(path)
            if node_id is None:
                node_id = path_to_node.get(f"intent_function.{path}")
            if node_id is None:
                # A complete accepted construction has one n* row for every
                # lexical operation.  Do not invent an ``n_anon_*`` diagnostic
                # namespace when the path table disagrees: that would make a
                # masked Gate1 projection appear structurally complete while
                # silently using a second node identity authority.
                raise SchemaError(
                    "canonical masked projection has no node identity for "
                    f"operation path {path!r}"
                )
            masked_op: dict[str, Any] = {"op": "<masked>", "result": node_id}
            body = op.get("body")
            if isinstance(body, list) and body:
                masked_op["body"] = mask_ops(body, f"{path}.body")
            if op.get("op") in {"effect_if", "value_if"}:
                positive = op.get("positive")
                negative = op.get("negative")
                if not isinstance(positive, Mapping) or not isinstance(
                    negative, Mapping
                ):
                    raise SchemaError(
                        "canonical conditional operation lacks both Replay arms"
                    )
                masked_op["predicate"] = "<masked>"
                masked_op["positive"] = {
                    "scope_id": scope_map[positive["scope_id"]],
                    "body": mask_ops(
                        positive.get("body", []),
                        f"{path}.positive.body",
                    ),
                }
                masked_op["negative"] = {
                    "scope_id": scope_map[negative["scope_id"]],
                    "body": mask_ops(
                        negative.get("body", []),
                        f"{path}.negative.body",
                    ),
                }
                if op.get("op") == "value_if":
                    operation_type = op.get("type")
                    if not isinstance(operation_type, Mapping):
                        raise SchemaError(
                            "masked value_if lacks a canonical result type"
                        )
                    masked_op["type"] = {
                        "dtype": "<masked>",
                        "shape": ["<masked>"]
                        * len(operation_type.get("shape", [])),
                    }
                    masked_op["positive"]["yield_value"] = opaque_reference(
                        positive["yield_value"]
                    )
                    masked_op["negative"]["yield_value"] = opaque_reference(
                        negative["yield_value"]
                    )
            result.append(masked_op)
        return result

    rewriter_returns, rewriter_mappings = _rewriter_output_views(
        artifact,
        opaque_reference,
        masked=True,
    )
    rewriter_scopes = _rewriter_scope_views(
        artifact,
        opaque_reference,
        masked=True,
    )
    masked_intent = {
        "intent_function": {
            "name": opaque_function,
            "inputs": [
                {
                    "name": tensor_map[tensor.name],
                    "dtype": "<masked>",
                    "shape": ["<masked>"] * len(tensor.shape),
                }
                for tensor in artifact.j.intent.inputs
            ],
            "outputs": [
                {
                    "name": tensor_map[tensor.name],
                    "dtype": "<masked>",
                    "shape": ["<masked>"] * len(tensor.shape),
                }
                for tensor in artifact.j.intent.outputs
            ],
            "returns": rewriter_returns,
            "body": mask_ops(top_level_operations, "body"),
        }
    }

    masked_boundary = {
        "domains": {domain_map[name]: "<masked>" for name in boundary.domains},
        "accesses": [
            {
                "access_id": access_map[access.access_id],
                "kind": "<masked>",
                "tensor": opaque_reference(access.tensor),
                "index_expression": "<masked>",
                **({"predicate": "<masked>"} if access.predicate is not None else {}),
                **({"fill_value": "<masked>"} if access.fill_value is not None else {}),
                **(
                    {"node_ref": opaque_reference(access.node_ref)}
                    if access.node_ref is not None
                    else {}
                ),
            }
            for access in boundary.accesses
        ],
        "broadcasts": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "operand": opaque_reference(item.operand),
                "axes": ["<masked>"],
            }
            for item in boundary.broadcasts
        ],
        "aliases": [
            {
                "left": opaque_reference(item.left),
                "right": opaque_reference(item.right),
                "kind": "<masked>",
            }
            for item in boundary.aliases
        ],
        "effects": [
            {
                "effect_id": effect_map[item.effect_id],
                "kind": "<masked>",
                "target": opaque_reference(item.target),
                **(
                    {"node_ref": opaque_reference(item.node_ref)}
                    if item.node_ref is not None
                    else {}
                ),
                "access_ref": (
                    opaque_reference(item.access_ref)
                    if item.access_ref is not None else None
                ),
            }
            for item in boundary.effects
        ],
        "output_mappings": rewriter_mappings,
        "reductions": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "tile_axis": "<masked>",
                **({"identity": "<masked>"} if item.identity is not None else {}),
            }
            for item in boundary.reductions
        ],
        "index_bounds": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "data": opaque_reference(item.data),
                "indices": opaque_reference(item.indices),
                "axis": "<masked>",
            }
            for item in boundary.index_bounds
        ],
        "updates": [
            {
                "node_ref": opaque_reference(item.node_ref),
                "base": opaque_reference(item.base),
                "indices": opaque_reference(item.indices),
                "values": opaque_reference(item.values),
                **(
                    {"compare": opaque_reference(item.compare)}
                    if item.compare is not None
                    else {}
                ),
                **(
                    {"observed": opaque_reference(item.observed)}
                    if item.observed is not None
                    else {}
                ),
                "axis": "<masked>",
                "combiner": "<masked>",
                "duplicate_policy": "<masked>",
                "order": "<masked>",
            }
            for item in boundary.updates
        ],
        "result_relations": [
            {
                "output": opaque_reference(item.output),
                "kind": "<masked>",
                **(
                    {
                        "update_node_ref": opaque_reference(
                            item.update_node_ref
                        )
                    }
                    if item.update_node_ref is not None
                    else {}
                ),
            }
            for item in boundary.result_relations
        ],
        "realization_authorizations": [
            {
                "auth_id": authorization_map[item.auth_id],
                "kind": "<masked>",
                "subject_ref": opaque_reference(item.subject_ref),
                **(
                    {
                        "source_dtype": "<masked>",
                        "target_dtype": "<masked>",
                    }
                    if item.source_dtype is not None
                    else {}
                ),
            }
            for item in boundary.realization_authorizations
        ],
        "consumer_extensions": [
            {
                "consumer_ref": opaque_reference(item.consumer_ref),
                "operand_index": item.operand_index,
                "access_ref": opaque_reference(item.access_ref),
                "tile_axis": "<masked>",
                "outside": {"dtype": "<masked>", "value": "<masked>"},
                "identity_law": "<masked>",
            }
            for item in boundary.consumer_extensions
        ],
        "access_guard_relations": [
            {
                "access_ref": opaque_reference(item.access_ref),
                "value_ref": opaque_reference(item.value_ref),
                "axes": ["<masked>"],
                "polarity": "<masked>",
            }
            for item in boundary.access_guard_relations
        ],
    }

    # The diagnostic arm masks J's semantic declarations while retaining the
    # same logical signature and complete implementation configuration C.
    normal_aliases, _bindings, _order = _opaque_reference_data(artifact)
    opaque_signature, opaque_c = _opaque_signature_and_c_view(
        artifact, normal_aliases
    )
    return {
        "j": {
            "kernel_id": opaque_function,
            "intent": masked_intent,
            "node_ids": {
                node_map[node_id]: f"node[{index}]"
                for index, node_id in enumerate(canonical_node_ids)
            },
            "boundary": masked_boundary,
            "signature": opaque_signature,
            "scopes": rewriter_scopes,
            "storage": [
                {
                    "storage_id": storage_map[item.storage_id],
                    "members": [opaque_reference(ref) for ref in item.members],
                    "scope_id": scope_map[item.scope_id],
                    "mutability": "<masked>",
                    "init_event": {
                        "kind": "<masked>",
                        "ref": opaque_reference(item.init_event.ref),
                    },
                    "last_use_event": {
                        "kind": "<masked>",
                        "ref": opaque_reference(item.last_use_event.ref),
                    },
                    "escapes": "<masked>",
                    "crosses_iteration": "<masked>",
                }
                for item in artifact.j.storage
            ],
            "dimension_constraints": [
                "<masked>" for _ in artifact.j.dimension_constraints
            ],
            "value_dimension_relations": [
                {
                    "value_ref": opaque_reference(item.value_ref),
                    "dimension_ref": opaque_reference(item.dimension_ref),
                    "relation": "<masked>",
                }
                for item in artifact.j.value_dimension_relations
            ],
            "dimension_relations": [
                {
                    "result": opaque_reference(item.result),
                    "expr": {
                        "op": "<masked>",
                        "arg": opaque_reference(item.expr.arg),
                        **(
                            {"rhs": opaque_reference(item.expr.rhs)}
                            if item.expr.rhs is not None
                            else {}
                        ),
                        **(
                            {"value": "<masked>"}
                            if item.expr.value is not None
                            else {}
                        ),
                    },
                }
                for item in artifact.j.dimension_relations
            ],
        },
        "c": opaque_c,
    }


__all__ = [
    "AccessGuardPolarity", "AccessGuardRelation", "AccessKind",
    "AliasKind", "AliasSpec",
    "BOUNDARY_ACCESS_REQUIRED_FIELDS", "BOUNDARY_ACCESS_WIRE_FIELDS",
    "BOUNDARY_WIRE_FIELDS", "Boundary",
    "BoundaryAccess", "BroadcastSpec", "CompileCase", "ConsumerExtension",
    "DuplicatePolicy", "EffectKind",
    "EffectSpec", "Frontend",
    "GATE1_ARTIFACT_MODEL_FIELDS", "GATE1_ARTIFACT_PRIVATE_FIELDS",
    "LEGACY_NESTED_J_FIELDS",
    "Gate1Artifact", "Gate1Status", "ImplementationConfig",
    "gate1_artifact_wire_diagnostics",
    "implementation_value_matches",
    "JNecessityStatus", "JsonValue",
    "IndexBoundsSpec", "LocationEntry", "LocationMap", "LogicalSignature",
    "LogicalUpdateSpec", "OutputMapping",
    "opaque_replay_abi",
    "ParameterKind", "ParameterMode", "ProvenanceRecord", "ProvenanceRole",
    "PortableScope", "RealizationAuthorization", "RealizationWitness",
    "ReductionBoundary", "RefinementKind",
    "ResultRelationKind", "ResultRelationSpec",
    "render_boundary_access_wire_grammar",
    "render_gate1_artifact_envelope_grammar", "RuntimeArgument",
    "ScheduleSlot", "ScheduleSlotKind", "SchemaError", "SignatureParameter",
    "SemanticArchiveCrossContractError",
    "ScopeKind",
    "opaque_replay_representation",
    "SEMANTIC_ARCHIVE_WIRE_FIELDS", "SemanticArchive", "ValueDimensionRelation",
    "DimensionRelation", "DimensionRelationExpr",
    "SemanticEventKind",
    "SemanticEventRef", "SourceSpan", "StorageMutability", "StorageObject",
    "StrictJSONMixin", "UpdateCombiner", "UpdateOrder",
    "TypedOutsideValue",
]
