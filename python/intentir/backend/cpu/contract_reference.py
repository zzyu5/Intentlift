"""J-only reference execution for relation-indexed pointwise graphs.

The ordinary Intent executor evaluates Layer A over complete logical tensors.
That is insufficient when Layer B explicitly gives one consumer a different
logical access view (for example, a sub-domain or an affine slice) and later
assembles an output through several stores.  This module lowers that closed
case from :class:`CanonicalContractIR` instead of recognizing kernel shapes.

The profile is intentionally conservative.  It supports flat, acyclic,
shape-preserving pointwise graphs, arbitrary typed ContractExpr indices and
predicates, explicit masked-load fills, and exhaustive non-overlapping output
stores.  Unsupported control, reduction, alias, or update relations fail
closed.  C, pi, Source, Replay, Gate observations, and kernel names are never
inputs to either projection or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from intentir.backend.cpu.abi import (
    CPUBackendUnsupportedError,
    numpy_dtype,
    validate_array,
)
from intentir.backend.cpu.pipeline import CPUExecutionResult
from intentir.backend.cpu.structured_reference import (
    _Evaluator,
    _resolved_shape,
)
from intentir.ir.canonical_j import (
    CanonicalValueType,
    ContractExpr,
    ContractSort,
    canonical_top_level_operations,
    canonical_broadcast_axes,
    direct_unit_dimension_refs,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.intent_op_contract import (
    _broadcast_projection,
    operation_attr_contract,
)
from intentir.ir.contract_ir import (
    CanonicalContractIR,
    ContractAccessIR,
    ContractIRProjectionError,
    project_contract_ir,
)
from intentir.ir.contract_semantics import ContractSemanticsError, TypedContractProgram
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.functional_boundary import (
    access_predicates_are_logical_domain_total,
    project_predicate_text,
)
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.ir.model import IntentOp

if TYPE_CHECKING:
    from intentir.verify.gate1.model import SemanticArchive


CONTRACT_REFERENCE_PROTOCOL = "intentir.j-contract-indexed-cpu-reference.v3"
_CONCAT_ATTR_CONTRACT = operation_attr_contract("concat")


class ContractReferenceUnsupportedError(CPUBackendUnsupportedError):
    """The accepted J is outside the closed relation-indexed profile."""


def _canonical_body(j: "SemanticArchive") -> tuple[IntentOp, ...]:
    """Return the top-level execution sequence from construction records.

    The typed rows preserve the original nested operation objects, so this is
    an execution view of the same canonical tree rather than a second body
    deserialization.  Keeping this accessor local to the reference backend
    prevents ``SemanticArchive.intent.body`` from becoming an accidental
    operation-order authority.
    """

    try:
        return canonical_top_level_operations(j)
    except (TypeError, ValueError) as exc:
        raise ContractReferenceUnsupportedError(
            "contract reference requires canonical typed operation records"
        ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _result_names(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(item for item in operation.result if isinstance(item, str))
    return ()


def _rectangular_domains(j: "SemanticArchive") -> dict[str, int | str]:
    """Return only explicitly authored zero-based rectangular domain extents."""

    domains: dict[str, int | str] = {}
    for domain_id, expression in j_elaboration(j).domains:
        try:
            projection = project_predicate_text(expression.to_text())
        except (NotImplementedError, ValueError) as exc:
            raise ContractReferenceUnsupportedError(
                "contract reference requires zero-based rectangular domains; "
                f"unsupported domain {domain_id!r}"
            ) from exc
        if projection.guards:
            raise ContractReferenceUnsupportedError(
                "contract reference does not implement guard-bearing domains"
            )
        upper_axes = {axis for axis, _extent in projection.upper_bounds}
        if upper_axes != set(projection.lower_axes):
            raise ContractReferenceUnsupportedError(
                "contract reference requires matching zero lower bounds"
            )
        for axis, extent in projection.upper_bounds:
            previous = domains.get(axis)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"contract domain axis {axis!r} has conflicting extents"
                )
            domains[axis] = extent
    if not domains:
        raise ContractReferenceUnsupportedError(
            "contract reference requires at least one authored logical domain"
        )
    return domains


@dataclass(frozen=True)
class IndexedLoadProjection:
    access: ContractAccessIR
    operand_index: int
    result_ref: str
    result_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_operand_view",
            "access": self.access.to_json(),
            "operand_index": self.operand_index,
            "result_ref": self.result_ref,
            "result_axes": list(self.result_axes),
        }


@dataclass(frozen=True)
class IndexedStoreProjection:
    access: ContractAccessIR
    value_ref: str
    value_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_output_store",
            "access": self.access.to_json(),
            "value_ref": self.value_ref,
            "value_axes": list(self.value_axes),
        }


@dataclass(frozen=True)
class IndexedOutputAssembly:
    """One output whose authored concat is realized by exact indexed stores."""

    output: str
    result_ref: str
    axis: int
    operand_refs: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_concat_output_assembly",
            "output": self.output,
            "result_ref": self.result_ref,
            "axis": self.axis,
            "operand_refs": list(self.operand_refs),
        }


@dataclass(frozen=True)
class IndexedResultLoadProjection:
    """One gather result whose exact masked access view is authored in Layer B."""

    access: ContractAccessIR
    result_ref: str
    result_axes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "proof": "typed_indexed_result_view",
            "access": self.access.to_json(),
            "result_ref": self.result_ref,
            "result_axes": list(self.result_axes),
        }


@dataclass(frozen=True)
class ContractReferencePlan:
    """Ephemeral source-free lowering plan derived one-to-one from J."""

    contract_ir: CanonicalContractIR
    domains: tuple[tuple[str, int | str], ...]
    loads: tuple[IndexedLoadProjection, ...]
    result_loads: tuple[IndexedResultLoadProjection, ...]
    stores: tuple[IndexedStoreProjection, ...]
    output_assemblies: tuple[IndexedOutputAssembly, ...]
    native_index_bounds: tuple[tuple[str, str, str, int], ...]
    broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]

    @property
    def domain_map(self) -> dict[str, int | str]:
        return dict(self.domains)

    @property
    def load_map(self) -> dict[tuple[str, int], IndexedLoadProjection]:
        return {
            (item.result_ref, item.operand_index): item
            for item in self.loads
        }

    @property
    def result_load_map(self) -> dict[str, IndexedResultLoadProjection]:
        return {item.result_ref: item for item in self.result_loads}

    @property
    def stores_by_output(self) -> dict[str, tuple[IndexedStoreProjection, ...]]:
        grouped: dict[str, list[IndexedStoreProjection]] = {}
        for item in self.stores:
            grouped.setdefault(item.access.tensor, []).append(item)
        return {name: tuple(items) for name, items in grouped.items()}

    @property
    def output_assembly_map(self) -> dict[str, IndexedOutputAssembly]:
        return {item.output: item for item in self.output_assemblies}

    @property
    def broadcast_map(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            result: dict(operands) for result, operands in self.broadcast_axes
        }

    def to_json(self) -> dict[str, Any]:
        record = {
            "protocol": CONTRACT_REFERENCE_PROTOCOL,
            "profile": "relation_indexed_graph_v2",
            "contract_ir_digest": self.contract_ir.digest,
            "domains": [list(item) for item in self.domains],
            "loads": [item.to_json() for item in self.loads],
            "result_loads": [item.to_json() for item in self.result_loads],
            "stores": [item.to_json() for item in self.stores],
            "output_assemblies": [
                item.to_json() for item in self.output_assemblies
            ],
            "native_index_bounds": [
                {
                    "node_ref": node_ref,
                    "data": data,
                    "indices": indices,
                    "axis": axis,
                }
                for node_ref, data, indices, axis in self.native_index_bounds
            ],
            "broadcast_axes": [
                [
                    result,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for result, operands in self.broadcast_axes
            ],
        }
        return {**record, "plan_digest": _digest(record)}


_RELATION_INDEXED_OPS = frozenset(
    {
        "cast",
        "concat",
        "const",
        "gather",
        "iota",
        "map_binary",
        "map_unary",
        "select",
    }
)


def project_contract_reference(j: "SemanticArchive") -> ContractReferencePlan:
    """Project a closed relation-indexed pointwise plan from canonical J only."""

    canonical = elaborate_j(j)
    boundary = canonical.boundary
    if (
        boundary.aliases
        or boundary.reductions
        or boundary.updates
        or boundary.consumer_extensions
        or boundary.access_guard_relations
    ):
        raise ContractReferenceUnsupportedError(
            "contract reference v1 does not implement alias, reduction, update, "
            "consumer-extension, or access-guard relations"
        )
    operations = canonical_top_level_operations(canonical)
    top_level_rows = tuple(
        (path, operation)
        for path, operation in j_elaboration(canonical).typed_operation_rows
        if path.startswith("body[") and "." not in path
    )
    if any(
        operation.body or operation.op not in _RELATION_INDEXED_OPS
        for operation in operations
    ):
        raise ContractReferenceUnsupportedError(
            "contract reference v2 requires a flat relation-indexed Layer-A graph"
        )
    if any(len(_result_names(operation)) != 1 for operation in operations):
        raise ContractReferenceUnsupportedError(
            "contract reference v1 requires one result per Layer-A operation"
        )

    operations_by_path = {
        path: operation for path, operation in top_level_rows
    }
    node_refs_by_path: dict[str, str] = {}
    for node_ref, path in canonical.node_ids.items():
        if path not in operations_by_path:
            continue
        previous = node_refs_by_path.setdefault(path, node_ref)
        if previous != node_ref:
            raise ValueError("contract reference has duplicate authored node paths")
    if set(node_refs_by_path) != set(operations_by_path):
        raise ValueError("contract reference requires one authored id per flat node")
    nodes = {
        node_refs_by_path[path]: operation
        for path, operation in operations_by_path.items()
    }
    value_types = j_elaboration(canonical).value_type_by_name
    domains = _rectangular_domains(canonical)
    unit_dimensions = direct_unit_dimension_refs(canonical.dimension_constraints)
    result_axes: dict[str, tuple[str, ...]] = {}

    def axes_for_access_result(
        operation: IntentOp,
        node_ref: str,
        access: ContractAccessIR,
    ) -> tuple[str, ...]:
        """Bind a producer domain from its own authored access programs."""

        result_ref = _result_names(operation)[0]
        cached = result_axes.get(result_ref)
        if cached is not None:
            return cached
        value_type = value_types.get(result_ref)
        if value_type is None:
            raise ValueError("contract reference operation lacks a closed result type")
        if len(access.index) != len(value_type.shape):
            raise ContractReferenceUnsupportedError(
                "contract access rank differs from its producer result"
            )
        axes: list[str] = []
        for dimension, (program, extent) in enumerate(
            zip(access.index, value_type.shape, strict=True)
        ):
            referenced_axes = [
                axis
                for axis in domains
                if axis in program.expression.references
            ]
            if len(referenced_axes) == 1:
                axis = referenced_axes[0]
                if domains[axis] != extent:
                    raise ContractReferenceUnsupportedError(
                        f"J node {node_ref!r} access axis {axis!r} differs from "
                        f"result dimension {dimension}"
                    )
                axes.append(axis)
                continue
            if len(referenced_axes) > 1:
                raise ContractReferenceUnsupportedError(
                    "one contract access component uses multiple producer axes"
                )
            if not (
                (isinstance(extent, int) and extent == 1)
                or (isinstance(extent, str) and extent in unit_dimensions)
            ):
                raise ContractReferenceUnsupportedError(
                    "contract producer dimension lacks one authored access axis"
                )
            axes.append(f"@unit:{result_ref}:{dimension}")
        if len(set(axes)) != len(axes):
            raise ContractReferenceUnsupportedError(
                "contract producer access axes are not unique"
            )
        predicate_axes = (
            set(access.predicate.expression.references).intersection(domains)
            if access.predicate is not None
            else set()
        )
        if not predicate_axes.issubset(set(axes)):
            raise ContractReferenceUnsupportedError(
                "contract access predicate uses axes outside its producer domain"
            )
        result_axes[result_ref] = tuple(axes)
        return tuple(axes)

    def axes_for_gather_result(
        operation: IntentOp,
        node_ref: str,
        access: ContractAccessIR,
        gather_axis: int,
    ) -> tuple[str, ...]:
        """Bind a gather result to the axes explicitly used by its J access."""

        result_ref = _result_names(operation)[0]
        data_type = value_types.get(access.tensor)
        index_type = value_types.get(operation.operands[1])
        result_type = value_types.get(result_ref)
        if data_type is None or index_type is None or result_type is None:
            raise ValueError("contract gather lacks a closed J type")
        if len(access.index) != len(data_type.shape):
            raise ContractReferenceUnsupportedError(
                "contract gather access rank differs from its data tensor"
            )
        if not 0 <= gather_axis < len(data_type.shape):
            raise ValueError("contract gather axis is outside its data rank")
        if len(index_type.shape) > 1:
            raise ContractReferenceUnsupportedError(
                "contract indexed result currently requires a scalar or rank-1 index"
            )

        axes: list[str] = []
        for dimension, (program, extent) in enumerate(
            zip(access.index, data_type.shape, strict=True)
        ):
            referenced_axes = [
                axis
                for axis in domains
                if axis in program.expression.references
            ]
            if dimension == gather_axis and not index_type.shape:
                if referenced_axes:
                    raise ContractReferenceUnsupportedError(
                        "scalar contract gather index unexpectedly carries a domain axis"
                    )
                continue
            if len(referenced_axes) == 1:
                axes.append(referenced_axes[0])
                continue
            if len(referenced_axes) > 1:
                raise ContractReferenceUnsupportedError(
                    "one contract gather result dimension uses multiple domain axes"
                )
            if dimension == gather_axis:
                raise ContractReferenceUnsupportedError(
                    "rank-1 contract gather index lacks its result domain axis"
                )
            if not (
                (isinstance(extent, int) and extent == 1)
                or (isinstance(extent, str) and extent in unit_dimensions)
            ):
                raise ContractReferenceUnsupportedError(
                    "constant contract gather coordinate requires a J-proved unit extent"
                )
            axes.append(f"@unit:{result_ref}:{dimension}")
        if len(axes) != len(result_type.shape) or len(set(axes)) != len(axes):
            raise ContractReferenceUnsupportedError(
                f"J node {node_ref!r} gather result axes are not uniquely closed"
            )
        result_axes[result_ref] = tuple(axes)
        return tuple(axes)

    try:
        contract_ir = project_contract_ir(canonical)
    except ContractIRProjectionError as exc:
        raise ContractReferenceUnsupportedError(
            "contract reference requires one closed typed ContractIR"
        ) from exc
    input_names = {item.name for item in canonical.intent.inputs}
    output_names = j_elaboration(canonical).output_names
    output_name_set = set(output_names)
    graph_operands = {
        operand for operation in operations for operand in operation.operands
    }
    input_load_counts: dict[str, int] = {}
    for access in contract_ir.accesses:
        if access.kind == "load":
            input_load_counts[access.tensor] = input_load_counts.get(access.tensor, 0) + 1
    predicates_are_domain_total = access_predicates_are_logical_domain_total(canonical)

    def is_closed_whole_input_load(access: ContractAccessIR) -> bool:
        """Recognize one complete logical input already present in the J env."""

        value_type = value_types.get(access.tensor)
        if (
            access.kind != "load"
            or access.node_ref is not None
            or access.tensor not in input_names
            or access.tensor not in graph_operands
            or input_load_counts.get(access.tensor) != 1
            or value_type is None
            or len(access.index) != len(value_type.shape)
            or (
                access.predicate is not None
                and access.fill_value is None
                and not predicates_are_domain_total
            )
            or not predicates_are_domain_total
        ):
            return False
        axes: list[str] = []
        for program, extent in zip(access.index, value_type.shape, strict=True):
            expression = program.expression
            if expression.op != "ref" or not isinstance(expression.value, str):
                return False
            axis = expression.value
            if axis in axes or domains.get(axis) != extent:
                return False
            axes.append(axis)
        return True

    loads: list[IndexedLoadProjection] = []
    result_loads: list[IndexedResultLoadProjection] = []
    stores: list[IndexedStoreProjection] = []
    seen_load_edges: set[tuple[str, int]] = set()
    seen_accesses: set[str] = set()
    consumed_index_bounds: set[tuple[str, str, str, int]] = set()
    for access in contract_ir.accesses:
        node_ref = access.node_ref
        operation = nodes.get(node_ref)
        if operation is None or node_ref is None:
            if is_closed_whole_input_load(access):
                seen_accesses.add(access.access_id)
                continue
            raise ContractReferenceUnsupportedError(
                "contract reference v1 requires every access to name one flat J node"
            )
        result_ref = _result_names(operation)[0]
        if access.kind == "load" and operation.op == "gather":
            axis = operation.attrs.get("axis")
            mode = operation.attrs["mode"]
            if (
                mode != "take"
                or not isinstance(axis, int)
                or isinstance(axis, bool)
                or len(operation.operands) != 2
                or operation.operands[0] != access.tensor
            ):
                raise ContractReferenceUnsupportedError(
                    "graph-native contract gather requires one typed take edge"
                )
            bound_key = (
                node_ref,
                access.tensor,
                operation.operands[1],
                axis,
            )
            matching_bounds = [
                item
                for item in boundary.index_bounds
                if (
                    item.node_ref,
                    item.data,
                    item.indices,
                    item.axis,
                )
                == bound_key
            ]
            if len(matching_bounds) != 1:
                raise ContractReferenceUnsupportedError(
                    "contract indexed result gather requires one exact index bound"
                )
            if (
                access.predicate is not None
                and access.fill_value is None
                and not predicates_are_domain_total
            ):
                raise ValueError("contract predicated gather load lacks its authored fill")
            result_loads.append(
                IndexedResultLoadProjection(
                    access,
                    result_ref,
                    axes_for_gather_result(
                        operation,
                        node_ref,
                        access,
                        axis,
                    ),
                )
            )
            consumed_index_bounds.add(bound_key)
            seen_accesses.add(access.access_id)
            continue

        axes = axes_for_access_result(operation, node_ref, access)
        axis_refs = set().union(
            *(set(program.expression.references) for program in access.index),
            (
                set(access.predicate.expression.references)
                if access.predicate is not None else set()
            ),
        )
        unknown_axes = (axis_refs & set(domains)) - set(axes)
        if unknown_axes:
            raise ContractReferenceUnsupportedError(
                "contract access uses domain axes outside its producer result: "
                + ", ".join(sorted(unknown_axes))
            )
        if access.kind == "load":
            if access.tensor not in input_names:
                raise ContractReferenceUnsupportedError(
                    "contract reference v1 loads only logical J inputs"
                )
            tensor_type = value_types.get(access.tensor)
            if tensor_type is None or len(access.index) != len(tensor_type.shape):
                raise ContractReferenceUnsupportedError(
                    "contract indexed load rank differs from its logical J tensor"
                )
            positions = [
                index
                for index, operand in enumerate(operation.operands)
                if operand == access.tensor
            ]
            if len(positions) != 1:
                raise ContractReferenceUnsupportedError(
                    "contract indexed load does not identify one unique operand edge"
                )
            edge = (node_ref, positions[0])
            if edge in seen_load_edges:
                raise ValueError("contract indexed loads duplicate one operand edge")
            seen_load_edges.add(edge)
            if (
                access.predicate is not None
                and access.fill_value is None
                and not predicates_are_domain_total
            ):
                raise ValueError("contract predicated load lacks its authored fill")
            loads.append(
                IndexedLoadProjection(
                    access,
                    positions[0],
                    result_ref,
                    axes,
                )
            )
        elif access.kind == "store":
            if access.tensor not in output_name_set:
                raise ContractReferenceUnsupportedError(
                    "contract reference v1 stores only logical J outputs"
                )
            tensor_type = value_types.get(access.tensor)
            if tensor_type is None or len(access.index) != len(tensor_type.shape):
                raise ContractReferenceUnsupportedError(
                    "contract indexed store rank differs from its logical J tensor"
                )
            if access.fill_value is not None:
                raise ValueError("contract store must not declare a fill")
            stores.append(
                IndexedStoreProjection(
                    access,
                    result_ref,
                    axes,
                )
            )
        else:
            raise ContractReferenceUnsupportedError(
                f"contract reference v1 does not implement access kind {access.kind!r}"
            )
        seen_accesses.add(access.access_id)

    if seen_accesses != {item.access_id for item in canonical.boundary.accesses}:
        raise ValueError("contract reference did not consume every authored access")
    authored_index_bounds = {
        (item.node_ref, item.data, item.indices, item.axis)
        for item in boundary.index_bounds
    }
    native_index_bounds: list[tuple[str, str, str, int]] = []
    for bound_key in sorted(authored_index_bounds - consumed_index_bounds):
        node_ref, data, indices, axis = bound_key
        operation = nodes.get(node_ref)
        if (
            operation is None
            or operation.op != "gather"
            or len(operation.operands) != 2
            or operation.operands[0] != data
            or operation.operands[1] != indices
            or operation.attrs.get("axis") != axis
            or operation.attrs.get("mode") not in {"along_axis", "take"}
        ):
            raise ContractReferenceUnsupportedError(
                "contract native index bound does not match one typed gather"
            )
        native_index_bounds.append(bound_key)
        consumed_index_bounds.add(bound_key)
    if consumed_index_bounds != authored_index_bounds:
        raise ValueError("contract reference left an authored index bound unconsumed")
    stores_by_output = {item.access.tensor for item in stores}
    if stores_by_output != output_name_set:
        raise ContractReferenceUnsupportedError(
            "contract reference v1 requires authored stores for every output"
        )
    output_bindings = {
        item.output: item.value for item in j_elaboration(canonical).output_bindings
    }
    producers = {
        _result_names(operation)[0]: operation for operation in operations
    }
    output_assemblies: list[IndexedOutputAssembly] = []
    assembly_results: set[str] = set()
    for output in output_names:
        binding = output_bindings.get(output)
        if binding == output:
            continue
        operation = producers.get(binding)
        output_stores = [item for item in stores if item.access.tensor == output]
        if (
            operation is None
            or operation.op != "concat"
            or not frozenset(operation.attrs).issubset(
                _CONCAT_ATTR_CONTRACT["allowed"]
            )
            or not frozenset(_CONCAT_ATTR_CONTRACT["required"]).issubset(
                operation.attrs
            )
            or not operation.operands
            or not isinstance(operation.attrs.get("axis"), int)
            or isinstance(operation.attrs.get("axis"), bool)
            or any(
                binding in candidate.operands
                for candidate in operations
                if candidate is not operation
            )
        ):
            raise ContractReferenceUnsupportedError(
                "contract indexed non-identity output requires one terminal concat"
            )
        axis = int(operation.attrs["axis"])
        output_type = value_types.get(output)
        if output_type is None or not 0 <= axis < len(output_type.shape):
            raise ValueError("contract output concat axis is outside output rank")
        stores_by_value: dict[str, list[IndexedStoreProjection]] = {}
        for store in output_stores:
            stores_by_value.setdefault(store.value_ref, []).append(store)
        if (
            len(operation.operands) != len(output_stores)
            or set(operation.operands) != set(stores_by_value)
            or any(len(stores_by_value[operand]) != 1 for operand in operation.operands)
        ):
            raise ContractReferenceUnsupportedError(
                "contract output concat operands differ from its indexed stores"
            )
        output_assemblies.append(
            IndexedOutputAssembly(
                output,
                binding,
                axis,
                tuple(operation.operands),
            )
        )
        assembly_results.add(binding)
    concat_results = {
        _result_names(operation)[0]
        for operation in operations
        if operation.op == "concat"
    }
    if concat_results != assembly_results:
        raise ContractReferenceUnsupportedError(
            "contract reference permits concat only as a closed output assembly"
        )

    effects = {item.access_ref: item for item in boundary.effects}
    if set(effects) != seen_accesses:
        raise ValueError("contract reference requires one effect per access")
    for access in contract_ir.accesses:
        effect = effects[access.access_id]
        expected = "read" if access.kind == "load" else "write"
        if (
            effect.kind.value != expected
            or effect.target != access.tensor
            or effect.node_ref != access.node_ref
        ):
            raise ValueError("contract access/effect relation is inconsistent")

    broadcasts = canonical_broadcast_axes(canonical)
    return ContractReferencePlan(
        contract_ir=contract_ir,
        domains=tuple(sorted(domains.items())),
        loads=tuple(loads),
        result_loads=tuple(result_loads),
        stores=tuple(stores),
        output_assemblies=tuple(output_assemblies),
        native_index_bounds=tuple(native_index_bounds),
        broadcast_axes=tuple(
            sorted(
                (
                    result,
                    tuple(sorted((operand, tuple(axes)) for operand, axes in operands.items())),
                )
                for result, operands in broadcasts.items()
            )
        ),
    )


def _resolved_domain_bindings(
    plan: ContractReferencePlan,
    dims: Mapping[str, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for axis, extent in plan.domains:
        if type(extent) is int:
            value = extent
        elif isinstance(extent, str) and extent in dims:
            value = dims[extent]
        else:
            raise ContractReferenceUnsupportedError(
                f"contract reference cannot resolve domain extent {extent!r}"
            )
        if value < 0:
            raise ValueError("contract reference domain has a negative extent")
        result[axis] = int(value)
    return result


def _scalar_bindings(
    inputs: Mapping[str, np.ndarray],
    dims: Mapping[str, int],
) -> dict[str, int | bool]:
    bindings: dict[str, int | bool] = dict(dims)
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in {"b", "i", "u"}:
            continue
        item = array.item()
        bindings[name] = bool(item) if array.dtype.kind == "b" else int(item)
    return bindings


def _evaluate_array_program(
    program: TypedContractProgram,
    bindings: Mapping[str, Any],
    *,
    context: str,
) -> int | bool | np.ndarray:
    """Evaluate one typed ContractExpr over broadcast NumPy index grids.

    This is the array denotation of the same closed opcode set implemented by
    ``TypedContractProgram.evaluate``.  It changes only execution granularity:
    no formula recognition, Source fact, C value, or pi relation enters here.
    """

    missing = set(program.expression.references) - set(bindings)
    if missing:
        raise ContractReferenceUnsupportedError(
            f"{context} lacks bindings for {sorted(missing)}"
        )
    reference_sorts = program.reference_sort_map

    def visit(expression: ContractExpr) -> Any:
        if expression.op == "ref":
            assert isinstance(expression.value, str)
            value = bindings[expression.value]
            array = np.asarray(value)
            expected = reference_sorts[expression.value]
            if expected is ContractSort.INTEGER and array.dtype.kind not in {"i", "u"}:
                raise ContractReferenceUnsupportedError(
                    f"{context} reference {expression.value!r} is not integer"
                )
            if expected is ContractSort.BOOLEAN and array.dtype.kind != "b":
                raise ContractReferenceUnsupportedError(
                    f"{context} reference {expression.value!r} is not boolean"
                )
            return value
        if expression.op in {"int", "bool"}:
            return expression.value

        values = tuple(visit(argument) for argument in expression.args)
        if expression.op == "pos":
            return values[0]
        if expression.op == "neg":
            return np.negative(values[0])
        if expression.op == "not":
            return np.logical_not(values[0])
        if expression.op == "add":
            return np.add(values[0], values[1])
        if expression.op == "sub":
            return np.subtract(values[0], values[1])
        if expression.op == "mul":
            return np.multiply(values[0], values[1])
        if expression.op in {"floordiv", "mod"}:
            if bool(np.any(np.asarray(values[1]) == 0)):
                raise ContractReferenceUnsupportedError(f"{context} divides by zero")
            return (
                np.floor_divide(values[0], values[1])
                if expression.op == "floordiv"
                else np.mod(values[0], values[1])
            )
        if expression.op == "left_shift":
            return np.left_shift(values[0], values[1])
        if expression.op == "right_shift":
            return np.right_shift(values[0], values[1])
        if expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
            return {
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
                "eq": np.equal,
                "ne": np.not_equal,
            }[expression.op](values[0], values[1])
        if expression.op in {"and", "or"}:
            reducer = np.logical_and if expression.op == "and" else np.logical_or
            result = values[0]
            for value in values[1:]:
                result = reducer(result, value)
            return result
        if expression.op == "call":
            left, right = values
            if expression.value == "ceil_div":
                if bool(np.any(np.asarray(right) == 0)):
                    raise ContractReferenceUnsupportedError(
                        f"{context} divides by zero"
                    )
                return np.negative(np.floor_divide(np.negative(left), right))
            if expression.value == "min":
                return np.minimum(left, right)
            if expression.value == "max":
                return np.maximum(left, right)
        raise ContractReferenceUnsupportedError(
            f"{context} has unsupported opcode {expression.op!r}"
        )

    result = visit(program.expression)
    result_array = np.asarray(result)
    if (
        program.result_sort is ContractSort.INTEGER
        and result_array.dtype.kind not in {"i", "u"}
    ):
        raise ValueError(f"{context} did not produce integer values")
    if program.result_sort is ContractSort.BOOLEAN and result_array.dtype.kind != "b":
        raise ValueError(f"{context} did not produce boolean values")
    return result


def _access_grid_bindings(
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    scalar_bindings: Mapping[str, int | bool],
) -> dict[str, Any]:
    if len(shape) != len(axes):
        raise ValueError("contract access axes differ from producer rank")
    grids = np.indices(shape, sparse=True, dtype=np.int64) if shape else ()
    return {
        **scalar_bindings,
        **dict(zip(axes, grids, strict=True)),
    }


def _broadcast_access_value(
    value: Any,
    shape: tuple[int, ...],
    *,
    boolean: bool,
    context: str,
) -> np.ndarray:
    array = np.asarray(value)
    if boolean and array.dtype.kind != "b":
        raise ValueError(f"{context} is not boolean")
    if not boolean and array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{context} is not integer")
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ContractReferenceUnsupportedError(
            f"{context} does not broadcast to its producer domain"
        ) from exc


def _vectorized_access(
    access: ContractAccessIR,
    *,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    scalar_bindings: Mapping[str, int | bool],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    bindings = _access_grid_bindings(shape, axes, scalar_bindings)
    predicate = (
        np.ones(shape, dtype=np.bool_)
        if access.predicate is None
        else _broadcast_access_value(
            _evaluate_array_program(
                access.predicate,
                bindings,
                context=f"J access {access.access_id!r} predicate",
            ),
            shape,
            boolean=True,
            context=f"J access {access.access_id!r} predicate",
        )
    )
    indices = tuple(
        _broadcast_access_value(
            _evaluate_array_program(
                program,
                bindings,
                context=f"J access {access.access_id!r} index[{position}]",
            ),
            shape,
            boolean=False,
            context=f"J access {access.access_id!r} index[{position}]",
        )
        for position, program in enumerate(access.index)
    )
    return predicate, indices


def _evaluate_predicate(
    access: ContractAccessIR,
    bindings: Mapping[str, int | bool],
) -> bool:
    if access.predicate is None:
        return True
    try:
        value = access.predicate.evaluate(
            bindings,
            context=f"J access {access.access_id!r} predicate",
        )
    except ContractSemanticsError as exc:
        raise ContractReferenceUnsupportedError(
            f"contract access {access.access_id!r} predicate has no closed "
            "runtime binding"
        ) from exc
    if type(value) is not bool:
        raise ValueError("contract access predicate did not produce boolean")
    return value


def _evaluate_index(
    access: ContractAccessIR,
    bindings: Mapping[str, int | bool],
) -> tuple[int, ...]:
    result: list[int] = []
    for position, program in enumerate(access.index):
        try:
            value = program.evaluate(
                bindings,
                context=f"J access {access.access_id!r} index[{position}]",
            )
        except ContractSemanticsError as exc:
            raise ContractReferenceUnsupportedError(
                f"contract access {access.access_id!r} index[{position}] has no "
                "closed runtime binding"
            ) from exc
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("contract access index did not produce integer")
        result.append(value)
    return tuple(result)


class _ContractEvaluator(_Evaluator):
    def __init__(
        self,
        j: "SemanticArchive",
        dims: Mapping[str, int],
        plan: ContractReferencePlan,
        scalar_bindings: Mapping[str, int | bool],
    ) -> None:
        super().__init__(j, dims, plan.broadcast_map)
        self.contract_loads = plan.load_map
        self.contract_result_loads = plan.result_load_map
        self.contract_scalar_bindings = dict(scalar_bindings)
        self.output_assembly_results = {
            item.result_ref for item in plan.output_assemblies
        }
        self.indexed_load_materialization_count = 0
        self.output_assembly_elision_count = 0

    def sequence(
        self,
        body: Sequence[IntentOp],
        env: Mapping[str, np.ndarray],
        *,
        allow_loop: bool,
    ) -> dict[str, np.ndarray]:
        """Execute the flat graph, deferring proven output concat to stores."""

        filtered: list[IntentOp] = []
        for operation in body:
            result_ref = self._result_name(operation)
            if result_ref in self.output_assembly_results:
                self.output_assembly_elision_count += 1
                continue
            filtered.append(operation)
        return super().sequence(filtered, env, allow_loop=allow_loop)

    def _materialize_indexed_load(
        self,
        projection: IndexedLoadProjection | IndexedResultLoadProjection,
        source: np.ndarray,
        op: IntentOp,
    ) -> np.ndarray:
        """Execute one complete J-authored Layer-B load view.

        Both consumer-local operand views and graph-native gather results have
        the same denotation: evaluate the typed access programs over the
        producer/result domain, read selected source indices, and place the
        authored fill everywhere the predicate is false.  The distinction is
        only where the resulting tensor enters Layer A.
        """

        declared = self._declared_type(projection.result_ref, op)
        if declared is None:
            raise ValueError("contract indexed load lacks a closed result type")
        result_shape = declared[1]
        if len(result_shape) != len(projection.result_axes):
            raise ValueError("contract indexed load result axes differ from result rank")
        fill: np.ndarray | None = None
        if projection.access.predicate is not None:
            try:
                fill = np.asarray(
                    projection.access.fill_value,
                    dtype=source.dtype,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ContractReferenceUnsupportedError(
                    "contract indexed load fill has no typed NumPy denotation"
                ) from exc
            if fill.shape != ():
                raise ContractReferenceUnsupportedError(
                    "contract indexed load v1 requires a scalar fill"
                )
        predicate, source_index = _vectorized_access(
            projection.access,
            shape=result_shape,
            axes=projection.result_axes,
            scalar_bindings=self.contract_scalar_bindings,
        )
        if len(source_index) != source.ndim or any(
            bool(np.any((item[predicate] < 0) | (item[predicate] >= source.shape[axis])))
            for axis, item in enumerate(source_index)
        ):
            raise ValueError("contract indexed load is outside its J tensor")
        result = (
            np.full(result_shape, fill, dtype=source.dtype)
            if fill is not None
            else np.empty(result_shape, dtype=source.dtype)
        )
        result[predicate] = source[
            tuple(item[predicate] for item in source_index)
        ]
        self.indexed_load_materialization_count += 1
        return result

    def _operand(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
        index: int,
    ) -> np.ndarray:
        projection = self.contract_loads.get((self._result_name(op), index))
        if projection is None:
            value = super()._operand(op, env, index)
            try:
                operand_name = op.operands[index]
            except IndexError as exc:
                raise ValueError("contract pointwise op lacks an operand") from exc
            axes = self.broadcast_axes.get(self._result_name(op), {}).get(
                operand_name, ()
            )
            if not axes:
                return value
            # ``_Evaluator``'s legacy insertion loop cannot distinguish a
            # declared axis already occupied by a singleton dimension from a
            # genuinely missing rank dimension.  Recompute the same canonical
            # alignment used by the shared Intent MLIR emitter, then materialize
            # the exact consumer-local view.
            raw = np.asarray(env[operand_name])
            declared = self._declared_type(self._result_name(op), op)
            if declared is None:
                raise ValueError("contract broadcast lacks a closed result type")
            projected_shape, _inserted = _broadcast_projection(
                list(raw.shape), axes, list(declared[1])
            )
            if projected_shape is None:  # pragma: no cover - helper is total/raises
                raise ValueError("contract broadcast shape is not closed")
            return np.broadcast_to(
                raw.reshape(tuple(projected_shape)), declared[1]
            )
        try:
            operand_name = op.operands[index]
            source = np.asarray(env[operand_name])
        except (IndexError, KeyError) as exc:
            raise ValueError("contract indexed load has an unavailable operand") from exc
        if operand_name != projection.access.tensor:
            raise ValueError("contract indexed load names a different operand")
        return self._materialize_indexed_load(projection, source, op)

    def _ordinary(
        self,
        op: IntentOp,
        env: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        projection = self.contract_result_loads.get(self._result_name(op))
        if projection is None:
            return super()._ordinary(op, env)
        if op.op != "gather" or not op.operands:
            raise ValueError("contract indexed result does not name one gather")
        try:
            source = np.asarray(env[op.operands[0]])
        except KeyError as exc:
            raise ValueError("contract indexed result source is unavailable") from exc
        if op.operands[0] != projection.access.tensor:
            raise ValueError("contract indexed result names a different source")
        return self._materialize_indexed_load(projection, source, op)


def _validate_indexed_concat_store_slice(
    *,
    predicate: np.ndarray,
    target_index: tuple[np.ndarray, ...],
    producer_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axis: int,
    offset: int,
) -> int:
    """Prove one store is the next ordered, complete concat-axis slice."""

    if len(producer_shape) != len(output_shape) or len(target_index) != len(output_shape):
        raise ContractReferenceUnsupportedError(
            "contract output concat store rank differs from its output"
        )
    if any(
        producer_shape[dimension] != output_shape[dimension]
        for dimension in range(len(output_shape))
        if dimension != axis
    ):
        raise ContractReferenceUnsupportedError(
            "contract output concat store changes a non-concat extent"
        )
    selected_count = int(np.count_nonzero(predicate))
    if selected_count == 0:
        raise ContractReferenceUnsupportedError(
            "contract output concat contains an empty authored slice"
        )
    grids = np.indices(producer_shape, sparse=False, dtype=np.int64)
    for dimension in range(len(output_shape)):
        selected_target = target_index[dimension][predicate]
        selected_source = grids[dimension][predicate]
        expected = (
            selected_source + offset
            if dimension == axis
            else selected_source
        )
        if not np.array_equal(selected_target, expected):
            raise ContractReferenceUnsupportedError(
                "contract output concat store is not an ordered identity slice"
            )
    active_source_axis = np.unique(grids[axis][predicate])
    if (
        active_source_axis.size == 0
        or active_source_axis[0] != 0
        or not np.array_equal(
            active_source_axis,
            np.arange(active_source_axis.size, dtype=np.int64),
        )
    ):
        raise ContractReferenceUnsupportedError(
            "contract output concat predicate is not a zero-based contiguous slice"
        )
    width = int(active_source_axis.size)
    expected_count = width
    for dimension, extent in enumerate(output_shape):
        if dimension != axis:
            expected_count *= extent
    if selected_count != expected_count:
        raise ContractReferenceUnsupportedError(
            "contract output concat predicate does not cover every outer coordinate"
        )
    if offset + width > output_shape[axis]:
        raise ValueError("contract output concat slice exceeds the output extent")
    return width


def execute_contract_reference(
    j: "SemanticArchive",
    dims: Mapping[str, int],
    inputs: Sequence[np.ndarray],
    *,
    plan: ContractReferencePlan | None = None,
) -> CPUExecutionResult:
    """Execute one relation-indexed plan using only J and invocation bindings."""

    canonical = elaborate_j(j)
    selected = plan or project_contract_reference(canonical)
    if len(inputs) != len(canonical.intent.inputs):
        raise ValueError("contract reference input arity differs from J")
    checked_dims = {
        name: int(extent)
        for name, extent in dims.items()
        if isinstance(name, str)
        and isinstance(extent, int)
        and not isinstance(extent, bool)
        and extent >= 0
    }
    if checked_dims != dict(dims):
        raise ValueError(
            "contract reference dims must be non-negative integer bindings"
        )
    checked_dims = resolve_dimension_relations(canonical, checked_dims)
    checked_inputs = {
        tensor.name: validate_array(
            np.asarray(value),
            shape=_resolved_shape(tensor.shape, checked_dims),
            dtype=tensor.dtype,
            context=f"contract input {tensor.name!r}",
        )
        for tensor, value in zip(canonical.intent.inputs, inputs, strict=True)
    }
    validate_value_dimension_instantiation(canonical, checked_dims, checked_inputs)
    domain_extents = _resolved_domain_bindings(selected, checked_dims)
    scalar_bindings = _scalar_bindings(checked_inputs, checked_dims)
    evaluator = _ContractEvaluator(
        canonical,
        checked_dims,
        selected,
        scalar_bindings,
    )
    env = evaluator.sequence(
        _canonical_body(canonical),
        checked_inputs,
        allow_loop=False,
    )

    value_types = j_elaboration(canonical).value_type_by_name
    outputs: dict[str, np.ndarray] = {}
    stores_by_output = selected.stores_by_output
    output_assemblies = selected.output_assembly_map
    executed_stores = 0
    assembled_slices = 0
    output_bindings = j_elaboration(canonical).output_bindings
    for binding in output_bindings:
        # The output record has already closed output/return/type/ABI order.
        # Do not reopen a second output type authority from intent.outputs.
        shape = _resolved_shape(binding.output_type.shape, checked_dims)
        output = np.empty(shape, dtype=numpy_dtype(binding.output_type.dtype))
        covered = np.zeros(shape, dtype=np.bool_)
        assembly = output_assemblies.get(binding.output)
        output_stores = stores_by_output[binding.output]
        if assembly is not None:
            by_value = {item.value_ref: item for item in output_stores}
            output_stores = tuple(by_value[item] for item in assembly.operand_refs)
        concat_offset = 0
        for store in output_stores:
            try:
                value = np.asarray(env[store.value_ref])
            except KeyError as exc:
                raise ValueError("contract output store producer is unavailable") from exc
            declared = value_types.get(store.value_ref)
            if declared is None:
                raise ValueError("contract output store producer lacks a closed type")
            producer_shape = _resolved_shape(declared.shape, checked_dims)
            if value.shape != producer_shape or len(value.shape) != len(store.value_axes):
                raise ValueError("contract output store producer type is inconsistent")
            predicate, target_index = _vectorized_access(
                store.access,
                shape=producer_shape,
                axes=store.value_axes,
                scalar_bindings=scalar_bindings,
            )
            if len(target_index) != len(shape) or any(
                bool(np.any((item[predicate] < 0) | (item[predicate] >= shape[axis])))
                for axis, item in enumerate(target_index)
            ):
                raise ValueError("contract output store is outside its J tensor")
            if assembly is not None:
                concat_offset += _validate_indexed_concat_store_slice(
                    predicate=predicate,
                    target_index=target_index,
                    producer_shape=producer_shape,
                    output_shape=shape,
                    axis=assembly.axis,
                    offset=concat_offset,
                )
                assembled_slices += 1
            linear = np.zeros(producer_shape, dtype=np.int64)
            for extent, index_values in zip(shape, target_index, strict=True):
                linear = linear * extent + index_values
            selected_indices = linear[predicate]
            if np.unique(selected_indices).size != selected_indices.size:
                raise ValueError("contract output store contains overlapping indices")
            covered_flat = covered.reshape(-1)
            if bool(np.any(covered_flat[selected_indices])):
                raise ValueError("contract output stores overlap")
            output.reshape(-1)[selected_indices] = value.reshape(-1)[
                predicate.reshape(-1)
            ]
            covered_flat[selected_indices] = True
            executed_stores += int(selected_indices.size)
        if not bool(np.all(covered)):
            raise ContractReferenceUnsupportedError(
                "contract output stores do not exhaust the logical output"
            )
        if assembly is not None and concat_offset != shape[assembly.axis]:
            raise ContractReferenceUnsupportedError(
                "contract output concat slices do not exhaust the concat axis"
            )
        outputs[binding.output] = validate_array(
            output,
            shape=shape,
            dtype=binding.output_type.dtype,
            context=f"contract output {binding.output!r}",
        )

    plan_record = selected.to_json()
    return CPUExecutionResult(
        outputs,
        {
            "protocol": CONTRACT_REFERENCE_PROTOCOL,
            "authority": "canonical_J_contract_ir_only",
            "profile": "relation_indexed_graph_v2",
            "plan_digest": plan_record["plan_digest"],
            "contract_ir_digest": selected.contract_ir.digest,
            "domain_axis_count": len(domain_extents),
            "indexed_load_count": evaluator.indexed_load_materialization_count,
            "executed_store_element_count": executed_stores,
            "output_assembly_count": len(output_assemblies),
            "assembled_slice_count": assembled_slices,
            "elided_terminal_concat_count": (
                evaluator.output_assembly_elision_count
            ),
            "native_index_bound_count": len(selected.native_index_bounds),
            "operation_count": evaluator.operation_count,
            "input_count": len(checked_inputs),
            "output_count": len(outputs),
        },
    )


__all__ = [
    "CONTRACT_REFERENCE_PROTOCOL",
    "ContractReferencePlan",
    "ContractReferenceUnsupportedError",
    "IndexedLoadProjection",
    "IndexedOutputAssembly",
    "IndexedStoreProjection",
    "execute_contract_reference",
    "project_contract_reference",
]
