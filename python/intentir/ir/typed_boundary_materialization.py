"""J-only lowering of typed boundary indices, predicates, and writes.

The accepted archive keeps access coordinates and predicates in Layer B while
Layer A names the pure producer/consumer graph.  This module joins those two
parts into ephemeral compiler state.  It consumes ``CanonicalContractIR`` and
emits ordinary Intent operations (iota/arithmetic/gather/select); it never
consults Source, a kernel name, C, a target, or runtime values.

This is intentionally a lowering, not another capability matcher.  The closed
``ContractExpr`` AST is the only expression input.  Unsupported index forms
fail before MLIR emission instead of falling back to string formula matching.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from intentir.ir.canonical_j import ContractExpr, elaborate_j, j_elaboration
from intentir.ir.contract_ir import ContractAccessIR, project_contract_ir
from intentir.ir.contract_materialization import (
    ContractMaterializationProjection,
    project_contract_materialization,
)
from intentir.ir.model import (
    IntentFunction,
    IntentOp,
    IntentReturn,
    walk_intent_op_paths,
)
from intentir.verify.gate1.model import SemanticArchive


TYPED_BOUNDARY_MATERIALIZATION_PROTOCOL = (
    "intentir.typed-boundary-materialization.v1"
)


class TypedBoundaryMaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class TypedBoundaryMaterializationRecord:
    access_id: str
    kind: str
    node_ref: str
    generated_values: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "kind": self.kind,
            "node_ref": self.node_ref,
            "generated_values": list(self.generated_values),
        }


@dataclass(frozen=True)
class TypedBoundaryMaterializationProjection:
    logical_projection: ContractMaterializationProjection
    records: tuple[TypedBoundaryMaterializationRecord, ...]
    synthetic_proven_index_bounds: frozenset[str] = frozenset()
    synthetic_broadcast_axes: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ] = ()
    structured_loop_nodes: tuple[str, ...] = ()

    @property
    def profile(self) -> str:
        return "typed_contract_index_and_effect_lowering"

    @property
    def runtime_predicate_proofs(self) -> tuple[Any, ...]:
        return ()

    @property
    def unsupported_lowering(self) -> tuple[str, ...]:
        resolved = {item.access_id for item in self.records}
        return tuple(
            issue
            for issue in self.logical_projection.unsupported_lowering
            if not any(issue.startswith(f"access {access_id} ") for access_id in resolved)
        )

    @property
    def proven_index_bounds(self) -> frozenset[str]:
        return frozenset(
            {
                *self.logical_projection.proven_index_bounds,
                *self.synthetic_proven_index_bounds,
            }
        )

    @property
    def consumed(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.logical_projection.consumed,
                    *(f"typed_boundary:{item.access_id}" for item in self.records),
                    *(f"loop:{item}" for item in self.structured_loop_nodes),
                }
            )
        )

    @property
    def predicated_access_count(self) -> int:
        return self.logical_projection.predicated_access_count

    @property
    def erased_fill_count(self) -> int:
        return self.logical_projection.erased_fill_count

    def emitter_broadcast_axes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        result = self.logical_projection.emitter_broadcast_axes()
        for output, operands in self.synthetic_broadcast_axes:
            target = result.setdefault(output, {})
            for operand, axes in operands:
                target[operand] = axes
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": TYPED_BOUNDARY_MATERIALIZATION_PROTOCOL,
            "profile": self.profile,
            "logical_projection": self.logical_projection.to_json(),
            "records": [item.to_json() for item in self.records],
            "synthetic_proven_index_bounds": sorted(
                self.synthetic_proven_index_bounds
            ),
            "synthetic_broadcast_axes": [
                [
                    output,
                    [[operand, list(axes)] for operand, axes in operands],
                ]
                for output, operands in self.synthetic_broadcast_axes
            ],
            "structured_loop_nodes": list(self.structured_loop_nodes),
            "consumed": list(self.consumed),
        }


@dataclass(frozen=True)
class TypedBoundaryFunctionMaterialization:
    function: IntentFunction
    projection: TypedBoundaryMaterializationProjection


def _walk(
    body: Sequence[IntentOp], prefix: str = "body"
) -> tuple[tuple[str, IntentOp], ...]:
    return tuple(walk_intent_op_paths(body, prefix))


def _canonical_walk(
    j: SemanticArchive,
) -> tuple[tuple[str, IntentOp], ...]:
    """Return typed operations materialized from canonical J records only."""

    rows = j_elaboration(j).typed_operation_by_path
    if rows is None:
        raise TypedBoundaryMaterializationError(
            "canonical operation records have no typed operation view"
        )
    return tuple(rows.items())


def _results(operation: IntentOp) -> tuple[str, ...]:
    if isinstance(operation.result, str):
        return (operation.result,)
    if isinstance(operation.result, list):
        return tuple(item for item in operation.result if isinstance(item, str))
    return ()


def _shape(value: Sequence[int | str]) -> tuple[int | str, ...]:
    return tuple(value)


def _broadcast_shape(
    lhs: tuple[int | str, ...], rhs: tuple[int | str, ...]
) -> tuple[int | str, ...]:
    if not lhs:
        return rhs
    if not rhs:
        return lhs
    rank = max(len(lhs), len(rhs))
    left = (1,) * (rank - len(lhs)) + lhs
    right = (1,) * (rank - len(rhs)) + rhs
    result: list[int | str] = []
    for a, b in zip(left, right, strict=True):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise TypedBoundaryMaterializationError(
                f"typed boundary expression cannot broadcast {lhs} and {rhs}"
            )
    return tuple(result)


class _Fresh:
    def __init__(self, occupied: set[str]) -> None:
        self.occupied = set(occupied)
        self.ordinal = 0

    def __call__(self, stem: str) -> str:
        while True:
            name = f"__intentir_typed_boundary_{stem}_{self.ordinal}"
            self.ordinal += 1
            if name not in self.occupied:
                self.occupied.add(name)
                return name


@dataclass(frozen=True)
class _Value:
    name: str
    dtype: str
    shape: tuple[int | str, ...]


class _ExpressionBuilder:
    """Lower one typed ContractExpr into ordinary tensor Intent operations."""

    def __init__(
        self,
        operations: list[IntentOp],
        fresh: _Fresh,
        value_types: Mapping[str, tuple[str, tuple[int | str, ...]]],
        extents: Mapping[str, int],
        axis_values: Mapping[str, _Value],
    ) -> None:
        self.operations = operations
        self.fresh = fresh
        self.value_types = value_types
        self.extents = extents
        self.axis_values = dict(axis_values)

    def constant(
        self, value: bool | int | float, dtype: str = "i64"
    ) -> _Value:
        name = self.fresh("const")
        self.operations.append(
            IntentOp(
                op="const",
                result=name,
                attrs={"value": value, "dtype": dtype},
                type={"dtype": dtype, "shape": []},
            )
        )
        return _Value(name, dtype, ())

    def iota(self, extent: int | str, axis_ref: str) -> _Value:
        name = self.fresh(f"axis_{axis_ref}")
        self.operations.append(
            IntentOp(
                op="iota",
                result=name,
                attrs={"n": extent, "dtype": "i64"},
                type={"dtype": "i64", "shape": [extent]},
            )
        )
        return _Value(name, "i64", (extent,))

    def unary(self, fn: str, value: _Value, dtype: str) -> _Value:
        name = self.fresh(fn)
        self.operations.append(
            IntentOp(
                op="map_unary",
                result=name,
                operands=[value.name],
                attrs={"fn": fn},
                type={"dtype": dtype, "shape": list(value.shape)},
            )
        )
        return _Value(name, dtype, value.shape)

    def binary(self, fn: str, lhs: _Value, rhs: _Value, dtype: str) -> _Value:
        shape = _broadcast_shape(lhs.shape, rhs.shape)
        name = self.fresh(fn)
        self.operations.append(
            IntentOp(
                op="map_binary",
                result=name,
                operands=[lhs.name, rhs.name],
                attrs={"fn": fn},
                type={"dtype": dtype, "shape": list(shape)},
            )
        )
        return _Value(name, dtype, shape)

    def select(
        self,
        condition: _Value,
        positive: _Value,
        negative: _Value,
        *,
        result_shape: tuple[int | str, ...] | None = None,
    ) -> _Value:
        if condition.dtype != "i1":
            raise TypedBoundaryMaterializationError(
                "typed boundary select condition must be boolean"
            )
        shape = result_shape or _broadcast_shape(
            condition.shape, _broadcast_shape(positive.shape, negative.shape)
        )
        dtype = positive.dtype
        if negative.dtype != dtype:
            raise TypedBoundaryMaterializationError(
                "typed boundary select arms have different dtypes"
            )
        name = self.fresh("select")
        self.operations.append(
            IntentOp(
                op="select",
                result=name,
                operands=[condition.name, positive.name, negative.name],
                type={"dtype": dtype, "shape": list(shape)},
            )
        )
        return _Value(name, dtype, shape)

    def emit(self, expression: ContractExpr) -> _Value:
        if expression.op == "ref":
            assert isinstance(expression.value, str)
            if expression.value in self.axis_values:
                return self.axis_values[expression.value]
            if expression.value in self.extents:
                return self.constant(int(self.extents[expression.value]))
            value_type = self.value_types.get(expression.value)
            if value_type is None:
                raise TypedBoundaryMaterializationError(
                    f"typed boundary expression has unbound ref {expression.value!r}"
                )
            dtype, shape = value_type
            if shape:
                raise TypedBoundaryMaterializationError(
                    f"typed boundary ref {expression.value!r} is not scalar"
                )
            normalized = dtype.removeprefix("torch.").lower()
            if normalized in {"i8", "i16", "i32"}:
                name = self.fresh("index_widen")
                self.operations.append(
                    IntentOp(
                        op="cast",
                        result=name,
                        operands=[expression.value],
                        attrs={"to_dtype": "i64"},
                        type={"dtype": "i64", "shape": []},
                    )
                )
                return _Value(name, "i64", ())
            return _Value(expression.value, dtype, shape)
        if expression.op == "int":
            assert isinstance(expression.value, int)
            return self.constant(expression.value)
        if expression.op == "bool":
            assert isinstance(expression.value, bool)
            return self.constant(expression.value, "i1")
        if expression.op == "pos":
            return self.emit(expression.args[0])
        if expression.op == "neg":
            return self.binary(
                "sub", self.constant(0), self.emit(expression.args[0]), "i64"
            )
        if expression.op == "not":
            return self.unary("logical_not", self.emit(expression.args[0]), "i1")
        if expression.op in {"and", "or"}:
            values = [self.emit(item) for item in expression.args]
            result = values[0]
            fn = "logical_and" if expression.op == "and" else "logical_or"
            for value in values[1:]:
                result = self.binary(fn, result, value, "i1")
            return result
        if expression.op in {
            "add", "sub", "mul", "floordiv", "mod", "left_shift", "right_shift"
        }:
            lhs, rhs = (self.emit(item) for item in expression.args)
            if expression.op == "floordiv":
                # Contract indices are mathematical integers; admitted access
                # domains in this lowering are non-negative, so signed integer
                # division is the exact floor operation.
                return self.binary("div", lhs, rhs, "i64")
            if expression.op == "mod":
                quotient = self.binary("div", lhs, rhs, "i64")
                product = self.binary("mul", quotient, rhs, "i64")
                return self.binary("sub", lhs, product, "i64")
            return self.binary(expression.op, lhs, rhs, "i64")
        if expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
            lhs, rhs = (self.emit(item) for item in expression.args)
            return self.binary(expression.op, lhs, rhs, "i1")
        if expression.op == "call":
            lhs, rhs = (self.emit(item) for item in expression.args)
            if expression.value == "ceil_div":
                one = self.constant(1)
                numerator = self.binary(
                    "sub", self.binary("add", lhs, rhs, "i64"), one, "i64"
                )
                return self.binary("div", numerator, rhs, "i64")
            if expression.value in {"min", "max"}:
                return self.binary(str(expression.value), lhs, rhs, "i64")
        raise TypedBoundaryMaterializationError(
            f"unsupported typed boundary opcode {expression.op!r}"
        )


def _flatten_and(expression: ContractExpr) -> tuple[ContractExpr, ...]:
    if expression.op != "and":
        return (expression,)
    return tuple(
        nested for argument in expression.args for nested in _flatten_and(argument)
    )


def _ref(expression: ContractExpr) -> str | None:
    return expression.value if expression.op == "ref" else None


def _int(expression: ContractExpr) -> int | None:
    value = expression.value
    return value if expression.op == "int" and isinstance(value, int) else None


def _positive_axis_scale(expression: ContractExpr, axis_ref: str) -> int | None:
    """Recognize ``axis`` or a positive integer multiple of one axis."""

    if _ref(expression) == axis_ref:
        return 1
    if expression.op != "mul" or len(expression.args) != 2:
        return None
    lhs, rhs = expression.args
    if _ref(lhs) == axis_ref:
        scale = _int(rhs)
    elif _ref(rhs) == axis_ref:
        scale = _int(lhs)
    else:
        return None
    return scale if scale is not None and scale > 0 else None


def _is_full_domain_term(
    expression: ContractExpr,
    axis_extents: Mapping[str, int | str],
    value_dimensions: Mapping[str, str],
) -> bool:
    if len(expression.args) != 2:
        return False
    lhs, rhs = expression.args
    if expression.op == "le" and _int(lhs) == 0 and _ref(rhs) in axis_extents:
        return True
    if expression.op == "ge" and _ref(lhs) in axis_extents and _int(rhs) == 0:
        return True
    axis = _ref(lhs)
    bound = _ref(rhs)
    if bound is not None:
        bound = value_dimensions.get(bound, bound)
    return (
        expression.op == "lt"
        and axis in axis_extents
        and axis_extents[axis] == bound
    )


def _predicate(
    expression: ContractExpr | None,
    builder: _ExpressionBuilder,
    axis_extents: Mapping[str, int | str],
    value_dimensions: Mapping[str, str],
) -> _Value | None:
    if expression is None:
        return None
    residual = [
        term
        for term in _flatten_and(expression)
        if not _is_full_domain_term(term, axis_extents, value_dimensions)
    ]
    if not residual:
        return None
    result = builder.emit(residual[0])
    for term in residual[1:]:
        result = builder.binary(
            "logical_and", result, builder.emit(term), "i1"
        )
    return result


def _combine_guard(
    guard: _Value | None,
    predicate: _Value | None,
    builder: _ExpressionBuilder,
) -> _Value | None:
    if guard is None:
        return predicate
    if predicate is None:
        return guard
    return builder.binary("logical_and", guard, predicate, "i1")


def _axis_candidates(j: SemanticArchive) -> frozenset[str]:
    contract = project_contract_ir(j)
    dimensions = {
        extent
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for extent in tensor.shape
        if isinstance(extent, str) and extent != "?"
    }
    logical_values = {
        name for name, _value_type in j_elaboration(j).value_types
    }
    refs = {
        ref
        for domain in contract.domains
        for ref in domain.predicate.expression.references
    }
    return frozenset(refs - dimensions - logical_values)


def _axis_extents(
    j: SemanticArchive, axis_candidates: frozenset[str]
) -> dict[str, int | str]:
    accesses = j_elaboration(j).access_by_id
    tensors = {
        tensor.name: tensor for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    result: dict[str, int | str] = {}
    authored = {item.access_id: item for item in j.boundary.accesses}
    for access_id, structured in accesses.items():
        tensor = tensors[authored[access_id].tensor]
        for position, component in enumerate(structured.index.components):
            axis = _ref(component)
            if axis not in axis_candidates:
                continue
            extent = tensor.shape[position]
            previous = result.get(axis)
            if previous is not None and previous != extent:
                raise TypedBoundaryMaterializationError(
                    f"logical axis {axis!r} has inconsistent tensor extents"
                )
            result[axis] = extent
    return result


def _requires_typed_lowering(j: SemanticArchive) -> bool:
    elaboration = j_elaboration(j)
    if any(
        operation.op == "effect_if"
        and any(
            nested.op == "loop"
            for arm in (operation.positive, operation.negative)
            if arm is not None
            for _nested_path, nested in _walk(arm.body)
        )
        for _path, operation in _canonical_walk(j)
    ):
        return True
    axes = _axis_candidates(j)
    authored = {item.access_id: item for item in j.boundary.accesses}
    by_path = dict(_canonical_walk(j))
    nodes = {
        node_ref: by_path[path]
        for node_ref, path in j.node_ids.items()
        if path in by_path
    }
    # The first production profile handles separable one-axis remaps.  A
    # mixed-radix/linearized coordinate that combines several logical axes is a
    # different lowering and remains with the existing closed projection until
    # that profile is migrated; it must not be partially interpreted here.
    if any(
        authored[access.access_id].kind.value == "store"
        and any(
            component.op not in {"ref", "int"}
            and len(component.references & axes) == 1
            for component in access.index.components
        )
        for access in elaboration.accesses
    ):
        return True
    return any(
        access.index.direct_coordinates is None
        and authored[access.access_id].node_ref in nodes
        and nodes[authored[access.access_id].node_ref].op == "gather"
        for access in elaboration.accesses
    )


def materialize_typed_boundary(
    j: SemanticArchive,
    function: IntentFunction,
    concrete_extents: Mapping[str, int],
    *,
    consume_loop_storage: bool = False,
) -> TypedBoundaryFunctionMaterialization | None:
    """Lower nontrivial typed accesses/effects into an executable function copy."""

    j = elaborate_j(j)
    if not _requires_typed_lowering(j):
        return None
    contract = project_contract_ir(j)
    structured_loop_nodes = tuple(
        sorted(
            node_ref
            for node_ref, path in j.node_ids.items()
            if path in dict(_canonical_walk(j))
            and dict(_canonical_walk(j))[path].op == "loop"
        )
    )
    logical_projection = project_contract_materialization(
        j,
        consume_loop_storage=(consume_loop_storage or bool(structured_loop_nodes)),
    )
    value_types = {
        name: (item.dtype, _shape(item.shape))
        for name, item in j_elaboration(j).value_types
    }
    tensor_types = {
        tensor.name: (tensor.dtype, _shape(tensor.shape))
        for tensor in [*j.intent.inputs, *j.intent.outputs]
    }
    value_types.update(tensor_types)
    occupied = set(value_types)
    for _path, operation in _walk(function.body):
        occupied.update(_results(operation))
        occupied.update(operation.operands)
    fresh = _Fresh(occupied)
    axis_candidates = _axis_candidates(j)
    axis_extents = _axis_extents(j, axis_candidates)
    value_dimensions = {
        relation.value_ref: relation.dimension_ref
        for relation in j.value_dimension_relations
    }
    path_to_node = j_elaboration(j).authored_node_by_path(j.node_ids)
    operation_records = j_elaboration(j).operation_records
    record_by_path = {
        record["path"]: record for record in operation_records
    }
    typed_by_path = dict(_canonical_walk(j))
    if set(record_by_path) != set(typed_by_path):
        raise TypedBoundaryMaterializationError(
            "typed boundary operation paths disagree with canonical J records"
        )
    canonical_node_results: dict[str, str] = {}
    for path, record in record_by_path.items():
        results = tuple(record.get("results", ()))
        if not results:
            continue
        operation = typed_by_path[path]
        typed_results = _results(operation)
        if tuple(typed_results) != results:
            raise TypedBoundaryMaterializationError(
                f"typed boundary result record disagrees at {path!r}"
            )
        node_ref = record["id"]
        authored_ref = path_to_node.get(path)
        if authored_ref is not None and authored_ref != node_ref:
            raise TypedBoundaryMaterializationError(
                f"typed boundary node record disagrees at {path!r}"
            )
        if len(results) == 1:
            canonical_node_results[node_ref] = results[0]
    accesses_by_node: dict[str, list[ContractAccessIR]] = {}
    for access in contract.accesses:
        if access.node_ref is not None:
            accesses_by_node.setdefault(access.node_ref, []).append(access)
    authored_access = {item.access_id: item for item in j.boundary.accesses}

    records: list[TypedBoundaryMaterializationRecord] = []
    synthetic_bounds: set[str] = set()
    base_broadcast_axes = logical_projection.emitter_broadcast_axes()
    synthetic_broadcasts: dict[str, dict[str, tuple[int, ...]]] = {}
    node_results: dict[str, str] = {}
    node_guards: dict[str, str | None] = {}
    rename: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in rename and name not in seen:
            seen.add(name)
            name = rename[name]
        return name

    def value(name: str) -> _Value:
        resolved = resolve(name)
        dtype, shape = value_types.get(
            resolved, value_types.get(name, (None, None))
        )
        if dtype is None or shape is None:
            raise TypedBoundaryMaterializationError(
                f"typed boundary value {resolved!r} has no type"
            )
        return _Value(resolved, dtype, shape)

    def register(operation: IntentOp) -> None:
        if not isinstance(operation.result, str) or not isinstance(operation.type, dict):
            return
        dtype = operation.type.get("dtype")
        shape = operation.type.get("shape", [])
        if isinstance(dtype, str) and isinstance(shape, list):
            value_types[operation.result] = (dtype, _shape(shape))

    def append_generated(target: list[IntentOp], generated: list[IntentOp]) -> None:
        for operation in generated:
            register(operation)
            target.append(operation)

    def make_axis_values(
        expression: ContractExpr | None,
        target: list[IntentOp],
    ) -> dict[str, _Value]:
        if expression is None:
            return {}
        result: dict[str, _Value] = {}
        builder = _ExpressionBuilder(
            target, fresh, value_types, concrete_extents, result
        )
        for axis in sorted(expression.references & axis_candidates):
            extent = axis_extents.get(axis)
            if extent is None:
                raise TypedBoundaryMaterializationError(
                    f"logical axis {axis!r} lacks a closed extent"
                )
            result[axis] = builder.iota(extent, axis)
            register(target[-1])
        return result

    def access_value(
        access: ContractAccessIR,
        execution_guard: str | None,
        target: list[IntentOp],
    ) -> _Value:
        authored = authored_access[access.access_id]
        current = value(access.tensor)
        components = [program.expression for program in access.index]
        constant_positions: list[int] = []
        generated_names: list[str] = []

        # Literal coordinates select a semantic slice and remove that axis.
        for position, component in enumerate(components):
            literal = _int(component)
            if literal is None:
                continue
            generated: list[IntentOp] = []
            builder = _ExpressionBuilder(
                generated, fresh, value_types, concrete_extents, {}
            )
            index = builder.constant(literal)
            axis = position - sum(item < position for item in constant_positions)
            result_shape = list(current.shape)
            result_shape.pop(axis)
            result_name = fresh("constant_gather")
            node_id = f"typed_boundary:{access.access_id}:constant:{position}"
            generated.append(
                IntentOp(
                    op="gather",
                    result=result_name,
                    operands=[current.name, index.name],
                    attrs={"axis": axis, "mode": "take"},
                    type={"dtype": current.dtype, "shape": result_shape},
                    node_id=node_id,
                )
            )
            append_generated(target, generated)
            synthetic_bounds.add(node_id)
            generated_names.extend(item.result for item in generated if isinstance(item.result, str))
            current = _Value(result_name, current.dtype, tuple(result_shape))
            constant_positions.append(position)

        transformed = [
            (position, component)
            for position, component in enumerate(components)
            if _ref(component) is None and _int(component) is None
        ]
        if len(transformed) > 1:
            raise TypedBoundaryMaterializationError(
                f"access {access.access_id!r} transforms more than one axis"
            )

        predicate_axes = make_axis_values(
            access.predicate.expression if access.predicate is not None else None,
            target,
        )
        generated: list[IntentOp] = []
        builder = _ExpressionBuilder(
            generated, fresh, value_types, concrete_extents, predicate_axes
        )
        predicate = _predicate(
            access.predicate.expression if access.predicate is not None else None,
            builder,
            axis_extents,
            value_dimensions,
        )
        if execution_guard is not None:
            predicate = _combine_guard(
                value(execution_guard), predicate, builder
            )

        if transformed:
            position, component = transformed[0]
            axes = component.references & axis_candidates
            if len(axes) != 1:
                raise TypedBoundaryMaterializationError(
                    f"access {access.access_id!r} transformed index needs one logical axis"
                )
            axis_ref = next(iter(axes))
            if axis_ref not in predicate_axes:
                extent = axis_extents[axis_ref]
                predicate_axes[axis_ref] = builder.iota(extent, axis_ref)
                builder.axis_values[axis_ref] = predicate_axes[axis_ref]
            index = builder.emit(component)
            if predicate is not None:
                index = builder.select(predicate, index, builder.constant(0))
            axis = position - sum(item < position for item in constant_positions)
            result_shape = list(current.shape)
            result_shape[axis:axis + 1] = list(index.shape)
            result_name = fresh("indexed_gather")
            node_id = f"typed_boundary:{access.access_id}:index"
            generated.append(
                IntentOp(
                    op="gather",
                    result=result_name,
                    operands=[current.name, index.name],
                    attrs={"axis": axis, "mode": "take"},
                    type={"dtype": current.dtype, "shape": result_shape},
                    node_id=node_id,
                )
            )
            current = _Value(result_name, current.dtype, tuple(result_shape))
            synthetic_bounds.add(node_id)

        if predicate is not None:
            fill = authored.fill_value
            if fill is None:
                fill = False if current.dtype == "i1" else 0
            if isinstance(fill, str) and fill in value_types:
                fill_value = value(fill)
            else:
                fill_value = builder.constant(fill, current.dtype)
            consumer_result = canonical_node_results.get(access.node_ref)
            axes = None
            if consumer_result is not None:
                axes = base_broadcast_axes.get(consumer_result, {}).get(
                    fill_value.name
                )
            current = builder.select(
                predicate,
                current,
                fill_value,
                result_shape=current.shape if axes is not None else None,
            )
            if axes is not None:
                synthetic_broadcasts.setdefault(current.name, {})[
                    fill_value.name
                ] = tuple(axes)

        append_generated(target, generated)
        generated_names.extend(
            item.result for item in generated if isinstance(item.result, str)
        )
        records.append(
            TypedBoundaryMaterializationRecord(
                access.access_id,
                access.kind,
                access.node_ref or "",
                tuple(generated_names),
            )
        )
        return current

    def combine_guard_names(
        outer: str | None,
        inner: str,
        target: list[IntentOp],
    ) -> str:
        if outer is None:
            return inner
        name = fresh("branch_and")
        operation = IntentOp(
            op="map_binary",
            result=name,
            operands=[resolve(outer), resolve(inner)],
            attrs={"fn": "logical_and"},
            type={"dtype": "i1", "shape": []},
        )
        register(operation)
        target.append(operation)
        return name

    def rewrite_body(
        body: Sequence[IntentOp],
        prefix: str,
        execution_guard: str | None,
    ) -> list[IntentOp]:
        rewritten: list[IntentOp] = []
        for index, original in enumerate(body):
            path = f"{prefix}[{index}]"
            node_ref = original.node_id or path_to_node.get(path)
            if original.op == "effect_if":
                if (
                    not isinstance(original.predicate, str)
                    or original.positive is None
                    or original.negative is None
                ):
                    raise TypedBoundaryMaterializationError(
                        "effect_if requires a predicate and two explicit arms"
                    )
                condition = resolve(original.predicate)
                negated = fresh("branch_not")
                not_op = IntentOp(
                    op="map_unary",
                    result=negated,
                    operands=[condition],
                    attrs={"fn": "logical_not"},
                    type={"dtype": "i1", "shape": []},
                )
                register(not_op)
                rewritten.append(not_op)
                positive_guard = combine_guard_names(
                    execution_guard, condition, rewritten
                )
                negative_guard = combine_guard_names(
                    execution_guard, negated, rewritten
                )
                rewritten.extend(
                    rewrite_body(
                        original.positive.body,
                        f"{path}.positive.body",
                        positive_guard,
                    )
                )
                rewritten.extend(
                    rewrite_body(
                        original.negative.body,
                        f"{path}.negative.body",
                        negative_guard,
                    )
                )
                continue

            if original.op == "loop":
                if original.trip is None:
                    raise TypedBoundaryMaterializationError(
                        "semantic loop requires a closed trip axis"
                    )
                if original.trip.is_dynamic:
                    # A runtime-bounded loop cannot be inlined into the
                    # target-neutral boundary projection: doing so would turn
                    # a data-dependent recurrence into one arbitrary host
                    # iteration.  Preserve the loop ABI and recursively lower
                    # only its J-owned body.  The MLIR emitter and CPU/native
                    # consumers then execute the same upper_bound/index form.
                    operation = copy.deepcopy(original)
                    operation.node_id = node_ref
                    operation.trip.upper_bound = resolve(
                        operation.trip.upper_bound
                    )
                    for scan in operation.scan:
                        scan.source = resolve(scan.source)
                    for state in operation.carry:
                        state.init_value = resolve(state.init_value)
                    operation.body = rewrite_body(
                        original.body,
                        f"{path}.body",
                        execution_guard,
                    )
                    rewritten.append(operation)
                    if node_ref is not None and _results(operation):
                        node_results[node_ref] = _results(operation)[0]
                        node_guards[node_ref] = execution_guard
                    continue
                # M0 chooses one deterministic full-axis step.  Every scan slice
                # is therefore the complete source tensor and the structured
                # carry recurrence has exactly one iteration; inline that one
                # step as ordinary SSA while retaining the loop consumption in
                # the projection.  Schedule search may later retile the same
                # semantic loop, but the direct lowering stays deterministic.
                for scan in original.scan:
                    rename[scan.name] = resolve(scan.source)
                for state in original.carry:
                    rename[state.body_argument] = resolve(state.init_value)
                rewritten.extend(
                    rewrite_body(
                        original.body,
                        f"{path}.body",
                        execution_guard,
                    )
                )
                for state in original.carry:
                    rename[state.result_value] = resolve(state.next_value)
                if node_ref is not None and original.carry:
                    node_results[node_ref] = resolve(
                        original.carry[0].next_value
                    )
                    node_guards[node_ref] = execution_guard
                continue

            operation = copy.deepcopy(original)
            operation.node_id = node_ref
            original_operands = list(operation.operands)
            operation.operands = [resolve(item) for item in operation.operands]
            for scan in operation.scan:
                scan.source = resolve(scan.source)
            for state in operation.carry:
                state.init_value = resolve(state.init_value)

            node_accesses = accesses_by_node.get(node_ref or "", [])
            load_accesses = [item for item in node_accesses if item.kind == "load"]
            post_accesses: list[ContractAccessIR] = []
            for access in load_accesses:
                # A gather already carries its index as Layer-A dataflow.  Keep
                # that operation and materialize only its guard/fill after it.
                if operation.op == "gather" and operation.operands[0] == resolve(
                    access.tensor
                ):
                    post_accesses.append(access)
                    continue
                positions = [
                    pos
                    for pos, operand in enumerate(operation.operands)
                    if operand == resolve(access.tensor)
                ]
                if len(positions) != 1:
                    predicate_expression = (
                        access.predicate.expression
                        if access.predicate is not None else None
                    )
                    direct = all(
                        _ref(program.expression) is not None
                        or _int(program.expression) is not None
                        for program in access.index
                    )
                    residual = (
                        []
                        if predicate_expression is None
                        else [
                            term
                            for term in _flatten_and(predicate_expression)
                            if not _is_full_domain_term(
                                term, axis_extents, value_dimensions
                            )
                        ]
                    )
                    # A loop scan is an indirect consumer of its source tensor.
                    # When both its index and predicate are exactly the complete
                    # logical domain, the structured loop already materializes
                    # that access and no extra operand rewrite is needed.
                    if direct and not residual:
                        continue
                    raise TypedBoundaryMaterializationError(
                        f"access {access.access_id!r} is not one unique consumer operand"
                    )
                loaded = access_value(access, execution_guard, rewritten)
                operation.operands[positions[0]] = loaded.name

            if isinstance(operation.result, str):
                authored_axes = base_broadcast_axes.get(operation.result, {})
                for previous, current in zip(
                    original_operands, operation.operands, strict=True
                ):
                    axes = authored_axes.get(previous)
                    if axes is not None and current != previous:
                        synthetic_broadcasts.setdefault(operation.result, {})[
                            current
                        ] = tuple(axes)

            # A branch-local gather must not evaluate an out-of-range index on
            # the arm that is semantically inactive.
            if operation.op == "gather" and execution_guard is not None:
                generated: list[IntentOp] = []
                builder = _ExpressionBuilder(
                    generated, fresh, value_types, concrete_extents, {}
                )
                index = value(operation.operands[1])
                safe = builder.select(
                    value(execution_guard), index, builder.constant(0, index.dtype)
                )
                append_generated(rewritten, generated)
                operation.operands[1] = safe.name

            rewritten.append(operation)
            register(operation)
            if node_ref is not None and _results(operation):
                node_results[node_ref] = _results(operation)[0]
                node_guards[node_ref] = execution_guard

            for access in post_accesses:
                if not isinstance(operation.result, str):
                    raise TypedBoundaryMaterializationError(
                        "guarded gather requires one result"
                    )
                generated: list[IntentOp] = []
                axes = make_axis_values(
                    access.predicate.expression
                    if access.predicate is not None else None,
                    generated,
                )
                builder = _ExpressionBuilder(
                    generated, fresh, value_types, concrete_extents, axes
                )
                predicate = _predicate(
                    access.predicate.expression
                    if access.predicate is not None else None,
                    builder,
                    axis_extents,
                    value_dimensions,
                )
                if execution_guard is not None:
                    predicate = _combine_guard(
                        value(execution_guard), predicate, builder
                    )
                if predicate is None:
                    # The Layer-A gather already owns the typed index dataflow;
                    # a J-proved full-domain predicate needs no extra select,
                    # but the access is nevertheless completely consumed.
                    records.append(
                        TypedBoundaryMaterializationRecord(
                            access.access_id,
                            access.kind,
                            access.node_ref or "",
                            (),
                        )
                    )
                    continue
                authored = authored_access[access.access_id]
                result = value(operation.result)
                fill = authored.fill_value
                if fill is None:
                    fill = False if result.dtype == "i1" else 0
                masked = builder.select(
                    predicate, result, builder.constant(fill, result.dtype)
                )
                append_generated(rewritten, generated)
                rename[operation.result] = masked.name
                records.append(
                    TypedBoundaryMaterializationRecord(
                        access.access_id,
                        access.kind,
                        access.node_ref or "",
                        tuple(
                            item.result
                            for item in generated
                            if isinstance(item.result, str)
                        ),
                    )
                )
        return rewritten

    transformed_body = rewrite_body(function.body, "body", None)

    # Materialize boundary stores only when the canonical Layer-A return is an
    # identity/unproduced destination.  Explicit produced returns (for example
    # concat-based MRoPE) already encode their writeback value in Layer A.
    input_names = {item.name for item in function.inputs}
    output_by_name = {item.name: item for item in function.outputs}
    output_bindings = j_elaboration(j).output_bindings
    if tuple(item.output for item in output_bindings) != tuple(output_by_name):
        raise TypedBoundaryMaterializationError(
            "canonical output bindings do not cover materialized function outputs "
            "in ABI order"
        )
    canonical_outputs = {
        item.output: item.value for item in output_bindings
    }
    store_accesses: dict[str, list[ContractAccessIR]] = {}
    for access in contract.accesses:
        if access.kind == "store":
            store_accesses.setdefault(access.tensor, []).append(access)

    final_outputs: dict[str, str] = {}
    for binding in output_bindings:
        output_name = binding.output
        output = output_by_name[output_name]
        try:
            canonical_value = resolve(canonical_outputs[output_name])
        except KeyError as exc:
            raise TypedBoundaryMaterializationError(
                f"output {output_name!r} has no complete canonical return binding"
            ) from exc
        stores = store_accesses.get(output_name, [])
        if not stores:
            final_outputs[output_name] = canonical_value
            continue

        output_value = _Value(output.name, output.dtype, _shape(output.shape))
        if output_name in input_names:
            accumulator = output_value
        else:
            generated: list[IntentOp] = []
            builder = _ExpressionBuilder(
                generated, fresh, value_types, concrete_extents, {}
            )
            zero = fresh("output_zero")
            zero_op = IntentOp(
                op="const",
                result=zero,
                attrs={"value": 0, "dtype": output.dtype, "shape": list(output.shape)},
                type={"dtype": output.dtype, "shape": list(output.shape)},
            )
            generated.append(zero_op)
            append_generated(transformed_body, generated)
            accumulator = _Value(zero, output.dtype, _shape(output.shape))

        for access in stores:
            if access.node_ref not in node_results:
                raise TypedBoundaryMaterializationError(
                    f"store {access.access_id!r} producer is unavailable"
                )
            producer = value(node_results[access.node_ref])
            components = [program.expression for program in access.index]
            transformed = [
                (position, component)
                for position, component in enumerate(components)
                if _ref(component) is None and _int(component) is None
            ]
            if len(transformed) > 1 or any(_int(item) is not None for item in components):
                raise TypedBoundaryMaterializationError(
                    f"store {access.access_id!r} has unsupported multidimensional remap"
                )
            generated: list[IntentOp] = []
            axis_values: dict[str, _Value] = {}
            builder = _ExpressionBuilder(
                generated, fresh, value_types, concrete_extents, axis_values
            )
            candidate = producer
            bounds: _Value | None = None

            if transformed:
                position, component = transformed[0]
                axes = component.references & axis_candidates
                if len(axes) != 1:
                    raise TypedBoundaryMaterializationError(
                        f"store {access.access_id!r} remap needs one logical axis"
                    )
                axis_ref = next(iter(axes))
                scale = _positive_axis_scale(component, axis_ref)
                offset_expression: ContractExpr | None = None
                if scale is None and component.op == "add" and len(component.args) == 2:
                    lhs, rhs = component.args
                    lhs_scale = _positive_axis_scale(lhs, axis_ref)
                    rhs_scale = _positive_axis_scale(rhs, axis_ref)
                    if lhs_scale is not None and axis_ref not in rhs.references:
                        scale, offset_expression = lhs_scale, rhs
                    elif rhs_scale is not None and axis_ref not in lhs.references:
                        scale, offset_expression = rhs_scale, lhs
                if scale is None:
                    raise TypedBoundaryMaterializationError(
                        f"store {access.access_id!r} remap is not invertible affine"
                    )
                source_extent = axis_extents[axis_ref]
                physical_extent = output.shape[position]
                physical = builder.iota(physical_extent, f"store_{axis_ref}")
                offset = (
                    builder.emit(offset_expression)
                    if offset_expression is not None
                    else builder.constant(0)
                )
                shifted = builder.binary("sub", physical, offset, "i64")
                if scale == 1:
                    source_index = shifted
                    aligned = None
                else:
                    divisor = builder.constant(scale)
                    source_index = builder.binary("div", shifted, divisor, "i64")
                    quotient_product = builder.binary(
                        "mul", source_index, divisor, "i64"
                    )
                    remainder = builder.binary(
                        "sub", shifted, quotient_product, "i64"
                    )
                    aligned = builder.binary(
                        "eq", remainder, builder.constant(0), "i1"
                    )
                axis_values[axis_ref] = source_index
                builder.axis_values[axis_ref] = source_index
                zero = builder.constant(0)
                high = builder.constant(
                    int(concrete_extents[source_extent])
                    if isinstance(source_extent, str)
                    else int(source_extent)
                )
                lower_ok = builder.binary("ge", source_index, zero, "i1")
                upper_ok = builder.binary("lt", source_index, high, "i1")
                bounds = builder.binary("logical_and", lower_ok, upper_ok, "i1")
                bounds = _combine_guard(aligned, bounds, builder)
                safe_index = builder.select(bounds, source_index, zero)
                gathered = fresh("store_inverse_gather")
                node_id = f"typed_boundary:{access.access_id}:store_inverse"
                gather_shape = list(producer.shape)
                gather_shape[position:position + 1] = list(safe_index.shape)
                generated.append(
                    IntentOp(
                        op="gather",
                        result=gathered,
                        operands=[producer.name, safe_index.name],
                        attrs={"axis": position, "mode": "take"},
                        type={"dtype": producer.dtype, "shape": gather_shape},
                        node_id=node_id,
                    )
                )
                candidate = _Value(gathered, producer.dtype, tuple(gather_shape))
                synthetic_bounds.add(node_id)
            else:
                predicate_expression = (
                    access.predicate.expression
                    if access.predicate is not None else None
                )
                for axis in sorted(
                    (predicate_expression.references if predicate_expression else set())
                    & axis_candidates
                ):
                    extent = axis_extents[axis]
                    axis_values[axis] = builder.iota(extent, axis)
                    builder.axis_values[axis] = axis_values[axis]

            predicate = _predicate(
                access.predicate.expression if access.predicate is not None else None,
                builder,
                axis_extents,
                value_dimensions,
            )
            predicate = _combine_guard(bounds, predicate, builder)
            guard_name = node_guards.get(access.node_ref)
            if guard_name is not None:
                predicate = _combine_guard(value(guard_name), predicate, builder)
            if predicate is not None:
                candidate = builder.select(predicate, candidate, accumulator)
            accumulator = candidate
            append_generated(transformed_body, generated)
            records.append(
                TypedBoundaryMaterializationRecord(
                    access.access_id,
                    access.kind,
                    access.node_ref or "",
                    tuple(
                        item.result
                        for item in generated
                        if isinstance(item.result, str)
                    ),
                )
            )
        final_outputs[output_name] = accumulator.name

    # Layer A may contain a logical concat whose operands are full-shape masked
    # packets while Layer B assigns each packet to a disjoint destination
    # partition.  Once stores have been materialized, that logical concat (and
    # any other now-unreachable pure scaffolding) must not be reinterpreted as a
    # physical full-width concatenation by the ordinary tensor emitter.
    live: set[str] = set(final_outputs.values())
    compact_body: list[IntentOp] = []
    for operation in reversed(transformed_body):
        produced = set(_results(operation))
        keep = not produced or bool(produced & live)
        if not keep:
            continue
        compact_body.append(operation)
        live.update(operation.operands)
        if operation.trip is not None and operation.trip.is_dynamic:
            live.add(operation.trip.upper_bound)
        if operation.predicate is not None:
            live.add(operation.predicate)
        for scan in operation.scan:
            live.add(scan.source)
        for state in operation.carry:
            live.add(state.init_value)
    compact_body.reverse()

    transformed = IntentFunction(
        name=function.name,
        inputs=copy.deepcopy(function.inputs),
        outputs=copy.deepcopy(function.outputs),
        body=compact_body,
        returns=[
            IntentReturn(output=binding.output, value=final_outputs[binding.output])
            for binding in output_bindings
        ],
    )
    return TypedBoundaryFunctionMaterialization(
        transformed,
        TypedBoundaryMaterializationProjection(
            logical_projection,
            tuple(records),
            frozenset(synthetic_bounds),
            tuple(
                (
                    output,
                    tuple(sorted(operands.items())),
                )
                for output, operands in sorted(synthetic_broadcasts.items())
            ),
            structured_loop_nodes,
        ),
    )


__all__ = [
    "TYPED_BOUNDARY_MATERIALIZATION_PROTOCOL",
    "TypedBoundaryFunctionMaterialization",
    "TypedBoundaryMaterializationError",
    "TypedBoundaryMaterializationProjection",
    "materialize_typed_boundary",
]
