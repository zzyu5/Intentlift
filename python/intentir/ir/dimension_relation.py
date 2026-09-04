"""Closed validation and consumption of J-owned dimension definitions.

The relations are authored by Lift.  Nothing in this module infers a relation
from concrete shapes, Harness values, grid, C, pi, Source, or compiler output.
Consumers may only validate or evaluate the exact closed expression carried by
J.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .dim_expr import dimension_expression_references


DIMENSION_INT64_MIN = -(2**63)
DIMENSION_INT64_MAX = 2**63 - 1
DIMENSION_RELATION_CONST_OPS = frozenset(
    {
        "add_const",
        "sub_const",
        "mul_const",
        "floordiv_const",
        "mod_const",
        "ceil_div_const",
        "round_up_const",
    }
)
DIMENSION_RELATION_OPS = frozenset(
    {"identity", "mul_dims", *DIMENSION_RELATION_CONST_OPS}
)


class DimensionRelationError(ValueError):
    """A J dimension relation or its public valuation is not closed."""


def declared_dimension_handles(j: Any) -> frozenset[str]:
    """Return every symbolic dimension handle declared by canonical J.

    Shapes, logical signature bindings, and derived-dimension relation results
    declare the same J-owned namespace.  Relation operands and
    value/dimension equalities only refer to that namespace; they do not create
    declarations.  Keep this collection in one place so TypeCheck and runtime
    relation consumers cannot disagree about whether an expression is closed.
    ``?`` is a data-dependent extent marker, not a public symbolic dimension.
    """

    dimensions = {
        reference
        for tensor in [*j.intent.inputs, *j.intent.outputs]
        for dimension in tensor.shape
        # ``?`` is the closed dynamic-extent marker for a data-dependent
        # result (currently ragged_member).  It is not a J dimension handle and
        # must not become a missing public dimension valuation.
        for reference in dimension_expression_references(dimension)
        if reference != "?"
    }
    signature = getattr(j, "signature", None)
    for parameter in getattr(signature, "parameters", ()):
        logical_ref = getattr(parameter, "logical_ref", None)
        if isinstance(logical_ref, str) and re.fullmatch(r"x[0-9]+", logical_ref):
            dimensions.add(logical_ref)
    # A relation result is a declaration of a derived dimension.  Expression
    # arguments are references to declarations, not declarations themselves;
    # treating them as declarations would make a misspelled or self-referential
    # x* handle appear closed.  The same rule applies to a value/dimension
    # equality: its dimension_ref consumes an already declared dimension and
    # must never bootstrap one.
    for relation in getattr(j, "dimension_relations", ()):
        reference = getattr(relation, "result", None)
        if isinstance(reference, str) and reference != "?":
            dimensions.add(reference)
    return frozenset(dimensions)


def _relation_map(j: Any) -> dict[str, Any]:
    relations: dict[str, Any] = {}
    for index, relation in enumerate(j.dimension_relations):
        path = f"j.dimension_relations[{index}]"
        if relation.result in relations:
            raise DimensionRelationError(
                f"{path}.result duplicates an earlier dimension definition"
            )
        relations[relation.result] = relation
    return relations


def validate_dimension_relations(j: Any) -> None:
    """Validate references, closed operations, unique results, and acyclicity."""

    declared = declared_dimension_handles(j)
    relations = _relation_map(j)
    for index, relation in enumerate(j.dimension_relations):
        path = f"j.dimension_relations[{index}]"
        if relation.result not in declared:
            raise DimensionRelationError(
                f"{path}.result must name a declared symbolic x* dimension"
            )
        if relation.expr.arg not in declared:
            raise DimensionRelationError(
                f"{path}.expr.arg must name a declared symbolic x* dimension"
            )
        if relation.expr.rhs is not None and relation.expr.rhs not in declared:
            raise DimensionRelationError(
                f"{path}.expr.rhs must name a declared symbolic x* dimension"
            )
        if relation.expr.op == "identity":
            if relation.expr.value is not None or relation.expr.rhs is not None:
                raise DimensionRelationError(
                    f"{path}.expr identity forbids value/rhs"
                )
        elif relation.expr.op == "mul_dims":
            if relation.expr.rhs is None or relation.expr.value is not None:
                raise DimensionRelationError(
                    f"{path}.expr mul_dims requires rhs and forbids value"
                )
        elif relation.expr.op in DIMENSION_RELATION_CONST_OPS:
            if (
                isinstance(relation.expr.value, bool)
                or not isinstance(relation.expr.value, int)
                or relation.expr.value <= 0
                or relation.expr.value > DIMENSION_INT64_MAX
            ):
                raise DimensionRelationError(
                    f"{path}.expr requires a positive signed-int64 constant"
                )
        else:
            raise DimensionRelationError(
                f"{path}.expr.op is outside the closed dimension grammar"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(result: str) -> None:
        if result in visited:
            return
        if result in visiting:
            raise DimensionRelationError(
                "j.dimension_relations contains a dependency cycle at "
                f"{result!r}"
            )
        visiting.add(result)
        expr = relations[result].expr
        dependencies = [expr.arg]
        if expr.rhs is not None:
            dependencies.append(expr.rhs)
        for argument in dependencies:
            if argument in relations:
                visit(argument)
        visiting.remove(result)
        visited.add(result)

    for result in relations:
        visit(result)


def _closed_extent(value: Any, *, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > DIMENSION_INT64_MAX
    ):
        raise DimensionRelationError(
            f"{context} must be a non-negative signed-int64 dimension"
        )
    return value


def evaluate_dimension_relation_operation(
    op: str,
    argument: int,
    *,
    value: int | None = None,
    rhs_argument: int | None = None,
    context: str = "dimension relation",
) -> int:
    """Evaluate one operation in the shared closed dimension grammar."""

    if op not in DIMENSION_RELATION_OPS:
        raise DimensionRelationError(
            f"{context} uses unsupported dimension relation operation {op!r}"
        )
    argument = _closed_extent(argument, context=f"{context}.arg")

    def result_extent(result: int) -> int:
        if result < 0:
            raise DimensionRelationError(
                f"{context} produces a negative dimension extent"
            )
        if result < DIMENSION_INT64_MIN or result > DIMENSION_INT64_MAX:
            raise DimensionRelationError(
                f"{context} exceeds the signed 64-bit dimension domain"
            )
        return result

    if op == "identity":
        if value is not None or rhs_argument is not None:
            raise DimensionRelationError(
                f"{context} identity forbids value/rhs"
            )
        return argument
    if op == "mul_dims":
        if value is not None or rhs_argument is None:
            raise DimensionRelationError(
                f"{context} mul_dims requires rhs and forbids value"
            )
        rhs_argument = _closed_extent(
            rhs_argument, context=f"{context}.rhs"
        )
        return result_extent(argument * rhs_argument)
    if op not in DIMENSION_RELATION_CONST_OPS:
        raise DimensionRelationError(
            f"{context} uses unsupported dimension relation operation {op!r}"
        )
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > DIMENSION_INT64_MAX
        or rhs_argument is not None
    ):
        raise DimensionRelationError(
            f"{context} {op} requires one positive signed-int64 value"
        )
    if op == "add_const":
        result = argument + value
    elif op == "sub_const":
        result = argument - value
    elif op == "mul_const":
        result = argument * value
    elif op == "floordiv_const":
        result = argument // value
    elif op == "mod_const":
        result = argument % value
    elif op == "ceil_div_const":
        # Avoid argument + value - 1 overflowing before the checked result.
        result = argument // value + int(argument % value != 0)
    else:
        quotient = argument // value + int(argument % value != 0)
        result = quotient * value
    return result_extent(result)


def evaluate_dimension_relation_expr(
    expr: Any, argument: int, rhs_argument: int | None = None
) -> int:
    """Evaluate one already-validated closed expression."""

    return evaluate_dimension_relation_operation(
        expr.op,
        argument,
        value=expr.value,
        rhs_argument=rhs_argument,
        context=f"dimension relation {expr.op}",
    )


def resolve_dimension_relations(
    j: Any,
    dimensions: Mapping[str, int],
) -> dict[str, int]:
    """Evaluate authored relations from public base valuations, failing closed.

    A supplied result valuation is checked rather than overwritten.  Missing
    derived results are filled only from the exact J relation.  A missing base
    dimension remains an error; no Harness or tensor-shape inference occurs here.
    """

    validate_dimension_relations(j)
    declared = declared_dimension_handles(j)
    resolved = {
        name: _closed_extent(value, context=f"dimension {name!r}")
        for name, value in dimensions.items()
        if name in declared
    }
    unexpected = sorted(set(dimensions) - declared)
    if unexpected:
        raise DimensionRelationError(
            "public dimension valuation contains unknown J dimension(s): "
            + ", ".join(unexpected)
        )
    relations = _relation_map(j)

    pending = set(relations)
    while pending:
        progress = False
        for result in tuple(sorted(pending)):
            relation = relations[result]
            dependencies = [relation.expr.arg]
            if relation.expr.rhs is not None:
                dependencies.append(relation.expr.rhs)
            if any(dependency not in resolved for dependency in dependencies):
                continue
            expected = evaluate_dimension_relation_expr(
                relation.expr,
                resolved[relation.expr.arg],
                (
                    resolved[relation.expr.rhs]
                    if relation.expr.rhs is not None
                    else None
                ),
            )
            if result in resolved and resolved[result] != expected:
                raise DimensionRelationError(
                    "J dimension relation mismatch: "
                    f"{result}={resolved[result]}, expected {expected} from "
                    f"{relation.expr.op}({relation.expr.arg}"
                    + (
                        f", {relation.expr.rhs})"
                        if relation.expr.rhs is not None
                        else ")"
                    )
                )
            resolved[result] = expected
            pending.remove(result)
            progress = True
        if not progress:
            missing = sorted({
                dependency
                for result in pending
                for dependency in (
                    relations[result].expr.arg,
                    *((relations[result].expr.rhs,) if relations[result].expr.rhs is not None else ()),
                )
                if dependency not in resolved
            })
            raise DimensionRelationError(
                "J dimension relations lack base valuation(s): " + ", ".join(missing)
            )

    missing_declared = sorted(declared - set(resolved))
    if missing_declared:
        raise DimensionRelationError(
            "public instantiation lacks J dimension(s): "
            + ", ".join(missing_declared)
        )
    return {name: resolved[name] for name in sorted(resolved)}


def validate_dimension_relation_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
) -> None:
    """Check one complete public valuation against every authored relation."""

    resolve_dimension_relations(j, dimensions)


__all__ = [
    "DIMENSION_INT64_MAX",
    "DIMENSION_INT64_MIN",
    "DIMENSION_RELATION_CONST_OPS",
    "DIMENSION_RELATION_OPS",
    "DimensionRelationError",
    "declared_dimension_handles",
    "evaluate_dimension_relation_expr",
    "evaluate_dimension_relation_operation",
    "resolve_dimension_relations",
    "validate_dimension_relation_instantiation",
    "validate_dimension_relations",
]
