"""Closed dimension expressions and affine legal-domain constraints.

The Lift model authors both the expression and every non-trivial legal-domain
constraint.  This module is deliberately only a parser and exact arithmetic
checker: it never receives a Harness valuation and never invents a missing
bound.  Its normalized forms are private compiler data, not a second J.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical_j import ContractExpr, ContractSort, parse_contract_expr


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class DimExprError(ValueError):
    """A dimension expression has no closed interpretation."""


def dimension_expression_references(value: Any) -> frozenset[str]:
    """Return symbolic references in one shape/dimension surface value.

    Invalid expressions are returned as one opaque reference so callers that
    build diagnostic environments remain total; the authoritative parser still
    reports the malformed expression at the operation/constraint site.
    """

    if not isinstance(value, str) or value == "?":
        return frozenset()
    try:
        expression = parse_contract_expr(
            value,
            expected=ContractSort.INTEGER,
            context="dimension reference",
        )
    except ValueError:
        return frozenset({value})
    return expression.references


def _checked_integer(value: int, *, context: str) -> int:
    if value < _INT64_MIN or value > _INT64_MAX:
        raise DimExprError(f"{context} exceeds the signed 64-bit dimension domain")
    return value


def _checked_dimension_binding(value: Any, *, name: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DimExprError(f"{context} valuation for {name!r} must be an integer")
    if value < 0:
        raise DimExprError(
            f"{context} valuation for {name!r} must be non-negative"
        )
    return _checked_integer(value, context=context)


def _evaluate_contract_dimension(
    expression: ContractExpr,
    dimensions: Mapping[str, int],
    *,
    context: str,
) -> int:
    """Evaluate one closed integer ContractExpr with checked intermediates."""

    op = expression.op
    if op == "int":
        if not isinstance(expression.value, int) or isinstance(
            expression.value, bool
        ):
            raise DimExprError(f"{context} contains an invalid integer literal")
        return _checked_integer(expression.value, context=context)
    if op == "ref":
        if not isinstance(expression.value, str):
            raise DimExprError(f"{context} contains an invalid dimension reference")
        if expression.value not in dimensions:
            raise DimExprError(
                f"{context} lacks a concrete valuation for {expression.value!r}"
            )
        return _checked_dimension_binding(
            dimensions[expression.value],
            name=expression.value,
            context=context,
        )
    if op in {"pos", "neg"}:
        operand = _evaluate_contract_dimension(
            expression.args[0], dimensions, context=context
        )
        result = operand if op == "pos" else -operand
        return _checked_integer(result, context=context)
    if op in {"add", "sub", "mul", "floordiv", "mod", "left_shift", "right_shift"}:
        left = _evaluate_contract_dimension(
            expression.args[0], dimensions, context=context
        )
        right = _evaluate_contract_dimension(
            expression.args[1], dimensions, context=context
        )
        if op == "add":
            result = left + right
        elif op == "sub":
            result = left - right
        elif op == "mul":
            result = left * right
        elif op in {"floordiv", "mod"}:
            if right == 0:
                raise DimExprError(f"{context} contains a zero divisor")
            result = left // right if op == "floordiv" else left % right
        else:
            if right < 0:
                raise DimExprError(f"{context} contains a negative shift amount")
            # Avoid constructing an unbounded Python integer for an expression
            # that cannot fit the signed int64 result domain.  Zero shifted by
            # any non-negative amount is still zero.
            if op == "left_shift" and left != 0 and right > 63:
                raise DimExprError(
                    f"{context} exceeds the signed 64-bit dimension domain"
                )
            result = left << right if op == "left_shift" else left >> right
        return _checked_integer(result, context=context)
    if op == "call":
        if expression.value not in {"ceil_div", "min", "max"}:
            raise DimExprError(
                f"{context} uses unsupported integer function {expression.value!r}"
            )
        left = _evaluate_contract_dimension(
            expression.args[0], dimensions, context=context
        )
        right = _evaluate_contract_dimension(
            expression.args[1], dimensions, context=context
        )
        if expression.value == "ceil_div":
            if right <= 0:
                raise DimExprError(
                    f"{context} ceil_div denominator must be positive"
                )
            # This form is correct for negative as well as non-negative
            # numerators; the latter is the common dimension-extent case.
            result = -((-left) // right)
        elif expression.value == "min":
            result = min(left, right)
        else:
            result = max(left, right)
        return _checked_integer(result, context=context)
    raise DimExprError(
        f"{context} uses unsupported dimension operator {op!r}"
    )


def _validate_contract_dimension_domain(
    expression: ContractExpr, *, context: str
) -> None:
    """Reject statically impossible divisors and shift amounts."""

    for argument in expression.args:
        _validate_contract_dimension_domain(argument, context=context)
    if expression.op in {"floordiv", "mod"}:
        divisor = expression.args[1]
        if divisor.op == "int" and divisor.value == 0:
            raise DimExprError(f"{context} contains a zero divisor")
    elif expression.op == "left_shift" or expression.op == "right_shift":
        amount = expression.args[1]
        if amount.op == "int" and isinstance(amount.value, int) and amount.value < 0:
            raise DimExprError(f"{context} contains a negative shift amount")
    elif expression.op == "call" and expression.value == "ceil_div":
        denominator = expression.args[1]
        if denominator.op == "int" and (
            not isinstance(denominator.value, int) or denominator.value <= 0
        ):
            raise DimExprError(
                f"{context} ceil_div denominator must be positive"
            )


def _normalized_contract_dimension(expression: ContractExpr) -> ContractExpr:
    """Normalize arithmetic identities without changing non-affine semantics."""

    if expression.op in {"int", "ref", "bool"}:
        return expression
    args = tuple(_normalized_contract_dimension(argument) for argument in expression.args)
    if expression.op == "pos":
        return args[0]
    if expression.op == "neg":
        operand = args[0]
        if operand.op == "int":
            assert isinstance(operand.value, int) and not isinstance(
                operand.value, bool
            )
            return ContractExpr(
                "int", value=-operand.value, sort=ContractSort.INTEGER
            )
        if operand.op == "neg":
            return operand.args[0]
        return ContractExpr("neg", (operand,), sort=ContractSort.INTEGER)
    if expression.op == "sub":
        return _normalized_contract_dimension(
            ContractExpr(
                "add",
                (
                    args[0],
                    ContractExpr("neg", (args[1],), sort=ContractSort.INTEGER),
                ),
                sort=ContractSort.INTEGER,
            )
        )
    if expression.op == "add":
        terms: list[ContractExpr] = []
        constant = 0

        def collect(term: ContractExpr) -> None:
            nonlocal constant
            if term.op == "add":
                for child in term.args:
                    collect(child)
                return
            if term.op == "int":
                assert isinstance(term.value, int) and not isinstance(
                    term.value, bool
                )
                constant += term.value
                return
            terms.append(term)

        for argument in args:
            collect(argument)
        if constant != 0:
            terms.append(
                ContractExpr("int", value=constant, sort=ContractSort.INTEGER)
            )
        terms.sort(key=lambda item: item.to_text())
        if not terms:
            return ContractExpr("int", value=0, sort=ContractSort.INTEGER)
        if len(terms) == 1:
            return terms[0]
        return ContractExpr("add", tuple(terms), sort=ContractSort.INTEGER)
    if expression.op == "mul":
        left, right = args
        if left.op == "int" and right.op == "int":
            assert isinstance(left.value, int) and isinstance(right.value, int)
            return ContractExpr(
                "int", value=left.value * right.value, sort=ContractSort.INTEGER
            )
        if left.op == "int" and left.value == 0:
            return left
        if right.op == "int" and right.value == 0:
            return right
        if left.op == "int" and left.value == 1:
            return right
        if right.op == "int" and right.value == 1:
            return left
    return ContractExpr(
        expression.op,
        args,
        value=expression.value,
        sort=expression.sort,
    )


@dataclass(frozen=True)
class DimensionExpr:
    """A checked integer dimension expression, affine or non-affine."""

    expression: ContractExpr

    @property
    def references(self) -> frozenset[str]:
        return self.expression.references

    @property
    def is_constant(self) -> bool:
        return not self.references

    def to_text(self) -> str:
        return self.expression.to_text()

    def to_json(self) -> dict[str, Any]:
        return self.expression.to_json()

    def evaluate(self, dimensions: Mapping[str, int], *, context: str) -> int:
        return _evaluate_contract_dimension(
            self.expression, dimensions, context=context
        )

    def scaled(self, factor: int, *, context: str) -> "DimensionExpr":
        """Return this expression multiplied by one checked integer factor."""

        if isinstance(factor, bool) or not isinstance(factor, int):
            raise DimExprError(f"{context} scale factor must be an integer")
        _checked_integer(factor, context=context)
        scale = ContractExpr(
            "int", value=factor, sort=ContractSort.INTEGER
        )
        product = ContractExpr(
            "mul",
            (scale, self.expression),
            sort=ContractSort.INTEGER,
        )
        return DimensionExpr(_normalized_contract_dimension(product))

    def plus(
        self,
        other: "DimensionExpr",
        *,
        factor: int = 1,
        context: str,
    ) -> "DimensionExpr":
        """Return ``self + factor * other`` with normalized expression syntax."""

        if isinstance(factor, bool) or not isinstance(factor, int):
            raise DimExprError(f"{context} combination factor must be an integer")
        _checked_integer(factor, context=context)
        combined = ContractExpr(
            "add",
            (self.expression, other.scaled(factor, context=context).expression),
            sort=ContractSort.INTEGER,
        )
        return DimensionExpr(_normalized_contract_dimension(combined))


def equivalent_dimension_expr(left: DimensionExpr, right: DimensionExpr) -> bool:
    return _normalized_contract_dimension(left.expression) == _normalized_contract_dimension(
        right.expression
    )


def parse_dimension_expr(
    value: int | str,
    *,
    dimensions: frozenset[str],
    context: str,
) -> DimensionExpr:
    """Parse one integer extent, including checked non-affine expressions."""

    if isinstance(value, bool):
        raise DimExprError(f"{context} must be an integer dimension expression")
    if isinstance(value, int):
        _checked_integer(value, context=context)
        return DimensionExpr(
            ContractExpr("int", value=value, sort=ContractSort.INTEGER)
        )
    if not isinstance(value, str):
        raise DimExprError(f"{context} must be an integer or ContractExpr string")
    if value.strip() == "?":
        raise DimExprError(
            f"{context} uses '?' as a dynamic extent marker, not a symbolic dimension"
        )
    try:
        parsed = parse_contract_expr(
            value,
            expected=ContractSort.INTEGER,
            context=context,
        )
    except ValueError as exc:
        raise DimExprError(str(exc)) from exc
    unknown = sorted(parsed.references - dimensions)
    if unknown:
        raise DimExprError(
            f"{context} references unknown symbolic dimension(s): "
            + ", ".join(unknown)
        )
    _validate_contract_dimension_domain(parsed, context=context)
    return DimensionExpr(_normalized_contract_dimension(parsed))


@dataclass(frozen=True)
class AffineDimExpr:
    """Exact affine form ``constant + sum(coeff_i * x_i)``."""

    constant: int = 0
    coefficients: tuple[tuple[str, int], ...] = ()

    @classmethod
    def create(
        cls,
        constant: int,
        coefficients: Iterable[tuple[str, int]] = (),
        *,
        context: str,
    ) -> "AffineDimExpr":
        combined: dict[str, int] = {}
        for name, coefficient in coefficients:
            combined[name] = combined.get(name, 0) + coefficient
        normalized = tuple(
            (name, _checked_integer(coefficient, context=context))
            for name, coefficient in sorted(combined.items())
            if coefficient != 0
        )
        return cls(_checked_integer(constant, context=context), normalized)

    @property
    def is_constant(self) -> bool:
        return not self.coefficients

    def scaled(self, factor: int, *, context: str) -> "AffineDimExpr":
        return AffineDimExpr.create(
            self.constant * factor,
            ((name, coefficient * factor) for name, coefficient in self.coefficients),
            context=context,
        )

    def plus(
        self,
        other: "AffineDimExpr",
        *,
        factor: int = 1,
        context: str,
    ) -> "AffineDimExpr":
        return AffineDimExpr.create(
            self.constant + factor * other.constant,
            (
                *self.coefficients,
                *(
                    (name, factor * coefficient)
                    for name, coefficient in other.coefficients
                ),
            ),
            context=context,
        )

    def to_text(self) -> str:
        terms: list[str] = []
        for name, coefficient in self.coefficients:
            if coefficient == 1:
                terms.append(name)
            elif coefficient == -1:
                terms.append(f"-{name}")
            else:
                terms.append(f"{coefficient}*{name}")
        if self.constant != 0 or not terms:
            terms.append(str(self.constant))
        return " + ".join(terms).replace("+ -", "- ")

    def evaluate(self, dimensions: dict[str, int], *, context: str) -> int:
        value = self.constant
        for name, coefficient in self.coefficients:
            if name not in dimensions:
                raise DimExprError(
                    f"{context} lacks a concrete valuation for {name!r}"
                )
            extent = dimensions[name]
            if isinstance(extent, bool) or not isinstance(extent, int):
                raise DimExprError(
                    f"{context} valuation for {name!r} must be an integer"
                )
            value += coefficient * extent
        return _checked_integer(value, context=context)


def _affine_from_contract(
    expression: ContractExpr,
    *,
    dimensions: frozenset[str],
    context: str,
) -> AffineDimExpr:
    if expression.op == "int":
        assert isinstance(expression.value, int) and not isinstance(
            expression.value, bool
        )
        return AffineDimExpr.create(expression.value, context=context)
    if expression.op == "ref":
        assert isinstance(expression.value, str)
        if expression.value not in dimensions:
            raise DimExprError(
                f"{context} references unknown symbolic dimension "
                f"{expression.value!r}"
            )
        return AffineDimExpr.create(
            0, ((expression.value, 1),), context=context
        )
    if expression.op in {"pos", "neg"}:
        operand = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        return operand if expression.op == "pos" else operand.scaled(
            -1, context=context
        )
    if expression.op in {"add", "sub"}:
        left = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        right = _affine_from_contract(
            expression.args[1], dimensions=dimensions, context=context
        )
        return left.plus(
            right,
            factor=1 if expression.op == "add" else -1,
            context=context,
        )
    if expression.op == "mul":
        left = _affine_from_contract(
            expression.args[0], dimensions=dimensions, context=context
        )
        right = _affine_from_contract(
            expression.args[1], dimensions=dimensions, context=context
        )
        if left.is_constant:
            return right.scaled(left.constant, context=context)
        if right.is_constant:
            return left.scaled(right.constant, context=context)
        raise DimExprError(
            f"{context} is non-affine: multiplication requires one constant operand"
        )
    raise DimExprError(
        f"{context} uses unsupported non-affine operator {expression.op!r}"
    )


def parse_affine_dim_expr(
    value: int | str,
    *,
    dimensions: frozenset[str],
    context: str,
) -> AffineDimExpr:
    """Parse one int/string surface value into an exact affine form."""

    if isinstance(value, bool):
        raise DimExprError(f"{context} must be an integer dimension expression")
    if isinstance(value, int):
        return AffineDimExpr.create(value, context=context)
    if not isinstance(value, str):
        raise DimExprError(f"{context} must be an integer or ContractExpr string")
    try:
        parsed = parse_contract_expr(
            value,
            expected=ContractSort.INTEGER,
            context=context,
        )
    except ValueError as exc:
        raise DimExprError(str(exc)) from exc
    return _affine_from_contract(
        parsed, dimensions=dimensions, context=context
    )


def _constraint_lower_forms(
    expression: ContractExpr,
    *,
    dimensions: frozenset[str],
    context: str,
) -> tuple[AffineDimExpr, ...]:
    """Return authored atoms normalized to integer forms known ``>= 0``."""

    if expression.op == "and":
        return tuple(
            form
            for argument in expression.args
            for form in _constraint_lower_forms(
                argument, dimensions=dimensions, context=context
            )
        )
    if expression.op not in {"le", "lt", "ge", "gt", "eq"}:
        raise DimExprError(
            f"{context} permits only affine comparisons and conjunction"
        )
    left = _affine_from_contract(
        expression.args[0], dimensions=dimensions, context=context
    )
    right = _affine_from_contract(
        expression.args[1], dimensions=dimensions, context=context
    )
    if expression.op in {"ge", "gt"}:
        lower = left.plus(right, factor=-1, context=context)
    else:
        lower = right.plus(left, factor=-1, context=context)
    if expression.op in {"lt", "gt"}:
        lower = lower.plus(
            AffineDimExpr.create(1, context=context),
            factor=-1,
            context=context,
        )
    if expression.op == "eq":
        return (lower, lower.scaled(-1, context=context))
    return (lower,)


@dataclass(frozen=True)
class DimensionConstraintSet:
    """Parsed affine legal-domain facts with a small, sound bound prover."""

    lower_forms: tuple[AffineDimExpr, ...]
    lower_bounds: tuple[tuple[str, int], ...]
    upper_bounds: tuple[tuple[str, int], ...]

    @classmethod
    def parse(
        cls,
        expressions: Iterable[str],
        *,
        dimensions: frozenset[str],
        context: str = "j.dimension_constraints",
    ) -> "DimensionConstraintSet":
        forms: list[AffineDimExpr] = []
        for index, value in enumerate(expressions):
            item_context = f"{context}[{index}]"
            try:
                parsed = parse_contract_expr(
                    value,
                    expected=ContractSort.BOOLEAN,
                    context=item_context,
                )
            except ValueError as exc:
                raise DimExprError(str(exc)) from exc
            forms.extend(
                _constraint_lower_forms(
                    parsed,
                    dimensions=dimensions,
                    context=item_context,
                )
            )

        lower = {name: 0 for name in dimensions}
        upper: dict[str, int] = {}
        for form in forms:
            if not form.coefficients:
                if form.constant < 0:
                    raise DimExprError(
                        f"{context} contains an unsatisfiable constant constraint"
                    )
                continue
            if len(form.coefficients) != 1:
                continue
            name, coefficient = form.coefficients[0]
            if coefficient > 0:
                # coefficient*x + constant >= 0
                bound = (-form.constant + coefficient - 1) // coefficient
                lower[name] = max(lower[name], bound)
            else:
                divisor = -coefficient
                # -divisor*x + constant >= 0
                bound = form.constant // divisor
                upper[name] = min(upper.get(name, bound), bound)
        for name in dimensions:
            if name in upper and lower[name] > upper[name]:
                raise DimExprError(
                    f"{context} is unsatisfiable for {name!r}: "
                    f"lower bound {lower[name]} exceeds upper bound {upper[name]}"
                )
        return cls(
            tuple(forms),
            tuple(sorted(lower.items())),
            tuple(sorted(upper.items())),
        )

    def proves_nonnegative(self, expression: AffineDimExpr) -> bool:
        """Prove ``expression >= 0`` without any concrete runtime values."""

        if expression in self.lower_forms:
            return True
        lower = dict(self.lower_bounds)
        upper = dict(self.upper_bounds)
        minimum = expression.constant
        for name, coefficient in expression.coefficients:
            if coefficient >= 0:
                minimum += coefficient * lower.get(name, 0)
                continue
            if name not in upper:
                return False
            minimum += coefficient * upper[name]
        return minimum >= 0

    def proves_dimension_nonnegative(self, expression: DimensionExpr) -> bool:
        """Prove a general dimension expression is non-negative.

        The affine prover remains authoritative for affine expressions.  The
        structural rules below are deliberately conservative: they cover only
        operators whose non-negativity follows from non-negative dimensions
        and a positive divisor, and never invent a bound for subtraction or an
        unknown denominator.
        """

        try:
            affine = _affine_from_contract(
                expression.expression,
                dimensions=expression.references,
                context="dimension expression",
            )
        except DimExprError:
            affine = None
        if affine is not None:
            return self.proves_nonnegative(affine)

        lower = dict(self.lower_bounds)

        def nonnegative(node: ContractExpr) -> bool:
            if node.op == "int":
                return isinstance(node.value, int) and not isinstance(
                    node.value, bool
                ) and node.value >= 0
            if node.op == "ref":
                return (
                    isinstance(node.value, str)
                    and lower.get(node.value, 0) >= 0
                )
            if node.op == "pos":
                return nonnegative(node.args[0])
            if node.op == "add":
                return all(nonnegative(argument) for argument in node.args)
            if node.op == "mul":
                return nonnegative(node.args[0]) and nonnegative(node.args[1])
            if node.op == "floordiv":
                return nonnegative(node.args[0]) and positive(node.args[1])
            if node.op in {"left_shift", "right_shift"}:
                return nonnegative(node.args[0]) and nonnegative(node.args[1])
            if node.op == "mod":
                return nonnegative(node.args[0]) and positive(node.args[1])
            if node.op == "call" and node.value == "ceil_div":
                return nonnegative(node.args[0]) and positive(node.args[1])
            if node.op == "call" and node.value == "min":
                return nonnegative(node.args[0]) and nonnegative(node.args[1])
            if node.op == "call" and node.value == "max":
                return nonnegative(node.args[0]) or nonnegative(node.args[1])
            return False

        def positive(node: ContractExpr) -> bool:
            if node.op == "int":
                return isinstance(node.value, int) and not isinstance(
                    node.value, bool
                ) and node.value > 0
            # A declared dimension may be zero, so a lower bound of one is
            # required before it can serve as a divisor.
            if node.op == "ref":
                return (
                    isinstance(node.value, str)
                    and lower.get(node.value, 0) > 0
                )
            return False

        return nonnegative(expression.expression)

    def validate_valuation(self, dimensions: dict[str, int]) -> None:
        """Reject one public instantiation outside the authored legal domain."""

        for index, form in enumerate(self.lower_forms):
            if form.evaluate(
                dimensions,
                context=f"j.dimension_constraints atom {index}",
            ) < 0:
                raise DimExprError(
                    "public dimension valuation violates "
                    f"J.dimension_constraints atom {form.to_text()!r} >= 0"
                )


def validate_dimension_instantiation(
    j: Any,
    dimensions: Mapping[str, int],
) -> None:
    """Validate one public J dimension valuation and every resolved pad extent."""

    # Keep public instantiation validation on the same declaration authority as
    # dimension relations.  Signature-bound x* handles and authored relation
    # results are J-owned declarations too; reconstructing the set from tensor
    # shapes alone silently made those two consumers disagree about whether a
    # valuation was closed.  The import is local because dimension_relation
    # itself uses dimension_expression_references from this module.
    from .dimension_relation import declared_dimension_handles

    declared = declared_dimension_handles(j)
    concrete = dict(dimensions)
    unexpected = sorted(set(concrete) - declared)
    if unexpected:
        raise DimExprError(
            "public dimension valuation contains unknown J dimension(s): "
            + ", ".join(unexpected)
        )
    missing = sorted(declared - set(concrete))
    if missing:
        raise DimExprError(
            "public instantiation lacks J dimension(s): " + ", ".join(missing)
        )
    for name in sorted(declared):
        value = concrete[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DimExprError(
                f"public dimension {name!r} must be a non-negative integer"
            )

    constraints = DimensionConstraintSet.parse(
        j.dimension_constraints,
        dimensions=declared,
    )
    constraints.validate_valuation(concrete)

    def validate_pad_extents(operations: list[Mapping[str, Any]]) -> None:
        for operation in operations:
            if operation.get("op") == "pad":
                attrs = operation.get("attrs")
                if isinstance(attrs, Mapping):
                    for field in ("lo", "hi"):
                        values = attrs.get(field)
                        if not isinstance(values, list):
                            continue
                        for index, expression in enumerate(values):
                            context = f"public pad {field}[{index}]"
                            resolved = parse_dimension_expr(
                                expression,
                                dimensions=declared,
                                context=context,
                            ).evaluate(concrete, context=context)
                            if resolved < 0:
                                raise DimExprError(
                                    "public dimension valuation resolves "
                                    f"pad {field}[{index}] to a negative extent"
                                )
            body = operation.get("body")
            if isinstance(body, list):
                validate_pad_extents(body)

    intent = j.intent.to_json()["intent_function"]
    validate_pad_extents(intent.get("body", []))


__all__ = [
    "AffineDimExpr",
    "DimensionExpr",
    "DimExprError",
    "DimensionConstraintSet",
    "dimension_expression_references",
    "equivalent_dimension_expr",
    "parse_affine_dim_expr",
    "parse_dimension_expr",
    "validate_dimension_instantiation",
]
