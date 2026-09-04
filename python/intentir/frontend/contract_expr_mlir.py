"""Generic MLIR lowering for one typed canonical ContractExpr program.

This module maps the closed target-neutral semantics from
``intentir.ir.contract_semantics`` to ordinary ``arith`` SSA operations.  It is
formula-generic: no access rank, radix, tensor shape, kernel name, or node id is
part of the lowering contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from intentir.ir.canonical_j import CanonicalIndex, ContractExpr, ContractSort
from intentir.ir.contract_semantics import TypedContractProgram


class ContractMLIRLoweringError(ValueError):
    """A typed ContractExpr cannot be represented by the generic arith lowering."""


@dataclass(frozen=True)
class ContractMLIRObligation:
    """A proof obligation required before finite-width MLIR is semantics preserving.

    ContractExpr itself denotes mathematical integers.  A target lowerer must
    discharge every obligation over J's declared logical domain; merely
    emitting the fragment is not a membership proof.
    """

    kind: Literal["representable", "nonzero_divisor"]
    expression: ContractExpr


@dataclass(frozen=True)
class ContractMLIRFragment:
    operations: tuple[str, ...]
    results: tuple[str, ...]
    result_types: tuple[str, ...]
    obligations: tuple[ContractMLIRObligation, ...]

    def indented(self, spaces: int = 4) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + operation for operation in self.operations)


class _ContractMLIRBuilder:
    def __init__(
        self,
        reference_values: Mapping[str, str],
        reference_sorts: Mapping[str, ContractSort],
        *,
        integer_type: str,
        ssa_prefix: str,
    ) -> None:
        if not integer_type or not isinstance(integer_type, str):
            raise ContractMLIRLoweringError("integer MLIR type must be non-empty")
        if not ssa_prefix.startswith("%") or len(ssa_prefix) == 1:
            raise ContractMLIRLoweringError("SSA prefix must start with %")
        if set(reference_values) != set(reference_sorts):
            raise ContractMLIRLoweringError(
                "MLIR reference values must exactly match typed references"
            )
        if any(
            not isinstance(value, str) or not value.startswith("%")
            for value in reference_values.values()
        ):
            raise ContractMLIRLoweringError(
                "MLIR reference values must be SSA identifiers"
            )
        self.reference_values = dict(reference_values)
        self.reference_sorts = dict(reference_sorts)
        self.integer_type = integer_type
        self.ssa_prefix = ssa_prefix
        self.operations: list[str] = []
        self.obligations: list[ContractMLIRObligation] = []
        self._counter = 0
        self._cache: dict[ContractExpr, tuple[str, str]] = {}

    def _require_representable(self, expression: ContractExpr) -> None:
        obligation = ContractMLIRObligation("representable", expression)
        if obligation not in self.obligations:
            self.obligations.append(obligation)

    def _require_nonzero(self, expression: ContractExpr) -> None:
        obligation = ContractMLIRObligation("nonzero_divisor", expression)
        if obligation not in self.obligations:
            self.obligations.append(obligation)

    def _ssa(self) -> str:
        result = f"{self.ssa_prefix}{self._counter}"
        self._counter += 1
        return result

    def _append(self, operation: str, result_type: str) -> tuple[str, str]:
        result = self._ssa()
        self.operations.append(f"{result} = {operation}")
        return result, result_type

    def emit(self, expression: ContractExpr) -> tuple[str, str]:
        cached = self._cache.get(expression)
        if cached is not None:
            return cached
        if expression.op == "ref":
            assert isinstance(expression.value, str)
            result = (
                self.reference_values[expression.value],
                "i1"
                if self.reference_sorts[expression.value] is ContractSort.BOOLEAN
                else self.integer_type,
            )
            if result[1] == self.integer_type:
                self._require_representable(expression)
            self._cache[expression] = result
            return result
        if expression.op == "int":
            assert isinstance(expression.value, int) and not isinstance(
                expression.value, bool
            )
            result = self._append(
                f"arith.constant {expression.value} : {self.integer_type}",
                self.integer_type,
            )
            self._require_representable(expression)
            self._cache[expression] = result
            return result
        if expression.op == "bool":
            assert isinstance(expression.value, bool)
            result = self._append(
                f"arith.constant {'true' if expression.value else 'false'}",
                "i1",
            )
            self._cache[expression] = result
            return result

        operands = tuple(self.emit(argument) for argument in expression.args)
        operand_values = tuple(item[0] for item in operands)
        operand_types = tuple(item[1] for item in operands)
        if expression.op == "pos":
            result = operands[0]
        elif expression.op == "neg":
            zero, _ = self._append(
                f"arith.constant 0 : {self.integer_type}", self.integer_type
            )
            result = self._append(
                f"arith.subi {zero}, {operand_values[0]} : {self.integer_type}",
                self.integer_type,
            )
        elif expression.op == "not":
            true, _ = self._append("arith.constant true", "i1")
            result = self._append(
                f"arith.xori {operand_values[0]}, {true} : i1", "i1"
            )
        elif expression.op in {
            "add", "sub", "mul", "floordiv", "left_shift", "right_shift"
        }:
            opcode = {
                "add": "addi",
                "sub": "subi",
                "mul": "muli",
                "floordiv": "floordivsi",
                "left_shift": "shli",
                "right_shift": "shrsi",
            }[expression.op]
            result = self._append(
                f"arith.{opcode} {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                self.integer_type,
            )
            if expression.op == "floordiv":
                self._require_nonzero(expression.args[1])
        elif expression.op == "mod":
            # ContractExpr follows Python floor-division modulo.  Build it from
            # floorDiv instead of arith.remsi, whose signed remainder follows
            # truncating division for negative operands.
            quotient, _ = self._append(
                f"arith.floordivsi {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                self.integer_type,
            )
            self._require_nonzero(expression.args[1])
            product, _ = self._append(
                f"arith.muli {quotient}, {operand_values[1]} : {self.integer_type}",
                self.integer_type,
            )
            result = self._append(
                f"arith.subi {operand_values[0]}, {product} : {self.integer_type}",
                self.integer_type,
            )
        elif expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
            predicate = {
                "lt": "slt",
                "le": "sle",
                "gt": "sgt",
                "ge": "sge",
                "eq": "eq",
                "ne": "ne",
            }[expression.op]
            result = self._append(
                f"arith.cmpi {predicate}, {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                "i1",
            )
        elif expression.op in {"and", "or"}:
            opcode = "andi" if expression.op == "and" else "ori"
            current = operand_values[0]
            for operand in operand_values[1:]:
                current, _ = self._append(
                    f"arith.{opcode} {current}, {operand} : i1", "i1"
                )
            result = current, "i1"
        elif expression.op == "call" and expression.value == "ceil_div":
            result = self._append(
                f"arith.ceildivsi {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                self.integer_type,
            )
            self._require_nonzero(expression.args[1])
        elif expression.op == "call" and expression.value in {"min", "max"}:
            predicate = "slt" if expression.value == "min" else "sgt"
            condition, _ = self._append(
                f"arith.cmpi {predicate}, {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                "i1",
            )
            result = self._append(
                f"arith.select {condition}, {operand_values[0]}, {operand_values[1]} "
                f": {self.integer_type}",
                self.integer_type,
            )
        else:
            raise ContractMLIRLoweringError(
                f"unsupported typed ContractExpr opcode {expression.op!r}"
            )
        expected_type = (
            "i1" if expression.sort is ContractSort.BOOLEAN else self.integer_type
        )
        if result[1] != expected_type:
            raise ContractMLIRLoweringError(
                f"ContractExpr {expression.op!r} lowered to {result[1]}, "
                f"expected {expected_type}"
            )
        if any(
            operand_type not in {self.integer_type, "i1"}
            for operand_type in operand_types
        ):
            raise ContractMLIRLoweringError("unexpected ContractExpr operand type")
        if result[1] == self.integer_type:
            self._require_representable(expression)
        self._cache[expression] = result
        return result


def lower_typed_contract_program(
    program: TypedContractProgram,
    reference_values: Mapping[str, str],
    *,
    integer_type: str,
    ssa_prefix: str = "%ce",
) -> ContractMLIRFragment:
    """Lower one fully typed expression to formula-generic arith operations."""

    if not isinstance(program, TypedContractProgram):
        raise TypeError("ContractExpr MLIR lowering requires TypedContractProgram")
    builder = _ContractMLIRBuilder(
        reference_values,
        program.reference_sort_map,
        integer_type=integer_type,
        ssa_prefix=ssa_prefix,
    )
    result, result_type = builder.emit(program.expression)
    return ContractMLIRFragment(
        tuple(builder.operations),
        (result,),
        (result_type,),
        tuple(builder.obligations),
    )


def lower_contract_index(
    index: CanonicalIndex,
    reference_sorts: Mapping[str, ContractSort],
    reference_values: Mapping[str, str],
    *,
    integer_type: str,
    ssa_prefix: str = "%ce",
) -> ContractMLIRFragment:
    """Lower every component of an arbitrary canonical index vector."""

    if not isinstance(index, CanonicalIndex):
        raise TypeError("index MLIR lowering requires CanonicalIndex")
    used_references = {
        reference
        for component in index.components
        for reference in component.references
    }
    closed_sorts = {
        name: sort for name, sort in reference_sorts.items()
        if name in used_references
    }
    missing_sorts = sorted(used_references - set(closed_sorts))
    if missing_sorts:
        raise ContractMLIRLoweringError(
            f"index lowering lacks reference sorts for {missing_sorts}"
        )
    missing_values = sorted(used_references - set(reference_values))
    if missing_values:
        raise ContractMLIRLoweringError(
            f"index lowering lacks MLIR values for {missing_values}"
        )
    builder = _ContractMLIRBuilder(
        {name: reference_values[name] for name in closed_sorts},
        closed_sorts,
        integer_type=integer_type,
        ssa_prefix=ssa_prefix,
    )
    results: list[str] = []
    for position, component in enumerate(index.components):
        program = TypedContractProgram.compile(
            component,
            closed_sorts,
            expected_sort=ContractSort.INTEGER,
            context=f"index_expression[{position}]",
        )
        result, result_type = builder.emit(program.expression)
        if result_type != integer_type:
            raise ContractMLIRLoweringError(
                "index component did not lower to the requested integer type"
            )
        results.append(result)
    return ContractMLIRFragment(
        tuple(builder.operations),
        tuple(results),
        (integer_type,) * len(results),
        tuple(builder.obligations),
    )


__all__ = [
    "ContractMLIRFragment",
    "ContractMLIRLoweringError",
    "ContractMLIRObligation",
    "lower_contract_index",
    "lower_typed_contract_program",
]
