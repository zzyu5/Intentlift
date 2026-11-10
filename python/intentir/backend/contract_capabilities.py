"""Target-neutral optimization capabilities derived from immutable Layer B."""

from __future__ import annotations

from intentir.backend.materialize import MaterializedJ
from intentir.ir.canonical_j import j_elaboration
from intentir.verify.gate1.model import EffectKind, ParameterMode


B_CAP_READONLY_INPUTS = "b.readonly_nonalias_inputs.v1"
B_CAP_RANK0_HOST_CAPTURE = "b.rank0_readonly_host_capture.v1"


def derive_search_capabilities(base: MaterializedJ) -> tuple[str, ...]:
    """Derive compiler permissions from closed B facts, never from C.

    Absence is conservative: an input with neither a read effect nor an
    explicit rank-0 value ABI is not treated as read-only.  Alias/update facts
    always block these capabilities.
    """

    j = base.j
    input_names = {item.name for item in j.intent.inputs}
    effects_by_target: dict[str, set[EffectKind]] = {
        name: set() for name in input_names
    }
    for effect in j.boundary.effects:
        if effect.target in effects_by_target:
            effects_by_target[effect.target].add(effect.kind)
    parameters = {
        item.logical_ref: item
        for item in j.signature.parameters
        if item.logical_ref is not None
    }
    tensors = {item.name: item for item in j.intent.inputs}

    # A closed signature may preserve unused frontend ABI parameters.  They are
    # not compiler-visible state when they neither occur in the typed A DAG nor
    # carry a B effect.  Requiring a read effect for such dead carriers made one
    # unrelated unused tensor suppress whole-graph fusion for every live input.
    # Walk every structured region so this projection is derived from J itself,
    # never from a runtime sample or kernel-name allowlist.
    typed_rows = j_elaboration(j).typed_operation_rows
    if typed_rows is None:
        # Capability derivation is a compiler permission boundary.  An
        # incomplete canonical view must not acquire permissions from a raw
        # body walk.
        return ()
    dag_operands = {
        operand
        for _path, operation in typed_rows
        for operand in operation.operands
    }
    observable_inputs = {
        binding.output
        for binding in j_elaboration(j).output_bindings
        if binding.output in input_names
    }
    active_input_names = {
        name
        for name in input_names
        if name in dag_operands
        or name in observable_inputs
        or bool(effects_by_target[name])
    }

    def is_readonly_input(name: str) -> bool:
        kinds = effects_by_target[name]
        if kinds == {EffectKind.READ}:
            return True
        parameter = parameters.get(name)
        return bool(
            not kinds
            and not tensors[name].shape
            and parameter is not None
            and parameter.mode is ParameterMode.VALUE
            and parameter.alias_of is None
        )

    all_inputs_readonly = bool(active_input_names) and all(
        is_readonly_input(name) for name in active_input_names
    )
    capabilities: list[str] = []
    if all_inputs_readonly and not j.boundary.aliases and not j.boundary.updates:
        capabilities.append(B_CAP_READONLY_INPUTS)

    runtime_names = set(base.runtime_input_names)
    rank0_names = {
        item.name
        for item in j.intent.inputs
        if item.name in runtime_names and not item.shape
    }
    rank0_value_abi = bool(rank0_names) and all(
        name in parameters
        and parameters[name].mode in {ParameterMode.INPUT, ParameterMode.VALUE}
        and parameters[name].alias_of is None
        for name in rank0_names
    )
    rank0_readonly = all(is_readonly_input(name) for name in rank0_names)
    if (
        B_CAP_READONLY_INPUTS in capabilities
        and rank0_value_abi
        and rank0_readonly
    ):
        capabilities.append(B_CAP_RANK0_HOST_CAPTURE)
    return tuple(sorted(capabilities))


__all__ = [
    "B_CAP_RANK0_HOST_CAPTURE",
    "B_CAP_READONLY_INPUTS",
    "derive_search_capabilities",
]
