"""Simple backend-neutral ``E_J`` that consumes the complete canonical J."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from intentir.backend.cpu.abi import CPUABIValidationError, CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.backend.cpu.structured_reference import (
    execute_structured_reference,
    requires_structured_reference,
)
from intentir.ir.canonical_j import (
    canonical_broadcast_axes,
    elaborate_j,
    j_elaboration,
)
from intentir.ir.dim_expr import validate_dimension_instantiation
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.functional_boundary import (
    project_functional_boundary,
    static_index_bounds_status,
)
from intentir.ir.integer_range import proves_index_bounds
from intentir.ir.value_dimension import validate_value_dimension_instantiation
from intentir.verify.gate1.model import SemanticArchive

from .model import ExecutionResult, output_binding_digest, output_digest


SCALAR_LOWERING_PROTOCOL = "j_scalar_cpu_mlir_v24"


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _canonical_int_mapping(value: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for key, item in value.items()
    ):
        raise ValueError("scalar dims must map names to non-negative integers")
    return tuple(sorted(value.items()))


def _canonical_str_mapping(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in value.items()
    ):
        raise ValueError("scalar bindings must map non-empty names")
    return tuple(sorted(value.items()))


def _validate_complete_boundary(j: SemanticArchive) -> str:
    """Compatibility wrapper over the shared J boundary projection."""

    return project_functional_boundary(j).boundary_digest


def scalar_index_bounds(j: SemanticArchive) -> frozenset[str]:
    """Return the exact emitter projection authorized by canonical J."""

    return project_functional_boundary(j).proven_index_bounds


def validate_index_bounds_inputs(
    j: SemanticArchive,
    values: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int] | None = None,
) -> None:
    """Check J's indexed-domain preconditions before native scalar execution."""

    for item in j.boundary.index_bounds:
        static_status = static_index_bounds_status(j, item)
        if static_status is True:
            continue
        if static_status is False:
            raise ValueError(
                f"J index-bounds precondition failed at node {item.node_ref!r}"
            )
        if proves_index_bounds(
            j,
            item,
            logical_inputs=values,
            dimension_extents=dimensions,
        ):
            continue
        if item.data not in values or item.indices not in values:
            # Internal indexed values are checked at the exact gather/scatter
            # operation by the J-only structured/contract evaluator.  The
            # runner forces that checked path below; no precondition is assumed.
            continue
        data = np.asarray(values[item.data])
        indices = np.asarray(values[item.indices])
        if item.axis >= data.ndim:
            raise ValueError("E_J indexed data rank violates canonical J")
        if indices.dtype.kind not in {"i", "u"}:
            raise ValueError("E_J indices must have an integer dtype")
        if bool(np.any(indices < 0)) or bool(np.any(indices >= data.shape[item.axis])):
            raise ValueError(
                f"J index-bounds precondition failed at node {item.node_ref!r}"
            )


def scalar_broadcast_axes(
    j: SemanticArchive,
) -> dict[str, dict[str, tuple[int, ...]]]:
    return project_functional_boundary(j).emitter_broadcast_axes()


@dataclass(frozen=True)
class ScalarLoweringSpec:
    """Invocation request holding the one elaborated J used by all consumers."""

    j: SemanticArchive
    dims: tuple[tuple[str, int], ...]
    input_bindings: tuple[tuple[str, str], ...]
    output_bindings: tuple[tuple[str, str], ...]
    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("scalar lowering requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        dims = resolve_dimension_relations(self.j, dict(self.dims))
        inputs = dict(self.input_bindings)
        outputs = dict(self.output_bindings)
        object.__setattr__(self, "dims", _canonical_int_mapping(dims))
        object.__setattr__(self, "input_bindings", _canonical_str_mapping(inputs))
        object.__setattr__(self, "output_bindings", _canonical_str_mapping(outputs))
        validate_dimension_instantiation(self.j, dims)
        expected_inputs = {item.name for item in self.j.intent.inputs}
        expected_outputs = {
            binding.output
            for binding in j_elaboration(self.j).output_bindings
        }
        if set(inputs) != expected_inputs:
            raise ValueError("scalar input bindings differ from J inputs")
        if set(outputs) != expected_outputs:
            raise ValueError("scalar output bindings differ from J outputs")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("scalar output destinations must be unique")

    @classmethod
    def create(
        cls,
        j: SemanticArchive,
        *,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
    ) -> "ScalarLoweringSpec":
        return cls(
            j,
            tuple(dims.items()),
            tuple(input_bindings.items()),
            tuple(output_bindings.items()),
        )

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

    @property
    def dims_map(self) -> dict[str, int]:
        return dict(self.dims)

    @property
    def input_binding_map(self) -> dict[str, str]:
        return dict(self.input_bindings)

    @property
    def output_binding_map(self) -> dict[str, str]:
        return dict(self.output_bindings)

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "j_digest": self.j_digest,
                "dims": self.dims_map,
                "input_bindings": self.input_binding_map,
                "output_bindings": self.output_binding_map,
            }
        )


@dataclass(frozen=True)
class ScalarLoweringUnavailable:
    j: SemanticArchive
    error: str
    failure_class: str = "j_lowering_setup"
    def __post_init__(self) -> None:
        if not isinstance(self.j, SemanticArchive):
            raise ValueError("unavailable RefEval request requires canonical J")
        object.__setattr__(self, "j", elaborate_j(self.j))
        if not self.error:
            raise ValueError("unavailable E_J request requires an error")

    @property
    def j_digest(self) -> str:
        return _digest(self.j.to_json())

    @property
    def spec_digest(self) -> str:
        return _digest(
            {
                "schema": SCALAR_LOWERING_PROTOCOL,
                "availability": "unavailable",
                "j_digest": self.j_digest,
                "failure_class": self.failure_class,
                "error": self.error,
            }
        )

    def execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            "j",
            False,
            error=self.error,
            unsupported=True,
            metadata={
                "protocol": SCALAR_LOWERING_PROTOCOL,
                "j_digest": self.j_digest,
                "scalar_spec_digest": self.spec_digest,
                "failure_class": self.failure_class,
            },
        )


ScalarLoweringRequest = ScalarLoweringSpec | ScalarLoweringUnavailable


_REFERENCE_UNSUPPORTED_FAILURE_PREFIXES = (
    (
        "functional dense access requires",
        "j_reference_dense_access_projection_unsupported",
    ),
    (
        "functional bounded gather",
        "j_reference_bounded_gather_projection_unsupported",
    ),
    (
        "functional constant access",
        "j_reference_constant_access_projection_unsupported",
    ),
    (
        "functional constant load partition",
        "j_reference_constant_load_partition_unsupported",
    ),
    (
        "functional constant gather",
        "j_reference_constant_gather_projection_unsupported",
    ),
    (
        "functional mask projection does not implement update padding",
        "j_reference_mask_update_padding_unsupported",
    ),
    (
        "functional mask projection does not implement guard-bearing access "
        "predicates",
        "j_reference_mask_guard_predicate_unsupported",
    ),
    (
        "functional mask projection does not implement cross-axis logical "
        "access predicates",
        "j_reference_mask_cross_axis_predicate_unsupported",
    ),
    (
        "functional mask projection requires a complete logical domain",
        "j_reference_mask_domain_incomplete",
    ),
    (
        "functional mask projection requires an explicit masked-load fill",
        "j_reference_masked_load_fill_missing",
    ),
    (
        "functional mask projection",
        "j_reference_mask_projection_unsupported",
    ),
    (
        "functional guarded load requires an explicit fill",
        "j_reference_guarded_load_fill_missing",
    ),
    (
        "functional guarded load consumer extension requires one single-result consumer",
        "j_reference_guarded_load_extension_consumer_result_unsupported",
    ),
    (
        "functional guarded load consumer extension requires one direct operand edge",
        "j_reference_guarded_load_extension_operand_unsupported",
    ),
    (
        "functional guarded load consumer extension has no access-to-operand provenance",
        "j_reference_guarded_load_extension_provenance_missing",
    ),
    (
        "functional guarded load requires one single-result consumer",
        "j_reference_guarded_load_consumer_result_unsupported",
    ),
    (
        "functional guarded load requires one unambiguous direct consumer operand",
        "j_reference_guarded_load_consumer_operand_ambiguous",
    ),
    (
        "functional guarded load requires an explicit consumer node or consumer extension",
        "j_reference_guarded_load_consumer_binding_missing",
    ),
    (
        "functional guarded load consumer edges must be unique",
        "j_reference_guarded_load_consumer_edge_duplicate",
    ),
    (
        "functional guarded load conflicts with another broadcast mapping",
        "j_reference_guarded_load_broadcast_conflict",
    ),
    (
        "functional guarded load",
        "j_reference_guarded_load_projection_unsupported",
    ),
    (
        "functional guarded store",
        "j_reference_guarded_store_projection_unsupported",
    ),
    (
        "functional output-only guarded store",
        "j_reference_output_only_guarded_store_unsupported",
    ),
    (
        "functional boundary requires exactly one full store",
        "j_reference_full_output_store_projection_unsupported",
    ),
    (
        "functional boundary requires exactly one store",
        "j_reference_output_store_projection_unsupported",
    ),
    (
        "functional boundary must cover every live logical tensor input",
        "j_reference_input_access_projection_unsupported",
    ),
    (
        "predicate-bearing reduce/scan operand requires",
        "j_reference_consumer_extension_unsupported",
    ),
    (
        "functional padded reduce/scan requires",
        "j_reference_reduction_padding_projection_unsupported",
    ),
    (
        "functional projection has no closed reduction neutral element",
        "j_reference_reduction_identity_unsupported",
    ),
    (
        "node-less store is supported only",
        "j_reference_node_less_store_unsupported",
    ),
    (
        "structured reference graph contains IntentOp(s) outside",
        "j_reference_operation_capability_unsupported",
    ),
    (
        "structured reference requires a structured op",
        "j_reference_structured_dispatch_unsupported",
    ),
    (
        "structured ",
        "j_reference_structured_execution_unsupported",
    ),
)


def _reference_failure_class(
    exc: BaseException,
    *,
    unsupported: bool = False,
) -> str:
    """Map one E_J failure to a bounded, source-free rule class.

    The text is used only inside the process to select a stable category.  The
    category deliberately does not include node ids, authored expressions,
    candidate text, or values.  Unknown failures remain execution failures.
    """

    if unsupported:
        return _reference_unsupported_failure_class(exc)
    if isinstance(exc, CPUABIValidationError):
        return "j_reference_cpu_abi_failure"
    message = str(exc).casefold()
    if "predicate" in message or "effect" in message:
        return "j_reference_predicate_effect_mismatch"
    if "dtype" in message or "shape" in message or "typed" in message:
        return "j_reference_typed_relation_mismatch"
    if "input" in message or "binding" in message or "output" in message:
        return "j_reference_binding_mismatch"
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return "j_reference_validation_failed"
    return "j_reference_execution_error"


def _reference_failure_metadata(
    exc: BaseException,
    *,
    unsupported: bool = False,
) -> dict[str, str]:
    """Return only fixed metadata safe for Gate2 summaries."""

    return {
        "phase": "reference_execution",
        "failure_class": _reference_failure_class(
            exc, unsupported=unsupported
        ),
        "exception_type": type(exc).__name__,
    }


def _reference_unsupported_failure_class(exc: BaseException) -> str:
    """Classify RefEval capability gaps without exporting diagnostic text."""

    message = str(exc)
    for prefix, failure_class in _REFERENCE_UNSUPPORTED_FAILURE_PREFIXES:
        if message.startswith(prefix):
            return failure_class
    if isinstance(exc, CPUBackendUnsupportedError):
        return "j_reference_cpu_backend_unsupported"
    return "j_reference_capability_unsupported"


class ScalarLoweringRunner:
    def run(self, spec: ScalarLoweringSpec, values: Mapping[str, Any]) -> ExecutionResult:
        try:
            raw_inputs = [
                values[spec.input_binding_map[tensor.name]]
                for tensor in spec.j.intent.inputs
            ]
        except KeyError as exc:
            return ExecutionResult("j", False, error=f"missing E_J input {exc.args[0]!r}")
        try:
            raw_inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(
                    spec.j.intent.inputs, raw_inputs, strict=True
                )
            }
            validate_value_dimension_instantiation(
                spec.j, spec.dims_map, raw_inputs_by_ref
            )
            inputs = [
                (
                    value.detach().cpu().numpy()
                    if type(value).__module__.startswith("torch")
                    and hasattr(value, "detach")
                    else np.asarray(value)
                )
                for value in raw_inputs
            ]
            inputs_by_ref = {
                tensor.name: value
                for tensor, value in zip(spec.j.intent.inputs, inputs, strict=True)
            }
            validate_index_bounds_inputs(
                spec.j,
                inputs_by_ref,
                spec.dims_map,
            )
            # Gate2 is the numeric execution judge.  It consumes the already
            # accepted canonical J and its public ABI/output bindings; it does
            # not rerun Source owner authorization, update-padding theorems, or
            # physical-store partition proofs. Those remain ingress and
            # canonical-construction responsibilities. The execution backend
            # still validates typed inputs, index bounds, operation results,
            # declared output dtype/shape, and complete observable outputs.
            try:
                broadcast_axes = canonical_broadcast_axes(spec.j)
            except (NotImplementedError, ValueError):
                broadcast_axes = {}
            guarded_loads = tuple(
                item
                for item in spec.j.boundary.access_guard_relations
                if item.access_kind == "load"
            )
            consumer_extensions = tuple(spec.j.boundary.consumer_extensions)
            proven_index_bounds = frozenset(
                spec.j.node_ids.get(item.node_ref, item.node_ref)
                for item in spec.j.boundary.index_bounds
                if static_index_bounds_status(spec.j, item) is True
            )
            boundary_profile = "canonical_j_numeric_execution_v1"
            boundary_projection_record = {
                "profile": boundary_profile,
                "mode": "numeric_execution_only",
                "physical_partition_proof": "not_consumed_by_gate2",
                "update_padding_proof": "not_consumed_by_gate2",
                "owner_authorization": "not_repeated_by_gate2",
                "output_bindings": sorted(spec.output_binding_map.items()),
            }
            boundary_digest = _digest(boundary_projection_record)
            predicated_access_count = sum(
                item.predicate is not None
                for item in spec.j.boundary.accesses
            )
            erased_fill_count = 0
            runtime_predicate_proof_count = 0
            deferred_index_bounds = any(
                item.data not in inputs_by_ref or item.indices not in inputs_by_ref
                for item in spec.j.boundary.index_bounds
            )
            structured = (
                requires_structured_reference(spec.j)
                or bool(guarded_loads)
                or bool(spec.j.boundary.updates)
                or deferred_index_bounds
            )
            execution = (
                execute_structured_reference(
                    spec.j,
                    spec.dims_map,
                    inputs,
                    broadcast_axes=broadcast_axes,
                    access_guard_relations=guarded_loads,
                    consumer_extensions=consumer_extensions,
                )
                if structured
                else execute_on_cpu_outputs(
                    spec.j,
                    spec.dims_map,
                    inputs,
                    broadcast_axes=broadcast_axes,
                    proven_index_bounds=proven_index_bounds,
                )
            )
            output_bindings = spec.output_binding_map
            outputs = {
                output_bindings[name]: np.asarray(value)
                for name, value in execution.outputs.items()
            }
            if set(outputs) != set(output_bindings.values()):
                raise ValueError("E_J did not produce every observable output")
            return ExecutionResult(
                "j",
                True,
                outputs,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "path": (
                        "canonical J -> typed structured CPU reference"
                        if structured
                        else "canonical J -> intent MLIR -> scalar CPU"
                    ),
                    "function": spec.j.intent.name,
                    "j_digest": spec.j_digest,
                    "boundary_digest": boundary_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    "boundary_profile": boundary_profile,
                    "boundary_projection": boundary_projection_record,
                    "domain_relation_count": len(spec.j.boundary.domains),
                    "access_relation_count": len(spec.j.boundary.accesses),
                    "effect_relation_count": len(spec.j.boundary.effects),
                    "predicated_access_count": predicated_access_count,
                    "erased_fill_count": erased_fill_count,
                    "runtime_predicate_proof_count": runtime_predicate_proof_count,
                    "broadcast_consumer_count": len(broadcast_axes),
                    "index_bounds_count": len(spec.j.boundary.index_bounds),
                    "result_relation_count": len(
                        spec.j.boundary.result_relations
                    ),
                    "execution_evidence": dict(execution.metadata),
                    "output_digest": output_digest(outputs),
                    "output_binding_digest": output_binding_digest(outputs),
                },
            )
        except (CPUBackendUnsupportedError, NotImplementedError) as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    **_reference_failure_metadata(exc, unsupported=True),
                },
            )
        except CPUABIValidationError as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    **_reference_failure_metadata(exc),
                },
            )
        except Exception as exc:
            return ExecutionResult(
                "j",
                False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "protocol": SCALAR_LOWERING_PROTOCOL,
                    "j_digest": spec.j_digest,
                    "scalar_spec_digest": spec.spec_digest,
                    **_reference_failure_metadata(exc),
                },
            )


__all__ = [
    "SCALAR_LOWERING_PROTOCOL", "ScalarLoweringRequest",
    "ScalarLoweringRunner", "ScalarLoweringSpec", "ScalarLoweringUnavailable",
    "scalar_broadcast_axes", "scalar_index_bounds",
    "validate_index_bounds_inputs",
]
