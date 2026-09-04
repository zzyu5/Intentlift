"""Deterministic value generation for SCRC Gate2 execution cases.

Gate2 input values are evidence, not incidental test fixtures.  This module
keeps the supported distributions explicit and reproducible so live SCRC
runs can record which numerical domain was exercised without falling back to
the archived JSON interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from intentir.ir.value_dimension import (
    RaggedRuntimeError,
    materialize_ragged_constant,
    ragged_member_length,
    validate_ragged_runtime_carrier,
    validate_ragged_runtime_invocation,
)
from intentir.ir.model import walk_intent_op_scopes
from intentir.runtime_dtype import is_numpy_bfloat16, numpy_bfloat16_dtype


class InputDistribution(str, Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    POSITIVE = "positive"
    NEAR_ONE = "near_one"
    LARGE_DYNAMIC = "large_dynamic"
    IEEE_SPECIAL = "ieee_special"
    INTEGER_BOUNDARY = "integer_boundary"


@dataclass(frozen=True)
class BoundaryExtent:
    label: str
    extent: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Gate2 boundary extent label must not be empty")
        if isinstance(self.extent, bool) or int(self.extent) <= 0:
            raise ValueError("Gate2 boundary extent must be positive")


def boundary_1d_extents(block: int) -> tuple[BoundaryExtent, ...]:
    """Canonical SCRC shape axis from section 9.3 of the design."""

    if isinstance(block, bool) or int(block) < 3:
        raise ValueError("Gate2 boundary block must be an integer >= 3")
    block = int(block)
    return (
        BoundaryExtent("one", 1),
        BoundaryExtent("block_minus_one", block - 1),
        BoundaryExtent("block", block),
        BoundaryExtent("block_plus_one", block + 1),
        BoundaryExtent("non_divisible", 2 * block + 3),
        BoundaryExtent("far_multitile", 8 * block + 17),
    )


_DTYPE_ALIASES = {
    "bool": "bool",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "index": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
}


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    key = dtype.removeprefix("torch.").lower()
    key = _DTYPE_ALIASES.get(key, key)
    if key == "bfloat16":
        return numpy_bfloat16_dtype()
    try:
        return np.dtype(key)
    except TypeError as exc:
        raise ValueError(f"unsupported Gate2 input dtype {dtype!r}") from exc


@dataclass(frozen=True)
class TensorInputSpec:
    shape: tuple[int, ...]
    dtype: str
    distribution: InputDistribution = InputDistribution.NORMAL
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, InputDistribution):
            object.__setattr__(
                self, "distribution", InputDistribution(self.distribution)
            )
        if any(int(extent) < 0 for extent in self.shape):
            raise ValueError("Gate2 input extents must be non-negative")
        if not np.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Gate2 input scale must be finite and non-negative")
        _numpy_dtype(self.dtype)


@dataclass(frozen=True)
class InputCaseTemplate:
    template_id: str
    tensors: Mapping[str, TensorInputSpec]
    scalars: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("Gate2 input template_id must not be empty")
        overlap = sorted(set(self.tensors) & set(self.scalars))
        if overlap:
            raise ValueError(
                "Gate2 input template tensor/scalar keys overlap: "
                + ", ".join(overlap)
            )
        for key, spec in self.tensors.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Gate2 tensor input keys must be non-empty strings")
            if not isinstance(spec, TensorInputSpec):
                raise TypeError("Gate2 input template tensors require TensorInputSpec")


@dataclass(frozen=True)
class GeneratedInputCase:
    case_id: str
    seed: int
    values: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _floating_values(
    rng: np.random.Generator, spec: TensorInputSpec, dtype: np.dtype[Any]
) -> np.ndarray:
    shape = tuple(int(extent) for extent in spec.shape)
    distribution = spec.distribution
    scale = float(spec.scale)
    if distribution is InputDistribution.NORMAL:
        values = rng.standard_normal(shape) * scale
    elif distribution is InputDistribution.UNIFORM:
        values = rng.uniform(-scale, scale, size=shape)
    elif distribution is InputDistribution.POSITIVE:
        values = np.abs(rng.standard_normal(shape)) * scale
    elif distribution is InputDistribution.NEAR_ONE:
        values = 1.0 + rng.uniform(-scale, scale, size=shape)
    elif distribution is InputDistribution.LARGE_DYNAMIC:
        count = int(np.prod(shape, dtype=np.int64))
        if count == 0:
            values = np.empty(shape, dtype=np.float64)
        else:
            exponents = np.linspace(-12.0, 12.0, count, dtype=np.float64)
            signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
            values = (signs * np.power(2.0, exponents) * scale).reshape(shape)
    elif distribution is InputDistribution.IEEE_SPECIAL:
        pattern = np.asarray(
            [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan],
            dtype=np.float64,
        )
        count = int(np.prod(shape, dtype=np.int64))
        values = np.resize(pattern, count).reshape(shape)
    elif distribution is InputDistribution.INTEGER_BOUNDARY:
        raise ValueError("integer_boundary requires an integer dtype")
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(distribution)
    return np.asarray(values, dtype=dtype)


def generate_tensor_input(
    spec: TensorInputSpec,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one deterministic tensor value from an explicit input spec."""

    if (seed is None) == (rng is None):
        raise ValueError("provide exactly one of seed or rng")
    generator = rng if rng is not None else np.random.default_rng(seed)
    assert generator is not None
    dtype = _numpy_dtype(spec.dtype)
    shape = tuple(int(extent) for extent in spec.shape)

    if np.issubdtype(dtype, np.floating) or is_numpy_bfloat16(dtype):
        return _floating_values(generator, spec, dtype)
    if spec.distribution in {
        InputDistribution.LARGE_DYNAMIC,
        InputDistribution.IEEE_SPECIAL,
        InputDistribution.NEAR_ONE,
    }:
        raise ValueError(
            f"distribution {spec.distribution.value!r} requires a floating dtype"
        )
    if np.issubdtype(dtype, np.bool_):
        if spec.distribution is InputDistribution.INTEGER_BOUNDARY:
            raise ValueError("integer_boundary requires an integer dtype")
        return generator.integers(0, 2, size=shape, dtype=np.int8).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        bound = max(1, int(spec.scale))
        if spec.distribution is InputDistribution.INTEGER_BOUNDARY:
            pattern = np.asarray(
                [0, bound, bound // 2, 1, max(0, bound - 1), 0],
                dtype=dtype,
            )
            count = int(np.prod(shape, dtype=np.int64))
            return np.resize(pattern, count).reshape(shape)
        low = (
            0
            if dtype.kind == "u"
            or spec.distribution is InputDistribution.POSITIVE
            else -bound
        )
        return generator.integers(low, bound + 1, size=shape, dtype=dtype)
    raise ValueError(f"unsupported Gate2 input dtype {spec.dtype!r}")


def generate_input_matrix(
    templates: Sequence[InputCaseTemplate], seeds: Sequence[int]
) -> tuple[GeneratedInputCase, ...]:
    """Expand explicit shape/value templates across reproducible seeds."""

    if not templates:
        raise ValueError("Gate2 input matrix requires at least one template")
    if not seeds:
        raise ValueError("Gate2 input matrix requires at least one seed")
    template_ids = [template.template_id for template in templates]
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("Gate2 input matrix template_id values must be unique")
    normalized_seeds: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Gate2 matrix seeds must be integers")
        normalized_seeds.append(seed)
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("Gate2 input matrix seeds must be unique")

    generated: list[GeneratedInputCase] = []
    for template in templates:
        for seed in normalized_seeds:
            rng = np.random.default_rng(seed)
            values: dict[str, Any] = dict(template.scalars)
            tensor_metadata: dict[str, Any] = {}
            for key in sorted(template.tensors):
                spec = template.tensors[key]
                values[key] = generate_tensor_input(spec, rng=rng)
                tensor_metadata[key] = {
                    "shape": list(spec.shape),
                    "dtype": spec.dtype,
                    "distribution": spec.distribution.value,
                    "scale": spec.scale,
                }
            generated.append(
                GeneratedInputCase(
                    f"{template.template_id}.seed{seed}",
                    seed,
                    values,
                    {
                        **dict(template.metadata),
                        "template_id": template.template_id,
                        "seed": seed,
                        "tensors": tensor_metadata,
                    },
                )
            )
    return tuple(generated)


def materialize_ragged_input_values(
    j: Any,
    values: Mapping[str, Any],
    input_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Make direct Ragged carrier inputs valid for every Gate2 leg.

    Generic integer generation cannot satisfy the offsets prefix contract by
    chance.  This workload-boundary helper derives the carrier shapes from the
    accepted J graph, creates a balanced non-decreasing offsets prefix, keeps
    the terminal offset equal to the indices length, and chooses outer group
    zero for direct selectors.  Source, Replay, and RefJ therefore consume the
    same J-derived values; no backend is asked to repair malformed input.

    Constant carriers/selectors are left untouched because the shared runtime
    validator already materializes and checks them.  A direct carrier that
    cannot be made valid is rejected as an invalid workload valuation.
    """

    if not hasattr(j, "intent") or not hasattr(j.intent, "body"):
        raise TypeError("Ragged workload input materialization requires canonical J")
    result = dict(values)
    j_inputs = {item.name: item for item in j.intent.inputs}

    scoped_operations = tuple(
        walk_intent_op_scopes(
            j.intent.body,
            visible=(item.name for item in j.intent.inputs),
        )
    )
    operations = tuple(item[1] for item in scoped_operations)
    has_ragged_operations = any(
        getattr(operation, "op", None) in {"ragged", "ragged_outer", "ragged_member"}
        for operation in operations
    )
    descriptors: dict[int, tuple[str, str]] = {}
    descriptor_names: set[str] = set()
    descriptor_contexts: dict[int, Mapping[str, Any]] = {}
    selectors: list[tuple[Any, str, Mapping[str, Any]]] = []
    for _path, operation, visible_bindings in scoped_operations:
        op_name = getattr(operation, "op", None)
        operands = tuple(getattr(operation, "operands", ()) or ())
        result_name = getattr(operation, "result", None)
        if op_name == "ragged":
            if (
                not isinstance(result_name, str)
                or len(operands) != 2
                or result_name in descriptor_names
            ):
                raise ValueError("Ragged workload contains an incomplete descriptor")
            descriptors[id(operation)] = (operands[0], operands[1])
            descriptor_names.add(result_name)
            descriptor_contexts[id(operation)] = visible_bindings
        elif op_name == "ragged_member":
            if len(operands) != 2:
                raise ValueError("Ragged workload contains an incomplete member query")
            descriptor_producer = visible_bindings.get(operands[0])
            if (
                descriptor_producer is None
                or getattr(descriptor_producer, "op", None) != "ragged"
            ):
                raise ValueError(
                    "Ragged member selector has no visible local descriptor producer"
                )
            selectors.append(
                (descriptor_producer, operands[1], visible_bindings)
            )

    if not descriptors:
        if selectors:
            raise ValueError("Ragged member selector has no descriptor producer")
        if not has_ragged_operations:
            # This helper is shared by dense and Ragged Gate2 workloads.  A
            # dense J has nothing to materialize; treating it as a malformed
            # Ragged graph would block every ordinary Gate2 case before any
            # source/replay/reference comparison can run.
            return result
        raise ValueError("Ragged operation set has no descriptor producer")

    # Derive the symbolic extents already fixed by direct workload values.  A
    # constant carrier is allowed to use those same J dimensions, but it must
    # still be materialized and checked before a direct selector is rewritten.
    dimensions: dict[str, int] = {}
    for name, tensor in j_inputs.items():
        runtime_name = input_bindings.get(name)
        if runtime_name is None or runtime_name not in result:
            continue
        value = np.asarray(result[runtime_name])
        if value.ndim != len(tensor.shape):
            continue
        for declared, actual in zip(tensor.shape, value.shape, strict=True):
            if isinstance(declared, str) and declared != "?":
                previous = dimensions.setdefault(declared, int(actual))
                if previous != int(actual):
                    raise ValueError(
                        f"Ragged workload dimension {declared!r} is inconsistent"
                    )

    def declared_field_dtype(operation: Any, field_index: int) -> str | None:
        operation_type = getattr(operation, "type", None)
        fields = (
            operation_type.get("fields")
            if isinstance(operation_type, Mapping)
            else None
        )
        if not isinstance(fields, list) or len(fields) != 2:
            return None
        field = fields[field_index]
        field_type = field.get("type") if isinstance(field, Mapping) else None
        dtype = field_type.get("dtype") if isinstance(field_type, Mapping) else None
        return dtype if isinstance(dtype, str) else None

    def constant_array(name: str, operation: Any, *, expected: str | None) -> np.ndarray:
        raw_shape = getattr(operation, "attrs", {}).get("shape", [])
        attrs = getattr(operation, "attrs", {})
        raw_value = attrs.get("value") if isinstance(attrs, Mapping) else None
        dtype = expected
        if dtype is None:
            operation_type = getattr(operation, "type", None)
            if isinstance(operation_type, Mapping):
                dtype = operation_type.get("dtype")
        try:
            return materialize_ragged_constant(
                raw_shape,
                raw_value,
                dtype=dtype,
                dims=dimensions,
                context=f"Ragged constant {name!r}",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Ragged constant {name!r} cannot be materialized") from exc

    def value_for(
        name: str,
        *,
        expected: str | None,
        role: str,
        visible_bindings: Mapping[str, Any],
    ) -> tuple[np.ndarray, str | None]:
        tensor = j_inputs.get(name)
        runtime_name = input_bindings.get(name)
        if tensor is not None and runtime_name is not None:
            if runtime_name not in result:
                raise ValueError(f"Ragged {role} input {name!r} lacks its workload value")
            return np.asarray(result[runtime_name]), runtime_name
        operation = visible_bindings.get(name)
        if operation is not None and getattr(operation, "op", None) == "const":
            return constant_array(name, operation, expected=expected), None
        raise ValueError(
            f"Ragged {role} {name!r} must be a direct input or a materializable const"
        )

    # Resolve each carrier once.  In particular, two descriptors sharing an
    # offsets vector must agree on the member-vector length because the final
    # offset is part of that shared carrier's value contract.
    carrier_records: dict[str, dict[str, Any]] = {}
    descriptor_records: dict[int, tuple[str, str]] = {}
    for descriptor_id, (offsets_name, indices_name) in descriptors.items():
        producer_context = descriptor_contexts[descriptor_id]
        producer = next(
            operation
            for _path, operation, _visible in scoped_operations
            if id(operation) == descriptor_id
        )
        expected_offsets = (
            declared_field_dtype(producer, 0) if producer is not None else None
        )
        expected_indices = (
            declared_field_dtype(producer, 1) if producer is not None else None
        )
        offsets, offsets_runtime = value_for(
            offsets_name,
            expected=expected_offsets or getattr(j_inputs.get(offsets_name), "dtype", None),
            role="offsets",
            visible_bindings=producer_context,
        )
        indices, indices_runtime = value_for(
            indices_name,
            expected=expected_indices or getattr(j_inputs.get(indices_name), "dtype", None),
            role="indices",
            visible_bindings=producer_context,
        )
        if offsets.ndim != 1 or offsets.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Ragged offsets input {offsets_name!r} is not a rank-1 integer carrier"
            )
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Ragged indices input {indices_name!r} is not a rank-1 integer carrier"
            )
        record = carrier_records.setdefault(
            offsets_name,
            {
                "offsets": offsets,
                "runtime": offsets_runtime,
                "indices": [],
            },
        )
        if record["offsets"].shape != offsets.shape or record["offsets"].dtype != offsets.dtype:
            raise ValueError(
                f"Ragged shared offsets carrier {offsets_name!r} is inconsistent"
            )
        record["indices"].append(
            (indices_name, indices, indices_runtime, expected_indices)
        )
        descriptor_records[descriptor_id] = (offsets_name, indices_name)

    group_counts: dict[str, int] = {}
    for offsets_name, record in carrier_records.items():
        offsets = record["offsets"]
        index_records = record["indices"]
        if offsets.size == 0:
            raise ValueError("Ragged offsets input must contain at least one value")
        index_counts = {int(indices.size) for _name, indices, _runtime, _dtype in index_records}
        if len(index_counts) != 1:
            raise ValueError(
                f"Ragged descriptors sharing offsets {offsets_name!r} disagree on index count"
            )
        index_count = next(iter(index_counts))
        group_count = int(offsets.size) - 1
        if group_count == 0 and index_count != 0:
            raise ValueError(
                "Ragged offsets with no outer groups cannot carry indices"
            )

        valid = True
        for _indices_name, indices, _runtime, expected_indices in index_records:
            try:
                validate_ragged_runtime_carrier(
                    offsets,
                    indices,
                    context=f"Gate2 ragged {offsets_name}",
                    offsets_dtype=(
                        j_inputs[offsets_name].dtype
                        if offsets_name in j_inputs
                        else None
                    ),
                    indices_dtype=expected_indices,
                )
            except RaggedRuntimeError:
                valid = False
                break

        if not valid and record["runtime"] is not None:
            if group_count:
                base, remainder = divmod(index_count, group_count)
                lengths = np.full(group_count, base, dtype=np.int64)
                if remainder:
                    lengths[:remainder] += 1
                rebuilt = np.concatenate(
                    (
                        np.asarray([0], dtype=np.int64),
                        np.cumsum(lengths, dtype=np.int64),
                    )
                )
            else:
                rebuilt = np.asarray([0], dtype=np.int64)
            limits = np.iinfo(offsets.dtype)
            if rebuilt[-1] > limits.max:
                raise ValueError(
                    f"Ragged offsets input {offsets_name!r} cannot represent its index count"
                )
            rebuilt = rebuilt.astype(offsets.dtype, copy=False)
            result[record["runtime"]] = rebuilt
            offsets = rebuilt
            record["offsets"] = offsets
        elif not valid:
            raise ValueError(
                f"Ragged constant offsets carrier {offsets_name!r} violates its runtime contract"
            )
        group_counts[offsets_name] = group_count

    for descriptor_producer, selector_name, selector_context in selectors:
        descriptor_id = id(descriptor_producer)
        carrier = descriptor_records.get(descriptor_id)
        if carrier is None:
            raise ValueError(
                "Ragged member selector references an unknown descriptor producer"
            )
        offsets_name, indices_name = carrier
        if selector_name in {offsets_name, indices_name}:
            raise ValueError("Ragged selector cannot alias a carrier field")
        selector_tensor = j_inputs.get(selector_name)
        expected_selector = getattr(selector_tensor, "dtype", None)
        selector, selector_runtime = value_for(
            selector_name,
            expected=expected_selector,
            role="selector",
            visible_bindings=selector_context,
        )
        if selector.shape != () or selector.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Ragged selector input {selector_name!r} must be a rank-0 integer"
            )
        group_count = group_counts[offsets_name]
        if group_count <= 0:
            raise ValueError("Ragged member requires at least one outer group")
        try:
            ragged_member_length(
                carrier_records[offsets_name]["offsets"],
                next(
                    indices
                    for name, indices, _runtime, _dtype in carrier_records[offsets_name]["indices"]
                    if name == indices_name
                ),
                selector,
                context=(
                    "Gate2 ragged member "
                    f"{getattr(descriptor_producer, 'result', '<anonymous>')}"
                ),
            )
        except (RaggedRuntimeError, StopIteration) as exc:
            if selector_runtime is None:
                raise ValueError(
                    f"Ragged constant selector {selector_name!r} is invalid"
                ) from exc
            result[selector_runtime] = np.asarray(0, dtype=selector.dtype).reshape(())

    runtime_inputs = {
        name: result[input_bindings[name]]
        for name in j_inputs
        if name in input_bindings and input_bindings[name] in result
    }
    try:
        validate_ragged_runtime_invocation(
            j.intent,
            runtime_inputs,
            dims=dimensions,
        )
    except RaggedRuntimeError as exc:
        raise ValueError(f"Gate2 Ragged input contract failed: {exc}") from exc
    return result


__all__ = [
    "BoundaryExtent",
    "GeneratedInputCase",
    "InputCaseTemplate",
    "InputDistribution",
    "TensorInputSpec",
    "boundary_1d_extents",
    "generate_input_matrix",
    "generate_tensor_input",
    "materialize_ragged_input_values",
]
