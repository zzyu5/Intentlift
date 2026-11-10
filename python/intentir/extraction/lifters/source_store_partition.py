"""Shared validation for Source-projected physical-store assembly relations."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

try:
    from source_projection_contracts import (
        is_canonical_ordinal,
        provider_shape_from_canonical,
    )
except ImportError:
    from intentir.ir.source_projection_contracts import (
        is_canonical_ordinal,
        provider_shape_from_canonical,
    )


def validate_source_store_partition_assembly(
    partition: Mapping[str, Any],
    *,
    access_rows: Sequence[Any],
    output_name: str,
    output_type: Any,
    expected_stores: Sequence[Mapping[str, Any]],
    dimension_aliases: Mapping[str, str],
) -> None:
    """Validate one Source-projected physical-store assembly.

    This check is shared by Source projection, the mounted MCP context reader,
    and the host MCP environment. It checks an already Source-derived locator;
    it never chooses producers or constructs a logical output binding.
    """

    assembly = partition.get("assembly") if isinstance(partition, Mapping) else None
    if assembly is None:
        return
    if (
        not isinstance(assembly, Mapping)
        or set(assembly)
        != {
            "canonical_op",
            "axis",
            "operand_store_ordinals",
            "operand_coordinates",
            "operand_type",
            "result_type",
        }
        or assembly.get("canonical_op") != "concat"
        or type(assembly.get("axis")) is not int
        or not isinstance(assembly.get("operand_store_ordinals"), list)
        or not isinstance(assembly.get("operand_coordinates"), list)
        or len(assembly["operand_store_ordinals"])
        != len(assembly["operand_coordinates"])
        or len(assembly["operand_store_ordinals"]) != len(expected_stores)
    ):
        raise ValueError("canonical Source output assembly requirement is malformed")
    output_dtype = output_type.get("dtype") if isinstance(output_type, Mapping) else None
    output_shape = output_type.get("shape") if isinstance(output_type, Mapping) else None
    axis = assembly["axis"]
    if (
        not isinstance(output_dtype, str)
        or not isinstance(output_shape, list)
        or axis < 0
        or axis >= len(output_shape)
        or assembly.get("result_type")
        != {"dtype": output_dtype, "shape": copy.deepcopy(output_shape)}
        or not isinstance(assembly.get("operand_type"), Mapping)
        or assembly["operand_type"].get("dtype") != output_dtype
        or assembly["operand_type"].get("shape")
        != [item for index, item in enumerate(output_shape) if index != axis]
    ):
        raise ValueError("canonical Source output assembly type is not output-bound")
    by_ordinal = {
        item["ordinal"]: item
        for item in expected_stores
        if isinstance(item, Mapping)
        and is_canonical_ordinal(item.get("ordinal"))
    }
    if (
        len(by_ordinal) != len(expected_stores)
        or any(
            not isinstance(item, Mapping)
            or not is_canonical_ordinal(item.get("ordinal"))
            or not isinstance(item.get("coordinates"), list)
            or len(item["coordinates"]) != len(output_shape)
            for item in expected_stores
        )
    ):
        raise ValueError("canonical Source output assembly coordinates are malformed")
    ordered = sorted(expected_stores, key=lambda item: item["coordinates"][axis])
    if (
        assembly["operand_store_ordinals"]
        != [item["ordinal"] for item in ordered]
        or assembly["operand_coordinates"]
        != [item["coordinates"] for item in ordered]
        or any(type(item["coordinates"][axis]) is not int for item in ordered)
        or any(item["ordinal"] not in by_ordinal for item in ordered)
        or [item["coordinates"][axis] for item in ordered]
        != list(range(len(ordered)))
    ):
        raise ValueError("canonical Source output assembly is not a complete ordered partition")
    if any(
        any(
            item["coordinates"][dimension]
            != ordered[0]["coordinates"][dimension]
            for item in ordered
        )
        for dimension in range(len(output_shape))
        if dimension != axis
    ):
        raise ValueError("canonical Source output assembly coordinates are inconsistent")
    expected_slice_shape = [
        item for index, item in enumerate(output_shape) if index != axis
    ]
    access_by_ordinal = {
        row.get("ordinal"): row
        for row in access_rows
        if isinstance(row, Mapping) and is_canonical_ordinal(row.get("ordinal"))
    }
    for store in ordered:
        access = access_by_ordinal.get(store["ordinal"])
        source = access.get("source_semantics") if isinstance(access, Mapping) else None
        authorizations = (
            source.get("semantic_authorizations")
            if isinstance(source, Mapping)
            else None
        )
        owner = authorizations.get("owner") if isinstance(authorizations, Mapping) else None
        outline = owner.get("authoring_outline") if isinstance(owner, Mapping) else None
        graph = outline.get("graph") if isinstance(outline, Mapping) else None
        root = graph[-1] if isinstance(graph, list) and graph else None
        try:
            root_shape = (
                provider_shape_from_canonical(root[2], dimension_aliases)
                if isinstance(root, (list, tuple))
                and len(root) == 5
                else None
            )
        except ValueError:
            root_shape = None
        if (
            not isinstance(source, Mapping)
            or source.get("kind") != "store"
            or source.get("tensor") != output_name
            or source.get("masked") is True
            or source.get("predicate") is not None
            or not isinstance(root, (list, tuple))
            or len(root) != 5
            or root[1] != output_dtype
            or root_shape != expected_slice_shape
        ):
            raise ValueError("canonical Source output assembly owner type is not closed")
