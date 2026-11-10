"""Compact, source-neutral lookup over the supplied IntentIR references."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


_BUNDLED_INPUT_ROOT = (
    Path("/input")
    if Path("/input").is_dir()
    else Path(__file__).resolve().parents[1]
)
_DEFAULT_SCHEMA = str(_BUNDLED_INPUT_ROOT / "schema" / "artifact.schema.json")
_COMMANDS = {
    "top",
    "catalog",
    "sections",
    "authoring-brief",
    "op",
    "section",
    "bundle",
}


def _load_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("schema root must be an object")
    return value


def _object_line(label: str, value: dict[str, Any]) -> None:
    required = value.get("required", [])
    properties = value.get("properties", {})
    print(f"{label}: required={required}; fields={list(properties)}")


def _top(schema: dict[str, Any]) -> None:
    _object_line("root", schema)
    root = schema["properties"]
    j = root["j"]
    _object_line("j", j)
    _object_line("j.intent", j["properties"]["intent"])
    _object_line(
        "j.intent.intent_function",
        j["properties"]["intent"]["properties"]["intent_function"],
    )
    _object_line("j.boundary", j["properties"]["boundary"])
    _object_line("j.signature", j["properties"]["signature"])
    _object_line("c", root["c"])
    _object_line("pi", root["pi"])
    print(
        "Use `section NAME` for one leaf schema, `catalog` for operation "
        "names, and `op NAME` for one operation contract."
    )


def _operation_catalog(schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    catalog = schema.get("x-intentir-operation-catalog")
    if not isinstance(catalog, Mapping):
        raise ValueError("schema contains no canonical operation catalog")
    ordinary = catalog.get("ordinary")
    structured = catalog.get("structured")
    if (
        not isinstance(ordinary, list)
        or not isinstance(structured, list)
        or not ordinary
        or not structured
        or any(not isinstance(name, str) for name in (*ordinary, *structured))
        or len(set(ordinary)) != len(ordinary)
        or len(set(structured)) != len(structured)
        or set(ordinary) & set(structured)
    ):
        raise ValueError("canonical operation catalog is malformed")
    return list(ordinary), list(structured)


def schema_operation_catalog(
    schema: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return the exact ordinary/structured operation catalog from schema."""

    return _operation_catalog(schema)


def schema_operation_contract(
    schema: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Return one operation contract without consulting prose references."""

    variants = schema.get("$defs", {}).get("intent_op", {}).get("anyOf", [])
    for variant in variants:
        op_schema = variant.get("properties", {}).get("op", {})
        if name not in op_schema.get("enum", []):
            continue
        result = copy.deepcopy(variant)
        result.setdefault("properties", {}).setdefault("op", {})["enum"] = [name]
        if "allOf" in result:
            result["allOf"] = [
                condition
                for condition in result.get("allOf", [])
                if condition.get("if", {})
                .get("properties", {})
                .get("op", {})
                .get("const")
                == name
            ]
        return result
    ordinary, structured = _operation_catalog(schema)
    raise KeyError(f"unknown operation: {name}; choose {ordinary} or {structured}")


def _semantic_provider_wire(schema_path: str) -> dict[str, Any]:
    """Load the host-published semantic submission wire beside one schema.

    ``artifact.schema.json`` describes the completed canonical artifact.  It
    is deliberately not returned to a Lift owner as an op-authoring grammar:
    that grammar would require host-allocated n*/v*/q* fields.  The mounted
    canonical bundle contains the schema-derived semantic projection used by
    source-brief and ingress, so schema_lookup must consume that same object.
    """

    bundle_path = Path(schema_path).with_name("canonical_wire_bundle.json")
    value = _load_json(str(bundle_path))
    profile = value.get("profile")
    if not isinstance(profile, Mapping):
        raise TypeError("canonical wire bundle lacks canonical profile")
    contracts = profile.get("semantic_operation_contracts")
    semantic_candidate = profile.get("semantic_candidate")
    template = (
        semantic_candidate.get("template")
        if isinstance(semantic_candidate, Mapping)
        else None
    )
    operation_namespace = (
        semantic_candidate.get("operation_namespace")
        if isinstance(semantic_candidate, Mapping)
        else None
    )
    if not isinstance(contracts, Mapping):
        raise TypeError(
            "canonical wire bundle lacks semantic operation lookup contracts"
        )
    # Keep the same top-level shape consumed by ``_op`` while sourcing exact
    # leaves from the canonical profile's bounded lookup projection.  The
    # first-turn semantic wire is intentionally smaller and must not be used
    # as a substitute for an exact operation leaf.
    if not isinstance(template, Mapping) or not isinstance(operation_namespace, Mapping):
        raise TypeError("canonical wire bundle lacks semantic authoring contract")
    return {
        "semantic_candidate": copy.deepcopy(dict(template)),
        "operation_namespace": copy.deepcopy(dict(operation_namespace)),
        "operations": copy.deepcopy(dict(contracts)),
    }


def _catalog(schema: dict[str, Any]) -> None:
    ordinary, structured = schema_operation_catalog(schema)
    print("ordinary operations: " + ", ".join(ordinary))
    print("structured operations: " + ", ".join(structured))
    print("Use `op NAME` to print the exact schema-derived operation contract.")


def _section_catalog(schema: dict[str, Any]) -> None:
    names = list(_sections(schema))
    print("artifact sections: " + ", ".join(names))
    print("Use `section NAME` to print the exact closed schema for one item.")


def _authoring_brief(schema: dict[str, Any]) -> None:
    """Print one compact source-neutral navigation index, never a template."""

    print(
        "These sections describe the completed host artifact, not the Lift "
        "submission. Author semantic rows from the mounted provider wire; "
        "omit host-owned semantic_owner_authorization."
    )
    _top(schema)
    _section_catalog(schema)
    _catalog(schema)


def _op(schema: dict[str, Any], name: str, semantic_wire: Mapping[str, Any]) -> None:
    """Print one semantic submission op contract, never final artifact wire."""

    operations = semantic_wire.get("operations")
    contract = operations.get(name) if isinstance(operations, Mapping) else None
    if not isinstance(contract, Mapping):
        if name in {"load", "store", "atomic_add", "atomic_cas"}:
            raise KeyError(
                f"{name} is a Source boundary access kind, not a J operation; "
                "write it as j.boundary.accesses[].kind"
            )
        if name in {"sigmoid", "exp", "log", "rsqrt", "sqrt", "tanh"}:
            raise KeyError(
                f"{name} is an API function, not a J operation; use "
                "map_unary with attrs.fn"
            )
        ordinary, structured = _operation_catalog(schema)
        raise KeyError(f"unknown semantic operation: {name}; choose {ordinary} or {structured}")
    template = semantic_wire.get("semantic_candidate")
    namespace = semantic_wire.get("operation_namespace")
    if not isinstance(template, Mapping) or not isinstance(namespace, Mapping):
        raise TypeError("schema lookup lacks canonical semantic authoring contract")
    print(
        json.dumps(
            {
                "semantic_candidate": dict(template),
                "operation_namespace": dict(namespace),
                "operation": dict(contract),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _schema_path(schema: Mapping[str, Any], path: list[str]) -> Any:
    value: Any = schema
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise TypeError(
                "canonical schema section path is unavailable: "
                + ".".join(path)
            )
        value = value[component]
    if not isinstance(value, dict):
        raise TypeError("canonical schema section must resolve to an object")
    return value


def _sections(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return sections from the schema-published canonical section index."""

    index = schema.get("x-intentir-schema-sections")
    if not isinstance(index, Mapping) or not index:
        raise TypeError("schema has no canonical section index")
    sections: dict[str, dict[str, Any]] = {}
    for name, descriptor in index.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise TypeError("canonical schema section index is malformed")
        path = descriptor.get("path")
        if (
            not isinstance(path, list)
            or not path
            or not all(isinstance(item, str) and item for item in path)
        ):
            raise TypeError(f"canonical schema section {name!r} has no path")
        sections[name] = _schema_path(schema, path)
    return sections


def schema_sections(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the exact source-neutral leaf sections used by CLI and agents."""

    return _sections(schema)


def _section(schema: dict[str, Any], name: str) -> None:
    sections = _sections(schema)
    if name not in sections:
        raise KeyError(f"unknown section: {name}; choose {sorted(sections)}")
    print(json.dumps(sections[name], indent=2, sort_keys=True))


def _bundle(
    schema: dict[str, Any],
    items: list[str],
    semantic_wire: Mapping[str, Any],
) -> None:
    """Render several independent source-neutral lookups in one tool action."""

    for index, item in enumerate(items):
        if index:
            print()
        print(f"=== {item} ===")
        try:
            if item == "top":
                _top(schema)
            elif item == "catalog":
                _catalog(schema)
            elif item == "sections":
                _section_catalog(schema)
            elif item == "authoring-brief":
                _authoring_brief(schema)
            elif item.startswith("op:") and len(item) > 3:
                _op(schema, item[3:], semantic_wire)
            elif item.startswith("section:") and len(item) > 8:
                _section(schema, item[8:])
            else:
                raise ValueError(
                    "bundle items must be top, catalog, sections, "
                    "authoring-brief, op:NAME, or section:NAME"
                )
        except (KeyError, ValueError) as exc:
            # One mistaken Source/API spelling must not discard the valid
            # operation contracts in the same bounded read.  This remains a
            # diagnostic lookup response; it never aliases or constructs a J
            # operation on the provider's behalf.
            print(f"lookup_error: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("top")
    subparsers.add_parser("catalog")
    subparsers.add_parser("sections")
    subparsers.add_parser("authoring-brief")
    op_parser = subparsers.add_parser("op")
    op_parser.add_argument("name")
    section_parser = subparsers.add_parser("section")
    section_parser.add_argument("name")
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("items", nargs="+")
    argv = sys.argv[1:]
    if argv and argv[0] in _COMMANDS:
        argv = [_DEFAULT_SCHEMA, *argv]
    args = parser.parse_args(argv)
    schema = _load_json(args.schema)
    # Only the provider-visible op query requires the mounted construction
    # bundle.  Keep source-neutral catalog/section inspection usable for an
    # explicitly supplied standalone schema, while the MCP environment itself
    # admits only ``op:NAME`` and therefore always takes the strict path.
    needs_semantic_wire = args.command == "op" or (
        args.command == "bundle"
        and any(item.startswith("op:") for item in args.items)
    )
    semantic_wire: Mapping[str, Any] = (
        _semantic_provider_wire(args.schema) if needs_semantic_wire else {}
    )
    if args.command == "top":
        _top(schema)
    elif args.command == "catalog":
        _catalog(schema)
    elif args.command == "sections":
        _section_catalog(schema)
    elif args.command == "authoring-brief":
        _authoring_brief(schema)
    elif args.command == "op":
        _op(schema, args.name, semantic_wire)
    elif args.command == "section":
        _section(schema, args.name)
    else:
        _bundle(schema, args.items, semantic_wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
