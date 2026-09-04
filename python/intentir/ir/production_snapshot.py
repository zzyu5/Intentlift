"""Strict transport snapshot for J plus an optional legacy Source-C prior.

The snapshot is not a new semantic or proof artifact.  It is a content-addressed
serialization of the already canonical J and a separate, erasable Source/search
prior retained for benchmark/export compatibility.  The ``ImplementationConfig``
stored here is not a target Layer-C plugin and is never accepted by the public
``LoweringRequest`` boundary.
It deliberately has no fields for source, replay, pi, comparison policy, or
certification evidence.  Consumers that need to establish acceptance do so
before constructing this object.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from intentir.ir.canonical_j import elaborate_j
from intentir.verify.gate1.model import ImplementationConfig, SemanticArchive


PRODUCTION_SNAPSHOT_SCHEMA = "intentir.production.jc-snapshot.v1"
PRODUCTION_CATALOG_SCHEMA = "intentir.production.jc-catalog.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ProductionSnapshotError(ValueError):
    """Raised when a production snapshot is malformed or not content exact."""


@dataclass(frozen=True)
class ProductionCatalogEntry:
    """Content-addressed location of one accepted J+Source-prior snapshot."""

    name: str
    case_id: str
    snapshot_file: str
    snapshot_digest: str
    j_digest: str
    c_digest: str | None
    receipt_digest: str
    evidence_kind: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionCatalogEntry":
        expected = {
            "name",
            "case_id",
            "snapshot_file",
            "snapshot_digest",
            "j_digest",
            "c_digest",
            "receipt_digest",
            "evidence_kind",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ProductionSnapshotError(
                "production catalog entry must contain exactly "
                f"{sorted(expected)}"
            )
        name = value["name"]
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise ProductionSnapshotError("production catalog name is unsafe")
        for field in ("snapshot_digest", "j_digest", "receipt_digest"):
            digest = value[field]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ProductionSnapshotError(
                    f"production catalog {field} is invalid"
                )
        c_digest = value["c_digest"]
        if c_digest is not None and (
            not isinstance(c_digest, str) or _SHA256.fullmatch(c_digest) is None
        ):
            raise ProductionSnapshotError("production catalog c_digest is invalid")
        snapshot_file = value["snapshot_file"]
        if snapshot_file != f"{value['snapshot_digest']}.json":
            raise ProductionSnapshotError(
                "production catalog snapshot_file is not content addressed"
            )
        for field in ("case_id", "evidence_kind"):
            if not isinstance(value[field], str) or not value[field]:
                raise ProductionSnapshotError(
                    f"production catalog {field} must be a non-empty string"
                )
        return cls(
            name=name,
            case_id=value["case_id"],
            snapshot_file=snapshot_file,
            snapshot_digest=value["snapshot_digest"],
            j_digest=value["j_digest"],
            c_digest=c_digest,
            receipt_digest=value["receipt_digest"],
            evidence_kind=value["evidence_kind"],
        )


def canonical_json(value: Any) -> str:
    """Return the one byte-stable JSON spelling used by snapshot digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProductionSnapshot:
    """Immutable J plus an optional legacy Source-C/search prior.

    The field remains named ``c`` for the persisted compatibility schema.  New
    production lowering must derive a ``TargetLayerCPlugin`` with ``BindTarget``
    instead of forwarding this value as target C.
    """

    j: SemanticArchive
    c: ImplementationConfig | None = None
    schema_version: str = PRODUCTION_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_SNAPSHOT_SCHEMA:
            raise ProductionSnapshotError(
                f"unsupported production snapshot schema {self.schema_version!r}"
            )
        if not isinstance(self.j, SemanticArchive):
            raise ProductionSnapshotError("production snapshot requires canonical J")
        # Decode/elaborate once, then detach the snapshot from any mutable object
        # owned by the caller.  This is still the same J, not a second IR.
        canonical_j = elaborate_j(self.j)
        detached_j = SemanticArchive.from_json(canonical_j.to_json())
        object.__setattr__(self, "j", detached_j)
        if self.c is not None:
            if not isinstance(self.c, ImplementationConfig):
                raise ProductionSnapshotError(
                    "production snapshot compatibility C must use the canonical "
                    "Source/search-prior schema"
                )
            detached_c = ImplementationConfig.from_json(self.c.to_json())
            object.__setattr__(self, "c", detached_c)

    @property
    def j_digest(self) -> str:
        return content_digest(self.j.to_json())

    @property
    def c_digest(self) -> str | None:
        return content_digest(self.c.to_json()) if self.c is not None else None

    @property
    def snapshot_digest(self) -> str:
        return content_digest(self._content())

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "j": self.j.to_json(),
            "c": self.c.to_json() if self.c is not None else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content(),
            "j_digest": self.j_digest,
            "c_digest": self.c_digest,
            "snapshot_digest": self.snapshot_digest,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        ) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionSnapshot":
        if not isinstance(value, Mapping):
            raise ProductionSnapshotError("production snapshot root must be an object")
        expected = {
            "schema_version",
            "j",
            "c",
            "j_digest",
            "c_digest",
            "snapshot_digest",
        }
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing:
            raise ProductionSnapshotError(
                f"production snapshot missing fields: {missing}"
            )
        if unknown:
            raise ProductionSnapshotError(
                f"production snapshot has forbidden fields: {unknown}"
            )
        for name in ("j_digest", "snapshot_digest"):
            if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
                raise ProductionSnapshotError(f"production snapshot {name} is invalid")
        if value["c_digest"] is not None and (
            not isinstance(value["c_digest"], str)
            or _SHA256.fullmatch(value["c_digest"]) is None
        ):
            raise ProductionSnapshotError("production snapshot c_digest is invalid")
        try:
            snapshot = cls(
                j=SemanticArchive.from_json(value["j"]),
                c=(
                    ImplementationConfig.from_json(value["c"])
                    if value["c"] is not None
                    else None
                ),
                schema_version=value["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ProductionSnapshotError(str(exc)) from exc
        if value["j_digest"] != snapshot.j_digest:
            raise ProductionSnapshotError("production snapshot J digest mismatch")
        if value["c_digest"] != snapshot.c_digest:
            raise ProductionSnapshotError("production snapshot C digest mismatch")
        if value["snapshot_digest"] != snapshot.snapshot_digest:
            raise ProductionSnapshotError("production snapshot content digest mismatch")
        return snapshot

    @classmethod
    def from_json(cls, payload: str) -> "ProductionSnapshot":
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProductionSnapshotError(
                f"invalid production snapshot JSON: {exc}"
            ) from exc
        return cls.from_dict(value)


def load_production_snapshot(
    directory: Path, name: str
) -> tuple[ProductionSnapshot, Path]:
    """Load a stable name from an export directory or frozen catalog.

    Acceptance exports retain human-readable ``<name>.json`` files.  Frozen
    catalogs retain only digest-named immutable objects.  Catalog metadata and
    the decoded J+Source-prior object are cross-checked before either is returned.
    """

    if not isinstance(directory, Path):
        directory = Path(directory)
    if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
        raise ProductionSnapshotError("production snapshot name is unsafe")
    named_path = directory / f"{name}.json"
    if named_path.is_file():
        return (
            ProductionSnapshot.from_json(named_path.read_text(encoding="utf-8")),
            named_path,
        )
    catalog_path = directory / "catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionSnapshotError(
            f"cannot read production catalog {catalog_path}: {exc}"
        ) from exc
    expected_root = {"protocol", "entry_count", "entries"}
    if not isinstance(catalog, Mapping) or set(catalog) != expected_root:
        raise ProductionSnapshotError(
            "production catalog root must contain protocol, entry_count, entries"
        )
    if catalog["protocol"] != PRODUCTION_CATALOG_SCHEMA:
        raise ProductionSnapshotError(
            f"unsupported production catalog protocol {catalog['protocol']!r}"
        )
    raw_entries = catalog["entries"]
    if (
        not isinstance(raw_entries, list)
        or isinstance(catalog["entry_count"], bool)
        or not isinstance(catalog["entry_count"], int)
        or catalog["entry_count"] != len(raw_entries)
    ):
        raise ProductionSnapshotError("production catalog entry_count mismatch")
    entries = [ProductionCatalogEntry.from_dict(item) for item in raw_entries]
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ProductionSnapshotError("production catalog contains duplicate names")
    entry = next((item for item in entries if item.name == name), None)
    if entry is None:
        raise ProductionSnapshotError(
            f"production catalog has no snapshot named {name!r}"
        )
    snapshot_path = directory / entry.snapshot_file
    try:
        snapshot = ProductionSnapshot.from_json(
            snapshot_path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise ProductionSnapshotError(
            f"cannot read production snapshot {snapshot_path}: {exc}"
        ) from exc
    for field, actual in (
        ("snapshot_digest", snapshot.snapshot_digest),
        ("j_digest", snapshot.j_digest),
        ("c_digest", snapshot.c_digest),
    ):
        if getattr(entry, field) != actual:
            raise ProductionSnapshotError(
                f"production catalog {name!r} {field} mismatch"
            )
    return snapshot, snapshot_path


__all__ = [
    "PRODUCTION_CATALOG_SCHEMA",
    "PRODUCTION_SNAPSHOT_SCHEMA",
    "ProductionCatalogEntry",
    "ProductionSnapshot",
    "ProductionSnapshotError",
    "canonical_json",
    "content_digest",
    "load_production_snapshot",
]
