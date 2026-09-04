#!/usr/bin/env python3
"""Diagnose current Gate1/Gate2 capability for exact historical bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.adapters import BoundGate1Adapter
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import StaticReplayWriter
from intentir.verify.gate2.orchestrate import Gate2Runner
from intentir.verify.scrc_workload_run import (
    build_workload_gate2_suite,
    validate_workload_gate1_recheck_parent,
)
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    prepare_triton_workload,
)
from intentir.verify.typecheck import type_check


GATE2_CAPABILITY_RECHECK_PROTOCOL = (
    "intentir.scrc.gate2-capability-recheck.v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="one active workload case id")
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="artifact.json from the exact parent attempt",
    )
    parser.add_argument(
        "--parent-evidence",
        required=True,
        type=Path,
        help="attempt directory containing artifact, gate1, replay and receipt",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/intentir-scrc-task11/gate2-capability-rechecks"),
        help="separate immutable evidence root",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def _read_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"parent Gate1 evidence lacks {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parent Gate1 {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"parent Gate1 {label} must be a JSON object")
    return value


def _load_parented_artifact(
    artifact_path: Path,
    parent_path: Path,
) -> tuple[Gate1Artifact, Path, Path]:
    parent = parent_path.expanduser().resolve()
    artifact_file = artifact_path.expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent evidence directory does not exist")
    if not artifact_file.is_file():
        raise ValueError("artifact file does not exist")
    try:
        artifact_file.relative_to(parent)
    except ValueError as exc:
        raise ValueError("artifact must be inside parent evidence") from exc
    payload = _read_object(artifact_file, "artifact.json")
    try:
        artifact = Gate1Artifact.from_json(payload)
    except Exception as exc:
        raise ValueError(
            f"artifact decode rejected: {type(exc).__name__}: {exc}"
        ) from exc
    return artifact, artifact_file, parent


def _load_exact_replay(parent: Path) -> tuple[KernelSource, str]:
    _read_object(parent / "gate1.json", "gate1.json")
    receipt = _read_object(parent / "replay_receipt.json", "replay_receipt.json")
    function_name = receipt.get("function_name")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError("parent replay receipt lacks function_name")
    replay_path = parent / "replay.py"
    try:
        replay_code = replay_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"parent replay.py is unreadable: {exc}") from exc
    try:
        replay = KernelSource.from_replay(
            replay_code,
            function_name,
        )
    except ValueError as exc:
        raise ValueError(f"parent Replay is invalid: {exc}") from exc
    return replay, replay_code


def _case_slug(case_id: str) -> str:
    identity = case_id.removeprefix("high_performance:")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", identity).strip("_").lower()
    if not slug:
        raise ValueError("workload id does not form a run directory name")
    return slug


def _next_recheck_dir(root: Path, case_id: str) -> Path:
    parent = root.expanduser().resolve() / _case_slug(case_id)
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = parent / f"gate2-recheck-{index:03d}"
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _emit(value: dict, *, output: Path | None, json_mode: bool) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if json_mode or output is None:
        print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        artifact, artifact_file, parent = _load_parented_artifact(
            args.artifact, args.parent_evidence
        )
        artifact = canonicalize_gate1_artifact(artifact)
        replay, replay_code = _load_exact_replay(parent)
        adapter = TRITON_WORKLOAD_ADAPTERS.get(args.case)
        if adapter is None:
            raise ValueError("case has no active Triton workload adapter")
        prepared = prepare_triton_workload(
            adapter,
            dataset_root=args.dataset_root,
        )
        validate_workload_gate1_recheck_parent(parent, prepared, artifact)
        recheck_dir = _next_recheck_dir(args.run_root, args.case)
        request_payload = {
            "protocol": GATE2_CAPABILITY_RECHECK_PROTOCOL,
            "capability_recheck": True,
            "model_calls": 0,
            "reruns_current_gate1": True,
            "reuses_exact_gate1_artifact": True,
            "reuses_exact_gate1_replay": True,
            "changes_parent_verdict": False,
            "case_id": args.case,
            "artifact": str(artifact_file),
            "parent_evidence": str(parent),
        }
        _write_json(recheck_dir / "request.json", request_payload)
        _write_json(recheck_dir / "artifact.json", artifact.to_json())
        (recheck_dir / "replay.py").write_text(replay_code, encoding="utf-8")

        checked = type_check(
            artifact.j,
            artifact.c,
            artifact.pi,
            require_program_axes=artifact.wire_schema == artifact.SCHEMA,
        )
        _write_json(recheck_dir / "typecheck.json", checked.to_json())
        if not checked.passed:
            raise ValueError("exact artifact no longer passes current TypeCheck")
        gate1 = BoundGate1Adapter(
            Gate1Runner(
                compiler=TritonCompiler(timeout=240.0),
                writer=StaticReplayWriter(
                    replay_code,
                    model="exact-parent-replay",
                ),
                config=Gate1Config(run_masked_j=False),
            )
        ).run(artifact, prepared.request)
        _write_json(recheck_dir / "gate1.json", gate1.to_json())

        gate2_passed = False
        status_counts: dict[str, int] = {}
        if gate1.passed:
            accepted = gate1.normal.generation
            if accepted is None:
                raise ValueError("closed Gate1 lacks its accepted Replay")
            replay = KernelSource.from_replay(
                accepted.code,
                accepted.function_name,
            )
            suite = build_workload_gate2_suite(prepared, artifact)
            result = Gate2Runner().run_suite(
                prepared.request.source.gate_source,
                replay,
                suite,
            )
            gate2_payload = result.to_json()
            _write_json(recheck_dir / "gate2.json", gate2_payload)
            gate2_passed = bool(result.passed)
            status_counts = dict(result.status_counts)
            status = (
                "capability_execution_pass"
                if result.passed
                else "gate2_failed"
            )
        else:
            status = "gate1_no_closure"
        summary = {
            **request_payload,
            "evidence_path": str(recheck_dir),
            "status": status,
            "gate1_passed": bool(gate1.passed),
            "gate2_passed": gate2_passed,
            "status_counts": status_counts,
        }
        _write_json(recheck_dir / "recheck.json", summary)
        (recheck_dir / "SUMMARY.md").write_text(
            "# Historical exact-byte current-contract capability diagnostic\n\n"
            f"- case: `{args.case}`\n"
            f"- status: `{status}`\n"
            "- model calls: `0`\n"
            f"- parent Gate1 evidence: `{parent}`\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"fixed-artifact Gate2 recheck failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    _emit(summary, output=args.output, json_mode=args.json)
    return 0 if summary["status"] == "capability_execution_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
