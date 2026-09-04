#!/usr/bin/env python3
"""Export one canonical J or Gate1 artifact as Intent Kernel MLIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.frontend.intent_kernel_mlir import (
    IntentKernelCapabilityError,
    export_intent_kernel_mlir,
)
from intentir.backend.protocol import (
    BackendTarget,
    bind_target_from_e_c,
    project_source_interface_to_target,
    serialize_target_artifact,
)
from intentir.ir.layer_c import LayerCInterface, source_layer_c_interface
from intentir.verify.gate1.model import Gate1Artifact, SemanticArchive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Gate1 artifact.json or canonical J JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write MLIR here, or the target-plugin artifact in bridge mode",
    )
    parser.add_argument(
        "--target-binding",
        type=Path,
        default=None,
        help=(
            "JSON bridge request with target, target_interface, capabilities, "
            "bindings, choice, required_capabilities, and contract"
        ),
    )
    return parser


def _load(path: Path) -> tuple[SemanticArchive, LayerCInterface]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    # Any artifact-shaped input must cross the one Gate1 envelope parser.  A
    # previous exact-set dispatch let a malformed artifact with an extra root
    # field fall through to the J parser, which obscured a nested c/pi layout
    # error and allowed different entrances to interpret ownership differently.
    if any(key in payload for key in ("schema", "j", "c", "pi")):
        if set(payload) != {"schema", "j", "c", "pi"}:
            raise ValueError(
                "Gate1 artifact input must contain exactly schema, j, c, and pi"
            )
        artifact = Gate1Artifact.from_json(payload)
        return artifact.j, source_layer_c_interface(artifact.j, artifact.c)
    return SemanticArchive.from_json(payload), LayerCInterface()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        archive, interface = _load(args.input)
        if args.target_binding is not None:
            request = json.loads(args.target_binding.read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                raise ValueError("target binding request must be a JSON object")
            required = {
                "target", "target_interface", "capabilities", "bindings",
                "choice", "required_capabilities", "contract",
            }
            optional = {"e_c", "e_c_mapping", "allowed_authorizations"}
            if set(request) - required - optional or not required <= set(request):
                raise ValueError(
                    "target binding request must contain the canonical bridge fields"
                )
            target_interface = LayerCInterface.from_dict(
                request["target_interface"]
            )
            if "e_c" in request:
                e_c = LayerCInterface.from_dict(request["e_c"])
                e_c_mapping = tuple(request.get("e_c_mapping", ()))
            else:
                # The Gate1 input carries the complete Source-C declaration
                # view.  Only declarations explicitly compatible with the
                # target interface enter shared E_C by default; the remaining
                # Source-only declarations stay on the Source/Gate1 side.
                e_c, inferred_mapping = project_source_interface_to_target(
                    interface,
                    target_interface,
                    e_c_mapping=(
                        tuple(request["e_c_mapping"])
                        if "e_c_mapping" in request
                        else None
                    ),
                )
                e_c_mapping = inferred_mapping
            artifact = bind_target_from_e_c(
                j=archive,
                e_c=e_c,
                target=BackendTarget.from_dict(request["target"]),
                target_interface=target_interface,
                capabilities=tuple(request["capabilities"]),
                bindings=tuple(request["bindings"]),
                choice=request["choice"],
                required_capabilities=tuple(request["required_capabilities"]),
                contract=request["contract"],
                allowed_authorizations=tuple(
                    request.get("allowed_authorizations", ())
                ),
                e_c_mapping=e_c_mapping,
            )
            rendered = json.dumps(
                serialize_target_artifact(artifact),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ) + "\n"
            if args.output is None:
                sys.stdout.write(rendered)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            return 0
        result = export_intent_kernel_mlir(archive, interface)
        if args.output is None:
            sys.stdout.write(result.mlir)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.mlir, encoding="utf-8")
    except (IntentKernelCapabilityError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
