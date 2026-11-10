#!/usr/bin/env python3
"""Pair an already selected IntentIR RVV winner with one external baseline.

This target-side tool deliberately skips search.  It reuses the production
runner's payload binding, Q comparator, cold-data eviction, object linker and
external-baseline compiler, then alternates only the two requested functions.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


_RUNNER_PATH = Path(__file__).with_name("run_rvv_benchmark_bundle.py")
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "intentir_rvv_pair_base_runner", _RUNNER_PATH
)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise RuntimeError(f"cannot import RVV runner from {_RUNNER_PATH}")
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner)


REPORT_PROTOCOL = "intentir.production.rvv-external-pair.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--external-baseline-sidecar", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--working-set-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cold-iterations", type=int, default=16)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--ggml-root", type=Path)
    parser.add_argument("--ggml-lib-dir", type=Path)
    return parser


def _only(items: list[Mapping[str, Any]], *, label: str) -> Mapping[str, Any]:
    if len(items) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(items)}")
    return items[0]


def _expected_outputs(
    entry: Mapping[str, Any], payload_root: Path
) -> dict[str, tuple[Any, int, int, str]]:
    result = {}
    for spec in entry["output_specs"]:
        name = spec["name"]
        record = entry["reference_outputs"][entry["output_bindings"][name]]
        payload = runner._decode_payload(record, payload_root)
        count = runner._numel([int(item) for item in spec["shape"]])
        byte_count = count * runner._dtype_size(spec["dtype"])
        if len(payload) != byte_count:
            raise ValueError(f"reference payload size differs from output {name!r}")
        result[name] = (
            ctypes.create_string_buffer(payload, len(payload)),
            count,
            byte_count,
            spec["dtype"],
        )
    return result


def _check(
    function: Any,
    arguments: list[Any],
    output_buffers: list[Any],
    entry: Mapping[str, Any],
    expected: Mapping[str, tuple[Any, int, int, str]],
    comparator: Any,
) -> list[dict[str, Any]]:
    for buffer in output_buffers:
        ctypes.memset(ctypes.addressof(buffer), 0, ctypes.sizeof(buffer))
    function(*arguments)
    records = []
    for spec, actual in zip(entry["output_specs"], output_buffers, strict=True):
        name = spec["name"]
        reference, count, byte_count, dtype = expected[name]
        passed, maximum = comparator.close(
            actual,
            reference,
            byte_count=byte_count,
            element_count=count,
            dtype=dtype,
            rule=entry["output_rules"][name],
        )
        records.append(
            {
                "output": name,
                "passed": passed,
                "max_abs_error": maximum,
                "comparator_protocol": runner.NATIVE_COMPARATOR_PROTOCOL,
            }
        )
    if not all(item["passed"] for item in records):
        raise RuntimeError(f"output comparison failed: {records}")
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(args.working_set_mib, args.warmup, args.cold_iterations, args.batches) <= 0:
        raise ValueError("all measurement counts must be positive")
    if (args.ggml_root is None) != (args.ggml_lib_dir is None):
        raise ValueError("--ggml-root and --ggml-lib-dir must be provided together")
    affinity = {args.cpu}
    if not affinity.issubset(os.sched_getaffinity(0)):
        raise ValueError(f"CPU {args.cpu} is outside available affinity")
    os.sched_setaffinity(0, affinity)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "CLOSE"

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("protocol") != runner.BUNDLE_PROTOCOL:
        raise ValueError("unsupported RVV benchmark bundle")
    entry = _only(
        [item for item in bundle["kernels"] if item["kernel"] == args.kernel],
        label="bundle kernel",
    )
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    selected_kernel = _only(
        [item for item in selection["kernels"] if item["kernel"] == args.kernel],
        label="selection-report kernel",
    )
    winner_id = selected_kernel["winner"]["candidate_id"]
    candidate = _only(
        [
            item
            for item in entry["candidates"]
            if item["candidate"]["candidate_id"] == winner_id
        ],
        label="selected Full candidate",
    )
    sidecar = runner._load_external_baseline_sidecar(
        args.external_baseline_sidecar
    )
    baseline = _only(
        [
            item
            for item in sidecar.get(args.kernel, ())
            if item["baseline_id"] == args.baseline_id
        ],
        label="external baseline",
    )

    payload_root = args.bundle.resolve().parent
    entry = {
        **entry,
        "output_specs": runner._resolve_runtime_output_specs(entry, payload_root),
    }
    argument_sets, keepalive, output_sets = runner._argument_replicas(
        entry, 1, payload_root
    )
    arguments = argument_sets[0]
    outputs = output_sets[0]
    descriptors = [item._obj for item in arguments]
    expected = _expected_outputs(entry, payload_root)
    eviction_bytes = args.working_set_mib * 1024 * 1024
    cache = runner._cpu_cache_record()
    if (
        cache["last_level_data_cache_bytes"]
        and eviction_bytes <= cache["last_level_data_cache_bytes"]
    ):
        raise ValueError("cold-data eviction buffer must exceed the last-level cache")
    eviction = ctypes.create_string_buffer(eviction_bytes)

    started = time.time_ns()
    with tempfile.TemporaryDirectory(prefix="intentir-rvv-pair-") as temporary:
        directory = Path(temporary)
        comparator = runner._NativeComparator(directory)
        full_path = runner._link_object(directory, candidate)
        full_library = ctypes.CDLL(str(full_path))
        full_function = getattr(
            full_library, candidate["candidate"]["c_interface_symbol"]
        )
        external_library, external_function, compile_record = (
            runner._compile_external_baseline(
                directory,
                baseline,
                ggml_root=args.ggml_root,
                ggml_lib_dir=args.ggml_lib_dir,
            )
        )
        argtypes = [ctypes.POINTER(type(item)) for item in descriptors]
        for function in (full_function, external_function):
            function.argtypes = argtypes
            function.restype = None
        full_correctness = _check(
            full_function, arguments, outputs, entry, expected, comparator
        )
        external_correctness = _check(
            external_function, arguments, outputs, entry, expected, comparator
        )

        full_rotation = runner._ArgumentRotation(argument_sets)
        external_rotation = runner._ArgumentRotation(argument_sets)
        runner._measure(
            full_function,
            full_rotation,
            regime="cold-data",
            eviction_buffer=eviction,
            warmup=args.warmup,
            warmup_ms=None,
            iterations=1,
            batches=1,
            timed_batch_ms=None,
        )
        runner._measure(
            external_function,
            external_rotation,
            regime="cold-data",
            eviction_buffer=eviction,
            warmup=args.warmup,
            warmup_ms=None,
            iterations=1,
            batches=1,
            timed_batch_ms=None,
        )

        samples = {"full": [], "external": []}
        prep_samples = {"full": [], "external": []}
        for batch in range(args.batches):
            pair = [
                ("external", external_function, external_rotation),
                ("full", full_function, full_rotation),
            ]
            if batch % 2:
                pair.reverse()
            for name, function, rotation in pair:
                latency, _raw, prep, _, _ = runner._measure(
                    function,
                    rotation,
                    regime="cold-data",
                    eviction_buffer=eviction,
                    warmup=0,
                    warmup_ms=None,
                    iterations=args.cold_iterations,
                    batches=1,
                    timed_batch_ms=None,
                )
                samples[name].append(latency)
                prep_samples[name].append(prep)
            print(
                json.dumps(
                    {
                        "progress": "batch_complete",
                        "batch": batch + 1,
                        "batches": args.batches,
                        "latest_full_us": samples["full"][-1],
                        "latest_external_us": samples["external"][-1],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

        full_us = float(statistics.median(samples["full"]))
        external_us = float(statistics.median(samples["external"]))
        result = {
            "protocol": REPORT_PROTOCOL,
            "kernel": args.kernel,
            "target": {
                "backend": "riscv",
                "arch": "rv64gcv",
                "thread_count": 1,
                "cpu": args.cpu,
            },
            "workload": {
                "profile": entry.get("workload_profile"),
                "extents": entry["extents"],
                "inputs": entry.get("input_specs"),
                "outputs": entry["output_specs"],
            },
            "measurement": {
                "regime": "code-warm/data-cold",
                "comparison_protocol": "paired-alternating.v1",
                "warmup": args.warmup,
                "cold_iterations_per_batch": args.cold_iterations,
                "batches": args.batches,
                "eviction_bytes": eviction_bytes,
                "cache_eviction_in_timed_region": False,
            },
            "full": {
                "candidate_id": winner_id,
                "latency_us": full_us,
                "samples_us": samples["full"],
                "correctness": full_correctness,
            },
            "external": {
                **compile_record,
                "latency_us": external_us,
                "samples_us": samples["external"],
                "correctness": external_correctness,
            },
            "external_over_full": external_us / full_us,
            "elapsed_seconds": (time.time_ns() - started) / 1_000_000_000.0,
        }
        # Keep input/output storage and both dynamic libraries alive through the
        # final sample and report construction.
        _ = (keepalive, full_library, external_library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
