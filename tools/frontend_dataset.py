#!/usr/bin/env python3
"""Inspect and validate the external CUDA/TileLang source Dataset catalog."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.verify.frontend_dataset import (
    load_frontend_dataset,
    resolve_cuda_include_dirs,
)
from intentir.verify.gate1.cuda_compiler import (
    CudaCompileCase,
    CudaCompiler,
    CudaKernelSource,
    FrontendParameter,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompiler,
    TileLangKernelSource,
)


def _probe_tilelang(catalog, *, workers: int, timeout: float) -> dict:
    rows = catalog.select(frontend="tilelang")
    # Factory compilation is shape/schedule self-contained.  This placeholder
    # parameter only satisfies the compile-case type; the worker reports the
    # actual PrimFunc/PTX interface in its result.
    case = TileLangCompileCase((FrontendParameter("pointer", "f32", 1, "input"),))

    def probe(row):
        result = TileLangCompiler(timeout=timeout).compile(
            TileLangKernelSource.from_factory_file(row.path, row.symbol),
            case,
        )
        lines = (result.diagnostic or "").splitlines()
        return {
            **row.to_json(),
            "compile_ok": result.ok,
            "diagnostic": lines[0] if lines else "",
            "compiler": dict(result.compiler),
            "target": dict(result.target),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    passed = sum(bool(item["compile_ok"]) for item in results)
    return {
        "protocol": "intentir.tilelang-dataset-compile-probe.v1",
        "dataset_root": str(catalog.root),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


def _probe_cuda(catalog, *, workers: int, timeout: float, arch: str) -> dict:
    rows = catalog.select(frontend="cuda")

    def probe(row):
        include_dirs = resolve_cuda_include_dirs(row)
        case = CudaCompileCase(
            (FrontendParameter("pointer", "f32", 1, "input"),),
            arch=arch,
            include_dirs=include_dirs,
        )
        result = CudaCompiler(timeout=timeout).compile(
            CudaKernelSource.from_file(row.path, row.symbol),
            case,
        )
        lines = (result.diagnostic or "").splitlines()
        return {
            **row.to_json(),
            "compile_ok": result.ok,
            "diagnostic": lines[0] if lines else "",
            "compiler": dict(result.compiler),
            "target": dict(result.target),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    passed = sum(bool(item["compile_ok"]) for item in results)
    return {
        "protocol": "intentir.cuda-dataset-compile-probe.v1",
        "dataset_root": str(catalog.root),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


_PTX_ENTRY_RE = re.compile(
    r"(?:\.visible\s+)?\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\("
)
_LAUNCH_RE = re.compile(
    r'T\.launch_thread\("(?P<axis>(?:blockIdx|threadIdx)\.[xyz])",\s*(?P<extent>[^)]+)\)'
)
_CUDA_PROTOTYPE_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*'
    r'\((?P<parameters>[^;]*)\)\s*;',
    re.DOTALL,
)


def _cuda_dtype(declaration: str) -> str:
    aliases = (
        ("CUtensorMap", "tensor_map"),
        ("nv_bfloat16", "bf16"),
        ("bfloat16_t", "bf16"),
        ("half_t", "f16"),
        ("double", "f64"),
        ("float", "f32"),
        ("uint64_t", "u64"),
        ("int64_t", "i64"),
        ("uint32_t", "u32"),
        ("int32_t", "i32"),
        ("unsigned int", "u32"),
        ("unsigned char", "u8"),
        ("signed char", "i8"),
        ("unsigned short", "u16"),
        ("short", "i16"),
        ("int", "i32"),
        ("bool", "i1"),
    )
    return next((dtype for spelling, dtype in aliases if spelling in declaration), declaration)


def _cuda_abi(cuda: str, interface: dict) -> dict:
    match = _CUDA_PROTOTYPE_RE.search(cuda)
    if match is None:
        raise ValueError("TileLang CUDA export lacks a kernel prototype")
    semantic = {
        str(item.get("name")): dict(item)
        for item in interface.get("parameters") or ()
        if isinstance(item, dict) and item.get("name")
    }
    parameters = []
    for position, raw in enumerate(match.group("parameters").split(",")):
        declaration = raw.strip()
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration)
        if name_match is None:
            raise ValueError(f"cannot parse CUDA parameter {declaration!r}")
        name = name_match.group(1)
        semantic_name = name.removesuffix("_desc")
        description = semantic.get(semantic_name, {})
        if "CUtensorMap" in declaration:
            kind = "opaque"
            mode = "value"
        elif "*" in declaration:
            kind = "pointer"
            mode = str(description.get("mode") or ("input" if "const" in declaration else "inout"))
        else:
            kind = "scalar"
            mode = "value"
        parameter = {
            "position": position,
            "name": name,
            "semantic_name": semantic_name,
            "kind": kind,
            "dtype": _cuda_dtype(declaration),
            "rank": int(description.get("rank") or 0),
            "mode": mode,
            "declaration": declaration,
        }
        if description.get("shape") is not None:
            parameter["shape"] = list(description["shape"])
        if kind == "opaque" and description.get("mode"):
            parameter["semantic_mode"] = str(description["mode"])
        parameters.append(parameter)
    return {"parameters": parameters}


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _export_cuda_snapshots(catalog, *, workers: int, timeout: float) -> dict:
    rows = catalog.select(frontend="tilelang")
    output = catalog.root / "cuda" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    case = TileLangCompileCase((FrontendParameter("pointer", "f32", 1, "input"),))

    def export(row):
        result = TileLangCompiler(timeout=timeout).compile(
            TileLangKernelSource.from_factory_file(row.path, row.symbol),
            case,
        )
        if not result.ok or result.cuda is None or result.ptx is None or result.tir is None:
            lines = (result.diagnostic or "TileLang export failed").splitlines()
            return {
                "case_id": row.case_id,
                "ok": False,
                "diagnostic": lines[0] if lines else "TileLang export failed",
            }
        entries = _PTX_ENTRY_RE.findall(result.ptx)
        if len(entries) != 1:
            return {
                "case_id": row.case_id,
                "ok": False,
                "diagnostic": f"expected one PTX entry, found {len(entries)}",
            }
        launch = {
            match.group("axis"): match.group("extent").strip()
            for match in _LAUNCH_RE.finditer(result.tir)
        }
        stem = row.case_id.split(":", 1)[1]
        cuda_path = output / f"{stem}.cu"
        ptx_path = output / f"{stem}.ptx"
        meta_path = output / f"{stem}.json"
        metadata = {
            "protocol": "intentir.tilelang-cuda-snapshot.v2",
            "origin_case_id": row.case_id,
            "origin_location": row.location,
            "entry_name": entries[0],
            "include_roots": [
                "tilelang/src",
                "tilelang/3rdparty/cutlass/include",
            ],
            "launch": launch,
            "interface": dict(result.interface),
            "cuda_abi": _cuda_abi(result.cuda, dict(result.interface)),
            "compiler": dict(result.compiler),
            "target": dict(result.target),
            "cuda_sha256": hashlib.sha256(result.cuda.encode()).hexdigest(),
            "ptx_sha256": hashlib.sha256(result.ptx.encode()).hexdigest(),
        }
        _atomic_text(cuda_path, result.cuda)
        _atomic_text(ptx_path, result.ptx)
        _atomic_text(
            meta_path,
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        return {
            "case_id": row.case_id,
            "ok": True,
            "cuda": str(cuda_path),
            "ptx": str(ptx_path),
            "metadata": str(meta_path),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(export, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    passed = sum(bool(item["ok"]) for item in results)
    return {
        "protocol": "intentir.tilelang-cuda-snapshot-export.v2",
        "output_dir": str(output),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--frontend", choices=("cuda", "tilelang"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate", action="store_true")
    action.add_argument("--probe-tilelang", action="store_true")
    action.add_argument("--probe-cuda", action="store_true")
    action.add_argument("--export-cuda-snapshots", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cuda-arch", default="sm_80")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    catalog = load_frontend_dataset(arguments.dataset_root)
    if arguments.export_cuda_snapshots:
        payload = _export_cuda_snapshots(
            catalog,
            workers=arguments.workers,
            timeout=arguments.timeout,
        )
    elif arguments.probe_cuda:
        payload = _probe_cuda(
            catalog,
            workers=arguments.workers,
            timeout=arguments.timeout,
            arch=arguments.cuda_arch,
        )
    elif arguments.probe_tilelang:
        payload = _probe_tilelang(
            catalog,
            workers=arguments.workers,
            timeout=arguments.timeout,
        )
    elif arguments.validate:
        payload = catalog.validate().to_json()
    else:
        selected = catalog.select(frontend=arguments.frontend)
        payload = {
            "protocol": "intentir.multifrontend-dataset.v1",
            "root": str(catalog.root),
            "kernels": [item.to_json() for item in selected],
        }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if (
        arguments.probe_tilelang
        or arguments.probe_cuda
        or arguments.export_cuda_snapshots
    ):
        return 0 if payload["failed"] == 0 else 2
    return 0 if not arguments.validate or payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
