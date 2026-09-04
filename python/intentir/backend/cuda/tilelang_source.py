"""Prepared execution of a bundled TileLang Source baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


class PreparedTileLangInvocation:
    """Compile one bundled factory and bind its tensor ABI once.

    The object deliberately mirrors the small part of ``PreparedTritonInvocation``
    used by the production benchmark: launch, thunk, copied outputs, and a
    no-allocation reset hook.  Compilation and allocation stay outside timing.
    """

    def __init__(
        self,
        record: Mapping[str, Any],
        entry: Mapping[str, Any],
        inputs: Sequence[Any],
    ) -> None:
        import tilelang
        import torch

        code = record.get("code")
        factory_name = record.get("factory")
        kwargs = record.get("factory_kwargs")
        logical_refs = record.get("argument_logical_refs")
        if (
            not isinstance(code, str)
            or not isinstance(factory_name, str)
            or not factory_name.isidentifier()
            or not isinstance(kwargs, Mapping)
            or not isinstance(logical_refs, list)
            or not all(isinstance(item, str) for item in logical_refs)
        ):
            raise ValueError("bundled TileLang Source record is malformed")

        self._temporary = tempfile.TemporaryDirectory(
            prefix="intentir-tilelang-source-"
        )
        source_path = Path(self._temporary.name) / "source.py"
        source_path.write_text(code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "_intentir_bundled_tilelang_source", source_path
        )
        if spec is None or spec.loader is None:
            self.close()
            raise RuntimeError("cannot import bundled TileLang Source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, factory_name)
        prim_func = factory(**dict(kwargs))
        self._kernel = tilelang.compile(
            prim_func,
            target="cuda",
            execution_backend="tvm_ffi",
        )

        input_order = entry.get("input_order")
        output_specs = entry.get("output_specs")
        output_keys = entry.get("output_keys")
        if (
            not isinstance(input_order, list)
            or len(input_order) != len(inputs)
            or not isinstance(output_specs, list)
            or not isinstance(output_keys, list)
            or len(output_specs) != len(output_keys)
        ):
            self.close()
            raise ValueError("TileLang Source entry has an invalid tensor ABI")
        tensors: dict[str, Any] = dict(zip(input_order, inputs, strict=True))
        self._outputs: dict[str, Any] = {}
        dtype_map = {
            "f16": torch.float16,
            "bf16": torch.bfloat16,
            "f32": torch.float32,
            "f64": torch.float64,
            "i1": torch.bool,
            "i32": torch.int32,
            "i64": torch.int64,
            "index": torch.int64,
            "u8": torch.uint8,
            "u16": torch.uint16,
            "u32": torch.uint32,
            "ui8": torch.uint8,
            "ui16": torch.uint16,
            "ui32": torch.uint32,
        }
        for output_key, output_spec in zip(
            output_keys, output_specs, strict=True
        ):
            logical_ref = str(output_spec["name"])
            dtype = dtype_map[str(output_spec["dtype"])]
            tensor = torch.empty(
                tuple(int(item) for item in output_spec["shape"]),
                dtype=dtype,
                device="cuda",
            )
            tensors[logical_ref] = tensor
            self._outputs[str(output_key)] = tensor
        try:
            self._arguments = tuple(tensors[item] for item in logical_refs)
        except KeyError as exc:
            self.close()
            raise ValueError(
                f"TileLang Source argument {exc.args[0]!r} is not a J boundary tensor"
            ) from exc

    def __enter__(self) -> "PreparedTileLangInvocation":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def launch(self) -> None:
        self._kernel(*self._arguments)

    def thunk(self):
        return self.launch

    def reset_mutable_inputs(self) -> None:
        # Both Stage-3 Source kernels have read-only inputs and fully overwrite
        # their output.  Target input reset remains owned by the outer harness.
        return None

    def copied_outputs(self) -> Mapping[str, Any]:
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self._outputs.items()
        }


__all__ = ["PreparedTileLangInvocation"]
