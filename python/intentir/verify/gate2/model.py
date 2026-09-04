"""Result model for source-anchored Gate2 dynamic execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

import numpy as np


def artifact_sha256(value: bytes | bytearray | memoryview) -> str:
    """Digest exact compiler-produced artifact bytes.

    This helper intentionally accepts bytes-like values only.  A source string,
    cache key, record id, or JSON description is not a compiled artifact and
    must never be silently promoted to one by a Gate2 consumer.
    """

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("compiled artifact digest requires bytes-like input")
    return hashlib.sha256(bytes(value)).hexdigest()


def _output_records(
    outputs: Mapping[str, Any],
) -> tuple[tuple[str, np.ndarray], ...]:
    if not isinstance(outputs, Mapping):
        raise TypeError("Gate2 outputs must be a mapping")
    result: list[tuple[str, np.ndarray]] = []
    for name in sorted(outputs):
        if not isinstance(name, str) or not name:
            raise ValueError("Gate2 output names must be non-empty strings")
        value = np.asarray(outputs[name])
        if value.dtype.hasobject:
            raise ValueError(f"Gate2 output {name!r} has object dtype")
        result.append((name, np.ascontiguousarray(value)))
    return tuple(result)


def output_binding_digest(outputs: Mapping[str, Any]) -> str:
    """Digest output names, dtypes, and shapes without their payload bytes."""

    records = [
        {
            "name": name,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        for name, value in _output_records(outputs)
    ]
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"intentir-gate2-output-bindings-v1\0" + payload).hexdigest()


def output_digest(outputs: Mapping[str, Any]) -> str:
    """Digest exact contiguous output bytes with name/dtype/shape framing."""

    digest = hashlib.sha256(b"intentir-gate2-outputs-v1\0")
    for name, value in _output_records(outputs):
        header = json.dumps(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        payload = value.tobytes(order="C")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def mapping_digest(values: Mapping[str, Any], *, purpose: str) -> str:
    """Digest a small JSON-safe name/value map with an explicit purpose."""

    if not isinstance(purpose, str) or not purpose:
        raise ValueError("mapping digest purpose must be non-empty")
    if not isinstance(values, Mapping) or any(
        not isinstance(name, str) or not name for name in values
    ):
        raise ValueError("mapping digest values require non-empty string keys")
    payload = json.dumps(
        dict(sorted(values.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"intentir-gate2-mapping-v1\0"
        + purpose.encode("utf-8")
        + b"\0"
        + payload
    ).hexdigest()


def json_sha256(value: Any) -> str:
    """Digest one canonical JSON value without silently dropping fields."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _binding_json(value: Any) -> Any:
    """Convert execution-contract values to deterministic JSON data."""

    if isinstance(value, Enum):
        return _binding_json(value.value)
    if is_dataclass(value):
        return {
            name: _binding_json(item)
            for name, item in asdict(value).items()
        }
    if isinstance(value, Mapping):
        return {
            str(name): _binding_json(item)
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_binding_json(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("execution binding does not allow non-finite floats")
        return value
    raise TypeError(
        "execution binding value is not JSON-safe: "
        f"{type(value).__name__}"
    )


def program_identity_digests(
    program: Any,
    *,
    frontend: str,
) -> tuple[str, str]:
    """Return raw-source and actually-executed-program digests.

    Triton applies its compatibility rewrite before importing a source module;
    CUDA and TileLang consume the source text as supplied.  Keeping both
    digests prevents an anonymized/normalized execution program from being
    confused with the selected source projection in the manifest.
    """

    reader = getattr(program, "read_code", None)
    if not callable(reader):
        raise TypeError("execution program has no read_code interface")
    value = reader()
    if isinstance(value, tuple):
        if not value or not isinstance(value[0], str):
            raise TypeError("execution program read_code returned invalid code")
        code = value[0]
    else:
        code = value
    if not isinstance(code, str) or not code:
        raise ValueError("execution program source must be non-empty")
    executed = code
    if frontend == "triton":
        from intentir.verify.triton_compat import normalize_deprecated_triton_dot

        executed, _compat = normalize_deprecated_triton_dot(code)
    elif frontend not in {"cuda", "tilelang"}:
        raise ValueError(f"unsupported execution frontend {frontend!r}")
    return (
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
        hashlib.sha256(executed.encode("utf-8")).hexdigest(),
    )


def execution_spec_contract(
    spec: Any,
    *,
    leg: str,
) -> dict[str, Any]:
    """Project a frontend execution spec onto its closed binding contract."""

    if leg not in {"source", "replay"}:
        raise ValueError("execution binding leg must be source or replay")
    arguments_for = getattr(spec, "arguments_for", None)
    arguments = tuple(
        arguments_for(leg) if callable(arguments_for) else getattr(spec, "arguments", ())
    )
    if not arguments:
        raise ValueError("execution binding requires ordered arguments")
    argument_order = tuple(
        str(getattr(argument, "name", f"arg{index}"))
        for index, argument in enumerate(arguments)
    )
    if len(argument_order) != len(set(argument_order)):
        raise ValueError("execution ABI argument order must be unique")
    argument_payload = [_binding_json(argument) for argument in arguments]
    abi_digest = json_sha256(
        {"argument_order": list(argument_order), "arguments": argument_payload}
    )

    if hasattr(spec, "schedule"):
        schedule = dict(getattr(spec, "schedule") or {})
    else:
        compile_case = getattr(spec, "compile_case", None)
        schedule = dict(getattr(compile_case, "schedule", {}) or {})
    schedule_names = tuple(str(name) for name in schedule)
    if len(schedule_names) != len(set(schedule_names)):
        raise ValueError("execution schedule names must be unique")
    schedule_digest = mapping_digest(
        schedule,
        purpose="gate2-execution-schedule",
    )

    compile_case = getattr(spec, "compile_case", None)
    if compile_case is None:
        compiler_identity = "triton"
        compiler_payload: Any = {"frontend": "triton"}
    else:
        compiler_type = type(compile_case)
        compiler_identity = (
            f"{compiler_type.__module__}.{compiler_type.__qualname__}"
        )
        compiler_payload = _binding_json(compile_case)
    compiler_digest = json_sha256(
        {"identity": compiler_identity, "case": compiler_payload}
    )

    launch_for = getattr(spec, "launch_for", None)
    if callable(launch_for):
        grid, block, shared_mem = launch_for(leg)
    else:
        grid = getattr(spec, "grid", ())
        block = ()
        shared_mem = 0
    grid = tuple(int(item) for item in grid)
    block = tuple(int(item) for item in block)
    shared_mem = int(shared_mem)

    output_keys: list[str] = []
    for index, argument in enumerate(arguments):
        role = getattr(getattr(argument, "role", None), "value", getattr(argument, "role", None))
        if role not in {"output", "inout"}:
            continue
        name = getattr(argument, "output_key", None)
        if not isinstance(name, str) or not name:
            name = str(getattr(argument, "name", f"output_{index}"))
        output_keys.append(name)
    return {
        "abi_order": argument_order,
        "abi_digest": abi_digest,
        "schedule": schedule,
        "schedule_names": schedule_names,
        "schedule_digest": schedule_digest,
        "compiler_identity": compiler_identity,
        "compiler_digest": compiler_digest,
        "grid": grid,
        "block": block,
        "shared_mem": shared_mem,
        "output_keys": tuple(output_keys),
    }


@dataclass(frozen=True)
class ExecutionBinding:
    """Strong Source/Replay identity consumed by a real Gate2 runner.

    Manifest identity and execution-program identity are deliberately separate:
    Triton Source executes an anonymized program while native Dataset Source
    executes the selected source file.  The runner must verify every field
    against the object and spec it is about to compile, then returns this exact
    binding with the real artifact and output evidence.
    """

    leg: str
    manifest_digest: str
    source_sha256: str
    source_slice_sha256: str
    source_location: str
    selected_device_entry: str
    program_source_sha256: str
    program_execution_sha256: str
    program_entry: str
    abi_order: tuple[str, ...]
    abi_digest: str
    schedule_names: tuple[str, ...]
    schedule_digest: str
    compiler_identity: str
    compiler_digest: str
    grid: tuple[int, ...]
    block: tuple[int, ...]
    shared_mem: int
    output_keys: tuple[str, ...]
    manifest_output_binding_digest: str

    def __post_init__(self) -> None:
        if self.leg not in {"source", "replay"}:
            raise ValueError("execution binding leg must be source or replay")
        for name, value in (
            ("manifest_digest", self.manifest_digest),
            ("source_sha256", self.source_sha256),
            ("source_slice_sha256", self.source_slice_sha256),
            ("program_source_sha256", self.program_source_sha256),
            ("program_execution_sha256", self.program_execution_sha256),
            ("abi_digest", self.abi_digest),
            ("schedule_digest", self.schedule_digest),
            ("compiler_digest", self.compiler_digest),
            ("manifest_output_binding_digest", self.manifest_output_binding_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"execution binding {name} must be a lowercase SHA-256")
        for name, value in (
            ("source_location", self.source_location),
            ("selected_device_entry", self.selected_device_entry),
            ("program_entry", self.program_entry),
            ("compiler_identity", self.compiler_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"execution binding {name} must be non-empty")
        if not self.program_entry.isidentifier() or "__" in self.program_entry:
            raise ValueError("execution binding program_entry is not an identifier")
        if not self.selected_device_entry.isidentifier():
            raise ValueError("execution binding selected_device_entry is not an identifier")
        abi_order = tuple(self.abi_order)
        schedule_names = tuple(self.schedule_names)
        output_keys = tuple(self.output_keys)
        if not abi_order or len(abi_order) != len(set(abi_order)):
            raise ValueError("execution binding ABI order must be non-empty and unique")
        if len(schedule_names) != len(set(schedule_names)):
            raise ValueError("execution binding schedule names must be unique")
        if not output_keys or len(output_keys) != len(set(output_keys)):
            raise ValueError("execution binding output keys must be non-empty and unique")
        if any(not isinstance(name, str) or not name for name in (*abi_order, *schedule_names, *output_keys)):
            raise ValueError("execution binding names must be non-empty strings")
        for name, dimensions in (("grid", self.grid), ("block", self.block)):
            dims = tuple(int(item) for item in dimensions)
            if any(item <= 0 for item in dims):
                raise ValueError(f"execution binding {name} dimensions must be positive")
            object.__setattr__(self, name, dims)
        if not isinstance(self.shared_mem, int) or self.shared_mem < 0:
            raise ValueError("execution binding shared memory must be non-negative")
        object.__setattr__(self, "abi_order", abi_order)
        object.__setattr__(self, "schedule_names", schedule_names)
        object.__setattr__(self, "output_keys", output_keys)

    def to_json(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "manifest_digest": self.manifest_digest,
            "source_sha256": self.source_sha256,
            "source_slice_sha256": self.source_slice_sha256,
            "source_location": self.source_location,
            "selected_device_entry": self.selected_device_entry,
            "program_source_sha256": self.program_source_sha256,
            "program_execution_sha256": self.program_execution_sha256,
            "program_entry": self.program_entry,
            "abi_order": list(self.abi_order),
            "abi_digest": self.abi_digest,
            "schedule_names": list(self.schedule_names),
            "schedule_digest": self.schedule_digest,
            "compiler_identity": self.compiler_identity,
            "compiler_digest": self.compiler_digest,
            "grid": list(self.grid),
            "block": list(self.block),
            "shared_mem": self.shared_mem,
            "output_keys": list(self.output_keys),
            "manifest_output_binding_digest": self.manifest_output_binding_digest,
        }

    @property
    def digest(self) -> str:
        return json_sha256(self.to_json())

    def validate_runtime(
        self,
        *,
        leg: str,
        program_source_sha256: str,
        program_execution_sha256: str,
        program_entry: str,
        contract: Mapping[str, Any],
    ) -> None:
        """Fail closed unless the runner's real pre-compile view matches."""

        if leg != self.leg:
            raise ValueError("execution binding leg differs from the runner leg")
        checks = (
            ("program source", program_source_sha256, self.program_source_sha256),
            ("program execution", program_execution_sha256, self.program_execution_sha256),
            ("program entry", program_entry, self.program_entry),
            ("ABI order", tuple(contract["abi_order"]), self.abi_order),
            ("ABI digest", contract["abi_digest"], self.abi_digest),
            ("schedule names", tuple(contract["schedule_names"]), self.schedule_names),
            ("schedule digest", contract["schedule_digest"], self.schedule_digest),
            ("compiler identity", contract["compiler_identity"], self.compiler_identity),
            ("compiler digest", contract["compiler_digest"], self.compiler_digest),
            ("grid", tuple(contract["grid"]), self.grid),
            ("block", tuple(contract["block"]), self.block),
            ("shared memory", int(contract["shared_mem"]), self.shared_mem),
            ("output keys", tuple(contract["output_keys"]), self.output_keys),
        )
        for name, actual, expected in checks:
            if actual != expected:
                raise ValueError(
                    f"execution binding {name} differs: actual={actual!r} expected={expected!r}"
                )


class Gate2Status(str, Enum):
    PASS = "gate2_pass"
    PAIRWISE_DIVERGED = "pairwise_diverged"
    COMPARISON_UNSUPPORTED = "comparison_unsupported"
    REFERENCE_UNSUPPORTED = "reference_unsupported"
    EXECUTION_FAILED = "execution_failed"


@dataclass
class ExecutionResult:
    leg: str
    ok: bool
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    error: str = ""
    unsupported: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "ok": self.ok,
            "unsupported": self.unsupported,
            "error": self.error,
            "outputs": {
                name: {
                    "dtype": str(np.asarray(value).dtype),
                    "shape": list(np.asarray(value).shape),
                }
                for name, value in self.outputs.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OutputAgreement:
    output: str
    passed: bool
    max_abs_error: float | None
    detail: str = ""
    relation_kind: str = "exact"
    equality_required: bool = True
    policy_id: str = ""
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "detail": self.detail,
            "relation_kind": self.relation_kind,
            "equality_required": self.equality_required,
            "policy_id": self.policy_id,
            "rtol": self.rtol,
            "atol": self.atol,
            "equal_nan": self.equal_nan,
        }


@dataclass(frozen=True)
class PairwiseAgreement:
    left: str
    right: str
    passed: bool
    outputs: tuple[OutputAgreement, ...]
    policy_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "passed": self.passed,
            "policy_id": self.policy_id,
            "outputs": [item.to_json() for item in self.outputs],
        }


@dataclass
class Gate2Result:
    case_id: str
    status: Gate2Status
    source: ExecutionResult
    replay: ExecutionResult
    reference: ExecutionResult
    source_replay: PairwiseAgreement | None = None
    source_j: PairwiseAgreement | None = None
    replay_j: PairwiseAgreement | None = None
    input_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is Gate2Status.PASS

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "source": self.source.to_json(),
            "replay": self.replay.to_json(),
            "reference": self.reference.to_json(),
            "source_replay": (
                self.source_replay.to_json()
                if self.source_replay is not None else None
            ),
            "source_j": (
                self.source_j.to_json() if self.source_j is not None else None
            ),
            "replay_j": (
                self.replay_j.to_json() if self.replay_j is not None else None
            ),
            "input_digest": self.input_digest,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Gate2SuiteResult:
    suite_id: str
    cases: tuple[Gate2Result, ...]
    c_values: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("Gate2 suite_id must not be empty")
        if not self.cases:
            raise ValueError("Gate2 suite requires at least one case result")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Gate2 suite result contains duplicate case_id values")
        names = [name for name, _ in self.c_values]
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(
                "Gate2 suite result c0 bindings must have unique non-empty slot names"
            )

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases if not case.passed)

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(case.status.value for case in self.cases).items())
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "passed": self.passed,
            "failed_case_ids": list(self.failed_case_ids),
            "status_counts": self.status_counts,
            "c0": dict(self.c_values),
            "cases": [case.to_json() for case in self.cases],
        }


__all__ = [
    "artifact_sha256",
    "ExecutionResult",
    "ExecutionBinding",
    "Gate2Result",
    "Gate2Status",
    "Gate2SuiteResult",
    "execution_spec_contract",
    "json_sha256",
    "mapping_digest",
    "OutputAgreement",
    "PairwiseAgreement",
    "program_identity_digests",
    "output_binding_digest",
    "output_digest",
]
