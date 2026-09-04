"""Fresh source-blind Replay episodes driven by the isolated Codex CLI runtime."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from intentir.extraction.providers.codex_cli import (
    CODEX_CLI_AGENT_RUNTIME,
    CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
    CodexCLICancelled,
    CodexCLIDiagnosticReady,
    CodexCLIError,
    CodexCLISubmissionAccepted,
    CodexCLISession,
    CodexCLIToolBridge,
    CodexCLITurn,
)
from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.provider import ProviderConfig, get_provider
from intentir.ir.intent_op_contract import RAGGED_OPS
from intentir.verify.gate1.compiler import KernelSource, TritonCompiler
from intentir.verify.gate1.ttir import fingerprint_ttir, runtime_argument_names

from .codegen import (
    TRITON_REPLAY_ABI_CONTRACT,
    TRITON_REPLAY_API_CONTRACT,
    TRITON_REPLAY_CONTROL_CONTRACT,
    TRITON_REPLAY_STRUCTURE_CONTRACT,
)
from .model import Frontend, Gate1Artifact
from .replay_semantic_core import (
    materialize_replay_semantics,
    query_replay_semantics,
)
from .replay import (
    REPLAY_CHECK_PROTOCOL,
    ReplayAttempt,
    ReplayGeneration,
    ReplayGenerationError,
    _context_receipt,
    _frontend_replay_guidance,
    build_source_blind_replay_compile_case,
    _replay_input_files,
    _trajectory_bytes,
    replay_source_ingress,
    source_blind_replay_compile_check,
)


CODEX_CLI_REPLAY_BACKEND = "openai.codex-cli.intentir-replay.v67"

_INITIAL_PROMPT = """You are the sole writer for one fresh source-blind IntentIR Replay episode.
Work only under the isolated /workspace. input/ is immutable; output/, work/, and
tmp/ are writable. There is no Source, Harness/compiler fact beyond optional complete C,
Gate result, earlier Replay, sibling state, repository, or network.

J is the only semantic authority. Read input/compiler-index.intent and the complete
read-only input/semantic-program.intent before drafting, then read input/frontend.json
and optional input/c.json when it exists. If frontend.json contains a
representation_contract, read its complete address_bindings and program_bindings before
choosing pointer or launch realization. S_J is a deterministic parameterized compiler
IR derived only from this J, not a target-source skeleton. Complete C values are exact
implementation bindings; keep every constexpr c* as a callable placeholder. Use `semantic_query` or
`packet_query` only to cross-reference one named J/C handle after reading those complete
small views. Pi, Source realization, Gate observations, and prior Replay are absent.
""" + TRITON_REPLAY_API_CONTRACT + TRITON_REPLAY_ABI_CONTRACT + TRITON_REPLAY_CONTROL_CONTRACT + TRITON_REPLAY_STRUCTURE_CONTRACT + """
For Triton Python, annotate only constexpr parameters with `tl.constexpr`; leave runtime
pointer and scalar parameters unannotated.
When a runtime scalar has a J logical binding and is used by an authored access index,
predicate, domain, or loop bound, include that scalar (and only that scalar) in the
Triton `do_not_specialize` list.  This preserves the J-owned runtime dependency in
your own compiled IR instead of folding the current Harness value into the target.
Never list an ABI-only scalar that has no J logical binding. Materialize a nonempty
list directly on `@triton.jit(do_not_specialize=[...])` before the first compile; do
not call `.to(...)` on a scalar that the frontend has already specialized to a Python
value. Use an explicit target cast only after preserving that parameter as runtime.
A J scalar used only as a pointwise arithmetic value is not an
access/domain/loop carrier: leave it specializable and use it directly in the
tensor expression (for example `rhs * p2`). Never call `p2.to(...)` in that
case; ordinary tensor arithmetic materializes the specialized scalar at its
typed consumer.

Act as the complete Replay compiler. Write one full target program to
output/replay.source yourself. You own instruction selection, addressing, program
mapping, control, loops/carriers, layout, memory, synchronization, target APIs and
whole-program restructuring while preserving all J-owned value semantics. Do not edit
S_J or infer a Source recipe. input/REPLAY_CONTRACT.md defines
the ownership boundary and input/TOOL.md defines the tools.

`compiler-index.intent` is only a compact cross-reference of facts already present in
S_J and optional complete C. Its dependency subsets are syntactic and neither select nor authorize a
program axis, loop, lane, guard, or target API. Use it to avoid repeated JSON joins,
then compile `boundary_ownership` as one whole-program contract. Preserve every
authored access/effect as a distinct logical occurrence and keep logical domain guards
distinct from access-local predicates; you still own every target realization choice.
For a masked load, `fill_value` is the only authority for a target `other=` operand:
materialize it exactly when present and omit `other=` when it is absent.  A
`consumer_extension` proves how an already-authored outside value is consumed; it does
not synthesize a missing fill value.
Ordinary algebraic instruction selection may fold a J `add` whose one operand is an
exact all-zero constant, including when that identity node exists only to express a
logical broadcasted output shape.  Preserve the other operand, output mapping, access
shape, predicate and effect, but do not emit target add arithmetic for the identity.
The declaration order of `boundary_ownership.access_effects` is J-owned observable
effect order: emit the corresponding loads, stores, atomics, and updates in that order.
Do not postpone an earlier authored statistics store until after later parameter loads or
another output store merely to group computations or outputs.
One authored access with a multi-axis index is one target transfer whose tensor result may
then be split; never replace it with one load/store per component. Do not implement an
ordinary finite J domain by a Python/Triton serial loop: specialization may unroll the loop
and silently change the authored access/effect multiplicity. The closed exception is a
contraction axis attached to an anonymous C `reduction_tile_extent`: that combination
authorizes a physical in-kernel contraction loop, while the J contraction remains the sole
semantic authority for the accumulator and result. A standalone J `reduce` still uses one
local tile reduction unless J itself authors a loop.
Lower the closed reducer kind with the corresponding public Triton reducer:
`sum -> tl.sum`, `prod -> tl.prod`, `max -> tl.max`, `min -> tl.min`,
`any -> tl.any`, and `all -> tl.all`. In particular, do not encode boolean
`any`/`all` as integer `max`/`min`; that changes the J-owned typed combiner and
adds conversion operations to the compiler-visible graph.
The other closed exception is an exact J `scatter` whose matching logical update
has `combiner=add`, `duplicate_policy=combine`, and `order=unordered`, targets a
singleton inout tensor, and whose contribution domain carries one anonymous C
`reduction_tile_extent`. Realize that source-neutral update with one program tile:
form `program_id(0) * tile + arange(0, tile)`, retain the authored access predicate,
locally sum the active contributions, and commit them with `tl.atomic_add`. The
atomic itself is the inout read-write effect; do not precede it with a load and
replace it by an ordinary store. This rule does not apply to overwrite/one-of
updates, non-singleton targets, or an unattached reduction domain.
For either global additive-update exception, preserve the J-authored integer
carrier width.  By default keep `program_id`, lane, tile arithmetic, and the
runtime upper bound at the width of the J domain/bound carrier; widen only when
an exact J coordinate or typed carrier requires it.  Do not introduce a wider
address expression merely as a defensive implementation preference.
The same source-neutral lowering applies when J explicitly defines a rank-0
inout result as `base + reduce(sum, contributions)` and the contribution domain
has one anonymous C `reduction_tile_extent`: use one program tile, retain the
J access predicates, locally reduce the active contributions, and realize the
single additive update with `tl.atomic_add`.  The atomic is the authored inout
read-write effect; do not separately load the base or emit an ordinary store.
This exception requires that exact closed J dependency shape and additive
combiner; it does not authorize atomic lowering for an output-only scalar,
another reducer, or an unattached domain.
For Ragged operations, preserve the J descriptor as the pair of rank-1 carrier
tensors `offsets` and `indices`: `ragged_outer` returns the outer-group count,
and `ragged_member` returns `indices[offsets[i]:offsets[i+1]]` for its scalar
selector.  The descriptor is a semantic intermediate, not a target ABI object;
do not pass a tuple descriptor across the target boundary.  A member result has
a runtime-dependent rank-1 length, so do not treat it as a scalar or assign it a
static length that is not authored by J.
Factor the exact J effect axes before choosing a realization: distinguish axes shared across
affected tensors, tensor-specific alternatives, and axes with anonymous C tile attachments.
For an authored finite J domain with a C `tile_extent` attachment, you may choose the standard
source-blind target realization `program_id * tile + arange(0, tile)` to cover that domain,
provided you retain the J upper-bound predicate at every affected access. This is target
instruction selection from J plus optional C, not evidence about the Source grid or a pi
instruction. You may instead product-linearize non-tiled shared axes, share one guarded
program coordinate across mutually exclusive tensor-specific axes, or choose another
whole-program mapping when it preserves the same J-owned effects. Never derive a guard
elision, a specialization branch, an exact grid cardinality, or an additional logical axis
from a C tile.
When independently authored parallel/output `d*` domains each carry their own anonymous
`tile_extent` attachment and there are at most three of them, retain that J partition:
assign one target `program_id` axis to each such domain in declaration order. A domain
attached through `reduction_tile_extent`, or used solely as a reducer/contraction axis,
is not a launch domain and MUST NOT receive an independent `program_id`, except for
the closed singleton additive-scatter update above. Do not merge
distinct parallel J domains into a product coordinate. Within one multi-axis parallel J
domain, you may instead use one linear `program_id(0)`, derive the per-axis tile counts,
and decode it with div/rem. These choices depend only on J domain ownership and optional C,
never on Source, grid, or pi.
For an untiled parallel/output domain whose affected J stores are scalar in that domain
coordinate, use one scalar `program_id` directly. Do not manufacture `arange(0, 1)`, a
singleton tensor coordinate, or singleton expand/broadcast operations around it. Introduce
lane tensors only for axes carrying an actual anonymous tile attachment or a J-authored
vector value.
When a rank-2-or-higher dense full-rank tensor has a closed anonymous
`block_order` attachment for every tensor axis, you MUST realize every access
to that tensor with `tl.make_block_ptr`, not expanded flat-pointer arithmetic:
use the
J-authored shape/dimension scalars and ABI-owned stride/address carriers, the decoded tile offsets, the attached
anonymous order slots, and `boundary_check` on the same axes. This is permitted
only when those access/axis attachments make the grouping unambiguous.
For this fully attached multi-axis case, compute every `num_tiles[d]` with
`tl.cdiv(extent[d], tile[d])`, decode one `tl.program_id(0)` from the trailing
axis toward the leading axis using `%` followed by `//` for each trailing
axis, and use the final residual quotient directly for the leading axis (do
not add a redundant `% num_tiles[0]`). When J authors a wider coordinate
carrier, widen both `program_id` and every `num_tiles[d]` before div/rem;
do not postpone that widening until the final offsets. Cast the resulting
per-axis tile offsets to `tl.int32`, pass the attached order tuple only to
`tl.make_block_ptr(order=...)`, and pass the logical tensor-axis tuple
`(0, 1, ...)` to `boundary_check`. Triton's boundary-check indices are logical
dimensions, not memory-order positions. Do not rebuild an
elementwise boolean mask beside these block pointers.
When a block-pointer order tuple is selected by comparing anonymous C stride/order
slots, express that compile-time choice directly as one `tl.constexpr` conditional
expression. Do not spell the same choice as Python `if`/`else` statement blocks,
which introduce compiler-visible control that J does not own.
For a J `contraction` domain attached by `reduction_tile_extent`, use the anonymous tile as
the local lane range and cover the contraction extent with an in-kernel target loop. Carry
the J-authored accumulator across tiles, form each global contraction coordinate from the
loop base plus the local range, retain the authored access predicate, then perform the final
local reduction and store. Even when the current anonymous tile equals the current extent,
retain this physical loop form rather than turning the contraction axis into a launch axis.
This is source-blind target lowering from J plus optional C, not a Source loop claim or a pi
instruction. Do not apply this contraction-loop rule to an ordinary standalone J `reduce`.
For the closed standalone diagonal-gather reduction case, however, a physical tiled
accumulator loop is the canonical target realization: J has one scalar reduction whose
operand depends on a gather/access with the same logical coordinate repeated across two
tensor axes, and that contribution domain has one anonymous C
`reduction_tile_extent`.  Allocate one vector accumulator of the anonymous tile size,
iterate over `tl.cdiv(extent, tile)` tile ordinals and form each loop base as
`ordinal * tile`, retain the exact J predicate/fill,
accumulate each gathered tile into the vector carrier, then perform the final local
reduction and scalar store.  Derive the diagonal address from J's repeated-coordinate
access; do not infer Source strides or read pi.
For the closed rank-2 column-reduction case with one C-tiled parallel/output axis and one
C-tiled axis-0 standalone reduction, use physical tile order
`[parallel_tile, reduction_tile]`: form the parallel coordinate as
`[program_id(0) * parallel_tile + arange(0, parallel_tile), 1]`, form the reduction
coordinate as `[1, loop_base + arange(0, reduction_tile)]`, carry a
`[parallel_tile, reduction_tile]` accumulator through the reduction loop, then reduce its
last axis. Compute each parallel and reduction upper-bound comparison once and broadcast
those same booleans into the joint access mask. If J authors a later second access to the
same tensor after intervening effects, emit a second physical loop in that authored effect
position; do not reuse the earlier transfer. This is a generic lowering of the exact J
axis/effect partition, not a Source recipe.
For two or more sibling standalone reductions over distinct inputs that share the
same logical reduction domain and the same anonymous C `reduction_tile_extent`,
realize them in one shared physical tile loop. Initialize one vector accumulator
per reduction with its authored identity, load each authored input once per tile
with its exact predicate/fill, update the corresponding accumulator, and perform
the authored local reductions only after the loop before evaluating the common
scalar epilogue. Retain the J-authored read and write order. This is a generic
multi-reduction lowering from J plus optional C; it does not expose Source or pi.
For the common rank-2-by-rank-1 contraction `ij,j->i`, keep the output coordinate as
`[parallel_tile, 1]` and the contracted coordinate as `[1, reduction_tile]` throughout
addressing and predicates. Compute each upper-bound comparison once at that expanded rank
and share it across every J access that authors the same predicate. Carry a
`[parallel_tile, reduction_tile]` partial-product accumulator through the physical loop,
add the loaded elementwise products to it, and perform the single axis-1 reduction only
after that loop; keep the singleton result dimension through the output store. This is the
canonical public Triton realization of that J subscript shape, not a Source-specific
template. For rank-2-by-rank-2 matrix multiplication, use the normal per-K-tile `tl.dot`
accumulation instead; do not apply the rank-1-output partial-product form.
For a J `consumer_extension` whose outside value is the reduction's two-sided
identity, use that identity only to preserve the reduction semantics.  Put it in
the predicated load's `other=` argument only when the matching J access also has
that explicit `fill_value`; when the access has no fill, omit `other=` and do not
add a post-load `where/select`.
When a J `reduce` declares a collapsed result shape, keep the target reduction
collapsed (`keep_dims=False`, normally the API default).  Reintroduce singleton
axes only at the exact broadcasted consumer/store that needs them; do not change
the reduction itself to `keep_dims=True`.
When product-linearizing J axes, decode every authored logical coordinate. A lower bound
that follows statically from public target semantics (`program_id`, nonnegative div/rem,
or `arange`) is already discharged and must not be emitted as a redundant runtime
comparison. Preserve every remaining authored upper bound. For a terminal complement of
their positive conjunction, emit the direct disjunction of the individual upper-bound
violations; do not replace it with either `not(combined_positive)` or one product-extent
comparison. The J domain owns those predicate edges even when an arithmetic shortcut is
extensionally equivalent for one launch.
Construct each logical coordinate in the integer dtype authored by its J `iota`.
For a boundary-only access axis without a J `iota`, first use the dtype of the
exact typed scalar carrier that J binds to that dimension (either directly in
the signature or through a value-dimension relation). Use i64 only when J has no
such typed carrier. When an
iota/coordinate is wider than the target-native program id or lane range, widen the base
and lane components before forming their sum, not the already-combined coordinate. Never
narrow from an anonymous valuation. For a multi-axis transfer whose J access uses
the Cartesian product of distinct logical domains and shares one conjunctive
predicate across that transfer, first expand each coordinate into the transfer
rank and then compare it with its own authored bound before combining the
predicates. Reuse those same expanded coordinates for address formation. This
canonical target form follows only from J access rank and predicate sharing; do
not compare rank-one coordinates and only later expand the boolean results. A
genuinely rank-one access remains rank-one.
In the common guarded fill case with complete C (no logical inputs, one guarded
output, one runtime extent, and one C tile attachment), cast `program_id`,
`arange`, and the runtime extent individually only when the J-selected coordinate
dtype requires it. Do not rely on implicit promotion of an already-combined coordinate.
Conversely, do not invent an explicit cast for a scalar operand when J does not
author one. In particular, an i32 stride multiplied by a J-authored i64 iota is
written directly as `coordinate * stride`; let the target's public mixed-integer
promotion realize that consumer rather than pre-casting the stride.
A `tile_extent` slot never authorizes guard elimination, a `c0 == logical_extent`
specialization branch, or an unmasked transfer. Only a J-authored predicate or a
separate anonymous C slot explicitly attached as guard specialization can do that.
An unattached anonymous `specialization_branch` slot is likewise only a physical
realization choice: it never changes a J-authored access predicate. Do not branch
between masked and unmasked transfers from that slot alone.
Do not invent a terminal early return, structured conditional, or additional domain
predicate unless J explicitly authors that control fact. Realize each authored access
predicate directly as that transfer's mask, sharing the same computed predicate across
accesses when J shares it. A nonnegative lower bound already discharged by `program_id`,
nonnegative div/rem, or `arange` must not be rebuilt as an `and` mask. Keep tensor-specific
effect guards as sibling masks and do not change their ownership.
Boundary domains delimit logical iteration but do not automatically become transfer
masks.  A load/store mask contains exactly the clauses of that J access predicate;
do not add an upper-bound comparison merely because another J domain names the
program coordinate.

Compile, inspect only your own Replay IR, and patch the complete program in this same
episode. First run `replay_compile_check`; once construction passes, call
`replay_ttir_inspect jc` for the bounded J/C structural summary. Compare its exact
typed J obligations against your own arithmetic, dot, reduction, loop, and cast
signatures before submit. Every operand and result dtype printed in S_J is authoritative:
preserve intermediate result dtypes at the corresponding operation boundary and insert
an explicit target cast when implicit promotion would skip that boundary. In particular,
an f16 intermediate later consumed by an f32 operation must remain f16 before that
consumer; matching only final store dtype is insufficient. Compare these typed edges;
matching only operation counts is insufficient.
For Triton/CUDA, preserve the public target realization of authored predicate
and transcendental nodes: lower `map_unary(fn=erf)` with
`tl.math.erf(x.to(tl.float32))`, not `libdevice.erf`,
`map_unary(fn=exp)` with `libdevice.exp`,
`map_unary(fn=sigmoid)` as `1 / (1 + libdevice.exp2(-x.to(tl.float32) *
1.4426950408889634))`,
importing `libdevice` from `triton.language.extra.cuda`; call
`libdevice.exp2` literally, not `tl.exp2` or `tl.sigmoid`, because those public
target APIs compile to a different compiler-visible operation graph,
`map_unary(fn=isnan)` with `libdevice.isnan(value.to(tl.float32))`, and an f32
`map_unary(fn=isfinite)` with `libdevice.finitef(value.to(tl.float32))` converted
to the authored i1 result. Do not replace authored `isnan` with `value != value`.
If the callable ABI physically stores a J-authored i1 tensor as bytes, cast each
load to `tl.int1` before its authored logical consumers. Retain an authored
comparison with integer one; neither using the raw byte directly as a predicate
nor comparing it to Python `True` preserves that typed J boundary.
Lower `map_unary(fn=logical_not)` as boolean conversion followed by boolean
negation (for example `~value.to(tl.int1)`); do not replace that authored node
with the algebraically equivalent direct comparison `value == 0`, which erases
the compiler-visible negation. A predicated J load with no authored
`fill_value` must omit Triton's `other=` argument; never invent zero merely to
silence inactive lanes.
When exact J directly spells `add(primary, neg(value))` or
`add(primary, sub(zero,value))` and that negated producer has no other J consumer,
lower the target expression as the direct subtraction `primary - value`; do not retain
an extra target zero/neg/add chain. This is local instruction selection, not a general
algebraic rewrite. Inspect your own IR again when a concrete J obligation remains
unclear; use targeted `find`, `raw`, or
`observer` queries instead of dumping the whole fingerprint. Keep computations for inactive constexpr
alternatives inside their branch and do not add unmentioned `volatile`, cache,
eviction, or other target attributes. In particular, when an anonymous constexpr
selects whether an already-authored logical-domain guard is elided, put the unguarded
and guarded memory operations in their respective lexical branches. Never encode the
unguarded arm as an always-true comparison or mask and then issue one masked operation
outside the branch: target IR may retain that predicate and turn an unconditional J
access/effect into a masked one. Then run
`submit_replay` on the reviewed exact bytes and repair until it
passes. Submit also rejects an unambiguous read/write-mode contradiction between an
explicit J boundary effect and your own compiled TTIR. Fixed code materializes S_J,
projects exact packet facts, checks and rejects;
it never constructs or repairs target source. Finish only with the transaction passing
and do not claim Gate1, Gate2, or certification. A passive host watcher may checkpoint
stable compilable bytes, but it never submits them and never ends this coding turn;
only your explicit `submit_replay` can finish early.
"""

_REPAIR_PROMPT = """The same fresh source-blind Replay episode continues in the same thread and workspace.
Read work/host_replay_check.json and the current output/replay.source. Repair the
complete program using the same read-only S_J, then rerun submit_replay.
Diagnostics reject only; they do not supply a semantic mapping or repair. Finish only
when ingress and compile both pass.
"""

_INCREMENTAL_PROMPT = """You are the sole writer for one source-blind incremental Replay compilation.
The exact new J in input/semantic-program.intent is the only semantic authority.  The
current output/replay.source is a compilation cache produced earlier from another J;
it contains no Source, pi, Gate observation, or Harness fact and is not an authority.

Read the complete new semantic program, the compact compiler index, frontend ABI,
optional complete C, and the current Replay. Patch or rewrite the Replay so it implements the new
J.  The supplied change record only locates the edited J field; it does not authorize
target semantics.  Preserve unchanged J-owned behavior, but never preserve old Replay
behavior when it conflicts with the new J.

Work only inside the isolated workspace.  Compile and inspect only your own Replay.
Use replay_compile_check and submit_replay, repairing the complete program until the
transaction passes.  Fixed tools diagnose and reject; they do not generate target
source or reveal Source.  input/REPLAY_CONTRACT.md and input/TOOL.md remain the closed
contract and tool reference.
"""


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(text.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_usage(turns: list[CodexCLITurn]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for turn in turns:
        for key, value in turn.usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = usage.get(key, 0) + value
    return usage


def _replay_startup_context(files: Mapping[str, str]) -> dict[str, Any]:
    """Return a path-only receipt; Codex reads the exact packet as a local file."""

    packet_file = files.get("packet.json")
    if not isinstance(packet_file, str):
        raise ValueError("Replay startup context requires packet.json")
    paths = {
        "packet": "input/packet.json",
        "semantic_program": "input/semantic-program.intent",
        "semantic_program_json": "input/semantic-program.json",
        "compiler_index": "input/compiler-index.intent",
        "compiler_index_json": "input/compiler-index.json",
        "frontend": "input/frontend.json",
        "public_contract": "input/REPLAY_CONTRACT.md",
        "tool_contract": "input/TOOL.md",
        "replay": "output/replay.source",
        "host_check": "work/host_replay_check.json",
    }
    if "c.json" in files:
        paths["c"] = "input/c.json"
    return {
        "protocol": "intentir.scrc.replay-startup-view.v6",
        "packet_bytes": len(packet_file.encode("utf-8")),
        "semantic_program_text_bytes": len(
            files["semantic-program.intent"].encode("utf-8")
        ),
        "semantic_program_json_bytes": len(
            files["semantic-program.json"].encode("utf-8")
        ),
        "compiler_index_text_bytes": len(
            files["compiler-index.intent"].encode("utf-8")
        ),
        "compiler_index_json_bytes": len(
            files["compiler-index.json"].encode("utf-8")
        ),
        "paths": paths,
        "authority": (
            "path and byte-count receipt only; exact source-isolated packet bytes "
            "remain in input/packet.json and no code or schedule is generated"
        ),
    }


def _jc_fingerprint_summary(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Project one bounded, source-blind structural view of Replay's own TTIR."""

    result: dict[str, Any] = {}
    for key in (
        "interface",
        "op_multiset",
        "effect_multiset",
        "reductions",
        "dots",
        "structured_ifs",
        "terminal_forks",
    ):
        if key in fingerprint:
            result[key] = fingerprint[key]

    loops = fingerprint.get("loops")
    if isinstance(loops, list):
        kept = {
            "block_argument_count",
            "body_effects",
            "body_op_multiset",
            "carried_operand_count",
            "result_count",
        }
        result["loops"] = [
            {key: row[key] for key in kept if key in row}
            for row in loops
            if isinstance(row, Mapping)
        ]

    for key in ("dataflow_edges", "memory_effects", "mask_predicates"):
        value = fingerprint.get(key)
        if isinstance(value, (list, tuple, dict)):
            result[f"{key}_count"] = len(value)
    return result


def _j_contract_summary(semantic_program: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact J-only checklist beside Replay's own TTIR summary.

    Counts here are logical authored facts, not required raw TTIR multiplicities:
    target lowering may fuse or split representation.  The view gives the coding
    agent a bounded omission checklist without reading Source, pi, or Gate state.
    """

    nodes = semantic_program.get("nodes", [])
    node_ops = Counter(
        str(item.get("op"))
        for item in nodes
        if isinstance(item, Mapping) and isinstance(item.get("op"), str)
    ) if isinstance(nodes, list) else Counter()
    value_types_value = semantic_program.get("value_types", [])
    value_types = {
        str(item["value"]): {
            "dtype": item.get("dtype"),
            "shape": item.get("shape"),
        }
        for item in value_types_value
        if isinstance(item, Mapping) and isinstance(item.get("value"), str)
    } if isinstance(value_types_value, list) else {}

    # These operations are exactly the places where a superficially plausible
    # target program most often changes a J-owned type, axis, temporal state, or
    # representation boundary.  The list is a read-only projection of S_J; it is
    # not a target recipe and it neither accepts nor rejects Replay.
    typed_kinds = {
        "cast",
        "const",
        "contraction",
        "gather",
        "iota",
        "loop",
        "map_binary",
        "map_unary",
        "reduce",
        "scan",
        "scatter",
        "select",
        "transpose",
        *RAGGED_OPS,
    }
    typed_obligations: list[dict[str, Any]] = []
    for item in nodes if isinstance(nodes, list) else []:
        if not isinstance(item, Mapping) or item.get("op") not in typed_kinds:
            continue
        raw_operands = item.get("operands", [])
        operands = raw_operands if isinstance(raw_operands, list) else []
        raw_result = item.get("result")
        results = raw_result if isinstance(raw_result, list) else [raw_result]
        record = {
            "id": item.get("id"),
            "op": item.get("op"),
            "operands": [
                {"value": value, **value_types.get(str(value), {})}
                for value in operands
            ],
            "results": [
                {"value": value, **value_types.get(str(value), {})}
                for value in results
                if value is not None
            ],
        }
        for key in ("attrs", "trip", "carry", "scan", "yield"):
            if key in item:
                record[key] = item[key]
        typed_obligations.append(record)

    ownership = semantic_program.get("boundary_ownership", {})
    accesses = (
        ownership.get("access_effects", [])
        if isinstance(ownership, Mapping)
        else []
    )
    access_kinds: Counter[str] = Counter()
    effect_kinds: Counter[str] = Counter()
    accesses_by_tensor: Counter[str] = Counter()
    guarded_access_kinds: Counter[str] = Counter()
    unguarded_access_kinds: Counter[str] = Counter()
    for item in accesses if isinstance(accesses, list) else []:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("kind"), str):
            kind = str(item["kind"])
            access_kinds[kind] += 1
            if item.get("predicate") is None:
                unguarded_access_kinds[kind] += 1
            else:
                guarded_access_kinds[kind] += 1
        if isinstance(item.get("tensor"), str):
            accesses_by_tensor[str(item["tensor"])] += 1
        effects = item.get("effects", [])
        for effect in effects if isinstance(effects, list) else []:
            if isinstance(effect, Mapping) and isinstance(effect.get("kind"), str):
                effect_kinds[str(effect["kind"])] += 1
    return {
        "authority": (
            "J-only authored-fact checklist; logical counts are not raw TTIR "
            "occurrence requirements and do not accept or reject Replay"
        ),
        "node_ops": dict(sorted(node_ops.items())),
        "typed_obligations": typed_obligations[:96],
        "typed_obligations_omitted": max(0, len(typed_obligations) - 96),
        "access_kinds": dict(sorted(access_kinds.items())),
        "guarded_access_kinds": dict(sorted(guarded_access_kinds.items())),
        "unguarded_access_kinds": dict(sorted(unguarded_access_kinds.items())),
        "effect_kinds": dict(sorted(effect_kinds.items())),
        "accesses_by_tensor": dict(sorted(accesses_by_tensor.items())),
        "shared_write_axis_refs": (
            ownership.get("shared_write_axis_refs", [])
            if isinstance(ownership, Mapping)
            else []
        ),
        "shared_write_domain_axis_refs": (
            ownership.get("shared_write_domain_axis_refs", [])
            if isinstance(ownership, Mapping)
            else []
        ),
        "domain_axis_refs": (
            ownership.get("domain_axis_refs", {})
            if isinstance(ownership, Mapping)
            else {}
        ),
        "outputs": semantic_program.get("output_bindings", []),
    }


def _j_boundary_effect_diagnostics(
    semantic_program: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    runtime_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Reject only unambiguous J-boundary read/write contradictions.

    The check consumes this episode's J-only ``S_J`` and Replay's own compiled
    interface.  It does not inspect Source, pi, Gate state, address formulas, or
    peer observations, and it cannot accept Replay.  Alias groups and incomplete
    boundary effect sets remain outside this narrow necessary condition.
    """

    signature = semantic_program.get("signature", {})
    raw_parameters = (
        signature.get("parameters", [])
        if isinstance(signature, Mapping)
        else []
    )
    parameters = {
        str(item["name"]): item
        for item in raw_parameters
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    refs: Counter[str] = Counter(
        str(item.get("logical_ref"))
        for item in parameters.values()
        if isinstance(item.get("logical_ref"), str)
    )
    boundary = semantic_program.get("boundary", {})
    raw_effects = (
        boundary.get("effects", []) if isinstance(boundary, Mapping) else []
    )
    effect_modes: dict[str, set[str]] = {}
    for effect in raw_effects if isinstance(raw_effects, list) else []:
        if not isinstance(effect, Mapping) or not isinstance(effect.get("target"), str):
            continue
        kind = effect.get("kind")
        modes = effect_modes.setdefault(str(effect["target"]), set())
        if kind in {"read", "read_write"}:
            modes.add("read")
        if kind in {"write", "read_write"}:
            modes.add("write")

    interface = fingerprint.get("interface", {})
    raw_arguments = (
        interface.get("arguments", [])
        if isinstance(interface, Mapping)
        else []
    )
    arguments = raw_arguments if isinstance(raw_arguments, list) else []

    # ``TTIRFingerprint.interface.arguments`` deliberately contains only
    # pointer/tensor boundary carriers.  Scalar carriers stay visible through
    # dataflow but are omitted from this hard effect interface (see ttir.py).
    # Recover each retained carrier by its original TTIR argument position;
    # comparing the retained count with the full runtime arity would reject
    # every kernel that has a scalar extent or stride.
    positioned_arguments: list[tuple[str, Mapping[str, Any]]] = []
    for argument in arguments:
        if not isinstance(argument, Mapping):
            continue
        position = argument.get("position")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position >= len(runtime_names)
        ):
            return [
                {
                    "rule_code": "compiled_j_boundary_interface_unavailable",
                    "path": (
                        "<synthetic-own-ttir> retained interface argument has "
                        f"invalid runtime position: {position!r}"
                    ),
                    "anonymous_handles": [],
                    "line": None,
                }
            ]
        positioned_arguments.append((runtime_names[position], argument))

    diagnostics: list[dict[str, Any]] = []
    for name, argument in positioned_arguments:
        parameter = parameters.get(name)
        if parameter is None:
            continue
        logical_ref = parameter.get("logical_ref")
        if (
            not isinstance(logical_ref, str)
            or refs[logical_ref] != 1
            or parameter.get("alias_of") is not None
        ):
            continue
        modes = effect_modes.get(logical_ref, set())
        expected = (
            "inout"
            if modes == {"read", "write"}
            else "input"
            if modes == {"read"}
            else "output"
            if modes == {"write"}
            else None
        )
        actual = argument.get("access")
        if expected is None or actual == expected:
            continue
        diagnostics.append(
            {
                "rule_code": "compiled_j_boundary_effect_mode",
                "path": (
                    f"<synthetic-own-ttir> {name}/{logical_ref}: "
                    f"expected={expected}, actual={actual}"
                ),
                "anonymous_handles": [name, logical_ref],
                "line": None,
            }
        )

    # A guarded J access cannot compile to a program with no masked transfer of
    # the corresponding direction.  This is intentionally only a presence
    # check: target lowering may fuse or split logical accesses, so raw TTIR
    # multiplicity is not a sound equality obligation here.  The check sees
    # only authored J ownership and this Replay's own TTIR fingerprint; it does
    # not use Source, pi, Gate1, or a target recipe.
    ownership = semantic_program.get("boundary_ownership", {})
    accesses = (
        ownership.get("access_effects", [])
        if isinstance(ownership, Mapping)
        else []
    )
    guarded_directions: set[str] = set()
    for access in accesses if isinstance(accesses, list) else []:
        if not isinstance(access, Mapping) or access.get("predicate") is None:
            continue
        kind = access.get("kind")
        if kind == "load":
            guarded_directions.add("read.masked")
        elif kind in {"store", "atomic", "update"}:
            guarded_directions.add("write.masked")
    effect_multiset = fingerprint.get("effect_multiset", {})
    observed_effects = effect_multiset if isinstance(effect_multiset, Mapping) else {}
    for effect in sorted(guarded_directions):
        observed = observed_effects.get(effect, 0)
        if isinstance(observed, int) and not isinstance(observed, bool) and observed > 0:
            continue
        diagnostics.append(
            {
                "rule_code": "compiled_j_guarded_access_missing",
                "path": (
                    "<synthetic-own-ttir> J authors a guarded access but Replay "
                    f"contains no {effect} transfer"
                ),
                "anonymous_handles": [effect],
                "line": None,
            }
        )
    return diagnostics[:20]


@dataclass
class CodexCLIReplayWriter:
    """Time-governed Codex writer; every ``generate`` is a fresh episode."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    model_timeout_seconds: float | None = None
    # Time, not a semantic-attempt count, bounds one fresh Replay episode.  High
    # High gets three full provider windows; xhigh/max get two of their longer
    # windows.
    # This prevents a bad direction from cycling for an hour while leaving every
    # useful coding-agent tool action inside those turns unrestricted.
    session_wall_time_limit_seconds: float | None = None
    # Codex owns its thread compaction.  Zero disables host-side thread rollover;
    # cumulative API input-token counters are not a measure of live context size.
    context_rollover_input_tokens: int = 0
    repair_effort: str | None = None
    compile_checker: Callable[[str, Gate1Artifact, str], Any] | None = None
    prior_replay_source: str | None = None
    j_change: Mapping[str, Any] | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
    session_factory: Callable[..., Any] = field(
        default=CodexCLISession,
        repr=False,
    )
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        if self.context_rollover_input_tokens < 0:
            raise ValueError("Replay context rollover token threshold cannot be negative")
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        episode_limit = self.session_wall_time_limit_seconds
        if episode_limit is None:
            episode_limit = (
                2.0 * XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() in {"xhigh", "max"}
                else 3.0 * DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        if episode_limit <= 0:
            raise ValueError("Replay episode wall-time limit must be positive")
        turn_limit = self.model_timeout_seconds
        if turn_limit is None:
            turn_limit = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() in {"xhigh", "max"}
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        files, packet, _instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        agent_packet = dict(packet)
        semantic_program = materialize_replay_semantics(packet["j"]).to_json()
        files["packet.json"] = json.dumps(
            agent_packet,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
        function_name = str(packet["u_f"]["replay_function"])
        if self.compile_checker is None and self.frontend is not Frontend.TRITON:
            raise ValueError(
                "native Codex Replay requires a frontend-specific compile checker"
            )

        try:
            from minisweagent.environments.intentir_replay import (
                INTENTIR_PACKET_QUERY,
                IntentIRReplayWorkspace,
                _query_packet,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Codex CLI Replay requires the replay-agent workspace dependency"
            ) from exc

        workspace = IntentIRReplayWorkspace.create(files)
        if self.prior_replay_source is not None:
            _atomic_write_text(workspace.replay_path, self.prior_replay_source)
        turns: list[CodexCLITurn] = []
        checks: list[dict[str, Any]] = []
        compile_checks: list[dict[str, Any]] = []
        ttir_inspections: list[dict[str, Any]] = []
        packet_queries: list[dict[str, Any]] = []
        semantic_queries: list[dict[str, Any]] = []
        timeout_events: list[dict[str, Any]] = []
        accepted_submission_interrupts: list[dict[str, Any]] = []
        diagnostic_ready_interrupts: list[dict[str, Any]] = []
        live_checkpoint_events: list[dict[str, Any]] = []
        host_commit_events: list[dict[str, Any]] = []
        accepted_submission_event = threading.Event()
        diagnostic_ready_event = threading.Event()
        live_checkpoint_stop = threading.Event()
        evaluation_lock = threading.RLock()
        source_text = ""
        formal_ingress = None
        formal_compile = None
        last_checked_good: str | None = None
        startup_context = _replay_startup_context(files)
        synthetic_compiler = (
            None
            if self.compile_checker is not None
            else TritonCompiler(
                timeout=120.0,
                cache_dir=workspace.tmp_dir / "synthetic-replay-compile-cache",
            )
        )
        ttir_observations: dict[str, dict[str, Any]] = {}
        reviewed_jc_sources: set[str] = set()
        prompt_body = (
            _INITIAL_PROMPT
            if self.prior_replay_source is None
            else _INCREMENTAL_PROMPT
            + "\n\n<j-change>\n"
            + json.dumps(
                dict(self.j_change or {}),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n</j-change>\n"
        )
        prompt_body += _frontend_replay_guidance(self.frontend)
        prompt = (
            prompt_body
            + "\n\n<source-blind-startup>\n"
            + json.dumps(
                startup_context,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n</source-blind-startup>\n"
        )
        selected_effort = effective_effort
        active_elapsed_ms = 0.0
        timed_out_calls = 0
        thread_id: str | None = None
        thread_ids: list[str] = []
        session: Any | None = None
        last_replay_check_source: str | None = None
        last_replay_check_signature: tuple[tuple[str, str], ...] | None = None
        repeated_replay_diagnostic = 0
        explicit_submission_accepted = False

        def write_host_check(check: Mapping[str, Any]) -> None:
            _atomic_write_text(
                workspace.scratch_dir / "host_replay_check.json",
                json.dumps(
                    check,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
            )

        def new_session() -> Any:
            return self.session_factory(
                provider=provider,
                input_dir=workspace.input_dir,
                output_dir=workspace.output_dir,
                scratch_dir=workspace.scratch_dir,
                tmp_dir=workspace.tmp_dir,
                model=provider.model,
                effort=effective_effort,
            )

        def ingress_current() -> Any:
            current = workspace.read_replay()
            return replay_source_ingress(
                current,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        def replay_check_tool() -> Mapping[str, Any]:
            nonlocal last_replay_check_source
            nonlocal last_replay_check_signature
            nonlocal repeated_replay_diagnostic
            try:
                current = workspace.read_replay()
                response = dict(ingress_current().to_check_response())
            except Exception as exc:
                current = None
                response = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_unavailable",
                            "path": "<replay>",
                            "anonymous_handles": [],
                            "observed_refs_or_types": [
                                f"{type(exc).__name__}: {str(exc)[:2048]}"
                            ],
                            "line": None,
                        }
                    ],
                }
            diagnostics = response.get("diagnostics")
            signature = tuple(
                (
                    str(item.get("rule_code", "")),
                    str(item.get("path", "")),
                )
                for item in diagnostics
                if isinstance(item, Mapping)
            ) if isinstance(diagnostics, list) else ()
            source_changed = (
                current is not None
                and last_replay_check_source is not None
                and current != last_replay_check_source
            )
            if signature and signature == last_replay_check_signature:
                repeated_replay_diagnostic += 1
            else:
                repeated_replay_diagnostic = 1 if signature else 0
            response["diagnostic_progress"] = {
                "same_rule_path_streak": repeated_replay_diagnostic,
                "replay_source_changed_since_previous_check": source_changed,
                "authority": (
                    "mechanical episode-local observation only; it neither edits "
                    "Replay nor changes ingress or Gate1 membership"
                ),
                **(
                    {
                        "next_inspection": (
                            "The same closed obligation persists. Read its expected/"
                            "observed fields, inspect the exact packet path and search "
                            "the relation name in input/REPLAY_CONTRACT.md before "
                            "another structural edit."
                        )
                    }
                    if repeated_replay_diagnostic >= 2
                    else {}
                ),
            }
            last_replay_check_source = current
            last_replay_check_signature = signature
            return response

        def replay_compile_tool() -> Mapping[str, Any]:
            nonlocal last_checked_good
            current = workspace.read_replay()
            ingress = replay_source_ingress(
                current,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )
            if ingress.status != "pass" or ingress.code is None:
                return ingress.to_check_response()
            checked = (
                self.compile_checker(ingress.code, artifact, function_name)
                if self.compile_checker is not None
                else source_blind_replay_compile_check(
                    ingress.code,
                    artifact,
                    function_name,
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            response = checked.to_check_response()
            if checked.status == "pass":
                # The checkpoint is the exact Replay-LLM-authored text that
                # passed both source-blind construction checks in this fresh
                # episode.  Fixed code may restore only these bytes.
                last_checked_good = current
            return response

        def submit_replay_tool() -> Mapping[str, Any]:
            """Run both construction checks as one exact-source transaction."""

            nonlocal explicit_submission_accepted

            ingress = dict(replay_check_tool())
            if ingress.get("status") != "pass":
                return {
                    "protocol": "intentir.codex-cli.replay-submit.v2",
                    "status": "fail",
                    "phase": "ingress",
                    "ingress": ingress,
                    "compile": None,
                    "authority": "diagnostic-only; no Replay source is generated or repaired",
                }
            compiled = dict(replay_compile_tool())
            necessary: Mapping[str, Any] | None = None
            if compiled.get("status") == "pass" and synthetic_compiler is not None:
                current_source = workspace.read_replay()
                reviewed_before_submit = current_source in reviewed_jc_sources
                necessary = replay_ttir_inspect_tool(("jc",))
                if not reviewed_before_submit:
                    return {
                        "protocol": "intentir.codex-cli.replay-submit.v2",
                        "status": "fail",
                        "phase": "j_owned_review_required",
                        "ingress": ingress,
                        "compile": compiled,
                        "j_owned_review": necessary,
                        "authority": (
                            "source-blind own-TTIR review checkpoint only; inspect the "
                            "returned J/C facts and current Replay TTIR, then explicitly "
                            "submit the same or revised exact bytes"
                        ),
                    }
                diagnostics = necessary.get("j_owned_necessary_diagnostics")
                if isinstance(diagnostics, list) and diagnostics:
                    return {
                        "protocol": "intentir.codex-cli.replay-submit.v2",
                        "status": "fail",
                        "phase": "j_owned_necessary",
                        "ingress": ingress,
                        "compile": compiled,
                        "j_owned_necessary": necessary,
                        "authority": (
                            "J-only necessary-condition diagnostic; no Replay source "
                            "is generated or repaired and formal Gate1 remains external"
                        ),
                    }
            if compiled.get("status") == "pass":
                explicit_submission_accepted = True
                accepted_submission_event.set()
            return {
                "protocol": "intentir.codex-cli.replay-submit.v2",
                "status": compiled.get("status", "fail"),
                "phase": "compile",
                "ingress": ingress,
                "compile": compiled,
                "authority": "diagnostic-only; no Replay source is generated or repaired",
            }

        def packet_query_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            """Project one bounded exact view from the immutable source-blind packet."""

            value = _query_packet(packet, [INTENTIR_PACKET_QUERY, *arguments])
            packet_queries.append({"query": list(arguments)})
            return {
                "protocol": "intentir.codex-cli.replay-packet-query.v1",
                "status": "pass",
                "query": list(arguments),
                "value": value,
                "authority": (
                    "exact bounded packet projection only; no Source, semantic "
                    "inference, realization choice, code generation, or Gate authority"
                ),
            }

        def semantic_query_tool(arguments: tuple[str, ...]) -> Mapping[str, Any]:
            value = query_replay_semantics(semantic_program, arguments)
            semantic_queries.append({"query": list(arguments)})
            return {
                **value,
                "authority": value.get(
                    "authority", semantic_program.get("authority")
                ),
            }

        def replay_ttir_inspect_tool(
            arguments: tuple[str, ...]
        ) -> Mapping[str, Any]:
            if synthetic_compiler is None:
                return {
                    "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                    "status": "unavailable",
                    "diagnostic": "custom compile checker exposes no Replay TTIR",
                }
            current = workspace.read_replay()
            observation = ttir_observations.get(current)
            if observation is None:
                ingress = replay_source_ingress(
                    current,
                    artifact=artifact,
                    frontend=self.frontend,
                    function_name=function_name,
                    packet=packet,
                )
                if ingress.status != "pass" or ingress.code is None:
                    return ingress.to_check_response()
                case = build_source_blind_replay_compile_case(artifact)
                result = synthetic_compiler.compile(
                    KernelSource.from_replay(ingress.code, function_name),
                    case,
                )
                if not result.ok or not isinstance(result.ttir, str):
                    observation = {
                        "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                        "status": "fail",
                        "diagnostic": result.diagnostic or "compiler returned no TTIR",
                    }
                else:
                    try:
                        fingerprint = fingerprint_ttir(
                            result.ttir,
                            python_interface=result.python_interface,
                            schedule=case.schedule,
                        )
                    except Exception as exc:
                        observation = {
                            "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                            "status": "fail",
                            "diagnostic": (
                                "own Replay TTIR observer rejected the compiled program: "
                                f"{type(exc).__name__}: {str(exc)[:2048]}"
                            ),
                        }
                    else:
                        observation = {
                            "protocol": "intentir.scrc.replay-own-ttir-view.v1",
                            "status": "pass",
                            "ttir": result.ttir,
                            "fingerprint": fingerprint.to_dict(),
                            "compiler": dict(result.compiler),
                            "target": dict(result.target),
                        }
                ttir_observations[current] = observation

            mode = arguments[0] if arguments else "summary"
            if mode not in {"jc", "summary", "raw", "find", "observer"}:
                raise ValueError(
                    "replay_ttir_inspect mode must be jc/summary/raw/find/observer"
                )
            if observation.get("status") != "pass":
                result_view = dict(observation)
            else:
                ttir = str(observation["ttir"])
                lines = ttir.splitlines()
                if mode == "jc":
                    if len(arguments) != 1:
                        raise ValueError("jc accepts no additional arguments")
                    fingerprint = observation.get("fingerprint")
                    if not isinstance(fingerprint, Mapping):
                        raise ValueError("compiled Replay exposes no fingerprint")
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "ttir_lines": len(lines),
                        "j_contract": _j_contract_summary(semantic_program),
                        "jc": _jc_fingerprint_summary(fingerprint),
                        "j_owned_necessary_diagnostics": (
                            _j_boundary_effect_diagnostics(
                                semantic_program,
                                fingerprint,
                                runtime_argument_names(ttir),
                            )
                        ),
                    }
                elif mode == "summary":
                    fingerprint = observation.get("fingerprint")
                    result_view = {
                        key: value
                        for key, value in observation.items()
                        if key not in {"ttir", "fingerprint"}
                    }
                    result_view["ttir_lines"] = len(lines)
                    result_view["fingerprint_sections"] = (
                        list(fingerprint)
                        if isinstance(fingerprint, Mapping)
                        else []
                    )
                elif mode == "raw":
                    if len(arguments) > 3:
                        raise ValueError("raw accepts START [COUNT]")
                    start = int(arguments[1]) if len(arguments) >= 2 else 1
                    count = int(arguments[2]) if len(arguments) >= 3 else 80
                    if start < 1 or count < 1 or count > 160:
                        raise ValueError("raw START/COUNT must be positive; COUNT <= 160")
                    selected = lines[start - 1 : start - 1 + count]
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "start": start,
                        "total": len(lines),
                        "lines": [
                            [ordinal, line]
                            for ordinal, line in enumerate(selected, start=start)
                        ],
                        "next_start": (
                            start + len(selected)
                            if start - 1 + len(selected) < len(lines)
                            else None
                        ),
                    }
                elif mode == "find":
                    if len(arguments) not in {2, 3}:
                        raise ValueError("find accepts TOKEN [CONTEXT]")
                    token = arguments[1]
                    context = int(arguments[2]) if len(arguments) == 3 else 2
                    if not token or len(token) > 256 or not 0 <= context <= 12:
                        raise ValueError("find TOKEN/CONTEXT is outside the closed bound")
                    matches = [index for index, line in enumerate(lines) if token in line]
                    rows: list[list[Any]] = []
                    seen: set[int] = set()
                    for index in matches[:32]:
                        for ordinal in range(
                            max(0, index - context),
                            min(len(lines), index + context + 1),
                        ):
                            if ordinal not in seen:
                                rows.append([ordinal + 1, lines[ordinal]])
                                seen.add(ordinal)
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "token": token,
                        "match_count": len(matches),
                        "lines": rows,
                    }
                else:
                    if len(arguments) != 2:
                        raise ValueError("observer accepts one dotted PATH")
                    value: Any = observation.get("fingerprint")
                    for component in arguments[1].split("."):
                        if not component:
                            raise ValueError("observer PATH contains an empty component")
                        if isinstance(value, Mapping) and component in value:
                            value = value[component]
                        elif isinstance(value, list) and component.isdigit():
                            value = value[int(component)]
                        else:
                            raise KeyError("observer PATH is absent")
                    result_view = {
                        "protocol": observation["protocol"],
                        "status": "pass",
                        "path": arguments[1],
                        "value": value,
                    }
            if mode == "jc" and result_view.get("status") == "pass":
                reviewed_jc_sources.add(current)
            ttir_inspections.append(
                {
                    "query": list(arguments) if arguments else ["summary"],
                    "status": result_view.get("status"),
                }
            )
            return {
                **result_view,
                "authority": (
                    "exact TTIR/fingerprint of this episode's current Replay under "
                    "the public synthetic profile only; no Source, Gate, inference, "
                    "or repair"
                ),
            }

        def _locked_tool_call(
            callback: Callable[..., Mapping[str, Any]],
            *arguments: Any,
        ) -> Mapping[str, Any]:
            with evaluation_lock:
                return callback(*arguments)

        bridge = CodexCLIToolBridge(
            workspace.scratch_dir / CODEX_CLI_TOOL_BRIDGE_ROOT_NAME,
            {
                "replay_check": lambda: _locked_tool_call(replay_check_tool),
                "replay_compile_check": lambda: _locked_tool_call(
                    replay_compile_tool
                ),
                "submit_replay": lambda: _locked_tool_call(submit_replay_tool),
            },
            argument_handlers={
                "packet_query": packet_query_tool,
                "semantic_query": semantic_query_tool,
                "replay_ttir_inspect": lambda arguments: _locked_tool_call(
                    replay_ttir_inspect_tool, arguments
                ),
            },
        )

        def _evaluate_current_unlocked(
            *, trigger: str
        ) -> tuple[bool, Mapping[str, Any]]:
            nonlocal source_text, formal_ingress, formal_compile
            try:
                source_text = workspace.read_replay()
            except Exception:
                source_text = ""
                formal_ingress = None
                formal_compile = None
                check: Mapping[str, Any] = {
                    "protocol": REPLAY_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "replay_unavailable",
                            "path": "<replay>",
                            "anonymous_handles": [],
                            "line": None,
                        }
                    ],
                }
            else:
                formal_ingress = replay_source_ingress(
                    source_text,
                    artifact=artifact,
                    frontend=self.frontend,
                    function_name=function_name,
                    packet=packet,
                )
                check = formal_ingress.to_check_response()
            checks.append(
                {
                    "trigger": trigger,
                    "status": check["status"],
                    "diagnostics": [dict(item) for item in check["diagnostics"]],
                }
            )
            if (
                formal_ingress is None
                or formal_ingress.status != "pass"
                or formal_ingress.code is None
            ):
                return False, check
            formal_compile = (
                self.compile_checker(
                    formal_ingress.code,
                    artifact,
                    function_name,
                )
                if self.compile_checker is not None
                else source_blind_replay_compile_check(
                    formal_ingress.code,
                    artifact,
                    function_name,
                    compiler=synthetic_compiler,
                    packet=packet,
                )
            )
            compile_check = formal_compile.to_check_response()
            compile_checks.append(
                {
                    "trigger": trigger,
                    "status": compile_check["status"],
                    "diagnostics": [
                        dict(item) for item in compile_check["diagnostics"]
                    ],
                }
            )
            return formal_compile.status == "pass", compile_check

        def evaluate_current(*, trigger: str) -> tuple[bool, Mapping[str, Any]]:
            with evaluation_lock:
                return _evaluate_current_unlocked(trigger=trigger)

        try:
            initial_replay = workspace.read_replay()
        except Exception:
            initial_replay = None
        live_checkpoint_attempted = (
            {initial_replay} if initial_replay is not None else set()
        )

        def live_checkpoint_watch() -> None:
            """Validate and atomically commit stable exact Agent-authored bytes.

            Fixed code performs only the same ingress, compile, and J-owned own-TTIR
            necessary checks exposed to the Agent.  It never generates or repairs
            target source; a failed checkpoint leaves the same Agent turn running.
            """

            nonlocal last_checked_good
            observed_replay: str | None = None
            stable_since = 0.0
            while not live_checkpoint_stop.wait(0.25):
                try:
                    observed_text = workspace.read_replay()
                except Exception:
                    observed_replay = None
                    stable_since = 0.0
                    continue
                if observed_text in live_checkpoint_attempted:
                    continue
                if observed_text != observed_replay:
                    observed_replay = observed_text
                    stable_since = time.monotonic()
                    continue
                if time.monotonic() - stable_since < 3.0:
                    continue
                live_checkpoint_attempted.add(observed_text)
                passed, check = evaluate_current(trigger="live_checkpoint")
                with evaluation_lock:
                    checked_text = source_text
                    accepted = passed and checked_text == observed_text
                    if accepted:
                        last_checked_good = checked_text
                own_ir: Mapping[str, Any] | None = None
                committed = False
                if accepted:
                    try:
                        still_current = workspace.read_replay() == observed_text
                    except Exception:
                        still_current = False
                    if synthetic_compiler is None:
                        # Native frontends have no Triton TTIR.  Their
                        # frontend-specific compile checker is the complete
                        # source-blind construction checkpoint; Gate1 remains
                        # the first native structural observer.
                        committed = still_current
                    else:
                        own_ir = _locked_tool_call(
                            replay_ttir_inspect_tool, ("jc",)
                        )
                        necessary = own_ir.get("j_owned_necessary_diagnostics")
                        committed = (
                            still_current
                            and own_ir.get("status") == "pass"
                            and not necessary
                        )
                live_checkpoint_events.append(
                    {
                        "replay_bytes": len(checked_text.encode()) if checked_text else 0,
                        "status": check.get("status", "fail"),
                        "accepted": accepted,
                        "j_owned_necessary": (
                            own_ir.get("status")
                            if own_ir is not None
                            else "not_applicable"
                            if synthetic_compiler is None
                            else "not_run"
                        ),
                        "committed": committed,
                    }
                )
                if checked_text != observed_text:
                    live_checkpoint_attempted.discard(observed_text)
                if committed:
                    host_commit_events.append(
                        {
                            "trigger": "live_checkpoint",
                            "replay_bytes": len(checked_text.encode()),
                            "ingress": "pass",
                            "compile": "pass",
                            "j_owned_necessary": (
                                "not_applicable"
                                if synthetic_compiler is None
                                else "pass"
                            ),
                            "authority": (
                                "atomic commit of exact Agent-authored bytes only; "
                                "fixed code generated and repaired no target source"
                            ),
                        }
                    )
                    accepted_submission_event.set()
                    continue
                if accepted:
                    observed_replay = None
                    stable_since = 0.0
                    continue
                if checked_text == observed_text:
                    write_host_check(check)
                    # The same closed diagnostic is available through the explicit
                    # tools. A passive failed draft must not split one native
                    # write-check-patch loop into another provider turn.
                    observed_replay = None
                    stable_since = 0.0

        live_checkpoint_thread: threading.Thread | None = None
        try:
            bridge.start()
            live_checkpoint_thread = threading.Thread(
                target=live_checkpoint_watch,
                name="intentir-codex-replay-live-checkpoint",
                daemon=True,
            )
            live_checkpoint_thread.start()
            session = new_session()
            while True:
                remaining = (
                    episode_limit
                    - active_elapsed_ms / 1000.0
                )
                if remaining <= 0:
                    raise RuntimeError("Replay owner episode wall-time exhausted")
                turn_timeout = min(remaining, turn_limit)
                try:
                    turn = session.run(
                        prompt,
                        timeout_seconds=turn_timeout,
                        effort=selected_effort,
                        stop_requested=self.stop_requested,
                        submission_accepted=accepted_submission_event.is_set,
                        diagnostic_ready=diagnostic_ready_event.is_set,
                    )
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                except CodexCLISubmissionAccepted as exc:
                    active_elapsed_ms += exc.elapsed_ms
                    accepted_submission_event.clear()
                    diagnostic_ready_event.clear()
                    thread_id = exc.thread_id
                    if thread_id not in thread_ids:
                        thread_ids.append(thread_id)
                    accepted_submission_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "pending_exact_checkpoint_recheck",
                        }
                    )
                    if last_checked_good is None:
                        raise CodexCLIError(
                            "accepted Replay submission has no exact checked checkpoint"
                        ) from exc
                    _atomic_write_text(workspace.replay_path, last_checked_good)
                    passed, _ = evaluate_current(
                        trigger="host_accepted_submission_restore"
                    )
                    if not passed:
                        raise CodexCLIError(
                            "accepted Replay failed independent host recheck"
                        ) from exc
                    accepted_submission_interrupts[-1]["recovery"] = (
                        "exact_checkpoint_restored"
                    )
                    break
                except CodexCLIDiagnosticReady as exc:
                    active_elapsed_ms += exc.elapsed_ms
                    diagnostic_ready_event.clear()
                    accepted_submission_event.clear()
                    thread_id = exc.thread_id
                    if thread_id not in thread_ids:
                        thread_ids.append(thread_id)
                    diagnostic_ready_interrupts.append(
                        {
                            "elapsed_ms": exc.elapsed_ms,
                            "thread_id": exc.thread_id,
                            "recovery": "same_thread_closed_diagnostic",
                        }
                    )
                    passed, check = evaluate_current(
                        trigger="host_live_diagnostic_recheck"
                    )
                    if passed:
                        break
                    write_host_check(check)
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
                except CodexCLIError as exc:
                    if "wall-time limit" not in str(exc):
                        raise
                    elapsed = turn_timeout * 1000.0
                    active_elapsed_ms += elapsed
                    timed_out_calls += 1
                    thread_id = getattr(session, "thread_id", None)
                    timeout_event = {
                        "turn_timeout_seconds": turn_timeout,
                        "thread_id": thread_id,
                        "recovery": None,
                    }
                    timeout_events.append(timeout_event)
                    passed, check = evaluate_current(
                        trigger="host_timeout_recheck"
                    )
                    remaining_after_timeout = (
                        episode_limit
                        - active_elapsed_ms / 1000.0
                    )
                    if passed:
                        timeout_event["recovery"] = (
                            "current_exact_replay"
                            if explicit_submission_accepted
                            else "turn_deadline_host_validated_current_exact_replay"
                        )
                        break
                    if last_checked_good is not None:
                        _atomic_write_text(
                            workspace.replay_path, last_checked_good
                        )
                        passed, check = evaluate_current(
                            trigger="host_timeout_checkpoint_restore"
                        )
                        if passed:
                            timeout_event["recovery"] = (
                                "same_episode_checked_checkpoint"
                                if explicit_submission_accepted
                                else "turn_deadline_host_validated_checked_checkpoint"
                            )
                            break
                    if remaining_after_timeout <= 0:
                        timeout_event["recovery"] = "episode_time_exhausted"
                        raise RuntimeError(
                            "Replay owner episode wall-time exhausted"
                        ) from exc
                    if thread_id is None:
                        timeout_event["recovery"] = "no_resumable_thread"
                        raise CodexCLIError(
                            "initial Codex Replay turn timed out without a "
                            "resumable thread or checked Replay"
                        ) from exc
                    write_host_check(check)
                    timeout_event["recovery"] = "same_thread_continuation"
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
                turns.append(turn)
                thread_id = turn.thread_id
                if thread_id not in thread_ids:
                    thread_ids.append(thread_id)
                active_elapsed_ms += turn.elapsed_ms
                passed, check = evaluate_current(trigger="host_post_turn")
                if passed and explicit_submission_accepted:
                    break
                if passed and synthetic_compiler is None:
                    last_checked_good = source_text
                    host_commit_events.append(
                        {
                            "trigger": "host_post_turn",
                            "replay_bytes": len(source_text.encode()),
                            "ingress": "pass",
                            "compile": "pass",
                            "j_owned_necessary": "not_applicable",
                            "authority": (
                                "exact Agent-authored native Replay bytes passed the "
                                "frontend-specific construction checker; formal Gate1 "
                                "remains external"
                            ),
                        }
                    )
                    break
                if passed:
                    last_checked_good = source_text
                    own_ir = replay_ttir_inspect_tool(("jc",))
                    necessary = own_ir.get("j_owned_necessary_diagnostics")
                    if own_ir.get("status") == "pass" and not necessary:
                        host_commit_events.append(
                            {
                                "trigger": "host_post_turn",
                                "replay_bytes": len(source_text.encode()),
                                "ingress": "pass",
                                "compile": "pass",
                                "j_owned_necessary": "pass",
                                "authority": (
                                    "atomic commit of exact Agent-authored bytes only; "
                                    "fixed code generated and repaired no target source"
                                ),
                            }
                        )
                        break
                    diagnostics = (
                        [dict(item) for item in necessary]
                        if isinstance(necessary, list) and necessary
                        else [
                            {
                                "rule_code": "replay_own_ttir_review_failed",
                                "path": "<replay-ttir>",
                                "anonymous_handles": [],
                                "line": None,
                            }
                        ]
                    )
                    write_host_check(
                        {
                            "protocol": REPLAY_CHECK_PROTOCOL,
                            "status": "fail",
                            "diagnostics": diagnostics,
                            "authority": (
                                "J-only necessary-condition diagnostics; fixed code "
                                "does not generate or repair Replay"
                            ),
                        }
                    )
                    prompt = _REPAIR_PROMPT
                    selected_effort = self.repair_effort or effective_effort
                    continue
                write_host_check(check)
                prompt = _REPAIR_PROMPT
                selected_effort = self.repair_effort or effective_effort

            assert formal_ingress is not None and formal_ingress.code is not None
            assert formal_compile is not None and formal_compile.status == "pass"
            usage = _aggregate_usage(turns)
            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_CLI_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "semantic_queries": semantic_queries,
                "timeout_events": timeout_events,
                "accepted_submission_interrupts": accepted_submission_interrupts,
                "diagnostic_ready_interrupts": diagnostic_ready_interrupts,
                "live_checkpoint": live_checkpoint_events,
                "host_commit": host_commit_events,
                "explicit_submission_accepted": explicit_submission_accepted,
                "tool_bridge": bridge.evidence,
                "formal_ingress_recheck": True,
                "isolation": {
                    "source_blind": True,
                    "input": "read-only",
                    "output_work_tmp": "write-only-scope",
                    "tool_network": "disabled",
                    "provider_secret_in_tool_env": False,
                },
            }
            # Detach the evidence object from episode-local lists that the live
            # checkpoint thread may finish updating while ``finally`` shuts it
            # down.  The stored trajectory and its canonical bytes must describe
            # the same immutable snapshot.
            trajectory = json.loads(_trajectory_bytes(trajectory))
            raw_trajectory = _trajectory_bytes(trajectory)
            context_receipt = _context_receipt({}, packet["u_f"])
            context_receipt.update(
                {
                    "backend_protocol": CODEX_CLI_REPLAY_BACKEND,
                    "fresh_session": True,
                    "thread_id": thread_id,
                    "thread_ids": list(thread_ids),
                    "context_rollover_input_tokens": (
                        self.context_rollover_input_tokens
                    ),
                    "episode_wall_time_limit_seconds": (
                        episode_limit
                    ),
                    "cli_turns": (
                        len(turns)
                        + timed_out_calls
                        + len(accepted_submission_interrupts)
                        + len(diagnostic_ready_interrupts)
                    ),
                    "model_segments_lower_bound": (
                        sum(turn.model_segments for turn in turns)
                        + timed_out_calls
                        + len(accepted_submission_interrupts)
                        + len(diagnostic_ready_interrupts)
                    ),
                    "tool_actions": sum(turn.tool_actions for turn in turns),
                    "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                    "formal_compile_recheck": True,
                }
            )
            generation = ReplayGeneration(
                code=formal_ingress.code,
                function_name=function_name,
                model=provider.model,
                response_id=thread_id,
                context_receipt=context_receipt,
                usage=usage,
                response_model=provider.model,
                entry_function_name=formal_ingress.entry_function_name,
                response_text=source_text,
                raw_response=raw_trajectory,
                attempts=(
                    len(turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                ),
                physical_provider_calls=(
                    sum(turn.model_segments for turn in turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                ),
                transport_retries=0,
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
            )
            self._records.append(generation)
            return generation
        except Exception as exc:
            cancelled_calls = 0
            if isinstance(exc, CodexCLICancelled):
                active_elapsed_ms += exc.elapsed_ms
                cancelled_calls = exc.physical_calls
            trajectory = {
                "protocol": CODEX_CLI_AGENT_RUNTIME,
                "backend": CODEX_CLI_REPLAY_BACKEND,
                "fresh_session": True,
                "thread_id": thread_id,
                "thread_ids": list(thread_ids),
                "turns": [turn.to_evidence() for turn in turns],
                "replay_check": checks,
                "replay_compile_check": compile_checks,
                "replay_ttir_inspections": ttir_inspections,
                "packet_queries": packet_queries,
                "semantic_queries": semantic_queries,
                "timeout_events": timeout_events,
                "accepted_submission_interrupts": accepted_submission_interrupts,
                "diagnostic_ready_interrupts": diagnostic_ready_interrupts,
                "live_checkpoint": live_checkpoint_events,
                "tool_bridge": bridge.evidence,
                "failure_class": type(exc).__name__,
                **(
                    {
                        "winner_branch_id": exc.winner_branch_id,
                        "cancelled_elapsed_ms": exc.elapsed_ms,
                    }
                    if isinstance(exc, CodexCLICancelled)
                    else {}
                ),
            }
            trajectory = json.loads(_trajectory_bytes(trajectory))
            raw_trajectory = _trajectory_bytes(trajectory)
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=source_text,
                response_id=thread_id,
                response_model=provider.model,
                usage=_aggregate_usage(turns),
                raw_response=raw_trajectory,
                attempts=(
                    len(turns)
                    + timed_out_calls
                    + len(accepted_submission_interrupts)
                    + len(diagnostic_ready_interrupts)
                ),
                elapsed_ms=active_elapsed_ms,
                trajectory=trajectory,
                backend_protocol=CODEX_CLI_REPLAY_BACKEND,
                physical_provider_calls=sum(
                    turn.model_segments for turn in turns
                ) + cancelled_calls + timed_out_calls + len(
                    accepted_submission_interrupts
                ) + len(diagnostic_ready_interrupts),
                transport_retries=0,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        finally:
            live_checkpoint_stop.set()
            if live_checkpoint_thread is not None:
                live_checkpoint_thread.join()
            bridge.close()
            if session is not None:
                session.close()
            workspace.cleanup()


__all__ = ["CODEX_CLI_REPLAY_BACKEND", "CodexCLIReplayWriter"]
