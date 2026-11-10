#ifndef INTENT_CONVERSION_CUDA_PASSES
#define INTENT_CONVERSION_CUDA_PASSES

include "mlir/Pass/PassBase.td"

def ConvertIntentRowReductionToGPU
    : Pass<"convert-intent-row-reduction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a row-local reduce+broadcast intent func to one "
                "read-once block-cooperative gpu.launch.";
  let description = [{
    Recognizes `func.func`s whose body is a straight-line graph of
    `intent.map_unary`/`intent.map_binary`/`intent.reduce` that computes a
    2-D `[M,N]` output where every `intent.reduce` is over the last axis
    (keepdims `[M,1]`) — the "row normalization" shape (rms_norm, softmax,
    layer_norm, ...). It rewrites such a func in place to a memref-arg function
    holding a single `gpu.launch`: one block per row, the feature dim split
    over the block's threads, each thread reading its columns **once** into an
    SSA register tile, compact warp/shared reduction groups, per-row scalars
    hoisted once, and the epilogue writing the output from the register tile
    (read-once = HBM parity with the library kernel). This is the verified block-cooperative
    path; it is general over the reduce/map DAG (NO per-kernel recognizer).
    Non-matching funcs are left untouched for `convert-intent-to-linalg`.

    Static shapes only. Run BEFORE `convert-intent-to-linalg`; the resulting
    memref+gpu.launch module goes straight to the device pipeline
    (`lower_memref_module_to_kernels`), bypassing bufferization. Transcendentals
    are emitted as `math.exp`/`math.sqrt`/`math.rsqrt` and rewritten to NVVM
    approx intrinsics by the `-lower-gpu-math-to-nvvm` target peephole.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::NVVM::NVVMDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the feature dim); 0 = auto "
           "(largest power-of-two <= min(N,256) dividing N). The PK harness "
           "passes the library kernel's geometry (e.g. 128 / 256).">,
    Option<"tileCols", "tile-cols", "unsigned", "32",
           "Max f32 columns held in registers per thread (register budget). "
           "When N/T exceeds this, the per-thread feature strip is split into "
           "num_tiles tiles of <= tile-cols each. Independent reductions share "
           "one scf.for over the tiles; dependent reductions start a new pass "
           "over the same workgroup-cached row frontier. Warp shuffles plus "
           "compact per-warp shared partials combine production geometries with "
           "one barrier per dependency level; the epilogue reads the frontier "
           "cache instead of re-reading it from HBM. num_tiles==1 "
           "(N/T <= tile-cols) keeps the register-tile path byte-identical.">,
    Option<"vectorWidth", "vector-width", "unsigned", "0",
           "Register-path contiguous memory width: 0 = structurally selected "
           "auto, 1 = force scalar coalesced loads/stores, 4 = require legal "
           "four-lane Row/Col loads and Row stores (vector<4xf32> or "
           "vector<4xf16/bf16>). This is a Layer-C candidate axis; unsupported "
           "forced-vector graphs fail closed.">,
    Option<"fuseFma", "fuse-fma", "bool", "false",
           "Emit explicit f32 math.fma for single-use mul+add epilogues and "
           "sum-of-products reduction accumulation. Disabled by default. The "
           "production search may enable it as an observation-changing concrete "
           "MLIR state; that state must pass Q on D_search before "
           "performance selection and the fixed winner must pass D_cert.">,
    Option<"approximateDiv", "approximate-div", "bool", "false",
           "Lower f32 division to multiplication by a native approximate "
           "reciprocal (or a rounded constant reciprocal for a proven static "
           "divisor). Disabled by default because its allowed observations differ "
           "from divf. Production search treats it as an observation-changing "
           "precision state and requires Q/D_search and D_cert.">,
    Option<"deferEpilogueColLoads", "defer-epilogue-col-loads", "bool", "true",
           "Delay epilogue-only contiguous Col loads until after all row "
           "collectives, reducing cross-barrier register pressure. Set false "
           "to preload them before the reduction so small register-resident "
           "rows can overlap global-memory latency with the collective. This "
           "is a Layer-C candidate and must independently pass P6.">,
    Option<"materializeSharedRowOutput", "materialize-shared-row-output", "bool", "true",
           "When a required full-row output also feeds a single reduction and "
           "its epilogue, true stores that row before the collective and reloads "
           "it afterwards to shorten register live ranges. False keeps the same "
           "pointwise values in SSA across the collective and emits only their "
           "required final store, eliminating one full-row reload. The false "
           "state is admitted only when that exact graph is proved; unsupported "
           "graphs fail closed.">,
    Option<"leaderFinalize", "leader-finalize", "bool", "false",
           "For a multi-warp register reduction with two or more reduction ops, "
           "let warp 0 fold the compact per-warp partials, publish one scalar per "
           "reduction, and broadcast it after a workgroup barrier. False retains "
           "the state in which every warp redundantly folds the same partials. "
           "Both states preserve the same arithmetic tree; unsupported graphs "
           "fail closed.">,
    Option<"promoteRank0Scalars", "promote-rank0-scalars", "bool", "true",
           "Load dynamic rank-0 inputs once in the host launch stub and pass "
           "them to the outlined kernel by value. False retains device-side "
           "uniform loads as an independently searchable ABI schedule.">,
    Option<"enableSortedPrefix", "enable-sorted-prefix", "bool", "true",
           "Enable the B-certified descending sparsemax early-stop fast path. "
           "Set false to force the general scan/reduction implementation for "
           "the same Intent graph, providing an independently executable "
           "generic correctness fallback.">,
    Option<"allowExecutionDomainSortedPrefix",
           "allow-execution-domain-sorted-prefix", "bool", "false",
           "Permit the same graph-structural sorted-prefix lowering without a "
           "Layer-B descending-order marker. This does not certify or infer the "
           "ordering fact: the emitted state is explicitly marked as an "
           "execution-domain assumption and production search must validate it "
           "with Q/D before timing or selection.">,
    Option<"requireSortedPrefix", "require-sorted-prefix", "bool", "false",
           "Require this invocation to materialize the sorted-prefix topology. "
           "If the exact structural matcher does not fire, fail the pass instead "
           "of silently relabeling the complete generic scan lowering.">,
    Option<"prefixThreads", "prefix-threads", "unsigned", "0",
           "Physical threads used by the sorted-prefix scan. 0 keeps the scan "
           "and output on the same block geometry. 32 assigns the prefix/support "
           "work to one warp while the full block writes the row epilogue, "
           "decoupling active-prefix work from full-row output parallelism. The "
           "nonzero state is accepted only by the exact sorted-prefix matcher.">,
    Option<"prefixPacketWidth", "prefix-packet-width", "unsigned", "4",
           "Sorted-input elements loaded by each active prefix thread per scan "
           "chunk. Legal values are 1, 2, or 4. This coordinate is independent "
           "of the full-row X/output packet, which remains vector<4xf32>: a "
           "narrow prefix packet can stop after 32 or 64 sorted values without "
           "degrading the coalesced epilogue. Accepted only by the exact sorted-"
           "prefix matcher.">,
    Option<"crossMultiplySupportPredicate",
           "cross-multiply-support-predicate", "bool", "false",
           "For the exact sorted-prefix support test, replace z > (prefix-1)/rank "
           "with rank*z > prefix-1 using the proved-positive rank. This removes "
           "per-element floating division but changes rounding observations, so "
           "it is an explicit Full numeric state that must pass Q/D. Unsupported "
           "graphs fail closed.">,
    Option<"replicatePrefixPerWarp", "replicate-prefix-per-warp", "bool", "false",
           "For a 32-thread sorted-prefix scan with a wider one-row CTA, let every "
           "output warp recompute the short active prefix and retain tau in warp-"
           "local SSA. This removes the shared tau publication and workgroup "
           "barrier at the cost of cache-local repeated prefix work. The state is "
           "accepted only by the exact sorted-prefix matcher.">,
    Option<"prefetchEpilogueInputBeforeSummary",
           "prefetch-epilogue-input-before-summary", "bool", "false",
           "For the exact sorted-prefix graph, issue each output thread's "
           "coalesced X packet loads before prefix/support/tau evaluation and "
           "keep those packets live until the epilogue. This overlaps the full-"
           "row input fetch with the serial prefix state at the cost of register "
           "pressure. Unsupported graphs fail closed.">,
    Option<"frontierBallotSupport", "frontier-ballot-support", "bool", "false",
           "For the exact descending sorted-prefix graph, replace the pairwise "
           "XOR reduction of every lane's best (rank,sum) with a warp ballot of "
           "threads containing supported values, followed by indexed broadcasts "
           "of the last supported frontier pair. The warp leader then computes "
           "tau once and broadcasts that live scalar instead of repeating the "
           "same division in every lane. Descending order proves support is a "
           "prefix; unsupported graphs fail closed.">,
    Option<"crossRowFinalize", "cross-row-finalize", "bool", "false",
           "For one float-nondeterministic axis-0 sum of row statistics, write "
           "one contribution per row to compiler-owned scratch and finalize it "
           "in a second GPU launch. False retains one atomic add per row. The "
           "graph matcher fails closed for any other output topology.">,
    Option<"rawMomentVariance", "raw-moment-variance", "bool", "false",
           "For a proven two-reduction mean/centered-square graph, evaluate raw "
           "sum and raw sum-of-squares in one independent reduction group, then "
           "form the variance from the two summaries. This is an observation-"
           "changing execution state and is admitted only for the exact typed "
           "pattern; otherwise the pass fails closed.">,
    Option<"compactReductionTopology", "compact-reduction-topology", "bool", "true",
           "Select the compiler-owned compact shuffle/shared reduction topology. "
           "False retains the standard MLIR gpu.all_reduce collective for every "
           "row reduction while preserving the same read-once producer/epilogue "
           "graph. C-Fixed and Schedule-Only use false; IntentIR may search both "
           "concrete topologies under the same Q/D policy.">,
    Option<"predicatePureValueIf", "predicate-pure-value-if", "bool", "true",
           "Convert a runtime rank-0 pure tensor scf.if into eager arm "
           "dataflow plus intent.select before row-graph admission. The "
           "transformation is restricted to typed side-effect-free MLIR arms, "
           "does not specialize the predicate, and records its actual count "
           "in the concrete MLIR attributes.">,
    Option<"rowsPerBlock", "rows-per-block", "unsigned", "1",
           "Independent logical rows assigned to one block. 1 preserves the "
           "one-block-per-row lowering. Values 2, 4, 8, 16, or 32 enable either "
           "the fail-closed one-warp-per-row register path or a sorted-prefix "
           "mapping with one prefix warp and a wider output group per row. "
           "Unsupported graphs remain unmatched rather than falling back under "
           "this candidate identity.">,
    Option<"reductionParts", "reduction-parts", "unsigned", "1",
           "Independent blocks assigned to one logical row for a true multi-stage "
           "single-reduction lowering. Values 2--128 (powers of two) emit partial, "
           "finalize, and—when a full-row result exists—epilogue launches sharing "
           "compiler-managed global scratch. 1 preserves the single-launch paths. "
           "Unsupported graphs fail closed; no atomic or Python-emitted fallback.">,
    Option<"accumulatorChains", "accumulator-chains", "unsigned", "1",
           "Independent per-thread accumulator chains used by the register row "
           "reduction before the subgroup/workgroup tree. Legal values are 1, 2 "
           "or 4. Values above one change the floating reduction tree, are Full-"
           "only execution-domain candidates, and fail closed outside the single-"
           "launch register path.">,
  ];
}

def ConvertIntentPaddedRowReductionToGPU
    : Pass<"convert-intent-padded-row-reduction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a masked last-axis row dot followed by zero padding to "
                "one gpu.launch.";
  let description = [{
    Recognizes the source-neutral structural family

      masked A/B [1,M,P,N] -> f32 multiply -> sum(axis=3)
      -> zero pad the P axis to Q

    where the two dense-load masks are the same conjunction of `p < validP`
    and `n < validN`.  It emits one destination-passing GPU kernel with one
    or more blocks per M row and one thread per padded output column.  Full
    short rows use aligned vector loads; runtime-short rows use guarded scalar
    loads. Q-P threads write the authored zero pad. The matcher uses only
    typed Intent/tensor dataflow and static shapes, never a function/kernel
    name.  Non-matching graphs remain untouched for other registered families
    or the generic lowering.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block; 0 selects the smallest warp-multiple power of "
           "two covering Q. Smaller values split Q across blocks; values must "
           "be warp multiples not exceeding 1024.">,
    Option<"vectorizeFullFeatures", "vectorize-full-features", "bool", "true",
           "Use aligned vector loads on the full-feature scalar-thread path.">,
    Option<"promoteRank0Scalars", "promote-rank0-scalars", "bool", "true",
           "Capture valid-row/feature scalars by value from the host stub; "
           "false loads their rank-0 buffers inside the GPU kernel.">,
    Option<"reductionTopology", "reduction-topology", "std::string",
           "\"serial-row\"",
           "Concrete dot-reduction mapping. serial-row assigns one thread to "
           "one padded output row and accumulates its feature strip serially. "
           "subgroup-feature assigns one subgroup to a row and uses subgroup "
           "shuffle reduction over contiguous feature lanes. The latter is an "
           "observation-changing compiler state and must pass the backend Q/D "
           "gate before performance selection.">,
    Option<"rowsPerSubgroup", "rows-per-subgroup", "unsigned", "1",
           "Number of independent output rows accumulated by each subgroup in "
           "the subgroup-feature topology. Legal values are 1, 2, 4, or 8; "
           "serial-row requires 1. This changes launch granularity and exposes "
           "row-level ILP without changing an individual reduction tree.">,
    Option<"lanesPerRow", "lanes-per-row", "unsigned", "32",
           "Power-of-two subgroup width used by subgroup-feature reduction. "
           "Legal values are 4, 8, 16, or 32 and must divide a hardware warp. "
           "Narrow subgroups let one warp reduce several independent rows.">,
  ];
}

def ConvertIntentElementwiseToGPU
    : Pass<"convert-intent-elementwise-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a pure-elementwise intent func to one read-once "
                "block-cooperative gpu.launch.";
  let description = [{
    The sibling of `convert-intent-row-reduction-to-gpu` for funcs whose body is
    a straight-line graph of `intent.map_unary`/`intent.map_binary` with NO
    `intent.reduce` — pure elementwise activations (dyt = tanh(alpha*x)*gamma,
    swiglu = (a*sigmoid(a))*b, geglu = the tanh-approx GELU gate * b). It
    rewrites such a func in place to a memref-arg function holding a single
    `gpu.launch`: normally one block per row, the feature dim split over the
    block's threads, each thread reading its columns **once** into SSA registers
    (read-once = HBM parity with the library kernel), per-row scalars hoisted
    once, and a multi-output epilogue writing each result from the register tile.
    Contiguous strips use vector<4xf32> when divisible by four, otherwise
    vector<2xf32> when divisible by two; scalar lowering remains the fallback.
    A fail-closed `flatten-contiguous` schedule is available when every tensor
    input/result is an equal-shape contiguous Row value and there are no
    Col/broadcast/indexed views: it maps the complete tensor as one linear vec4
    domain, reducing block count without changing the normalized op DAG.
    There is NO `gpu.all_reduce` — nothing is reduced. General over the map DAG
    (NO per-kernel recognizer); the two passes' match() predicates are disjoint
    (this pass returns failure on ANY reduce; row-reduction returns failure with
    no reduce), so they compose in either order.

    Static shapes only. The resulting memref+gpu.launch module goes straight to
    the device pipeline (`lower_memref_module_to_kernels`), bypassing
    bufferization. Transcendentals (`math.exp`/`math.tanh`/...) are rewritten to
    NVVM approx intrinsics by the `-lower-gpu-math-to-nvvm` target peephole.
    Rank-2 f16/bf16 results and Row inputs use half storage with per-map f32
    computation (`extf` before the scalar op, `truncf` at that op's declared
    result boundary), preserving typed intermediate rounding. Mixed f32
    Col/Scalar broadcasts are admitted without changing the result storage type.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the feature dim); 0 = auto "
           "(largest power-of-two <= min(N,256) dividing N). The PK harness "
           "passes the library kernel's geometry.">,
    Option<"flattenContiguous", "flatten-contiguous", "bool", "false",
           "Map an admitted equal-shape contiguous Row-only graph over one "
           "linear vec4 domain instead of launching one block per row.">,
    Option<"flatVectorWidth", "flat-vector-width", "unsigned", "4",
           "Vector width for flatten-contiguous (2, 4, or 8). Ignored by the "
           "default row schedule.">,
    Option<"rowVectorWidth", "row-vector-width", "unsigned", "0",
           "Vector width for the ordinary row elementwise schedule. 0 lets "
           "the pass choose the largest legal contiguous packet; 1 forces "
           "scalar row loads/stores and 2 or 4 requests an explicit packet. "
           "The request is checked against the exact static row tile and the "
           "same MLIR path, so it is a real schedule coordinate.">,
    Option<"gridStrideUnroll", "grid-stride-unroll", "unsigned", "1",
           "Static elements-per-thread unroll for the scalar flat grid-stride "
           "state. Legal values are 1, 2, 4, or 8. Each block covers "
           "block-threads*unroll contiguous logical elements before advancing "
           "by the complete grid stride; inactive tail elements are guarded. "
           "This is a target schedule coordinate, not a graph transformation.">,
    Option<"logicalTileElements", "logical-tile-elements", "unsigned", "0",
           "Logical contiguous elements assigned to one block before target "
           "thread mapping; 0 selects the compiler default. A nonzero tile must "
           "be a power of two, be divisible by block-threads, and assign at most "
           "32 scalar elements to each thread. This coordinate preserves a C "
           "tile independently from CUDA thread count and enables true retile.">,
    Option<"promoteUniformScalars", "promote-uniform-scalars", "bool", "false",
           "Capture rank-0 or singleton broadcast inputs by value from the "
           "host launch stub. False retains the historical per-thread device "
           "load as the schedule-only control.">,
    Option<"pairedHeadReuse", "paired-head-reuse", "bool", "false",
           "For an admitted paired last-axis rotation, map threads over "
           "(batch, sequence, head-group, rotary pair), load the shared angle "
           "once, and reuse it for the Q/K outputs in that bounded head group. "
           "This Full-only traversal preserves pair-contiguous memory access "
           "and each output's arithmetic order.">,
    Option<"pairedHeadGroupSize", "paired-head-group-size", "unsigned", "4",
           "Number of Q/K heads visited by one admitted paired-head-reuse "
           "lane. Legal values are 2, 4, 8, or 16. This changes the traversal/"
           "reuse topology and is searched only by Full; it is ignored when "
           "paired-head-reuse is false.">,
    Option<"pairedDirectPlane", "paired-direct-plane", "bool", "false",
           "For an admitted three-plane paired rotation, derive the selected "
           "plane index from the certified static sections and load only that "
           "plane. The original select/add arithmetic is retained, while "
           "unobserved read-only plane loads are eliminated.">,
    Option<"cancelShiftedExp", "cancel-shifted-exp", "bool", "false",
           "Recognize the typed execution-domain pattern exp(x-s)*exp(s) and "
           "replace it with exp(x) before the ordinary elementwise lowering. "
           "This is an observation-changing algebraic candidate: it is admitted "
           "only when the whole live graph becomes a pure elementwise graph and "
           "the backend Q/D correctness policy accepts the resulting numeric "
           "evaluation. The default is false.">,
    Option<"rowMaskZeroFill", "row-mask-zero-fill", "bool", "false",
           "For the exact typed graph cast(mask[M]:i1)->f32 followed by the "
           "pointwise contraction X[M,N]*mask[M], emit a uniform row branch. "
           "False preserves cast-and-multiply; true skips X loads for false rows "
           "and writes zero. The latter is an execution-domain Full state and "
           "must pass Q/D because NaN/Inf times zero is observably different.">,
    Option<"tanhApproximation", "tanh-approximation", "std::string",
           "\"stable-ex2\"",
           "Concrete f32 tanh evaluation state. stable-ex2 uses the existing "
           "overflow-safe ex2/reciprocal expansion. native-ptx marks tanh ops "
           "for the single-instruction PTX tanh.approx.f32 lowering. native-ptx "
           "is observation-changing and must pass Q/D before timing.">,
  ];
}

def ConvertIntentGatherToGPU
    : Pass<"convert-intent-gather-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower intent.gather / intent.scatter data-dependent addressing "
                "and an admitted two-level gather chain to gpu.launch.";
  let description = [{
    The C3 data-dependent-addressing pass. Recognizes a `func.func` whose body is
    exactly one `intent.gather` or one `intent.scatter` over rank-2 data producing
    a rank-2 result, and rewrites it in place to ONE block-cooperative
    `gpu.launch`: one block per output/scatter row, the free (column) axis split
    over the block's threads (T threads, K = cols/T columns each, the register
    tile unrolled over K). The new mechanism is a data-dependent address: the
    per-element load/store coordinate is read at runtime from the integer `index`
    memref (a `memref.load` of `index`, `arith.index_cast` to an index, then used
    as a `memref.load`/`memref.store` coordinate on `data`) — NOT an affine
    function of the iteration.

    * gather (`along_axis` = torch.gather, `take` = index_select/embedding
      row-gather): destination-passing — the appended `[Ro,Co]` output memref is
      written from the index-computed `data` load. `index` rank 1 (`take`) or 2
      (`along_axis`); `axis` selects which output coordinate the index replaces.
    * scatter (`along_axis`, reduce `none`/`add`): IN-PLACE on the (caller-cloned)
      `data` memref — one launch per `src`/`index` element does a data-dependent
      `memref.store` (reduce `none`) or legacy `memref.atomic_rmw addf` (implicit
      reduce `add`). An explicit atomic scatter uses the scoped LLVM/NVVM
      operation and ordering fences, and no output arg is appended.

    Correctness-only (no bank-conflict-free guarantee needed): the index stream is
    exact so verification is bit-faithful (rel < 1e-3). Static shapes, cols % T ==
    0 (tail-mask is a follow-up). Run BEFORE `convert-intent-to-linalg`; the
    resulting memref+gpu.launch module goes straight to the device pipeline
    (`lower_memref_module_to_kernels`), bypassing bufferization.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the free/column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
    Option<"composeGatherChain", "compose-gather-chain", "bool", "false",
           "For an exact take(axis=0) -> along_axis(axis=1) gather chain, "
           "compose the two runtime index maps and load only the final scalar. "
           "False keeps the deterministic eager first-gather row topology. "
           "This structural state is searched only by Full.">,
  ];
}

def ConvertIntentRaggedToGPU
    : Pass<"convert-intent-ragged-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a local ragged outer/member query to one gpu.launch.";
  let description = [{
    Recognizes a straight-line function containing one `intent.ragged` followed
    by either `intent.ragged_outer` or `intent.ragged_member`, with the queried
    value returned directly.  Offsets and indices may be normal pointer
    arguments or deterministic rank-one `arith.constant` carriers; a runtime
    selector remains a pointer argument, while an authorized constexpr
    selector is materialized as a device-side constant.
    The tuple descriptor never crosses the target ABI.
    `ragged_member` uses a destination-passing ranked memref whose allocation
    size and dynamic extent are resolved by the host from the authored offsets
    and selector.  The tuple is not an ABI value: after GPU lowering with
    kernel-bare-ptr-calling-convention=false, the device receives the flattened
    descriptor fields (allocated/aligned pointers, offset, size, and stride).
    Every store is guarded by the runtime slice length.  Functions outside this
    closed shape are left for another explicit lowering and are not silently
    erased.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "256",
           "Threads used by a ragged_member copy launch.">,
  ];
}

def ConvertIntentLayoutToGPU
    : Pass<"convert-intent-layout-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single structural-layout op (concat/pad/tile/flip/triu) "
                "to one block-cooperative gpu.launch.";
  let description = [{
    The C4 structural-layout pass. Recognizes a `func.func` whose body is exactly
    one `intent.concat` / `intent.pad` / `intent.tile` / `intent.flip` /
    `intent.triu` over rank-2 operands producing a rank-2 result, and rewrites it
    in place to ONE block-cooperative `gpu.launch`: one block per OUTPUT row, the
    output column axis split over the block's threads (T threads, K = cols/T
    columns each, unrolled). Destination-passing: the appended `[Ro,Co]` output
    memref is written; the inputs are passed as memrefs in operand order.

    The new mechanism is pure DATA MOVEMENT — each output element's SOURCE
    coordinate is a STATIC structural function of the output coord (NOT the
    data-dependent runtime index of gather), then a `memref.load` (or the pad
    constant):
    * concat: piecewise by the cumulative axis offsets — the output coord is owned
      by one input; a per-input `scf.if(off <= coord < off+extent)` loads that
      input at the offset-shifted coord and stores.
    * pad: `scf.if(in-bounds)` yields `input[coord-lo]` else the `value` constant.
    * tile: `input[coord mod input_extent]` via `arith.remui` (no guard).
    * flip: `input[N-1-coord]` on the flipped axes (affine remap).
    * triu: `arith.select(col >= row + k, input[row,col], 0)`.

    Correctness-only (pure movement -> bit-faithful, rel < 1e-3). Static shapes,
    cols % T == 0 (tail-mask is a follow-up). Run BEFORE
    `convert-intent-to-linalg`; the memref+gpu.launch module goes straight to the
    device pipeline, bypassing bufferization.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the output column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
  ];
}

def ConvertIntentRngToGPU
    : Pass<"convert-intent-rng-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a seeded-RNG intent func (a straight-line elementwise "
                "graph containing >=1 intent.rand) to one block-cooperative "
                "gpu.launch with an inline philox4x32-10 counter RNG.";
  let description = [{
    The C5 stateful-RNG pass. Recognizes a `func.func` whose body is a
    straight-line SCALAR-elementwise graph over rank-2 `[Ro,Co]` operands
    containing at least one `intent.rand` — the RNG family: `rand`/`rand_like`
    (one rand), `dropout` (`select(rand >= p, x/(1-p), 0)`), `randn`/`normal`
    (Box-Muller of two rands), `uniform(a,b)` (`a + (b-a)*rand`) — and rewrites
    it in place to ONE block-cooperative `gpu.launch`: one block per OUTPUT row,
    the column axis split over the block's threads (T threads, K = cols/T columns
    each, unrolled). Destination-passing: the appended `[Ro,Co]` output memref is
    written; any tensor inputs (e.g. dropout's `x`) are passed as memrefs.

    The new mechanism is a STATELESS counter-based RNG: each output element at
    `(row,col)` computes `philox4x32-10(seed, row*Co + col)` entirely in `arith`
    (10 rounds of integer mul-hi/xor over the public philox constants; the seed
    splits into the two key words, the element index is the counter), maps the
    first output word to a 24-bit-mantissa uniform[0,1) f32, then runs the
    composed scalar transform (map_unary / map_binary incl. compare / select /
    const, reusing the C0/M3 scalar emitters) and stores. No global RNG state, no
    visited-order dependence — same seed -> bit-identical, different seed ->
    decorrelated.

    Verification is matched-seed against the same philox in the interpreter
    (determinism / self-consistency) and STATISTICAL against torch/flag_gems
    (uniform mean~0.5 var~1/12, normal mean~0 var~1, dropout keeps ~(1-p)) — the
    philox counter layout does NOT match Triton's `tl.rand` bit-for-bit, so
    library parity is distributional, not byte-exact. Static shapes, cols % T ==
    0 (tail-mask is a follow-up). Run BEFORE `convert-intent-to-linalg`; the
    memref+gpu.launch module goes straight to the device pipeline.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (block size over the output column axis); 0 = auto "
           "(largest power-of-two <= min(cols,256) dividing cols).">,
  ];
}

def ConvertIntentArgReduceToGPU
    : Pass<"convert-intent-arg-reduce-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower an argmax/argmin intent func (one intent.arg_reduce over "
                "the last axis) to one block-cooperative gpu.launch returning the "
                "extremum's INDEX.";
  let description = [{
    The C6 order-statistics arg-reduce pass. Recognizes a `func.func` whose body
    is a single `intent.arg_reduce` (`argmax`/`argmin`) over a rank-2 `[M,N]`
    static-f32 row (`axis = 1`), producing the integer INDEX of the row extremum
    (`[M]` or keepdims `[M,1]` i32). It lowers to ONE block-cooperative
    `gpu.launch`: one block per row, the feature axis split over T threads
    (K = N/T contiguous columns each). Each thread scans its K columns for a local
    `(best_val, best_idx)` pair (ascending columns + a STRICT compare so ties keep
    the lowest column, matching torch's first-index tie-break); a power-of-two
    shared-memory tree then reduces the T per-thread pairs keeping the extremum
    (ties -> the lower index). Thread 0 stores the winning integer index.

    The new mechanism vs. `intent.reduce` (value-only, one `gpu.all_reduce`) is
    the PAIR carried through the reduction — an argmax/argmin tree over
    `(val, idx)` in two dynamic-shared buffers — and an INTEGER output. Verified
    exact-index vs torch.argmax/argmin and the real flag_gems kernel. Static
    shapes, N % T == 0, T a power of two.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the feature axis; 0 = auto.">,
  ];
}

def ConvertIntentSortToGPU
    : Pass<"convert-intent-sort-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower an intent.sort/topk/argsort func (one intent.sort over the "
                "last axis) to one block-cooperative bitonic-sort gpu.launch.";
  let description = [{
    The C6 HARD-tier order-statistics pass — the last order-stats gap. Recognizes
    a `func.func` whose body is a single `intent.sort` over a rank-2 `[M,N]`
    static-f32 row (`axis = 1`) producing sorted `values` `[M,W]` f32 and/or
    argsort `indices` `[M,W]` i32 (`W <= N`; `W < N` = topk). It lowers to ONE
    block-cooperative BLOCK BITONIC SORT `gpu.launch`: one block per row, `T`
    threads.

    Each block loads its row into a `[P]`-element dynamic-shared value buffer,
    padding `N -> P` (the next power of two) with a `+inf` (ascending) / `-inf`
    (descending) sentinel so the pad sorts to the END. It then runs the bitonic
    network: for `k = 2,4,...,P` (outer) and `j = k/2,...,1` (inner),
    `log2(P)*(log2(P)+1)/2` stages, each a `gpu.barrier` then a grid-stride sweep
    where the thread owning the lower index of each pair `(i, i^j)` compare-swaps
    under the direction bit `(i & k)==0` (inverted for `descending`). After the
    network thread `t` writes `shared[t] -> values[row,t]` for the first `W`. For
    the argsort `indices` an `i32` index buffer (init `idx[i]=i`) is swapped in
    lockstep and written the same way.

    The new mechanism vs. the value-only reductions/scans is the O(N log^2 N)
    barrier-separated compare-exchange NETWORK in shared memory (not a single
    tree/scan). Correctness-only: values exact vs torch.sort/topk + flag_gems; the
    index permutation reproduces the sorted values (ties unstable, like torch).
    Static shapes, `axis` last, `T` a power of two dividing `P`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
    "::mlir::math::MathDialect",
    "::mlir::NVVM::NVVMDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block (power of two dividing P=next_pow2(N)); 0 = auto.">,
  ];
}

def ConvertIntentPoolToGPU
    : Pass<"convert-intent-pool-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.pool (avg / max[+indices]) over NCHW to "
                "one block-cooperative gpu.launch (windowed reduce).";
  let description = [{
    The C7 windowed-reduce pass. Recognizes a `func.func` whose body is exactly
    one `intent.pool` over a rank-4 `[N,C,H,W]` f32 input producing `[N,C,OH,OW]`
    (avg) or `[N,C,OH,OW]` + an i64 argmax-index `[N,C,OH,OW]` (max_pool2d_with_
    indices), and rewrites it in place to ONE block-cooperative `gpu.launch`:
    grid = (N*C, OH, 1) (one block per channel-plane row of the output), block =
    (T,1,1) threads over the OW output columns (K = OW/T columns each, unrolled).

    The new mechanism vs. `intent.reduce` (a full-axis tree/all_reduce) is a
    BOUNDED per-output-cell reduction over the `kH x kW` window: each thread walks
    the static window `(ih,iw) = (oh*sH - padH + kh, ow*sW - padW + kw)`, guards
    each source coord with an in-bounds `scf.if` (out-of-bounds = skip), and
    accumulates. `avg` sums (padded cells contribute 0) and divides by `kH*kW`
    (torch's default count_include_pad). `max` keeps the running max and — when the
    `indices` output is present — the argmax FLAT index `ih*W + iw` carried as an
    i64 PAIR alongside the value (strict `>` update = first-index tie-break,
    matching torch/flag_gems). Destination-passing: the appended output memref(s)
    (`[N,C,OH,OW]` f32, then the i64 index memref for max-with-indices) are
    written; the input is passed as a rank-4 memref.

    Correctness-only (bit-faithful vs torch + flag_gems: rel < 1e-3 values,
    exact indices). Static shapes, OW % T == 0. Run BEFORE
    `convert-intent-to-linalg`; the memref+gpu.launch module goes straight to the
    device pipeline, bypassing bufferization.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the OW output-column axis; 0 = auto.">,
  ];
}

def ConvertIntentUpsampleToGPU
    : Pass<"convert-intent-upsample-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.upsample (nearest or bicubic) over NCHW "
                "to one block-cooperative gpu.launch.";
  let description = [{
    The C7 interpolation pass. Recognizes a `func.func` whose body is exactly one
    `intent.upsample{mode="nearest", scale=[sh,sw]}` or
    `intent.upsample{mode="bicubic", scale=[sh,sw]}` over a rank-4
    `[N,C,H,W]` f32 input producing
    `[N,C,OH,OW]`, and rewrites it in place to ONE block-cooperative `gpu.launch`:
    a flat grid-stride launch over the complete output, with block size `T`.

    For `nearest`, each output element's source coordinate is an affine `floor`
    index-remap — `ih = min(floor(oh*IH/OH), IH-1)` and
    `iw = min(floor(ow*IW/OW), IW-1)`. For `bicubic`, the pass evaluates the
    closed four-by-four cubic-convolution neighborhood with `align_corners=False`.
    Both modes use destination-passing: the appended `[N,C,OH,OW]` output memref
    is written and the input is a rank-4 memref.

    Correctness-only. Static shapes. Run BEFORE `convert-intent-to-linalg`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the OW output-column axis; 0 = auto.">,
  ];
}

def ConvertIntentPolarToGPU
    : Pass<"convert-intent-polar-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.polar (abs,angle -> real,imag) to one "
                "block-cooperative gpu.launch (elementwise complex build).";
  let description = [{
    The C7 complex-dtype pass. Recognizes a `func.func` whose body is exactly one
    `intent.polar` over rank-2 `[M,N]` f32 `abs`/`angle` producing two `[M,N]` f32
    results `real`/`imag`, and rewrites it in place to ONE block-cooperative
    `gpu.launch`: one block per row, the column axis split over T threads
    (K = N/T columns each, unrolled). Destination-passing: two appended `[M,N]`
    output memrefs (`real` then `imag`) are written.

    Elementwise: `real = abs*cos(angle)`, `imag = abs*sin(angle)`, reusing the C0
    `cos`/`sin` scalar emitters (`math.cos`/`math.sin` -> `cos.approx`/`sin.approx`
    NVVM hardware ops via the `-lower-gpu-math-to-nvvm` peephole). The complex
    tensor is represented as the trailing real/imag PAIR (two real results),
    compared against `torch.view_as_real(torch.polar(...))` (rel < 1e-3). Static
    shapes, N % T == 0. Run BEFORE `convert-intent-to-linalg`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the column axis; 0 = auto.">,
  ];
}

def ConvertIntentConvToGPU
    : Pass<"convert-intent-conv-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.conv (conv1d/2d/3d, forward) over "
                "NC*spatial to one block-cooperative gpu.launch (direct "
                "windowed-weighted accumulation).";
  let description = [{
    The C7 convolution pass. Recognizes a `func.func` whose body is exactly one
    `intent.conv` over a rank `2+R` (`R` = 1/2/3) f32 input `[N,Cin,*spatial]`
    with a `[Cout,Cin/groups,*kernel]` weight (+ optional `[Cout]` bias)
    producing `[N,Cout,*out_spatial]`, and rewrites it in place to ONE
    block-cooperative `gpu.launch`: grid = (N*Cout, prod(outer out-spatial), 1)
    (one block per output row = fixed `(n,co,*outer)`), block = (T,1,1) threads
    over the innermost out-spatial axis (K = O_last/T columns each, unrolled).

    The mechanism EXTENDS the `intent.pool` windowed per-output-cell reduce with
    an input-channel sum and a per-tap weight multiply (direct
    windowed-weighted-accumulation, no im2col): for output cell `(n,co,*o)`,
    `acc = bias[co] + sum_{ci_local} sum_{*k} in[n, g*(Cin/groups)+ci_local,
    *(o*stride - pad + k*dil)] * w[co, ci_local, *k]`, guarding every source
    spatial coord with a single in-bounds `scf.if` (out-of-bounds = zero pad).
    Channel groups pick the input-channel slice via `g = co/(Cout/groups)`. The
    accumulation core is parameterized by the spatial rank `R`, so conv1d (R=1),
    conv2d (R=2) and conv3d (R=3) share one code path (extra kernel/window
    loops). Destination-passing: the new signature is
    `(input, weight[, bias]) ++ [output]` memrefs.

    Correctness-only (rel < 1e-3 vs torch + flag_gems; fp accumulation order).
    Static shapes, O_last % T == 0. Run BEFORE `convert-intent-to-linalg`.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the innermost out-spatial axis; 0 = auto.">,
  ];
}

def ConvertIntentConvToGPUMMA
    : Pass<"convert-intent-conv-to-gpu-mma", "::mlir::ModuleOp"> {
  let summary = "Lower a single conv2d intent.conv as an IMPLICIT GEMM onto the "
                "nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16) tensor-core path.";
  let description = [{
    The tensor-core sibling of `convert-intent-conv-to-gpu` (the naive direct
    windowed f32-accumulation path). Recognizes a `func.func` whose body is
    exactly one forward conv2d `intent.conv` (rank-4 `[N,Cin,H,W]` f32 input,
    `[Cout,Cin,kH,kW]` weight, optional `[Cout]` bias, `groups == 1`) and rewrites
    it as an IMPLICIT GEMM `Y = im2col(X) . W^T`:

      M = N*OH*OW (output pixels)   N_gemm = Cout   K = Cin*kH*kW

    The im2col matrix is NEVER materialized: each `(m-row, k-col)` of the A shared
    tile is mapped back to an input pixel by the SAME OOB-guarded conv address
    math as the naive pass (`ih = oh*stride - pad + kh*dil`, out-of-bounds = zero
    pad) emitted as the smem tile-fill load predicate. W is already a real
    `[Cout, Cin*kH*kW]` row-major matrix so its tile-fill is a plain gather. Both
    operands are cast f32 -> f16 during the fill (fp16 input mode: rel ~1e-2 vs
    fp32 torch, matching the TF32 cuDNN reference). The register-level contraction
    (`nvgpu.ldmatrix` + `nvgpu.mma.sync` m16n8k16 -> per-warp `vector<2x2xf32>`
    accumulators carried across the K loop as `scf.for` iter_args) and the padded
    f16 shared tile geometry are reused verbatim from
    `convert-intent-contraction-to-gpu-mma`; the store scatters each accumulator
    element back to the NCHW output (`row -> (n,oh,ow)`, `col -> co`) with an
    optional bias add.

    stride/padding/dilation are fully general (folded into the address math).
    Static shapes, `groups == 1`, `K % 32 == 0` (else the pass declines and the
    naive `convert-intent-conv-to-gpu` fallback handles it); M and N are padded
    via a ceil grid + guarded store. Bare-ptr ABI; the result goes to the device
    pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "block C-tile cols BN (each warp owns BN/8 n8 mma column tiles).">,
    Option<"tf32", "tf32", "bool", /*default=*/"false",
           "precise input mode: keep f32 operands, round to TF32 and contract "
           "on the m16n8k8 nvgpu.mma.sync (tf32Enabled) tensor-core shape "
           "(rel ~1e-3 vs fp32 torch, cuDNN-matching). Default false = the fast "
           "fp16 m16n8k16 path (rel ~1e-2).">,
  ];
}

def ConvertIntentCompactToGPU
    : Pass<"convert-intent-compact-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single dynamic-shape compaction op (nonzero / "
                "masked_select / masked_scatter) to one block-cooperative "
                "prefix-sum gpu.launch.";
  let description = [{
    The C8 dynamic-shape-output pass — the ONE genuinely-new mechanism vs every
    prior cluster: a data-dependent output EXTENT. Recognizes a `func.func` whose
    body is exactly one `intent.nonzero` / `intent.masked_select` /
    `intent.masked_scatter` and rewrites it in place to ONE block-cooperative
    `gpu.launch` (grid = (1,1,1), block = (T,1,1)).

    The compaction primitive is an EXCLUSIVE PREFIX-SUM of the flattened 0/1
    mask: the flattened `Ntotal = prod(shape)` elements are split into T
    contiguous SEGMENTS (Ntotal % T == 0, seg = Ntotal/T); thread `t` first counts
    the true elements in ALL prior segments `[0, t*seg)` — its exclusive-prefix
    base slot — then re-scans its own segment `[t*seg, (t+1)*seg)` ascending,
    emitting each true element at the running slot (`base`, `base+1`, ...). Because
    the segments are contiguous ranges of the row-major flat index and each is
    scanned ascending, the global write order is EXACT row-major (matches
    torch.nonzero / torch.masked_select). No shared memory / barriers (each thread
    is independent); the last thread writes the total `nnz` to a `count` buffer.

    * nonzero (`cond`): dest-passing `(cond, out[MAXNNZ,ndim] i64, count[1] i64)`.
      Each true flat index `i` is unraveled to its `ndim`-vector coord (a divmod
      chain) and written as `out[slot, 0..ndim)`; `ndim = cond.rank`.
    * masked_select (`data`,`mask`): dest-passing `(data, mask, sel[MAXNNZ],
      count[1] i64)`. `sel[slot] = data[coord]` at each true position.
    * masked_scatter (`data`,`mask`,`src`): dest-passing `(data, mask, src, out)`,
      out DATA-SHAPED (static). `out[coord] = mask[coord] ? src[slot] :
      data[coord]` — the prefix-sum indexes `src`; no dynamic extent, no count.

    The runtime extent (`nnz`) flows HOST-side: the caller sizes the `[nnz,*]`
    allocation from a reduce over the mask BEFORE the launch (two-step), OR passes
    a MAX-sized buffer + reads the kernel's `count` and slices `[:nnz]` on the host
    (single-launch). Either way the destination leading dim is concrete at launch —
    orchestrate never allocates a dynamically-sized device buffer. A truth test of
    `!= 0` accepts f32 / integer / i1 masks. Correctness-only (bit-faithful, exact
    row-major order). Static shapes, Ntotal % T == 0, T a power of two. Run BEFORE
    `convert-intent-to-linalg`; the memref+gpu.launch module goes straight to the
    device pipeline.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"blockThreads", "block-threads", "unsigned", "0",
           "Threads per block over the flattened element axis; 0 = auto "
           "(largest power-of-two <= min(Ntotal,256) dividing Ntotal).">,
  ];
}

def ConvertIntentContractionToGPU
    : Pass<"convert-intent-contraction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to one tiled-K "
                "tensor-core gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is exactly one
    `intent.contraction{"mk,nk->mn"}` (A[M,K] f16, W[N,K] f16 -> Y[M,N] f32) and
    rewrites it in place to ONE tiled-K tensor-core `gpu.launch`: BM=BN=64,
    BK=32, a 4x4 grid of m16n16k16 WMMA tiles over 4 warps, with the per-warp
    fp32 accumulator fragments carried across the K loop as `scf.for` iter_args
    and the A/W sub-tiles staged through `gpu.dynamic_shared_memory`. The
    contraction is realized by NVIDIA WMMA (`gpu.subgroup_mma_*` ->
    wmma.mma.sync on sm_120a), NOT scalarized to a `linalg.generic` MAC.

    The B operand (W[N,K]) is loaded transposed so "mk,nk->mn" is exact:
    C[m,n] += sum_k A[m,k]*W[n,k]. Static shapes only, M%64==N%64==K%32==0
    (non-divisible tail-mask is a documented follow-up). Run BEFORE
    `convert-intent-to-linalg` (which would otherwise scalarize the contraction
    for the CPU/oracle leg). The resulting memref+gpu.launch module goes to the
    device pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`, so
    the gpu->nvvm conversion can legalize an scf.for carrying !gpu.mma_matrix
    iter_args).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"3",
           "cp.async multistage pipeline depth (N buffers; effective N is "
           "clamped to the K-tile count). N=1 disables prefetch overlap.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis, one "
           "16-row strip each). Falls back to 64 for shapes M%BM!=0.">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"128",
           "block C-tile cols BN (each warp owns BN/16 column fragments; wider "
           "BN raises A-load reuse). Falls back to 64 for shapes N%BN!=0.">,
  ];
}

def ConvertIntentContractionToGPUMMA
    : Pass<"convert-intent-contraction-to-gpu-mma", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to one tiled-K "
                "tensor-core gpu.launch via nvgpu.ldmatrix + nvgpu.mma.sync.";
  let description = [{
    The `nvgpu.mma.sync` sibling of `convert-intent-contraction-to-gpu` (which
    uses the opaque `gpu.subgroup_mma_*` / `!gpu.mma_matrix` WMMA path). Same
    recognizer (one `intent.contraction{"mk,nk->mn"}`, matching A/W storage in
    f16 or bf16 -> Y[M,N] f32, optional bias+activation epilogue) and the SAME cp.async
    multistage staging, but the contraction is realized by the register-level
    `nvgpu.ldmatrix` (shared->register warp-cooperative fragment load) +
    `nvgpu.mma.sync` (m16n8k16 warp MMA on `vector<...>` fragments with a KNOWN
    lane layout -> `vector<2x2xf32>` accumulator) primitive, NOT the opaque
    fragment. This is the parallel path explored for full compute-bound parity;
    the `gpu.mma_matrix` pass is left 100% intact as the default / fallback.

    Fragment recipe (validated against the MLIR --convert-vector-to-gpu
    use-nvgpu lowering on sm_120a): lane = tid%32, warp = tid/32, each warp owns
    a 16-row strip; A operand `ldmatrix aSh[warp*16 + lane%16, kOff + (lane/16)*8]`
    transpose=false -> vector<4x2xT>; W operand (natural [N,K] layout, the
    "nk" transpose folded into the load) `ldmatrix wSh[n*8 + lane%8, kOff +
    (lane/8)*8]` transpose=false -> vector<2x2xT>, where T is f16 or bf16; the
    m16n8 f32 accumulator
    stores back with the textbook layout d[i][j] -> row = lane/4 + i*8,
    col = (lane%4)*2 + j. Per-warp accumulator vectors ride the K loop as
    `scf.for` iter_args. Static shapes, M%BM==N%BN==K%BK==0. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`).

    The recognizer also consumes the canonical live-linear spelling in which
    A/W are explicitly widened from their matching 16-bit storage type to f32
    before the contraction, a same-storage rank-1 bias is widened and added in
    f32, and two same-storage results return the identical
    identity preactivation (`C` plus saved activation input). For this exact
    semantic pattern the known mma.sync lane layout enables a register epilogue:
    load/widen bias, add in f32, truncate once, and store the same value to both
    destinations. No f32 accumulator tile is spilled through shared memory.
    BF16 NVGPU fragments are completed by the target-side LLVM-20 compatibility
    pass `-lower-bf16-mma-to-nvvm`; other supported activation DAGs retain the
    generic shared-tile epilogue.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"3",
           "cp.async multistage pipeline depth (N buffers; clamped to the "
           "K-tile count). N=1 disables prefetch overlap.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numWarps = BM/16 warps tile the M axis).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"128",
           "block C-tile cols BN (each warp owns BN/8 n8 mma column tiles).">,
    Option<"blockK", "block-k", "unsigned", /*default=*/"32",
           "K staging step BK (must be a multiple of the m16n8k16 MMA_K=16; "
           "BK/16 mma sub-tiles per stage). Wider BK stages more K per cp.async "
           "wave; the smem budget caps nStages*(BM+BN)*BK*2 at ~100 KiB. "
           "Autotuner knob (was a hard constant 32).">,
    Option<"smemPadK", "smem-pad-k", "unsigned", /*default=*/"8",
           "Pad the shared A/W K-tile leading dim by this many f16 elements "
           "(swizzle-free bank-conflict avoidance for the ldmatrix reads). "
           "Must be a multiple of 8 (16-byte cp.async alignment) or 0 to "
           "disable. Autotuner knob.">,
    Option<"warpsN", "warps-n", "unsigned", /*default=*/"1",
           "Warp partitions along the block N tile. 1 keeps one 16-row warp "
           "responsible for the full block-N span. 2, 4, or 8 split that span "
           "between more warps, reducing accumulator registers per lane while "
           "preserving the same mma.sync and output tile.">,
    Option<"groupM", "group-m", "unsigned", /*default=*/"8",
           "Consecutive M tiles in the flat block-id L2 swizzle. This is a "
           "target schedule coordinate: larger groups retain one W/N tile "
           "across more M blocks, while smaller groups shorten A-tile reuse "
           "distance.">,
    Option<"splitKParts", "split-k-parts", "unsigned", /*default=*/"1",
           "Number of independent K partitions for the mma.sync contraction. "
           "1 emits the single-stage kernel. 2 or 4 emit an f32 partial "
           "scratch kernel followed by an ordered additive finalize/epilogue "
           "kernel; admitted only for the exact bias-plus-dual-output graph.">,
  ];
}

def ConvertIntentContractionToGPUMMAWS
    : Pass<"convert-intent-contraction-to-gpu-mma-ws", "::mlir::ModuleOp"> {
  let summary = "Lower a single intent.contraction matmul to a WARP-SPECIALIZED "
                "(producer/consumer) tiled-K nvgpu.mma.sync gpu.launch.";
  let description = [{
    M7g: the warp-specialized sibling of `convert-intent-contraction-to-gpu-mma`.
    Same recognizer and the same register-level `nvgpu.ldmatrix` +
    `nvgpu.mma.sync` (m16n8k16) contraction, but the block's warps are split into
    PRODUCER warps (the high warp ids -- loop the K-tiles issuing cp.async loads
    into an N-stage shared buffer, signal a per-stage `full` mbarrier, wait the
    matching `empty` mbarrier) and CONSUMER warps (the low warp ids, each owning
    a 16-row C strip as in M7f -- wait `full`, run ldmatrix + mma.sync into the
    register accumulator carried as scf.for iter_args, signal `empty`). The
    handshake uses the nvgpu.mbarrier primitives (init/arrive/try_wait.parity),
    which lower to mbarrier.*.shared.b64 on sm_120a. Phase parity is computed
    directly from the loop index ((ki/nStages)%2), no phase iter_args. The M7f
    pass is left 100% intact; this is gated by its own name. Static shapes,
    M%BM==N%BN==K%BK==0. Run BEFORE `convert-intent-to-linalg`; the result goes
    to the device pipeline (`lower_memref_module_to_kernels`, `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"4",
           "cp.async multistage pipeline depth (N shared buffers; clamped to "
           "the K-tile count).">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "block C-tile rows BM (numConsumerWarps = BM/16).">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "block C-tile cols BN (each consumer warp owns BN/8 n8 tiles).">,
    Option<"numProducerWarps", "num-producer-warps", "unsigned",
           /*default=*/"1",
           "warps dedicated to issuing cp.async loads (the rest are consumers).">,
  ];
}

def ConvertIntentLoopContractionToGPU
    : Pass<"convert-intent-loop-contraction-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a fused FlashAttention-forward intent.loop (online "
                "softmax + two contractions) to one tensor-core gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the FlashAttention forward as ONE
    `intent.loop` scanning the key axis: carries (running max `m[Sq,1]`, running
    sum `l[Sq,1]`, accumulator `O[Sq,D]`); the body computes `S = Q@Kj^T`
    (`intent.contraction{"qd,kd->qk"}`), a row-wise online softmax
    (`reduce{max}` -> `exp` -> `reduce{sum}`), and `O += P@Vj`
    (`intent.contraction{"qk,kd->qd"}`). It rewrites the func in place to ONE
    tiled tensor-core `gpu.launch`.

    The fusion SIDESTEPS the opaque-fragment rowmax/rescale problem by keeping
    the online-softmax state RESIDENT IN DYNAMIC SHARED MEMORY (S, P, O, m, l,
    alpha tiles) and the two matmuls on tensor cores: each K-tile does
    `S = Q@Kj^T` via WMMA (`gpu.subgroup_mma_*`) stored to a shared `[BM,BN]`
    f32 tile, a COOPERATIVE row-wise online softmax over that shared tile (per
    row: m_new = max(m, rowmax(scale*S)), alpha = exp(m-m_new), P = exp(S-m_new)
    cast to f16, l = l*alpha + rowsum), then `PV = P@Vj` via WMMA into a shared
    f32 tile and the cooperative update `O = O*alpha + PV`. The per-row alpha
    rescale and the rowmax/rowsum are plain cooperative shared-memory arithmetic
    over the BM rows -- NOT fragment-lane indexing. The scf.for over key tiles
    carries no iter_args (m/l/O live in shared, made visible across trips by
    `gpu.barrier`). Epilogue: `O/l` to global.

    Q/K/V are f16 (WMMA operands), the accumulator/softmax state is f32. Static
    shapes, BM=BN=64, D%16==0, Sq%64==0, Sk%64==0 (non-causal, no bias; tail-mask
    / causal / split-K are documented follow-ups). Run BEFORE
    `convert-intent-to-linalg`. The resulting memref+gpu.launch module goes to the
    device pipeline (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"1",
           "cp.async pipeline depth for the K/V key-tile loop (N rotating "
           "dynamic-shared K/V buffers; with N>1 the global->shared loads of "
           "tiles ki+1..ki+N-1 prefetch ahead of the WMMA+softmax compute of "
           "tile ki). Effective N is clamped to the K-tile count. DEFAULT 1: the "
           "win over M8a's scalar cooperative copy comes from the *vectorized* "
           "16-byte cp.async.cg staging (N=1, single buffer, ~30% faster), NOT "
           "from multistage prefetch -- this kernel is softmax/compute-bound, so "
           "N>=2 only adds async-copy + wait/barrier overhead and (D=64) grows "
           "shared 82688->99072 B (less L1), measurably REGRESSING latency. "
           "N=2 is the only depth >1 that still fits the 101376 B optin budget "
           "for D=64 (each extra stage adds BN*D*2*2 = 16384 B).">,
    Option<"capturedFlatOnly", "captured-flat-only", "bool", /*default=*/"false",
           "Require the exact flat split-K graph recognizer and reject the "
           "historical normalized intent.loop form. This makes automatic paper "
           "selection incapable of claiming the split-K fast-path ID for a "
           "different fused-attention graph.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "Query rows per CTA. Must be a positive multiple of the 16-row "
           "WMMA tile; the block uses block-m/16 consumer warps.">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "Key rows per staged tile. Must be a positive multiple of 16; a "
           "tail smaller than block-n is zero-filled and logit-masked.">,
    Option<"fastF32LSELog", "fast-f32-lse-log", "bool", /*default=*/"false",
           "For an f64 LSE destination whose online-softmax state is f32, "
           "evaluate log(l) with the native f32 fast path before extending to "
           "f64. The full f64 polynomial remains the default; this option is a "
           "paper fast path guarded by strict source-vs-generated P6 checks.">,
  ];
}

def ConvertIntentFlashMMAToGPU
    : Pass<"convert-intent-flash-mma-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower a fused FlashAttention-forward loop or the live B=H=1 "
                "flat rank-4 graph to one tensor-core gpu.launch with the "
                "online-softmax state REGISTER-RESIDENT via nvgpu.mma.sync.";
  let description = [{
    The register-resident `nvgpu.mma.sync` sibling of
    `convert-intent-loop-contraction-to-gpu` (M8a, which round-trips the
    online-softmax state O/S/P/m/l through dynamic shared). It recognizes either
    one `intent.loop` scanning the key axis (carries `m[Sq,1]`, `l[Sq,1]`,
    `O[Sq,D]`; body = `S=Q@Kj^T` -> online softmax -> `O+=P@Vj`) or the exact
    loop-free B=H=1 rank-4 graph emitted by the live FlashAttention P4 path:
    `bmhd,bnhd->nbhm`, max/sum on axis 0, LSE/TMP, f32-to-f16 P,
    `nbhm,bnhd->dbhm`, transpose, and f16 Out. The flat matcher is structural,
    never symbol-name based, and rejects extra mask/bias/tail operations. The
    rewrite keeps O/S/m/l RESIDENT IN REGISTERS so only Q/K/V touch shared:

    Per block (BM=BN=64, 4 warps, each warp a 16-row query strip):
      S = Q@Kj^T via nvgpu.ldmatrix + nvgpu.mma.sync (m16n8k16 f16->f32). The S
      accumulators are per-warp `vector<2x2xf32>` fragments with the KNOWN PTX
      lane layout `d[i][j] -> row = lane/4 + i*8, col = 8*tile + (lane%4)*2 + j`.
      Each lane therefore owns 2 query rows (lane/4 and lane/4+8); a row's BN key
      columns are spread across that lane (over the n8 tiles and j) and the 3
      other lanes of its quad (lane%4). Per-row rowmax/rowsum are computed with a
      quad `gpu.shuffle xor` (delta 1 then 2, width 32) -- NO shared round-trip.
      alpha = exp(m_old-m_new) per row; P = exp(scale*S-m_new) cast f16, all in
      registers. The P A-operand fragment for the PV matmul is assembled DIRECTLY
      from the S accumulators (the m16n8 acc layout == the m16n8k16 A-operand
      fragment layout, two consecutive n8 S-tiles -> one k16 A-subtile, NO
      shuffle, NO shared P). O accumulators (`vector<2x2xf32>`, carried across the
      key loop as `scf.for` iter_args) are rescaled by alpha (per-row scalar
      already lane-local) then `O += P@Vj` via mma.sync (V staged transposed in
      shared as the [D,Sk] B operand). Epilogue: `O/l` straight from registers to
      global.

    Q/K/V f16, accumulators f32. Static shapes, BM=BN=64, D%16==0, Sq%64==0,
    Sk%64==0. The legacy loop form remains rank-2 and supports the existing
    optional LSE/causal controls. The live flat form is non-causal B=H=1 and
    preserves all three destination outputs: Out f16, LSE f32, and TMP=1/l f32;
    its scale input is a rank-0 f32 memref loaded inside the launch. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`, the scf.for carries
    vector iter_args). M8a is left 100% intact (separate pass).
  }];
  let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"2",
           "cp.async software-pipeline depth for the K/V key-tile loop "
           "(N rotating dynamic-shared K/V buffers; clamped to the K-tile "
           "count). 1 = single-buffer (no multistage). N=2 hides the K/V "
           "global-load latency behind the mma.sync+softmax of the previous "
           "tile (~1.2x at >=512 keys); N>=3 plateaus and only wastes smem.">,
    Option<"emitLse", "emit-lse", "bool", /*default=*/"false",
           "Also emit the per-query log-sum-exp (LSE = m + log(l), the "
           "log-sum-exp of the SCALED scores) as an appended [Sq,1] f32 "
           "destination output. This is the statistic FlashAttention's backward "
           "(and flag_gems' flash_api) consume; it is free here because the "
           "online-softmax m (rowmax) and l (rowsum) are already register-"
           "resident per row. Written once per row (from the quad's tig==0 "
           "lane). R-FLASH-3 attention-coverage extension.">,
    Option<"causal", "causal", "bool", /*default=*/"false",
           "Apply causal (lower-triangular) masking: score S[q,k] is set to -inf "
           "when the GLOBAL key index exceeds the GLOBAL query index (assuming "
           "Sq==Sk diagonal alignment), so query q attends only to keys <= q -- "
           "the flag_gems _attn_fwd STAGE=3 diagonal mask. Applied to the scaled "
           "scores in the register-resident online softmax (each accumulator "
           "lane knows its (query-row, key-col) from the m16n8 PTX layout). This "
           "is the CORRECTNESS extension (every key tile is still visited + "
           "masked; the STAGE=1 fully-masked-tile skip is not applied). "
           "R-FLASH-3 attention-coverage extension.">,
    Option<"certifiedFullAxisOnly", "certified-full-axis-only", "bool",
           /*default=*/"false",
           "Require the exact batch/head-explicit full-axis graph materialized "
           "from a certified prefix/loop/epilogue J. This structural-search "
           "entry rejects the historical normalized loop and B=H=1 live graph "
           "so DirectLower cannot silently acquire the fused topology.">,
    Option<"blockM", "block-m", "unsigned", /*default=*/"64",
           "Query rows assigned to one CTA. Must be a positive multiple of the "
           "16-row MMA tile; changing it is a schedule choice and does not alter "
           "the certified key-scan recurrence.">,
    Option<"blockN", "block-n", "unsigned", /*default=*/"64",
           "Key rows consumed by one semantic scan/update step. The certified "
           "structured path requires this to equal intent.loop tile_hint; this "
           "prevents a backend schedule tile from silently changing the authored "
           "floating-point recurrence.">,
    Option<"directVMatrixLoad", "direct-v-matrix-load", "bool",
           /*default=*/"false",
           "Use ldmatrix.trans to form the PV B operand directly from natural "
           "shared V[key,feature]. False retains the canonical materialized V^T "
           "buffer; true is a Full structural state that removes that buffer, "
           "one complete shared-memory traversal and its barrier.">,
    Option<"directVRowPadding", "direct-v-row-padding", "unsigned",
           /*default=*/"0",
           "Extra f16 elements in each natural shared V row when using the direct "
           "ldmatrix.trans path. Legal values are 0 and 8. The padded state reuses "
           "the proven Flash-backward D-stride padding mechanism to change the "
           "shared-bank mapping without changing any J-visible value.">,
    Option<"materializeTmpRoundtrip", "materialize-tmp-roundtrip", "bool",
           /*default=*/"false",
           "Retain the certified loop's observable TMP store/load materialization "
           "at every recurrence step and before the output epilogue. DirectLower "
           "uses true; Full may remove those round trips only as an explicitly "
           "searched, Q/D-checked materialization state.">,
    Option<"pairedBMatrixLoads", "paired-b-matrix-loads", "bool",
           /*default=*/"false",
           "Pack two adjacent n8 B fragments into one ldmatrix.x4 for both QK "
           "and PV, then split the returned registers in SSA. This reuses the "
           "validated attention-backward loader and is a Full instruction-layout "
           "state; it requires direct-v-matrix-load.">,
    Option<"vectorizedQStaging", "vectorized-q-staging", "bool",
           /*default=*/"false",
           "Stage Q to shared memory with 16-byte vector load/store packets. "
           "False retains DirectLower's scalar canonical staging; true reuses "
           "the validated wide transaction shape from Flash backward as a Full "
           "instruction-layout state.">,
    Option<"streamPFragments", "stream-p-fragments", "bool",
           /*default=*/"false",
           "Generate each two-n8 probability fragment after the row maximum is "
           "known and consume it immediately in the corresponding PV MMA. "
           "False retains DirectLower's whole-tile P register materialization; "
           "true is a Full register-lifetime topology that preserves the authored "
           "128-key recurrence while avoiding a whole P tile live across PV.">,
    Option<"scoreFragmentColumns", "score-fragment-columns", "int64_t",
           /*default=*/"0",
           "When non-zero, keep one authored block-n recurrence update but compute "
           "its score matrix in smaller column fragments. The lowering first "
           "derives the row maximum across all fragments, then recomputes QK one "
           "fragment at a time and immediately consumes P in PV. This trades extra "
           "tensor-core work for a lower live-register peak without changing the "
           "authored reduction/update order. It must divide block-n, be a multiple "
           "of 16, be smaller than block-n, and requires stream-p-fragments.">,
    Option<"hopperWarpgroup", "hopper-warpgroup", "bool",
           /*default=*/"false",
           "Use the stock MLIR NVGPU warpgroup MMA abstraction for the two "
           "FlashAttention contractions. This is a Hopper-only Full structure: "
           "four warps cooperate on 64 query rows, QK is emitted as K*Q^T to "
           "reuse natural K staging, and the authored online-softmax recurrence "
           "is preserved between the two warpgroup contractions.">,
    Option<"hopperHybridPV", "hopper-hybrid-pv", "bool",
           /*default=*/"false",
           "With hopper-warpgroup, use WGMMA only for QK, assemble P in the "
           "existing register fragment layout, and use mma.sync with direct "
           "natural-V ldmatrix loads for PV. This Hopper-only Full state removes "
           "the P and delta shared-memory round trips while preserving the same "
           "authored 128-key online-softmax update.">,
    Option<"normalizedSoftmaxCarry", "normalized-softmax-carry", "bool",
           /*default=*/"false",
           "Represent the certified online-softmax recurrence with the standard "
           "register-resident (row max, scaled mass, weighted accumulator) state. "
           "This removes the per-key-tile log/exp round trip used by the Source "
           "realization and reconstructs LSE once in the epilogue. It is a "
           "Full-only execution-domain candidate and must pass Source/Q-D output "
           "checks before timing.">,
    Option<"compactQueryWarpMapping", "compact-query-warp-mapping", "bool",
           /*default=*/"false",
           "For block-m=128, keep one four-warp CTA and assign two 16-row query "
           "fragments to each warp. Both fragments reuse the same staged K/V "
           "tile while retaining independent register-resident softmax carries. "
           "This is a Full-only implementation-structure candidate; the default "
           "mapping keeps one 16-row fragment per warp.">,
    Option<"hopperRegisterQK", "hopper-register-qk", "bool",
           /*default=*/"false",
           "With the Hopper hybrid-PV path, emit the QK warpgroup contraction "
           "as NVVM WGMMA and consume its per-thread accumulator registers "
           "directly in online softmax. This removes the score shared-memory "
           "store, barrier, and reload while retaining MLIR/NVVM codegen.">,
    Option<"hopperRegisterPV", "hopper-register-pv", "bool",
           /*default=*/"false",
           "With hopper-register-qk, store the probability tile once to "
           "swizzled shared memory, execute PV with NVVM WGMMA, and keep the "
           "weighted output accumulator as per-thread registers across the key "
           "scan. This removes mma.sync PV and the delta shared round trip.">,
    Option<"hopperAsyncKV", "hopper-async-kv", "bool",
           /*default=*/"false",
           "Stage Hopper K/V tiles with 16-byte nvgpu.device_async_copy "
           "transactions instead of generic vector load/store pairs. The "
           "single-buffer state waits before WGMMA; it removes intermediate "
           "register traffic and is the prerequisite for a multistage ring.">,
  ];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::NVVM::NVVMDialect",
  ];
}

def ConvertIntentAttentionBackwardToGPU
    : Pass<"convert-intent-attention-backward-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the fused FlashAttention BACKWARD (one intent.loop over "
                "the query axis or the live flat rank-4 five-contraction DAG) "
                "to one tensor-core gpu.launch (serial-DQ).";
  let description = [{
    Recognizes either (1) a `func.func` whose body is FlashAttention backward as ONE
    `intent.loop` scanning the QUERY axis (scan inputs Q, dO, LSE, Delta; carries
    the per-key-row gradients `dk[Sk,D]`, `dv[Sk,D]`, init zeros). The body is the
    flash bwd inner block: recompute `P = exp(scale*S - LSE)` from the saved
    log-sum-exp, then FIVE contractions

        S  = Q@K^T        (`"qd,kd->qk"`)        recompute the scores
        dV += P^T@dO      (`"qk,qd->kd"`)        accumulate into the dv carry
        dP =  dO@V^T      (`"qd,kd->qk"`)
        dS =  P*(dP - Delta)*scale               (elementwise)
        dK += dS^T@Q      (`"qk,qd->kd"`)        accumulate into the dk carry
        dQ =  dS@K        (`"qk,kd->qd"`)        scatter-accumulate into DQ[start_m]

    or (2) the unchanged live B=H=1 flat rank-4 DAG with the equivalent
    `bmhd,bnhd->nbhm` / `nbhm,bmhd->bnhd` / `nbhm,bnhd->bmhd`
    contractions, explicit f16 round trips for P and dS, and
    `DQ = DQ_initial + dQ_update`. The flat matcher starts from the three return
    values and verifies every operation/type; it does not consult a function name
    or source symbol. Its distinct DQ destination is initialized from the nonzero
    DQ_initial input inside the same launch before serial accumulation.

    The match is STRUCTURAL (the 5 contraction subscripts + the recompute-P
    subtree + the carry shapes), NOT a kernel-name match -- it just happens to be
    attention-backward-shaped (hence the pass name).

    It rewrites the func in place to ONE tiled tensor-core `gpu.launch`. N4: the 5
    matmuls run on REGISTER-RESIDENT `nvgpu.mma.sync` m16n8k16 (the forward pass's
    proven recipe), NOT WMMA. The dV/dK accumulators are carried in registers
    (`scf.for` iter_args across the inner query loop); S/dP/dS and the online-
    softmax recompute run in registers; only P (then dS) is published to a single
    `[BM,BN]` f16 shared scratch, REQUIRED there for the dV/dK operand transpose.
    Both `dV=P^T@dO` and `dK=dS^T@Q` (and `dQ=dS@K`'s `K^T`) load their transposed
    operands DIRECTLY from the untransposed shared tiles via `nvgpu.ldmatrix
    transpose=true` (the `ldmatrix.trans` PTX) -- no separate transpose
    materialization, so the smem budget (~65KB) stays within the 99KB sm_120a cap.
    K/V/Q/dO stage via cp.async. 8 warps: the q-outputs (S/dP/dQ, M=BM=128) use 8
    row-strips; the k-outputs (dV/dK, M=BN=64) use 4 row-strips x 2 column-groups.
    The schedule is ONE block per (b,h): an OUTER `scf.for` over key tiles wrapping
    an INNER `scf.for` over query tiles. dQ uses the SERIAL accumulation mode
    (SEQUENCE_PARALLEL=false): one block owns the whole head and read-modify-
    writes the pre-zeroed global `DQ` deterministically across the outer key loop
    (the cross-program atomic_add mode + its serial==atomic gate are M9b).

    Q/K/V/dO are f16, LSE/Delta/scale and all accumulators are f32. Static shapes,
    BM=128, BN=64, D%16==0, Sq%128==0, Sk%64==0 (non-causal, no bias; tail-mask /
    causal / split-K are documented follow-ups). The pass appends three
    destination-passing outputs `dQ`, `dK`, `dV`; the legacy loop form expects
    host-preinitialized DQ, while the flat form copies its explicit DQ_initial
    input itself. Run BEFORE
    `convert-intent-to-linalg`; the result goes to the device pipeline
    (`lower_memref_module_to_kernels` with `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"dqMode", "dq-mode", "std::string", /*default=*/"\"serial\"",
           "DQ accumulation mode. 'serial' (SEQUENCE_PARALLEL=false): ONE block "
           "per (b,h) owns the whole head, an OUTER scf.for over key tiles "
           "read-modify-writes the pre-zeroed global DQ deterministically. "
           "'atomic' (SEQUENCE_PARALLEL=true): grid parallel over key blocks "
           "(grid.x = cdiv(Sk,BN)); each program owns ONE key block and "
           "accumulates its dQ contribution into pre-zeroed global DQ via "
           "memref.atomic_rmw addf (-> red.global.add.f32 on sm_120a), unlocking "
           "parallelism over key blocks. Both modes pre-zero DQ; they compute the "
           "same value modulo the float reassociation of the cross-program adds "
           "(the serial==atomic equivalence gate).">,
    Option<"flatSharedDPad", "flat-shared-d-pad", "unsigned", /*default=*/"8",
           "Extra f16 elements in each physical Q/K/V/dO shared-memory row for "
           "the exact flat-rank4 path. Must be a multiple of eight so every "
           "logical vector access remains 16-byte aligned; eight matches the "
           "captured Triton D-tile row pitch.">,
    Option<"flatSharedPdsPad", "flat-shared-pds-pad", "unsigned", /*default=*/"8",
           "Extra f16 elements in each physical P/dS scratch row for the exact "
           "flat-rank4 path. Must be a multiple of eight; the logical [BM,BN] "
           "tensor and all A/B semantics are unchanged. Eight is the measured "
           "bank-conflict minimum for the current standard-NVGPU schedule.">,
    Option<"flatDQInit", "flat-dq-init", "std::string",
           /*default=*/"\"in-kernel\"",
           "DQ initialization schedule for the exact flat-rank4 path. "
           "'in-kernel' copies DQ_initial inside the device launch; 'prelaunch' "
           "emits an explicit host memref.copy before the launch so kernel-only "
           "timing matches a baseline whose DQ is initialized before launch.">,
  ];
}

def ConvertIntentSplitKReduceToGPU
    : Pass<"convert-intent-splitk-reduce-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the split-K attention reduce (online-LSE merge across "
                "chunks, no matmul) to one block-cooperative gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the xformers `_splitK_reduce` /
    `_splitK_reduce_varargs` merge as ONE `intent.loop` scanning the split axis
    of a packed partial-output `O[S,Mq,D]` and partial log-sum-exp
    `LSE[S,Mq,1]` (the varargs list is normalized to this packed form upstream,
    so both reduce variants lift to the SAME IntentIR). The loop carries the
    online-merge state (`lse_max[Mq,1]`, `sumexp[Mq,1]`, `O_acc[Mq,D]`); the body
    is a pure elementwise/broadcast online update (NO contraction, NO reduce):
    `m_new = max(lse_max, LSE_s)`, `corr = exp(lse_max - m_new)`,
    `w = exp(LSE_s - m_new)`, `sumexp = sumexp*corr + w`,
    `O_acc = O_acc*corr + w*O_s`. Epilogue: `O = O_acc / sumexp`.

    Because `lse_max`/`sumexp` depend only on the per-(row) LSE scalars, every
    thread can recompute them redundantly with no shared memory or barriers: the
    rewrite emits grid = (Mq, 1, 1) (one block per output row) with one thread
    per head-dim element owning a single `O_acc[d]` register, scanning the split
    axis with a plain `scf.for`. Memory-bound (read S partials once, write O
    once) -> parity is the honest floor. Run BEFORE `convert-intent-to-linalg`;
    the result goes to the device pipeline (`lower_memref_module_to_kernels`,
    `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConvertIntentSplitKReduceVarargsToGPU
    : Pass<"convert-intent-splitk-reduce-varargs-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the exact xFormers four-way split-K varargs merge to a "
                "stride-aware warp-per-row gpu.launch.";
  let description = [{
    Structurally recognizes the live flat Intent graph for xFormers' four-way
    split-K merge: four independent partial-output tensors and four independent
    partial-LSE tensors are concatenated on the singleton G axis to form the
    canonical packed-B contract `[B,4,H,M,D]` / `[B,4,H,M]`; a stable
    max/exp-weighted merge produces both the transposed destination Out and LSE.
    Matching is based on operand provenance, concat/reduce axes, result shapes
    and output structure, never on the function symbol.

    The rewrite preserves the real unrolled runtime ABI rather than replacing it
    with the legacy H=G=1 packed surrogate: 10 independent pointers followed by
    all 45 i32 stride arguments. H=4, G=1, D=64 and WRITE_LSE=true are the
    captured constexpr specialization. Addressing consumes the runtime strides,
    so non-contiguous views remain semantic inputs. Four warps cooperate per
    block, one output row per warp, and every lane owns two D elements. The
    resulting single launch is compiled through the standard GPU/NVVM/PTX/cubin
    backend. If this structural contract does not match, the pass leaves the
    function unchanged for the generic `convert-intent-to-linalg` fallback.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"rowsPerBlock", "rows-per-block", "int64_t", /*default=*/"4",
           "Number of independent output rows assigned to one block. Must be "
           "one of 1, 2, 4 or 8; one warp owns each row.">,
    Option<"warpsPerRow", "warps-per-row", "int64_t", /*default=*/"1",
           "Number of warps cooperatively partitioning the contiguous D axis of "
           "one output row in the typed unrolled graph. Legal values are 1, 2, "
           "4 or 8. Values above one require rows-per-block=1 and form a distinct "
           "multi-warp D topology; the runtime-stride legacy graph fails closed.">,
    Option<"dPartitions", "d-partitions", "int64_t", /*default=*/"1",
           "Number of independent CTAs partitioning the contiguous D packets of "
           "one output row in the typed unrolled graph. Legal values are 1, 2, "
           "4 or 8. Values above one require rows-per-block=1 and "
           "warps-per-row=1; the runtime-stride legacy graph fails closed.">,
    Option<"vectorWidth", "vector-width", "int64_t", /*default=*/"1",
           "Contiguous output-D packet width for the typed unrolled graph. "
           "Legal values are 1, 2 or 4 and D must be divisible by the width. "
           "The runtime-stride legacy graph remains scalar unless contiguous "
           "D strides are proven by a future typed contract.">,
    Option<"warpBroadcastScalars", "warp-broadcast-scalars", "bool", "false",
           "Compute the per-row LSE maximum, split weights, and denominator in "
           "lane zero once, then broadcast them through warp shuffles before "
           "the cooperative D merge. False redundantly computes them in every "
           "lane and is the fixed-structure schedule baseline.">,
  ];
}

def ConvertIntentSplitKReduceBackwardToGPU
    : Pass<"convert-intent-splitk-reduce-backward-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower the split-K attention reduce BACKWARD (gradient of the "
                "online-LSE merge, no matmul) to one block-cooperative "
                "gpu.launch.";
  let description = [{
    Recognizes a `func.func` whose body is the xformers
    `_splitK_reduce_varargs_backward` -- the GRADIENT of the online log-sum-exp
    merge -- as a straight-line map/reduce DAG over the packed split-K partials
    (Out_splitK[S,Mq,D], LSE_splitK[S,Mq,1]) + the merged-output gradients
    (Out[Mq,D], LSE[Mq,1], DOut[Mq,D], DLSE[Mq,1]), producing the per-split
    gradients dOut_splitK[S,Mq,D], dLSE_splitK[S,Mq,1] (the varargs list is
    normalized to this packed RowList upstream, the SAME contract as the forward
    reduce). The DAG has exactly two `intent.reduce` -- a `max` over the split
    axis (lse_max) and a `sum` over the head dim (the corr term) -- and broadcast
    map ops; NO loop, NO contraction. The match is STRUCTURAL (shapes + the two
    reduce signatures), NOT a kernel-name match.

    It rewrites the func in place to ONE block-cooperative `gpu.launch`:
    grid = (Mq,1,1) (one block per output row), block = (D,1,1) (one thread per
    head-dim element). Two passes over the split axis (`scf.for`): pass 1
    computes `lse_max = max_s LSE_splitK[s]` (every thread recomputes the per-row
    scalar); `denom = exp(LSE - lse_max)`. pass 2 per split s computes
    `w_s = exp(LSE_splitK[s] - lse_max)/denom`, stores `dOut_splitK[s] = w_s*DOut`,
    and `dLSE_splitK[s] = w_s*DLSE + sum_d((Out_splitK[s]-Out)*w_s*DOut)`, where
    the head-dim sum is a `gpu.all_reduce` ADD across the block's D threads
    (inside the split-axis loop). dLSE_splitK is written once by thread 0.
    Memory-bound (read partials once, write per-split gradients once) -> parity is
    the honest floor. Run BEFORE `convert-intent-to-linalg`; the result goes to
    the device pipeline (`lower_memref_module_to_kernels`, `scf_to_cf=True`).
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif // INTENT_CONVERSION_CUDA_PASSES
