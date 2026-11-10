//===- TileLinalgByIteratorKind.cpp - generic iterator-aware tiling -------===//
//
// A target-neutral, source-blind schedule transformation.  It consumes only
// concrete compiler schedule parameters and the iterator kinds already present
// in Linalg IR.  C may order those parameters outside this pass; it never
// changes matching or legality here.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Common/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace intent {
#define GEN_PASS_DEF_TILELINALGBYITERATORKIND
#include "Intent/Conversion/Common/Passes.h.inc"
} // namespace intent

using namespace mlir;

namespace {

static bool hasReductionProducer(Operation *operation) {
  SmallVector<Operation *> worklist;
  llvm::SmallPtrSet<Operation *, 16> visited;
  for (Value operand : operation->getOperands())
    if (Operation *producer = operand.getDefiningOp())
      worklist.push_back(producer);
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val();
    if (!visited.insert(producer).second)
      continue;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(producer)) {
      if (llvm::is_contained(linalgOp.getIteratorTypesArray(),
                             utils::IteratorType::reduction))
        return true;
      for (Value operand : producer->getOperands())
        if (Operation *next = operand.getDefiningOp())
          worklist.push_back(next);
    }
  }
  return false;
}

static bool hasLinalgConsumer(Operation *operation) {
  return llvm::any_of(operation->getResults(), [](Value result) {
    return llvm::any_of(result.getUsers(), [](Operation *user) {
      return isa<linalg::LinalgOp>(user);
    });
  });
}

static int64_t existingScheduleAttr(ModuleOp module, StringRef name) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return 0;
}

static void addScheduleAttr(ModuleOp module, Builder &builder, StringRef name,
                            int64_t value) {
  module->setAttr(name, builder.getI64IntegerAttr(
                            existingScheduleAttr(module, name) + value));
}

struct TileLinalgByIteratorKindPass
    : ::intent::impl::TileLinalgByIteratorKindBase<
          TileLinalgByIteratorKindPass> {
  using TileLinalgByIteratorKindBase::TileLinalgByIteratorKindBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (parallelTile == 0 && reductionTile == 0) {
      // A split schedule may invoke this pass once per iterator kind.  A
      // no-op invocation must not erase evidence produced by the preceding
      // real stage.
      if (!module->hasAttr("intent.schedule.linalg_tiled_op_count"))
        module->setAttr(
            "intent.schedule.linalg_tiled_op_count",
            IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));
      return;
    }

    SmallVector<Operation *> worklist;
    module.walk([&](linalg::LinalgOp op) {
      // A tile-and-fuse path starts from terminal Linalg consumers and pulls
      // their producer slices into the same loop nest.  Tiling every original
      // producer again after it has been fused would duplicate the whole
      // computation and make the requested structural path a no-op after DCE.
      if (op.hasPureTensorSemantics() &&
          (!fuseProducers || !hasLinalgConsumer(op.getOperation())))
        worklist.push_back(op.getOperation());
    });

    IRRewriter rewriter(&getContext());
    unsigned tiledOpCount = 0;
    unsigned generatedLoopCount = 0;
    unsigned fusedProducerCount = 0;
    unsigned fusionOuterParallelOnlyCount = 0;
    unsigned fusionMultiUseRejectionCount = 0;
    bool parallelTileApplied = false;
    bool reductionTileApplied = false;
    for (Operation *operation : worklist) {
      if (!operation->getBlock())
        continue;
      auto linalgOp = dyn_cast<linalg::LinalgOp>(operation);
      auto tilingOp = dyn_cast<TilingInterface>(operation);
      if (!linalgOp || !tilingOp)
        continue;

      SmallVector<OpFoldResult> tileSizes;
      bool hasNonZeroTile = false;
      bool opUsesParallelTile = false;
      bool opUsesReductionTile = false;
      const bool outerParallelOnly =
          fuseProducers && hasReductionProducer(operation) &&
          llvm::none_of(linalgOp.getIteratorTypesArray(),
                        [](utils::IteratorType iterator) {
                          return iterator == utils::IteratorType::reduction;
                        });
      bool emittedOuterParallelTile = false;
      for (utils::IteratorType iterator : linalgOp.getIteratorTypesArray()) {
        unsigned tile = 0;
        if (iterator == utils::IteratorType::reduction) {
          tile = reductionTile;
        } else if (!outerParallelOnly || !emittedOuterParallelTile) {
          tile = parallelTile;
          emittedOuterParallelTile = true;
        }
        tileSizes.push_back(rewriter.getIndexAttr(tile));
        hasNonZeroTile |= tile != 0;
        opUsesReductionTile |=
            tile != 0 && iterator == utils::IteratorType::reduction;
        opUsesParallelTile |=
            tile != 0 && iterator != utils::IteratorType::reduction;
      }
      if (!hasNonZeroTile || tileSizes.empty())
        continue;
      fusionOuterParallelOnlyCount += outerParallelOnly;

      rewriter.setInsertionPoint(operation);
      scf::SCFTilingOptions options;
      options.setTileSizes(tileSizes).setLoopType(
          scf::SCFTilingOptions::LoopType::ForOp);
      if (fuseProducers) {
        scf::SCFTileAndFuseOptions fuseOptions;
        fuseOptions.setTilingOptions(options);
        fuseOptions.setFusionControlFn(
            [&](tensor::ExtractSliceOp candidateSlice, OpResult producer,
                bool isDestinationOperand)
                -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
              if (isDestinationOperand)
                return scf::SCFTileAndFuseOptions::ControlFnResult{};
              llvm::SmallPtrSet<Operation *, 4> originalConsumers;
              for (Operation *user : producer.getUsers()) {
                if (user == candidateSlice.getOperation() ||
                    isa<tensor::ExtractSliceOp>(user))
                  continue;
                originalConsumers.insert(user);
              }
              if (originalConsumers.size() > 1) {
                ++fusionMultiUseRejectionCount;
                return std::nullopt;
              }
              return scf::SCFTileAndFuseOptions::ControlFnResult{};
            });
        FailureOr<scf::SCFTileAndFuseResult> tiled =
            scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp,
                                                      fuseOptions);
        if (failed(tiled)) {
          operation->emitError("failed iterator-kind Linalg tile-and-fuse");
          signalPassFailure();
          return;
        }
        SmallVector<Value> replacements;
        replacements.reserve(operation->getNumResults());
        for (Value result : operation->getResults()) {
          auto replacement = tiled->replacements.find(result);
          if (replacement == tiled->replacements.end()) {
            operation->emitError(
                "tile-and-fuse produced an incomplete replacement set");
            signalPassFailure();
            return;
          }
          replacements.push_back(replacement->second);
        }
        generatedLoopCount += tiled->loops.size();
        fusedProducerCount += tiled->fusedProducers.size();
        rewriter.replaceOp(operation, replacements);
        for (Operation *producer : llvm::reverse(tiled->fusedProducers)) {
          if (producer->getBlock() && producer->use_empty())
            rewriter.eraseOp(producer);
        }
      } else {
        FailureOr<scf::SCFTilingResult> tiled =
            scf::tileUsingSCF(rewriter, tilingOp, options);
        if (failed(tiled)) {
          operation->emitError("failed iterator-kind Linalg tiling");
          signalPassFailure();
          return;
        }
        if (tiled->mergeResult.replacements.size() !=
            operation->getNumResults()) {
          operation->emitError("tiling produced an incomplete replacement set");
          signalPassFailure();
          return;
        }
        generatedLoopCount += tiled->loops.size();
        rewriter.replaceOp(operation, tiled->mergeResult.replacements);
      }
      ++tiledOpCount;
      parallelTileApplied |= opUsesParallelTile;
      reductionTileApplied |= opUsesReductionTile;
    }

    Builder builder(&getContext());
    // Preserve the other iterator kind when the caller deliberately splits a
    // combined schedule into two real passes.  Counters are cumulative across
    // those passes, so the final transformation record describes the actual
    // composed compiler state rather than whichever pass ran last.
    if (parallelTileApplied || module->hasAttr("intent.schedule.parallel_tile"))
      module->setAttr(
          "intent.schedule.parallel_tile",
          builder.getI64IntegerAttr(parallelTileApplied
                                        ? parallelTile
                                        : existingScheduleAttr(
                                              module,
                                              "intent.schedule.parallel_tile")));
    if (reductionTileApplied || module->hasAttr("intent.schedule.reduction_tile"))
      module->setAttr(
          "intent.schedule.reduction_tile",
          builder.getI64IntegerAttr(reductionTileApplied
                                        ? reductionTile
                                        : existingScheduleAttr(
                                              module,
                                              "intent.schedule.reduction_tile")));
    addScheduleAttr(module, builder, "intent.schedule.linalg_tiled_op_count",
                    tiledOpCount);
    addScheduleAttr(module, builder, "intent.schedule.generated_loop_count",
                    generatedLoopCount);
    addScheduleAttr(module, builder, "intent.schedule.fused_producer_count",
                    fusedProducerCount);
    addScheduleAttr(module, builder,
                    "intent.schedule.fusion_outer_parallel_only_count",
                    fusionOuterParallelOnlyCount);
    addScheduleAttr(module, builder,
                    "intent.schedule.fusion_multi_use_rejection_count",
                    fusionMultiUseRejectionCount);
  }
};

} // namespace
