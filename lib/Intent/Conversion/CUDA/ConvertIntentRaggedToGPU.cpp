//===- ConvertIntentRaggedToGPU.cpp - ragged queries -> gpu.launch ----------===//
//
// This is the target-specific consumer of the Layer-A ragged descriptor.  The
// descriptor is a builtin tuple only while the Intent graph is being checked;
// it is never passed through a CUDA function boundary.  A closed local graph
// is rewritten to a destination-passing memref function and one GPU launch:
//
//   ragged_outer  -> one scalar store of offsets.size - 1
//   ragged_member -> guarded parallel copy of
//                    indices[offsets[i] : offsets[i + 1]]
//
// The member result is data-dependent.  Its host allocation is therefore
// materialized from the input offsets and selector.  The lowered host
// signature keeps a dynamic rank-one memref descriptor.  The CUDA device
// lowering runs with kernel-bare-ptr-calling-convention=false for this pass,
// so the device receives the flattened allocated/aligned pointers, offset,
// size, and stride fields of each ranked memref.  The generated kernel never
// dereferences beyond the runtime slice length.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/CUDA/Passes.h"

#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTRAGGEDTOGPU
#include "Intent/Conversion/CUDA/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

static MemRefType memrefOf(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value toSignlessInteger(OpBuilder &builder, Location loc, Value value) {
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.isSignless())
    return value;
  Type signless = IntegerType::get(value.getContext(), integer.getWidth());
  return builder.create<UnrealizedConversionCastOp>(
                         loc, TypeRange{signless}, ValueRange{value})
      .getResult(0);
}

static Value toIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  if (!isa<IntegerType>(value.getType()))
    return Value();
  return builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), toSignlessInteger(builder, loc, value));
}

static FailureOr<int64_t> constantTensorInteger(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 0 ||
      (!type.getElementType().isIndex() &&
       !isa<IntegerType>(type.getElementType())))
    return failure();
  auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
  if (!dense || dense.getNumElements() != 1)
    return failure();
  APInt bits = dense.getSplatValue<APInt>();
  if (auto integer = dyn_cast<IntegerType>(type.getElementType()); integer &&
      integer.isUnsigned())
    return static_cast<int64_t>(bits.getZExtValue());
  return bits.getSExtValue();
}

static FailureOr<DenseIntElementsAttr> constantTensorElements(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      !isRaggedCarrierTensorType(type))
    return failure();
  auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
  if (!dense || dense.getNumElements() != type.getDimSize(0))
    return failure();
  return dense;
}

static void copyIntentFunctionAttrs(func::FuncOp source,
                                    func::FuncOp destination) {
  for (NamedAttribute attr : source->getAttrs()) {
    if (attr.getName().getValue().starts_with("intent."))
      destination->setAttr(attr.getName(), attr.getValue());
  }
}

struct RaggedLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  RaggedOp ragged;
  RaggedOuterOp outer;
  RaggedMemberOp member;
  int outerResultIndex = -1;
  int memberResultIndex = -1;
  bool selectorIsConstant = false;
  int64_t selectorConstant = 0;

  RaggedLowering(func::FuncOp fn, unsigned blockThreads)
      : fn(fn), blockThreads(blockThreads) {}

  static FailureOr<unsigned> argumentNumber(Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return failure();
    return argument.getArgNumber();
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    Operation *ret = body.getTerminator();
    if (!isa<func::ReturnOp>(ret))
      return failure();

    SmallVector<Operation *> operations;
    for (Operation &operation : body.without_terminator())
      operations.push_back(&operation);
    for (Operation *operation : operations) {
      if (auto candidate = dyn_cast<RaggedOp>(operation)) {
        if (ragged)
          return failure();
        ragged = candidate;
        continue;
      }
      if (auto candidate = dyn_cast<RaggedOuterOp>(operation)) {
        if (outer)
          return failure();
        outer = candidate;
        continue;
      }
      if (auto candidate = dyn_cast<RaggedMemberOp>(operation)) {
        if (member)
          return failure();
        member = candidate;
        continue;
      }
    }
    if (!ragged || (!outer && !member))
      return failure();

    if (outer && outer.getRagged() != ragged.getResult())
      return failure();
    if (member && member.getRagged() != ragged.getResult())
      return failure();
    if (ret->getNumOperands() != static_cast<int64_t>(!!outer + !!member))
      return failure();

    for (unsigned index = 0; index < ret->getNumOperands(); ++index) {
      Value value = ret->getOperand(index);
      if (outer && value == outer.getResult()) {
        if (outerResultIndex != -1)
          return failure();
        outerResultIndex = static_cast<int>(index);
        continue;
      }
      if (member && value == member.getResult()) {
        if (memberResultIndex != -1)
          return failure();
        memberResultIndex = static_cast<int>(index);
        continue;
      }
      return failure();
    }
    if ((outer && outerResultIndex == -1) ||
        (member && memberResultIndex == -1))
      return failure();

    // A constexpr selector is materialized as one arith.constant immediately
    // before the Ragged query.  It is part of the same closed graph, but it is
    // deliberately not a runtime memref argument.  No other operation may be
    // hidden by this allowance.
    Operation *selectorDefinition = nullptr;
    if (member)
      selectorDefinition = member.getIndex().getDefiningOp();
    Operation *offsetsDefinition = ragged.getOffsets().getDefiningOp();
    Operation *indicesDefinition = ragged.getIndices().getDefiningOp();
    for (Operation *operation : operations) {
      if (operation == ragged.getOperation() ||
          (outer && operation == outer.getOperation()) ||
          (member && operation == member.getOperation()) ||
          operation == selectorDefinition || operation == offsetsDefinition ||
          operation == indicesDefinition)
        continue;
      return failure();
    }

    auto offsetsTy = dyn_cast<RankedTensorType>(ragged.getOffsets().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(ragged.getIndices().getType());
    if (!offsetsTy || !indicesTy || !offsetsTy.hasStaticShape() ||
        !indicesTy.hasStaticShape() || !isRaggedCarrierTensorType(offsetsTy) ||
        !isRaggedCarrierTensorType(indicesTy) || offsetsTy.getDimSize(0) < 1)
      return failure();
    if (blockThreads == 0 || blockThreads > 1024)
      return failure();

    // A carrier is either a function input or a deterministic rank-one
    // arith.constant. Both forms are part of the shared Layer-A contract;
    // rejecting the latter here would make CUDA disagree with CPU/RISC-V
    // before any target-specific ABI decision is made.
    if (failed(argumentNumber(ragged.getOffsets())) &&
        failed(constantTensorElements(ragged.getOffsets())))
      return failure();
    if (failed(argumentNumber(ragged.getIndices())) &&
        failed(constantTensorElements(ragged.getIndices())))
      return failure();

    if (outer) {
      auto resultTy = dyn_cast<RankedTensorType>(outer.getResult().getType());
      if (!resultTy || resultTy.getRank() != 0 ||
          !isRaggedCarrierElementType(resultTy.getElementType()) ||
          resultTy.getElementType() != offsetsTy.getElementType())
        return failure();
    }

    if (member) {
      auto indexTy = dyn_cast<RankedTensorType>(member.getIndex().getType());
      auto resultTy = dyn_cast<RankedTensorType>(member.getResult().getType());
      if (!indexTy || !resultTy || !indexTy.hasStaticShape() ||
          !resultTy.getRank() || resultTy.getRank() != 1 ||
          !resultTy.isDynamicDim(0) || indexTy.getRank() != 0 ||
          !isRaggedCarrierElementType(indexTy.getElementType()) ||
          !isRaggedCarrierElementType(resultTy.getElementType()) ||
          resultTy.getElementType() != indicesTy.getElementType())
        return failure();
      if (auto constant = member.getIndex().getDefiningOp<arith::ConstantOp>()) {
        auto value = constantTensorInteger(member.getIndex());
        if (failed(value))
          return failure();
        selectorIsConstant = true;
        selectorConstant = *value;
        if (selectorConstant < 0 ||
            selectorConstant >= offsetsTy.getDimSize(0) - 1)
          return failure();
      } else if (failed(argumentNumber(member.getIndex()))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *context = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder builder(context);

    SmallVector<Type> inputTypes;
    inputTypes.reserve(fn.getNumArguments() + fn.getNumResults());
    for (Type type : fn.getArgumentTypes()) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || !tensor.hasStaticShape())
        return failure();
      inputTypes.push_back(memrefOf(tensor));
    }

    for (Type type : fn.getResultTypes()) {
      auto resultTy = dyn_cast<RankedTensorType>(type);
      if (!resultTy ||
          (resultTy.getRank() == 1 && !resultTy.isDynamicDim(0)) ||
          (resultTy.getRank() != 0 && resultTy.getRank() != 1))
        return failure();
    }
    for (Type type : fn.getResultTypes()) {
      auto resultTy = cast<RankedTensorType>(type);
      SmallVector<int64_t> outputShape(resultTy.getShape().begin(),
                                       resultTy.getShape().end());
      inputTypes.push_back(
          MemRefType::get(outputShape, resultTy.getElementType()));
    }

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    builder.setInsertionPoint(fn);
    auto lowered = builder.create<func::FuncOp>(loc, name,
                                                  FunctionType::get(
                                                      context, inputTypes, {}));
    copyIntentFunctionAttrs(fn, lowered);
    Block *entry = lowered.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto materializeCarrier = [&](Value source) -> Value {
      if (auto argument = argumentNumber(source); succeeded(argument))
        return entry->getArgument(*argument);
      auto elements = constantTensorElements(source);
      if (failed(elements))
        return Value();
      return builder.create<arith::ConstantOp>(loc, *elements).getResult();
    };
    Value offsetsValue = materializeCarrier(ragged.getOffsets());
    Value indicesValue = materializeCarrier(ragged.getIndices());
    if (!offsetsValue || !indicesValue)
      return failure();

    auto loadCarrier = [&](Value carrier, Value index) -> Value {
      if (isa<MemRefType>(carrier.getType()))
        return builder.create<memref::LoadOp>(loc, carrier, ValueRange{index});
      return builder.create<tensor::ExtractOp>(loc, carrier,
                                               ValueRange{index});
    };

    auto outputMemFor = [&](int resultIndex) -> Value {
      if (resultIndex < 0 ||
          resultIndex >= static_cast<int>(fn.getNumResults()))
        return Value();
      return entry->getArgument(fn.getNumArguments() + resultIndex);
    };

    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value oneBlock = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value threads = builder.create<arith::ConstantIndexOp>(
        loc, member ? static_cast<int64_t>(blockThreads) : 1);
    auto launch = builder.create<gpu::LaunchOp>(loc, oneBlock, oneBlock,
                                                oneBlock, threads, oneBlock,
                                                oneBlock);
    builder.setInsertionPointToStart(&launch.getBody().front());
    Value tx = launch.getThreadIds().x;

    if (outer) {
      auto offsetsTy = dyn_cast<RankedTensorType>(ragged.getOffsets().getType());
      auto resultTy = dyn_cast<RankedTensorType>(outer.getResult().getType());
      Value outputMem = outputMemFor(outerResultIndex);
      if (!offsetsTy || !resultTy || resultTy.getRank() != 0 ||
          resultTy.getElementType() != offsetsTy.getElementType() || !outputMem)
        return failure();
      int64_t groups = offsetsTy.getDimSize(0) - 1;
      Value value;
      if (resultTy.getElementType().isIndex())
        value = builder.create<arith::ConstantIndexOp>(loc, groups);
      else
        value = builder.create<arith::ConstantOp>(
            loc, IntegerAttr::get(resultTy.getElementType(), groups));
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value isLeader = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, tx, zero);
      auto guarded = builder.create<scf::IfOp>(loc, isLeader, false);
      guarded.getThenBodyBuilder().create<memref::StoreOp>(
          loc, value, outputMem, ValueRange{});
    }
    if (member) {
      Value outputMem = outputMemFor(memberResultIndex);
      if (!outputMem)
        return failure();
      Value selector;
      if (selectorIsConstant) {
        selector = builder.create<arith::ConstantIndexOp>(loc, selectorConstant);
      } else {
        auto selectorArgument = argumentNumber(member.getIndex());
        if (failed(selectorArgument))
          return failure();
        Value selectorMem = entry->getArgument(*selectorArgument);
        Value selectorRaw = builder.create<memref::LoadOp>(
            loc, selectorMem, ValueRange{});
        selector = toIndex(builder, loc, selectorRaw);
        if (!selector)
          return failure();
      }
      Value nextSelector = builder.create<arith::AddIOp>(loc, selector, one);
      Value startRaw = loadCarrier(offsetsValue, selector);
      Value endRaw = loadCarrier(offsetsValue, nextSelector);
      Value start = toIndex(builder, loc, startRaw);
      Value end = toIndex(builder, loc, endRaw);
      if (!start || !end)
        return failure();
      Value length = builder.create<arith::SubIOp>(loc, end, start);
      // One block is sufficient for the ABI, but a member may be longer than
      // the selected block size.  Each lane therefore walks its strided part
      // of the runtime slice instead of silently truncating at blockThreads.
      auto loop = builder.create<scf::ForOp>(loc, tx, length, threads);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Value position = loop.getInductionVar();
        Value sourceIndex = builder.create<arith::AddIOp>(loc, start, position);
        Value value = loadCarrier(indicesValue, sourceIndex);
        builder.create<memref::StoreOp>(loc, value, outputMem,
                                        ValueRange{position});
      }
      builder.setInsertionPointAfter(loop);
    }

    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPointAfter(launch);
    builder.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentRaggedToGPUPass
    : ::intent::impl::ConvertIntentRaggedToGPUBase<
          ConvertIntentRaggedToGPUPass> {
  using ConvertIntentRaggedToGPUBase::ConvertIntentRaggedToGPUBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> targets;
    bool failedMatch = false;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      RaggedLowering probe(fn, blockThreads);
      if (succeeded(probe.match())) {
        targets.push_back(fn);
        continue;
      }
      bool hasRagged = false;
      fn.walk([&](Operation *operation) {
        hasRagged |= isa<RaggedOp, RaggedOuterOp, RaggedMemberOp>(operation);
      });
      if (hasRagged) {
        fn.emitError("CUDA Ragged lowering only admits a local direct outer/member query");
        signalPassFailure();
        failedMatch = true;
      }
    }
    if (failedMatch)
      return;
    for (func::FuncOp fn : targets) {
      RaggedLowering lowering(fn, blockThreads);
      if (failed(lowering.match()) || failed(lowering.rewrite())) {
        fn.emitError("CUDA Ragged lowering failed after structural match");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
