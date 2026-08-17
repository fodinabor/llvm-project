//===- TensorToMimIRTranslation.cpp - Translate tensor to MimIR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Tensor dialect and
// MimIR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Tensor/TensorToMimIRTranslation.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include <mim/plug/core/core.h>
#include <mim/plug/tensor/autogen.h>

using namespace mlir;

namespace {

/// Builds a `«r; Nat»` tuple of literals.
static const mim::Def *natTuple(mim::World &world, ArrayRef<int64_t> values) {
  mim::DefVec nats;
  for (int64_t v : values)
    nats.push_back(world.lit_nat(v));
  return world.tuple(mim::Defs{nats});
}

/// Translates a row-major reshape (tensor.collapse_shape/expand_shape) to
/// %tensor.reshape.
static LogicalResult
convertReshape(Operation *op, Value src, RankedTensorType srcType,
               RankedTensorType resultType, mim::World &world,
               MimIR::ModuleTranslation &moduleTranslation) {
  if (!srcType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitError("dynamic tensor shapes are not supported");
  const auto *input = moduleTranslation.lookupValue(src);
  if (!input)
    return op->emitError("failed to lookup reshape source");
  const auto *elemType =
      moduleTranslation.convertType(srcType.getElementType());

  const auto *reshape = world.annex<mim::plug::tensor::reshape>();
  reshape = world.app(reshape,
                      mim::Defs{elemType, world.lit_nat(srcType.getRank()),
                                world.lit_nat(resultType.getRank())});
  reshape = world.app(reshape, natTuple(world, srcType.getShape()));
  reshape = world.app(reshape, natTuple(world, resultType.getShape()));
  moduleTranslation.mapValue(op->getResult(0), world.app(reshape, input));
  return success();
}

/// Builds the implicit arguments {T, r, s} of %tensor.get / %tensor.set and
/// the `«i: r; Idx (s#i)»` index tuple for the given tensor access. Indices
/// are `index`-typed (Idx 2^64) and are narrowed to `Idx (s#i)`.
static LogicalResult
buildAccess(Operation *op, RankedTensorType type, ValueRange indices,
            mim::World &world, MimIR::ModuleTranslation &moduleTranslation,
            const mim::Def *&implicits, const mim::Def *&indexTuple) {
  using namespace mim::plug;
  if (!type.hasStaticShape())
    return op->emitError("dynamic tensor shapes are not supported");

  const auto *elemType = moduleTranslation.convertType(type.getElementType());
  mim::DefVec shape, idxs;
  for (auto [dim, value] : llvm::zip(type.getShape(), indices)) {
    shape.push_back(world.lit_nat(dim));
    const auto *idx = moduleTranslation.lookupValue(value);
    if (!idx)
      return op->emitError("failed to lookup index");
    idxs.push_back(world.call(core::conv::u, world.lit_nat(dim), idx));
  }
  implicits = world.tuple({elemType, world.lit_nat(type.getRank()),
                           world.tuple(mim::Defs{shape})});
  indexTuple = world.tuple(mim::Defs{idxs});
  return success();
}

class TensorDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    // An empty tensor is an uninitialized value: bottom of the array type.
    if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {
      if (!emptyOp.getType().hasStaticShape())
        return op->emitError("dynamic tensor shapes are not supported");
      const auto *type = moduleTranslation.convertType(emptyOp.getType());
      moduleTranslation.mapValue(emptyOp.getResult(), world.bot(type));
      return success();
    }

    // tensor.extract -> %tensor.get.
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      auto type = cast<RankedTensorType>(extractOp.getTensor().getType());
      const auto *tensorDef =
          moduleTranslation.lookupValue(extractOp.getTensor());
      if (!tensorDef)
        return op->emitError("failed to lookup tensor");
      const mim::Def *implicits, *indexTuple;
      if (failed(buildAccess(op, type, extractOp.getIndices(), world,
                             moduleTranslation, implicits, indexTuple)))
        return failure();
      const auto *get = world.annex<mim::plug::tensor::get>();
      get = world.app(get, implicits);
      moduleTranslation.mapValue(
          extractOp.getResult(),
          world.app(get, mim::Defs{tensorDef, indexTuple}));
      return success();
    }

    // tensor.insert -> %tensor.set.
    if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
      auto type = cast<RankedTensorType>(insertOp.getDest().getType());
      const auto *destDef = moduleTranslation.lookupValue(insertOp.getDest());
      const auto *scalarDef =
          moduleTranslation.lookupValue(insertOp.getScalar());
      if (!destDef || !scalarDef)
        return op->emitError("failed to lookup operands");
      const mim::Def *implicits, *indexTuple;
      if (failed(buildAccess(op, type, insertOp.getIndices(), world,
                             moduleTranslation, implicits, indexTuple)))
        return failure();
      const auto *set = world.annex<mim::plug::tensor::set>();
      set = world.app(set, implicits);
      moduleTranslation.mapValue(
          insertOp.getResult(),
          world.app(set, mim::Defs{destDef, indexTuple, scalarDef}));
      return success();
    }

    // tensor.pad with a constant padding value -> %tensor.pad (mode 0).
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      auto srcType = padOp.getSourceType();
      if (!srcType.hasStaticShape() ||
          !padOp.getResultType().hasStaticShape())
        return op->emitError("dynamic tensor shapes are not supported");
      if (llvm::any_of(padOp.getStaticLow(), ShapedType::isDynamic) ||
          llvm::any_of(padOp.getStaticHigh(), ShapedType::isDynamic))
        return op->emitError("dynamic padding amounts are not supported");
      Value padValue = padOp.getConstantPaddingValue();
      if (!padValue)
        return op->emitError("only constant padding values are supported");
      const auto *source = moduleTranslation.lookupValue(padOp.getSource());
      const auto *value = moduleTranslation.lookupValue(padValue);
      if (!source || !value)
        return op->emitError("failed to lookup pad operands");
      const auto *elemType =
          moduleTranslation.convertType(srcType.getElementType());

      const auto *pad = world.annex<mim::plug::tensor::pad>();
      pad = world.app(pad,
                      mim::Defs{elemType, world.lit_nat(srcType.getRank())});
      pad = world.app(pad, natTuple(world, srcType.getShape()));
      pad = world.app(pad, mim::Defs{world.lit_nat(0),
                                     natTuple(world, padOp.getStaticLow()),
                                     natTuple(world, padOp.getStaticHigh())});
      moduleTranslation.mapValue(padOp.getResult(),
                                 world.app(pad, mim::Defs{source, value}));
      return success();
    }

    // Reassociating reshapes are row-major reshapes.
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op))
      return convertReshape(op, collapseOp.getSrc(), collapseOp.getSrcType(),
                            collapseOp.getResultType(), world,
                            moduleTranslation);
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op))
      return convertReshape(op, expandOp.getSrc(), expandOp.getSrcType(),
                            expandOp.getResultType(), world,
                            moduleTranslation);

    return op->emitError("unsupported tensor operation in MimIR translation");
  }
};

} // namespace

void mlir::registerTensorDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    dialect->addInterfaces<TensorDialectMimIRTranslationInterface>();
  });
}

void mlir::registerTensorDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerTensorDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
