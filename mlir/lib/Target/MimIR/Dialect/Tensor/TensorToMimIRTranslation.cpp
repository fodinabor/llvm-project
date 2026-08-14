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
