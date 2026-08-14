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

using namespace mlir;

namespace {

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
