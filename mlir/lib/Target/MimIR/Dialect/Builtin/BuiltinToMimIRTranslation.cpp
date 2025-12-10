//===- BuiltinToLLVMIRTranslation.cpp - Translate builtin to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR builtin dialect and MIM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Builtin/BuiltinToMimIRTranslation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"

using namespace mlir;

namespace {

class BuiltinDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    return success(isa<ModuleOp>(op));
  }
};

} // namespace

void mlir::registerBuiltinDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinDialectMimIRTranslationInterface>();
  });
}

void mlir::registerBuiltinDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerBuiltinDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
