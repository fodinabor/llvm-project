//===- FuncToLLVMIRTranslation.cpp - Translate Func to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Func dialect and MimIR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Func/FuncToMimIRTranslation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

class FuncDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      auto func = op->getParentOfType<func::FuncOp>();
      auto *lam = moduleTranslation.lookupFunction(func.getName());
      lam->dump();
      const auto *ret = lam->ret_var();
      assert(ret && "function must have a return continuation");

      mim::DefVec retArgs;
      std::transform(returnOp.getOperands().begin(),
                     returnOp.getOperands().end(), std::back_inserter(retArgs),
                     [&](Value v) {
                       const auto *def = moduleTranslation.lookupValue(v);
                       assert(def && "failed to lookup return operand");
                       return def;
                     });

      lam->app(false, ret, mim::Defs{retArgs});

      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::registerFuncDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncDialectMimIRTranslationInterface>();
  });
}

void mlir::registerFuncDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerFuncDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
