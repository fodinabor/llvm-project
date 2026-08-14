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
      const auto *ret = lam->ret_var();
      assert(ret && "function must have a return continuation");

      mim::DefVec retArgs{moduleTranslation.currentMem()};
      for (Value v : returnOp.getOperands()) {
        const auto *def = moduleTranslation.lookupValue(v);
        if (!def)
          return returnOp.emitError("failed to lookup return operand");
        retArgs.push_back(def);
      }

      moduleTranslation.currentLam()->app(false, ret, mim::Defs{retArgs});
      return success();
    }

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      mim::Lam *callee = moduleTranslation.lookupFunction(callOp.getCallee());
      if (!callee)
        return callOp.emitError("failed to lookup callee '")
               << callOp.getCallee() << "' in MimIR translation";

      mim::DefVec args{moduleTranslation.currentMem()};
      for (Value v : callOp.getOperands()) {
        const auto *def = moduleTranslation.lookupValue(v);
        if (!def)
          return callOp.emitError("failed to lookup call operand");
        args.push_back(def);
      }

      // Calls are sequenced in CPS: the rest of the block goes into a fresh
      // return continuation receiving the mem token and the call results.
      mim::DefVec resultTypes{moduleTranslation.memType()};
      for (Type t : callOp.getResultTypes())
        resultTypes.push_back(moduleTranslation.convertType(t));
      auto *retCon = world.mut_con(resultTypes)
                         ->set("ret." + callOp.getCallee().str());

      args.push_back(retCon);
      moduleTranslation.currentLam()->app(false, callee, mim::Defs{args});

      for (auto [index, result] : llvm::enumerate(callOp.getResults()))
        moduleTranslation.mapValue(result, retCon->var(index + 1));

      moduleTranslation.setCurrentLam(retCon);
      moduleTranslation.setCurrentMem(retCon->var(mim::nat_t(0)));
      return success();
    }

    return op->emitError("unsupported func operation in MimIR translation");
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
