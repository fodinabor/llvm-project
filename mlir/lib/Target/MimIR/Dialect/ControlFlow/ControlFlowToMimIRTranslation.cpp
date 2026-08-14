//===- ControlFlowToMimIRTranslation.cpp - Translate cf to MimIR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR ControlFlow dialect and
// MimIR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/ControlFlow/ControlFlowToMimIRTranslation.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

using namespace mlir;

namespace {

/// Looks up the MimIR values for the given block-argument operands. Returns
/// failure and emits an error at `op` if any operand is unmapped.
static LogicalResult
lookupOperands(Operation *op, OperandRange operands,
               MimIR::ModuleTranslation &moduleTranslation,
               mim::DefVec &defs) {
  for (Value v : operands) {
    const auto *def = moduleTranslation.lookupValue(v);
    if (!def)
      return op->emitError("failed to lookup operand in MimIR translation");
    defs.push_back(def);
  }
  return success();
}

/// Returns a mem-only continuation jumping to `dest` with `args`. If `args`
/// is empty, `dest` itself already has the right shape `Cn %mem.M`.
static const mim::Def *wrapSuccessor(mim::World &world,
                                     MimIR::ModuleTranslation &mt,
                                     mim::Lam *dest, mim::Defs args) {
  if (args.empty())
    return dest;
  auto *wrap = world.mut_con(mt.memType());
  mim::DefVec destArgs{wrap->var()};
  destArgs.insert(destArgs.end(), args.begin(), args.end());
  wrap->app(false, dest, mim::Defs{destArgs});
  return wrap;
}

class ControlFlowDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
      mim::Lam *dest = moduleTranslation.lookupBlock(branchOp.getDest());
      assert(dest && "successor block must have been created upfront");

      mim::DefVec args{moduleTranslation.currentMem()};
      if (failed(lookupOperands(op, branchOp.getDestOperands(),
                                moduleTranslation, args)))
        return failure();

      moduleTranslation.currentLam()->app(false, dest, mim::Defs{args});
      return success();
    }

    if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
      const auto *cond = moduleTranslation.lookupValue(condBranchOp.getCondition());
      if (!cond)
        return op->emitError("failed to lookup condition in MimIR translation");

      mim::Lam *trueDest =
          moduleTranslation.lookupBlock(condBranchOp.getTrueDest());
      mim::Lam *falseDest =
          moduleTranslation.lookupBlock(condBranchOp.getFalseDest());
      assert(trueDest && falseDest &&
             "successor blocks must have been created upfront");

      mim::DefVec trueArgs, falseArgs;
      if (failed(lookupOperands(op, condBranchOp.getTrueDestOperands(),
                                moduleTranslation, trueArgs)) ||
          failed(lookupOperands(op, condBranchOp.getFalseDestOperands(),
                                moduleTranslation, falseArgs)))
        return failure();

      // Lam::branch applies the selected continuation to the mem token, so
      // both targets must be of type `Cn %mem.M`; forward successor arguments
      // through wrappers where necessary.
      const auto *t = wrapSuccessor(world, moduleTranslation, trueDest,
                                    mim::Defs{trueArgs});
      const auto *f = wrapSuccessor(world, moduleTranslation, falseDest,
                                    mim::Defs{falseArgs});
      moduleTranslation.currentLam()->branch(false, cond, t, f,
                                             moduleTranslation.currentMem());
      return success();
    }

    return op->emitError(
        "unsupported ControlFlow operation in MimIR translation");
  }
};

} // namespace

void mlir::registerControlFlowDialectTranslationMimIR(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cf::ControlFlowDialect *dialect) {
    dialect->addInterfaces<ControlFlowDialectMimIRTranslationInterface>();
  });
}

void mlir::registerControlFlowDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerControlFlowDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
