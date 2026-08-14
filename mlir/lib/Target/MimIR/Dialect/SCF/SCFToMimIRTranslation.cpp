//===- SCFToMimIRTranslation.cpp - Translate SCF to MimIR ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR SCF dialect and MimIR:
//
// * scf.if  -> Lam::branch on continuations for the two regions that join in
//              a fresh continuation receiving the results
// * scf.for -> a loop-header continuation carrying (mem, iv, iter_args...)
// * scf.yield -> the jump back to the join/header continuation
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/SCF/SCFToMimIRTranslation.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/StateStack.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"

#include <mim/plug/core/core.h>

using namespace mlir;

namespace {

using namespace mim::plug;

/// Records where an `scf.yield` nested in `op` must jump to: the join
/// continuation of an `scf.if` or the header continuation of an `scf.for`
/// (in which case `prefix` carries the incremented induction variable).
struct ScfYieldFrame : public StateStackFrameBase<ScfYieldFrame> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScfYieldFrame)

  ScfYieldFrame(Operation *op, mim::Lam *target, mim::DefVec prefix)
      : op(op), target(target), prefix(std::move(prefix)) {}

  Operation *op;
  mim::Lam *target;
  mim::DefVec prefix;
};

class ScfToMimIRVisitor {
public:
  ScfToMimIRVisitor(mim::World &world,
                    MimIR::ModuleTranslation &moduleTranslation)
      : world_(world), moduleTranslation_(moduleTranslation) {}

  /// Converts the single block of `region` into `lam`.
  LogicalResult convertRegion(Region &region, mim::Lam *lam) {
    if (!region.hasOneBlock())
      return failure();
    moduleTranslation_.mapBlock(&region.front(), lam);
    return moduleTranslation_.convertBlock(region.front(),
                                           /*ignoreArguments=*/true);
  }

  // scf.yield: jump to the recorded target continuation.
  LogicalResult operator()(scf::YieldOp op) {
    ScfYieldFrame *frame = nullptr;
    moduleTranslation_.stackWalk<ScfYieldFrame>([&](ScfYieldFrame &f) {
      frame = &f;
      return WalkResult::interrupt();
    });
    if (!frame || frame->op != op->getParentOp())
      return op.emitError("no enclosing scf construct in MimIR translation");

    mim::DefVec args{moduleTranslation_.currentMem()};
    args.insert(args.end(), frame->prefix.begin(), frame->prefix.end());
    for (Value v : op.getOperands()) {
      const auto *def = moduleTranslation_.lookupValue(v);
      if (!def)
        return op.emitError("failed to lookup yielded value");
      args.push_back(def);
    }
    moduleTranslation_.currentLam()->app(false, frame->target,
                                         mim::Defs{args});
    return success();
  }

  // scf.if: branch on continuations for the regions; both jump to a fresh
  // join continuation receiving (mem, results...).
  LogicalResult operator()(scf::IfOp op) {
    const auto *cond = moduleTranslation_.lookupValue(op.getCondition());
    if (!cond)
      return op.emitError("failed to lookup condition");

    mim::DefVec joinDom{moduleTranslation_.memType()};
    for (Type t : op.getResultTypes())
      joinDom.push_back(moduleTranslation_.convertType(t));
    auto *join = world_.mut_con(joinDom)->set("if.join");

    auto *thenLam =
        world_.mut_con(moduleTranslation_.memType())->set("if.then");
    bool hasElse = !op.getElseRegion().empty();
    if (!hasElse && op.getNumResults() != 0)
      return op.emitError("expected an else region for an if with results");
    mim::Lam *elseLam =
        hasElse ? world_.mut_con(moduleTranslation_.memType())->set("if.else")
                : join;

    moduleTranslation_.currentLam()->branch(false, cond, thenLam, elseLam,
                                            moduleTranslation_.currentMem());

    {
      MimIR::ModuleTranslation::SaveStack<ScfYieldFrame> frame(
          moduleTranslation_, op.getOperation(), join, mim::DefVec{});
      if (failed(convertRegion(op.getThenRegion(), thenLam)))
        return op.emitError("failed to convert the then region");
      if (hasElse && failed(convertRegion(op.getElseRegion(), elseLam)))
        return op.emitError("failed to convert the else region");
    }

    moduleTranslation_.setCurrentLam(join);
    moduleTranslation_.setCurrentMem(join->var(mim::nat_t(0)));
    for (auto [index, result] : llvm::enumerate(op.getResults()))
      moduleTranslation_.mapValue(result, join->var(index + 1));
    return success();
  }

  // scf.for: a header continuation carrying (mem, iv, iter_args...) branches
  // between the body and an exit continuation receiving (mem, iter_args...).
  LogicalResult operator()(scf::ForOp op) {
    const auto *lb = moduleTranslation_.lookupValue(op.getLowerBound());
    const auto *ub = moduleTranslation_.lookupValue(op.getUpperBound());
    const auto *step = moduleTranslation_.lookupValue(op.getStep());
    if (!lb || !ub || !step)
      return op.emitError("failed to lookup loop bounds");

    const auto *ivType =
        moduleTranslation_.convertType(op.getInductionVar().getType());
    mim::DefVec headerDom{moduleTranslation_.memType(), ivType};
    mim::DefVec exitDom{moduleTranslation_.memType()};
    for (Value init : op.getInitArgs()) {
      const auto *type = moduleTranslation_.convertType(init.getType());
      headerDom.push_back(type);
      exitDom.push_back(type);
    }
    auto *header = world_.mut_con(headerDom)->set("for.head");
    auto *exit = world_.mut_con(exitDom)->set("for.exit");
    auto *body = world_.mut_con(moduleTranslation_.memType())->set("for.body");

    // Enter the loop.
    mim::DefVec entryArgs{moduleTranslation_.currentMem(), lb};
    for (Value init : op.getInitArgs()) {
      const auto *def = moduleTranslation_.lookupValue(init);
      if (!def)
        return op.emitError("failed to lookup iter_args initial value");
      entryArgs.push_back(def);
    }
    moduleTranslation_.currentLam()->app(false, header, mim::Defs{entryArgs});

    // The header compares the induction variable against the upper bound and
    // branches into the body or to the exit, forwarding the iteration state.
    const auto *iv = header->var(1);
    const auto *cont = world_.call(core::icmp::sl, mim::Defs{iv, ub});
    const mim::Def *exitTarget = exit;
    if (!op.getInitArgs().empty()) {
      auto *wrap = world_.mut_con(moduleTranslation_.memType());
      mim::DefVec exitArgs{wrap->var()};
      for (size_t i = 0; i < op.getInitArgs().size(); ++i)
        exitArgs.push_back(header->var(i + 2));
      wrap->app(false, exit, mim::Defs{exitArgs});
      exitTarget = wrap;
    }
    header->branch(false, cont, body, exitTarget,
                   header->var(mim::nat_t(0)));

    // The body region: the induction variable and iter_args map to the
    // header's vars; scf.yield jumps back to the header with the incremented
    // induction variable.
    moduleTranslation_.mapValue(op.getInductionVar(), iv);
    for (auto [index, arg] : llvm::enumerate(op.getRegionIterArgs()))
      moduleTranslation_.mapValue(arg, header->var(index + 2));

    const auto *nextIV = world_.call(core::wrap::add, world_.lit_nat(0),
                                     mim::Defs{iv, step});
    {
      MimIR::ModuleTranslation::SaveStack<ScfYieldFrame> frame(
          moduleTranslation_, op.getOperation(), header, mim::DefVec{nextIV});
      if (failed(convertRegion(op.getBodyRegion(), body)))
        return op.emitError("failed to convert the loop body");
    }

    moduleTranslation_.setCurrentLam(exit);
    moduleTranslation_.setCurrentMem(exit->var(mim::nat_t(0)));
    for (auto [index, result] : llvm::enumerate(op.getResults()))
      moduleTranslation_.mapValue(result, exit->var(index + 1));
    return success();
  }

  LogicalResult operator()(Operation *op) {
    return op->emitError("scf operation not supported yet");
  }

private:
  mim::World &world_;
  MimIR::ModuleTranslation &moduleTranslation_;
};

class SCFDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    ScfToMimIRVisitor visitor{world, moduleTranslation};
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<scf::YieldOp, scf::IfOp, scf::ForOp>(
            [&](auto typedOp) { return visitor(typedOp); })
        .Default([&](Operation *) { return visitor(op); });
  }
};

} // namespace

void mlir::registerSCFDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    dialect->addInterfaces<SCFDialectMimIRTranslationInterface>();
  });
}

void mlir::registerSCFDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerSCFDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
