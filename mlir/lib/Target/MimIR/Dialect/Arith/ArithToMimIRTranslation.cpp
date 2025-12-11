//===- ArithToLLVMIRTranslation.cpp - Translate Arith to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Arith dialect and MimIR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Arith/ArithToMimIRTranslation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "mim/plug/core/core.h"

using namespace mlir;

namespace {

class ArithDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
      const auto *lhs = moduleTranslation.lookupValue(addOp.getLhs());
      const auto *rhs = moduleTranslation.lookupValue(addOp.getRhs());
      if (!lhs || !rhs)
        return addOp.emitError(
            "failed to lookup operands in MimIR translation");
      // todo: handle overflow modes
      auto mode = mim::plug::core::Mode::none;
      const auto *result =
          world.call(mim::plug::core::wrap::add,
                     world.lit_nat((mim::nat_t)mode), mim::Defs{lhs, rhs});
      moduleTranslation.mapValue(addOp.getResult(), result);
      return success();
    }
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      auto valueAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!valueAttr)
        return constOp.emitError(
            "only integer constants are supported in MimIR translation");
      int64_t value = valueAttr.getInt();
      const auto *lit =
          world.lit(moduleTranslation.convertType(constOp.getType()), value);
      moduleTranslation.mapValue(constOp.getResult(), lit);
      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::registerArithDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    dialect->addInterfaces<ArithDialectMimIRTranslationInterface>();
  });
}

void mlir::registerArithDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerArithDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
