//===- MathToMimIRTranslation.cpp - Translate math to MimIR --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Math dialect and MimIR's
// %math plugin.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Math/MathToMimIRTranslation.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"

#include <utility>

#include <mim/plug/math/math.h>

using namespace mlir;

namespace {

namespace mimath = mim::plug::math;

/// Translates MLIR fastmath flags to a %math.Mode bit set (same mapping as in
/// the Arith dialect translation).
static mimath::Mode convertFastMathToMimMode(arith::FastMathFlags flags) {
  mim::nat_t mode = std::to_underlying(mimath::Mode::none);
  auto set = [&](arith::FastMathFlags flag, mimath::Mode m) {
    if (bitEnumContainsAll(flags, flag))
      mode |= std::to_underlying(m);
  };
  set(arith::FastMathFlags::reassoc, mimath::Mode::reassoc);
  set(arith::FastMathFlags::nnan, mimath::Mode::nnan);
  set(arith::FastMathFlags::ninf, mimath::Mode::ninf);
  set(arith::FastMathFlags::nsz, mimath::Mode::nsz);
  set(arith::FastMathFlags::arcp, mimath::Mode::arcp);
  set(arith::FastMathFlags::contract, mimath::Mode::contract);
  set(arith::FastMathFlags::afn, mimath::Mode::afn);
  set(arith::FastMathFlags::fast, mimath::Mode::fast);
  return (mimath::Mode)mode;
}

class MathToMimIRVisitor {
public:
  MathToMimIRVisitor(mim::World &world,
                     MimIR::ModuleTranslation &moduleTranslation)
      : world_(world), moduleTranslation_(moduleTranslation) {}

  template <class UnaryOp, class Id>
  LogicalResult convertUnaryOp(UnaryOp op, Id mimId) {
    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operand in MimIR translation");
    auto mode = convertFastMathToMimMode(op.getFastmath());
    const auto *result =
        world_.call(mimId, world_.lit_nat((mim::nat_t)mode), operand);
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }

  // %math.exp
  LogicalResult operator()(math::ExpOp op) {
    return convertUnaryOp(op, mimath::exp::exp);
  }
  LogicalResult operator()(math::Exp2Op op) {
    return convertUnaryOp(op, mimath::exp::exp2);
  }
  LogicalResult operator()(math::LogOp op) {
    return convertUnaryOp(op, mimath::exp::log);
  }
  LogicalResult operator()(math::Log2Op op) {
    return convertUnaryOp(op, mimath::exp::log2);
  }
  LogicalResult operator()(math::Log10Op op) {
    return convertUnaryOp(op, mimath::exp::log10);
  }

  // %math.tri
  LogicalResult operator()(math::SinOp op) {
    return convertUnaryOp(op, mimath::tri::sin);
  }
  LogicalResult operator()(math::CosOp op) {
    return convertUnaryOp(op, mimath::tri::cos);
  }
  LogicalResult operator()(math::TanOp op) {
    return convertUnaryOp(op, mimath::tri::tan);
  }
  LogicalResult operator()(math::SinhOp op) {
    return convertUnaryOp(op, mimath::tri::sinh);
  }
  LogicalResult operator()(math::CoshOp op) {
    return convertUnaryOp(op, mimath::tri::cosh);
  }
  LogicalResult operator()(math::TanhOp op) {
    return convertUnaryOp(op, mimath::tri::tanh);
  }
  LogicalResult operator()(math::AsinOp op) {
    return convertUnaryOp(op, mimath::tri::asin);
  }
  LogicalResult operator()(math::AcosOp op) {
    return convertUnaryOp(op, mimath::tri::acos);
  }
  LogicalResult operator()(math::AtanOp op) {
    return convertUnaryOp(op, mimath::tri::atan);
  }
  LogicalResult operator()(math::AsinhOp op) {
    return convertUnaryOp(op, mimath::tri::asinh);
  }
  LogicalResult operator()(math::AcoshOp op) {
    return convertUnaryOp(op, mimath::tri::acosh);
  }
  LogicalResult operator()(math::AtanhOp op) {
    return convertUnaryOp(op, mimath::tri::atanh);
  }

  // %math.rt
  LogicalResult operator()(math::SqrtOp op) {
    return convertUnaryOp(op, mimath::rt::sq);
  }
  LogicalResult operator()(math::CbrtOp op) {
    return convertUnaryOp(op, mimath::rt::cb);
  }

  // %math.er
  LogicalResult operator()(math::ErfOp op) {
    return convertUnaryOp(op, mimath::er::f);
  }
  LogicalResult operator()(math::ErfcOp op) {
    return convertUnaryOp(op, mimath::er::fc);
  }

  // %math.abs / %math.round
  LogicalResult operator()(math::AbsFOp op) {
    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operand in MimIR translation");
    auto mode = convertFastMathToMimMode(op.getFastmath());
    const auto *result = world_.call<mimath::abs>(
        world_.lit_nat((mim::nat_t)mode), operand);
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }
  LogicalResult operator()(math::FloorOp op) {
    return convertUnaryOp(op, mimath::round::f);
  }
  LogicalResult operator()(math::CeilOp op) {
    return convertUnaryOp(op, mimath::round::c);
  }
  LogicalResult operator()(math::RoundOp op) {
    return convertUnaryOp(op, mimath::round::r);
  }
  LogicalResult operator()(math::TruncOp op) {
    return convertUnaryOp(op, mimath::round::t);
  }

  // %math.pow
  LogicalResult operator()(math::PowFOp op) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getLhs());
    const auto *rhs = moduleTranslation_.lookupValue(op.getRhs());
    if (!lhs || !rhs)
      return op.emitError("failed to lookup operands in MimIR translation");
    auto mode = convertFastMathToMimMode(op.getFastmath());
    const auto *result = world_.call<mimath::pow>(
        world_.lit_nat((mim::nat_t)mode), mim::Defs{lhs, rhs});
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }

  // rsqrt(x) = 1 / sqrt(x), like the %math.slf composition.
  LogicalResult operator()(math::RsqrtOp op) {
    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operand in MimIR translation");
    const auto *modeNat = world_.lit_nat(
        (mim::nat_t)convertFastMathToMimMode(op.getFastmath()));
    const auto *type = moduleTranslation_.convertType(op.getType());
    const auto *pe = type->as<mim::App>()->arg();
    const auto *oneF64 = world_.lit(world_.annex<mimath::F64>(),
                                    llvm::bit_cast<uint64_t>(double(1.0)));
    const auto *one = world_.call(mimath::conv::f2f, pe, oneF64);
    const auto *sqrt = world_.call(mimath::rt::sq, modeNat, operand);
    const auto *result =
        world_.call(mimath::arith::div, modeNat, mim::Defs{one, sqrt});
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }

  LogicalResult operator()(Operation *op) {
    return op->emitError("math operation not supported yet");
  }

private:
  mim::World &world_;
  MimIR::ModuleTranslation &moduleTranslation_;
};

class MathDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    MathToMimIRVisitor visitor{world, moduleTranslation};
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<math::ExpOp, math::Exp2Op, math::LogOp, math::Log2Op,
              math::Log10Op, math::SinOp, math::CosOp, math::TanOp,
              math::SinhOp, math::CoshOp, math::TanhOp, math::AsinOp,
              math::AcosOp, math::AtanOp, math::AsinhOp, math::AcoshOp,
              math::AtanhOp, math::SqrtOp, math::CbrtOp, math::ErfOp,
              math::ErfcOp, math::AbsFOp, math::FloorOp, math::CeilOp,
              math::RoundOp, math::TruncOp, math::PowFOp, math::RsqrtOp>(
            [&](auto typedOp) { return visitor(typedOp); })
        .Default([&](Operation *) { return visitor(op); });
  }
};

} // namespace

void mlir::registerMathDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, math::MathDialect *dialect) {
    dialect->addInterfaces<MathDialectMimIRTranslationInterface>();
  });
}

void mlir::registerMathDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerMathDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
