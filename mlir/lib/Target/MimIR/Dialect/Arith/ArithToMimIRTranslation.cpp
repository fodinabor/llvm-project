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
#include "mlir/IR/Diagnostics.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <fe/enum.h>
#include <mim/plug/core/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/autogen.h>
#include <mim/plug/math/math.h>
#include <mim/util/types.h>

using namespace mlir;

namespace {

using namespace mim::plug;

core::Mode convertOverflowToMimMode(arith::IntegerOverflowFlags flags) {
  if (flags ==
      (arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw))
    return core::Mode::nusw;
  switch (flags) {
  case arith::IntegerOverflowFlags::none:
    return core::Mode::none;
  case arith::IntegerOverflowFlags::nsw:
    return core::Mode::nsw;
  case arith::IntegerOverflowFlags::nuw:
    return core::Mode::nuw;
  }
}

core::icmp convertCmpPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return core::icmp::e;
  case arith::CmpIPredicate::ne:
    return core::icmp::ne;
  case arith::CmpIPredicate::slt:
    return core::icmp::sl;
  case arith::CmpIPredicate::sle:
    return core::icmp::sle;
  case arith::CmpIPredicate::sgt:
    return core::icmp::sg;
  case arith::CmpIPredicate::sge:
    return core::icmp::sge;
  case arith::CmpIPredicate::ult:
    return core::icmp::ul;
  case arith::CmpIPredicate::ule:
    return core::icmp::ule;
  case arith::CmpIPredicate::ugt:
    return core::icmp::ug;
  case arith::CmpIPredicate::uge:
    return core::icmp::uge;
  }
}

math::Mode convertFastMathToMimMode(arith::FastMathFlags flags) {
  using namespace fe;
  mim::nat_t mode = (mim::nat_t)math::Mode::none;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::none))
    mode = mode | math::Mode::none;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::reassoc))
    mode = mode | math::Mode::reassoc;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::nnan))
    mode = mode | math::Mode::nnan;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::ninf))
    mode = mode | math::Mode::ninf;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::nsz))
    mode = mode | math::Mode::nsz;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::arcp))
    mode = mode | math::Mode::arcp;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::contract))
    mode = mode | math::Mode::contract;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::afn))
    mode = mode | math::Mode::afn;
  if (bitEnumContainsAll(flags, arith::FastMathFlags::fast))
    mode = mode | math::Mode::fast;
  return (math::Mode)mode;
}

math::cmp convertFCmpPredicate(arith::CmpFPredicate pred) {
  switch (pred) {
  case arith::CmpFPredicate::AlwaysFalse:
    return math::cmp::f;
  case arith::CmpFPredicate::OEQ:
    return math::cmp::e;
  case arith::CmpFPredicate::OGT:
    return math::cmp::g;
  case arith::CmpFPredicate::OGE:
    return math::cmp::ge;
  case arith::CmpFPredicate::OLT:
    return math::cmp::l;
  case arith::CmpFPredicate::OLE:
    return math::cmp::le;
  case arith::CmpFPredicate::ONE:
    return math::cmp::ne;
  case arith::CmpFPredicate::ORD:
    return math::cmp::o;
  case arith::CmpFPredicate::UEQ:
    return math::cmp::ue;
  case arith::CmpFPredicate::UGT:
    return math::cmp::ug;
  case arith::CmpFPredicate::UGE:
    return math::cmp::uge;
  case arith::CmpFPredicate::ULT:
    return math::cmp::ul;
  case arith::CmpFPredicate::ULE:
    return math::cmp::ule;
  case arith::CmpFPredicate::UNE:
    return math::cmp::une;
  case arith::CmpFPredicate::UNO:
    return math::cmp::u;
  case arith::CmpFPredicate::AlwaysTrue:
    return math::cmp::t;
  }
}

// TODO: no tensor/vector variants supported thus far!
// TODO: no integer div/rem support so far, missing mem!!
class ArithToMimIRVisitor {
public:
  ArithToMimIRVisitor(mim::World &world,
                      MimIR::ModuleTranslation &moduleTranslation)
      : world_(world), moduleTranslation_(moduleTranslation) {}

  template <class WrapOp>
  LogicalResult convertWrapOp(WrapOp &op, core::wrap mimId) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getLhs());
    const auto *rhs = moduleTranslation_.lookupValue(op.getRhs());
    if (!lhs || !rhs)
      return op.emitError("failed to lookup operands in MimIR translation");

    auto mode = convertOverflowToMimMode(op.getOverflowFlags());
    const auto *result = world_.call(mimId, world_.lit_nat((mim::nat_t)mode),
                                     mim::Defs{lhs, rhs});
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // %core.wrap(add, sub, mul, shl)
  LogicalResult operator()(arith::AddIOp &op) {
    return convertWrapOp(op, core::wrap::add);
  }
  LogicalResult operator()(arith::SubIOp &op) {
    return convertWrapOp(op, core::wrap::sub);
  }
  LogicalResult operator()(arith::MulIOp &op) {
    return convertWrapOp(op, core::wrap::mul);
  }
  LogicalResult operator()(arith::ShLIOp &op) {
    return convertWrapOp(op, core::wrap::shl);
  }

  template <class BinOp, class Id>
  LogicalResult convertBinaryOp(BinOp &op, Id mimId) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getLhs());
    const auto *rhs = moduleTranslation_.lookupValue(op.getRhs());
    if (!lhs || !rhs)
      return op.emitError("failed to lookup operands in MimIR translation");
    const auto *result = world_.call(mimId, mim::Defs{lhs, rhs});
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }

  // core.shr.(a,l)
  LogicalResult operator()(arith::ShRUIOp &op) {
    return convertBinaryOp(op, core::shr::l);
  }
  LogicalResult operator()(arith::ShRSIOp &op) {
    return convertBinaryOp(op, core::shr::a);
  }

  template <class WrapOp>
  LogicalResult convertBit2Op(WrapOp &op, core::bit2 mimId) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getLhs());
    const auto *rhs = moduleTranslation_.lookupValue(op.getRhs());
    if (!lhs || !rhs)
      return op.emitError("failed to lookup operands in MimIR translation");

    // Todo: why does bit2 have a mode??
    const auto *result =
        world_.call(mimId, world_.lit_nat((mim::nat_t)core::Mode::none),
                    mim::Defs{lhs, rhs});
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // core.bit2( f,      nor, nciff, nfst, niff, nsnd, xor_, nand,
  //            and_,  nxor,   snd,  iff,  fst, ciff,  or_,    t)
  LogicalResult operator()(arith::AndIOp &op) {
    return convertBit2Op(op, core::bit2::and_);
  }
  LogicalResult operator()(arith::OrIOp &op) {
    return convertBit2Op(op, core::bit2::or_);
  }
  LogicalResult operator()(arith::XOrIOp &op) {
    return convertBit2Op(op, core::bit2::xor_);
  }

  // core.icmp(xygle = f,  xyglE = e,   xygLe,      xygLE,
  //           xyGle,      xyGlE,       xyGLe,      xyGLE,
  //           xYgle,      xYglE,       xYgLe = sl, xYgLE = sle,
  //           xYGle = ug, xYGlE = uge, xYGLe,      xYGLE,
  //           Xygle,      XyglE,       XygLe = ul, XygLE = ule,
  //           XyGle = sg, XyGlE = sge, XyGLe,      XyGLE,
  //           XYgle,      XYglE,       XYgLe,      XYgLE,
  //           XYGle,      XYGlE,       XYGLe = ne, XYGLE = t):
  LogicalResult operator()(arith::CmpIOp &op) {
    return convertBinaryOp(op, convertCmpPredicate(op.getPredicate()));
  }

  // core.extrema(sm=umin, sM=umax, Sm=smin, SM=smax)
  LogicalResult operator()(arith::MinUIOp &op) {
    return convertBinaryOp(op, core::extrema::umin);
  }
  LogicalResult operator()(arith::MaxUIOp &op) {
    return convertBinaryOp(op, core::extrema::umax);
  }
  LogicalResult operator()(arith::MinSIOp &op) {
    return convertBinaryOp(op, core::extrema::smin);
  }
  LogicalResult operator()(arith::MaxSIOp &op) {
    return convertBinaryOp(op, core::extrema::smax);
  }

  template <class FastOp, class Id>
  LogicalResult convertBinaryFastOp(FastOp &op, Id mimId) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getLhs());
    const auto *rhs = moduleTranslation_.lookupValue(op.getRhs());
    if (!lhs || !rhs)
      return op.emitError("failed to lookup operands in MimIR translation");

    auto mode = convertFastMathToMimMode(op.getFastmath());
    const auto *result = world_.call(mimId, world_.lit_nat((mim::nat_t)mode),
                                     mim::Defs{lhs, rhs});
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // math.arith(add, sub, mul, div, rem)
  LogicalResult operator()(arith::AddFOp &op) {
    return convertBinaryFastOp(op, math::arith::add);
  }
  LogicalResult operator()(arith::SubFOp &op) {
    return convertBinaryFastOp(op, math::arith::sub);
  }
  LogicalResult operator()(arith::MulFOp &op) {
    return convertBinaryFastOp(op, math::arith::mul);
  }
  LogicalResult operator()(arith::DivFOp &op) {
    return convertBinaryFastOp(op, math::arith::div);
  }
  LogicalResult operator()(arith::RemFOp &op) {
    return convertBinaryFastOp(op, math::arith::rem);
  }

  // math.cmp(ugle =   f, uglE =   e, ugLe =   l, ugLE =  le,
  //          uGle =   g, uGlE =  ge, uGLe =  ne, uGLE =   o,
  //          Ugle =   u, UglE =  ue, UgLe =  ul, UgLE = ule,
  //          UGle =  ug, UGlE = uge, UGLe = une, UGLE =   t)
  LogicalResult operator()(arith::CmpFOp &op) {
    return convertBinaryFastOp(op, convertFCmpPredicate(op.getPredicate()));
  }

  // math.extrema(im = fmin,       iM = fmax,
  //              Im = ieee754min, IM = ieee754max)
  LogicalResult operator()(arith::MinimumFOp &op) {
    return convertBinaryFastOp(op, math::extrema::fmin);
  }
  LogicalResult operator()(arith::MaximumFOp &op) {
    return convertBinaryFastOp(op, math::extrema::fmax);
  }
  LogicalResult operator()(arith::MinNumFOp &op) {
    return convertBinaryFastOp(op, math::extrema::ieee754min);
  }
  LogicalResult operator()(arith::MaxNumFOp &op) {
    return convertBinaryFastOp(op, math::extrema::ieee754max);
  }

  template <class FastOp, class Id>
  LogicalResult convertUnaryFastOp(FastOp &op, Id mimId) {
    const auto *lhs = moduleTranslation_.lookupValue(op.getOperand());
    if (!lhs)
      return op.emitError("failed to lookup operands in MimIR translation");

    auto mode = convertFastMathToMimMode(op.getFastmath());
    const auto *result =
        world_.call(mimId, world_.lit_nat((mim::nat_t)mode), lhs);
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // math.minus
  LogicalResult operator()(arith::NegFOp &op) {
    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operands in MimIR translation");

    const auto *result = world_.call<math::minus>(
        convertFastMathToMimMode(op.getFastmath()), operand);
    moduleTranslation_.mapValue(op.getResult(), result);
    return success();
  }

  template <class ConvOp, class Id>
  LogicalResult convertConvOp(ConvOp &op, Type targetType, Id mimId) {
    const auto *dType = moduleTranslation_.convertType(targetType);
    const mim::App *fOrI = dType->as<mim::App>();

    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operands in MimIR translation");

    const auto *result = world_.call(mimId, fOrI->arg(), operand);
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // math.conv(f2f,u2f,s2f,f2s,f2u)
  // Todo: rounding modes?
  LogicalResult operator()(arith::TruncFOp &op) {
    return convertConvOp(op, op.getType(), math::conv::f2f);
  }
  LogicalResult operator()(arith::ExtFOp &op) {
    return convertConvOp(op, op.getType(), math::conv::f2f);
  }
  LogicalResult operator()(arith::FPToSIOp &op) {
    return convertConvOp(op, op.getType(), math::conv::f2s);
  }
  LogicalResult operator()(arith::FPToUIOp &op) {
    return convertConvOp(op, op.getType(), math::conv::f2u);
  }
  LogicalResult operator()(arith::UIToFPOp &op) {
    return convertConvOp(op, op.getType(), math::conv::u2f);
  }
  LogicalResult operator()(arith::SIToFPOp &op) {
    return convertConvOp(op, op.getType(), math::conv::s2f);
  }

  // core.conv(s, u)
  LogicalResult operator()(arith::TruncIOp &op) {
    if (bitEnumContainsAll(op.getOverflowFlags(),
                           arith::IntegerOverflowFlags::nsw))
      return convertConvOp(op, op.getType(), core::conv::s);
    return convertConvOp(op, op.getType(), core::conv::u);
  }
  LogicalResult operator()(arith::ExtSIOp &op) {
    return convertConvOp(op, op.getType(), core::conv::s);
  }
  LogicalResult operator()(arith::ExtUIOp &op) {
    return convertConvOp(op, op.getType(), core::conv::u);
  }

  // iMM to index
  LogicalResult operator()(arith::IndexCastOp &op) {
    return convertConvOp(op, op.getType(), core::conv::s);
  }
  LogicalResult operator()(arith::IndexCastUIOp &op) {
    return convertConvOp(op, op.getType(), core::conv::u);
  }

  // core.bitcast
  LogicalResult operator()(arith::BitcastOp &op) {
    const auto *dType = moduleTranslation_.convertType(op.getType());
    const auto *operand = moduleTranslation_.lookupValue(op.getOperand());
    if (!operand)
      return op.emitError("failed to lookup operands in MimIR translation");

    const auto *result = world_.call<core::bitcast>(dType, operand);
    moduleTranslation_.mapValue(op.getResult(), result);

    return success();
  }

  // special ops
  // int/float constant
  LogicalResult operator()(arith::ConstantOp &op) {
    if (auto valueAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      int64_t value = valueAttr.getInt();
      if (op.getType().isInteger() && op.getType().getIntOrFloatBitWidth() == 1)
        value = value ? 1 : 0;
      const auto *lit =
          world_.lit(moduleTranslation_.convertType(op.getType()), value);
      moduleTranslation_.mapValue(op.getResult(), lit);
      return success();
    }
    if (auto valueAttr = dyn_cast<FloatAttr>(op.getValue())) {
      auto value = valueAttr.getValue();
      const auto *lit = world_.lit(moduleTranslation_.convertType(op.getType()),
                                   value.bitcastToAPInt().getZExtValue());
      moduleTranslation_.mapValue(op.getResult(), lit);
      return success();
    }
    return failure();
  }

  // (ff, tt)#cond
  LogicalResult operator()(arith::SelectOp &op) {
    const auto *cond = moduleTranslation_.lookupValue(op->getOperand(0));
    const auto *tt = moduleTranslation_.lookupValue(op->getOperand(1));
    const auto *ff = moduleTranslation_.lookupValue(op->getOperand(2));
    moduleTranslation_.mapValue(op.getResult(), world_.select(cond, tt, ff));
    return success();
  }

  LogicalResult operator()(Operation *op) {
    return op->emitError("arith operation not supported, yet");
  }

private:
  mim::World &world_;
  MimIR::ModuleTranslation &moduleTranslation_;
};

class ArithDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    ArithToMimIRVisitor visitor{world, moduleTranslation};
    return llvm::TypeSwitch<Operation &, LogicalResult>(*op)
        .Case<
#define GET_OP_LIST
#include <mlir/Dialect/Arith/IR/ArithOps.cpp.inc>
            >(visitor)
        .Default([&](Operation &) {
          return op->emitError("arith operation not supported, yet");
        });
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
