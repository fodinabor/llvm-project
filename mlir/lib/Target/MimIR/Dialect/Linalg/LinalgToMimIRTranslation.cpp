//===- LinalgToMimIRTranslation.cpp - Translate linalg to MimIR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Linalg dialect and
// MimIR, targeting the high-level SSA-world axioms of MimIR's tensor plugin:
//
// * linalg.fill      -> a pack `‹shape; value›`
// * linalg.transpose -> %tensor.map_reduce with a permuted read map
// * linalg.matmul    -> %tensor.dot_product over a %tensor.Ring
// * linalg.generic   -> %tensor.map_reduce, translating the body region into
//                       the fold function
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/MimIR/Dialect/Linalg/LinalgToMimIRTranslation.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"

#include <mim/lattice.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/tuple.h>

using namespace mlir;

namespace {

using namespace mim::plug;

/// Extracts the dim positions of a projected permutation / broadcast map,
/// e.g. `(d0, d1) -> (d1, d0)` yields `[1, 0]`. Fails on non-dim results.
static FailureOr<SmallVector<int64_t>> projectedDims(AffineMap map) {
  SmallVector<int64_t> dims;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return failure();
    dims.push_back(dimExpr.getPosition());
  }
  return dims;
}

/// Returns true if `def` is a (nested) pack or literal that is all zero bits,
/// or an uninitialized (bottom) value. Used to detect `outs` operands that do
/// not contribute to the result.
static bool isZeroOrUndef(const mim::Def *def) {
  while (auto pack = def->isa<mim::Pack>())
    def = pack->body();
  if (auto lit = def->isa<mim::Lit>())
    return lit->get() == 0;
  return def->isa<mim::Bot>() != nullptr;
}

class LinalgToMimIRVisitor {
public:
  LinalgToMimIRVisitor(mim::World &world,
                       MimIR::ModuleTranslation &moduleTranslation)
      : world_(world), moduleTranslation_(moduleTranslation) {}

  //===--------------------------------------------------------------------===//
  // Helpers for the %tensor plugin.
  //===--------------------------------------------------------------------===//

  /// Builds a `«r; Nat»` tuple of literals.
  const mim::Def *natTuple(ArrayRef<int64_t> values) {
    mim::DefVec nats;
    for (int64_t v : values)
      nats.push_back(world_.lit_nat(v));
    return world_.tuple(mim::Defs{nats});
  }

  /// Builds the read/write map `%tensor.proj_map @(n, r) subs`: input axis `j`
  /// is indexed by loop variable `subs[j]` of the `n` loop variables.
  const mim::Def *projMap(uint64_t numLoops, ArrayRef<int64_t> subs) {
    const auto *pm = world_.annex<mim::plug::tensor::proj_map>();
    pm = world_.app(pm, mim::Defs{world_.lit_nat(numLoops),
                                  world_.lit_nat(subs.size())});
    return world_.app(pm, natTuple(subs));
  }

  /// Interprets a value as the loop-invariant scalar of a semi-affine
  /// operation: a non-negative constant becomes a Nat literal, a bound symbol
  /// value is bitcast to Nat. Returns nullptr otherwise.
  const mim::Def *semiAffineNat(AffineExpr expr,
                                ArrayRef<const mim::Def *> symbols) {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr))
      return cst.getValue() >= 0 ? world_.lit_nat(cst.getValue()) : nullptr;
    if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
      if (sym.getPosition() < symbols.size())
        return world_.call<core::bitcast>(world_.type_nat(),
                                          symbols[sym.getPosition()]);
    return nullptr;
  }

  /// Translates an affine expression into a `%affine.index` value over the
  /// given loop variables (`loopPos` maps MLIR dim positions to loop-vector
  /// positions). Symbols are the "curried" arguments of an affine map: they
  /// are bound to the loop-invariant `symbols` values, which the resulting
  /// map lambda simply closes over.
  FailureOr<const mim::Def *> affineExpr(Operation *op, AffineExpr expr,
                                         ArrayRef<const mim::Def *> loopVars,
                                         ArrayRef<int64_t> loopPos,
                                         ArrayRef<const mim::Def *> symbols) {
    namespace mimaffine = mim::plug::affine;
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      return loopVars[loopPos[dim.getPosition()]];
    if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
      if (sym.getPosition() >= symbols.size())
        return op->emitError("no value bound for affine symbol");
      // A loop-invariant "constant" index from the symbol's runtime value.
      return world_.call<mimaffine::constant>(world_.call<core::bitcast>(
          world_.type_nat(), symbols[sym.getPosition()]));
    }
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      const auto *c = world_.call<mimaffine::constant>(
          world_.lit_nat(std::abs(cst.getValue())));
      if (cst.getValue() < 0)
        c = world_.call(mimaffine::op::neg, c);
      return c;
    }
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin)
      return op->emitError("unsupported affine expression");

    if (expr.getKind() == AffineExprKind::Add) {
      auto lhs = affineExpr(op, bin.getLHS(), loopVars, loopPos, symbols);
      auto rhs = affineExpr(op, bin.getRHS(), loopVars, loopPos, symbols);
      if (failed(lhs) || failed(rhs))
        return failure();
      return world_.call(mimaffine::op::add, mim::Defs{*lhs, *rhs});
    }

    // Mul/Mod/FloorDiv/CeilDiv: the scalar side must be constant with regard
    // to the loop indices — a constant or a bound symbol. Affine canonical
    // form keeps it on the right (for Mul, allow either side).
    const mim::Def *scalar = semiAffineNat(bin.getRHS(), symbols);
    AffineExpr indexSide = bin.getLHS();
    if (!scalar && expr.getKind() == AffineExprKind::Mul) {
      scalar = semiAffineNat(bin.getLHS(), symbols);
      indexSide = bin.getRHS();
    }
    if (!scalar) {
      // A negative constant multiplier is expressible via negation.
      if (auto rhsCst = dyn_cast<AffineConstantExpr>(bin.getRHS());
          rhsCst && expr.getKind() == AffineExprKind::Mul) {
        auto lhs = affineExpr(op, bin.getLHS(), loopVars, loopPos, symbols);
        if (failed(lhs))
          return failure();
        const auto *m = world_.call(
            mimaffine::semiop::mul,
            mim::Defs{*lhs, world_.lit_nat(std::abs(rhsCst.getValue()))});
        return world_.call(mimaffine::op::neg, m);
      }
      return op->emitError(
          "expected a constant or symbol scalar in a semi-affine expression");
    }
    auto index = affineExpr(op, indexSide, loopVars, loopPos, symbols);
    if (failed(index))
      return failure();
    switch (expr.getKind()) {
    case AffineExprKind::Mul:
      return world_.call(mimaffine::semiop::mul, mim::Defs{*index, scalar});
    case AffineExprKind::Mod:
      return world_.call(mimaffine::semiop::mod, mim::Defs{*index, scalar});
    case AffineExprKind::FloorDiv:
      return world_.call(mimaffine::semiop::floordiv,
                         mim::Defs{*index, scalar});
    case AffineExprKind::CeilDiv:
      return world_.call(mimaffine::semiop::ceildiv,
                         mim::Defs{*index, scalar});
    default:
      return op->emitError("unsupported affine expression");
    }
  }

  /// Builds a `[«numLoops; %affine.index»] → «r; %affine.index»` read map for
  /// an arbitrary affine indexing map. Pure dim projections use
  /// %tensor.proj_map; everything else becomes a lambda over %affine ops.
  /// `symbols` supplies the values the map's symbols are bound to (linalg
  /// indexing maps carry none by construction, but e.g. affine.apply-style
  /// clients have operands for them).
  FailureOr<const mim::Def *>
  affineIndexMap(Operation *op, AffineMap map, uint64_t numLoops,
                 ArrayRef<int64_t> loopPos,
                 ArrayRef<const mim::Def *> symbols = {}) {
    if (auto dims = projectedDims(map); succeeded(dims)) {
      SmallVector<int64_t> subs;
      for (int64_t dim : *dims)
        subs.push_back(loopPos[dim]);
      return projMap(numLoops, subs);
    }

    const auto *indexType = world_.annex<mim::plug::affine::index>();
    auto *lam = world_.mut_lam(world_.arr(numLoops, indexType),
                               world_.arr(map.getNumResults(), indexType));
    lam->set("affine_map");
    SmallVector<const mim::Def *> loopVars;
    for (uint64_t i = 0; i < numLoops; ++i)
      loopVars.push_back(world_.extract(lam->var(), numLoops, i));

    mim::DefVec results;
    for (AffineExpr expr : map.getResults()) {
      auto def = affineExpr(op, expr, loopVars, loopPos, symbols);
      if (failed(def))
        return failure();
      results.push_back(*def);
    }
    lam->set(true, world_.tuple(mim::Defs{results}));
    return (const mim::Def *)lam;
  }

  /// Invokes `%tensor.map_reduce`, passing all implicit arguments explicitly
  /// (they are not inferable in general).
  const mim::Def *
  mapReduce(const mim::Def *outElemType, uint64_t outRank, uint64_t redRank,
            ArrayRef<int64_t> outShape, ArrayRef<int64_t> loopBounds,
            ArrayRef<const mim::Def *> inElemTypes,
            ArrayRef<int64_t> inRanks, ArrayRef<const mim::Def *> inShapes,
            const mim::Def *fold, const mim::Def *init, const mim::Def *mapOut,
            ArrayRef<const mim::Def *> maps,
            ArrayRef<const mim::Def *> inputs) {
    const auto *mr = world_.annex<mim::plug::tensor::map_reduce>();
    mr = world_.app(mr, world_.lit_nat(inputs.size()));
    mr = world_.app(mr, mim::Defs{outElemType, world_.lit_nat(outRank),
                                  world_.lit_nat(redRank)});
    mr = world_.app(mr, mim::Defs{natTuple(outShape), natTuple(loopBounds)});
    mr = world_.app(mr,
                    mim::Defs{world_.tuple(mim::Defs{inElemTypes}),
                              natTuple(inRanks),
                              world_.tuple(mim::Defs{inShapes})});
    mr = world_.app(mr, mim::Defs{fold, init});
    mr = world_.app(mr, mapOut);
    mr = world_.app(mr, world_.tuple(mim::Defs{maps}));
    return world_.app(mr, world_.tuple(mim::Defs{inputs}));
  }

  /// Builds a direct-style `[T, T] → T` lambda for the given binary axiom
  /// application, used for %tensor.Ring operations.
  template <class Id>
  const mim::Def *binaryRingOp(const mim::Def *elemType, Id mimId,
                               std::string_view name) {
    auto *lam = world_.mut_lam(mim::Defs{elemType, elemType}, elemType);
    lam->set(std::string(name));
    const auto *body =
        world_.call(mimId, world_.lit_nat(0),
                    mim::Defs{lam->var(mim::nat_t(0)), lam->var(1)});
    lam->set(true, body);
    return lam;
  }

  /// Builds the fold `f (acc, ys) = ys#0` that ignores its accumulator.
  const mim::Def *copyFold(const mim::Def *elemType, std::string_view name) {
    auto *copy =
        world_.mut_fun(mim::Defs{elemType, world_.arr(1, elemType)}, elemType);
    copy->set(std::string(name));
    const auto *ys = world_.extract(copy->var(mim::nat_t(0)), mim::u64(1));
    copy->app(true, copy->ret_var(), world_.extract(ys, mim::u64(0)));
    return copy;
  }

  /// Builds an identity permutation [0, ..., rank-1].
  SmallVector<int64_t> identityPerm(uint64_t rank) {
    SmallVector<int64_t> identity(rank);
    for (uint64_t i = 0; i < rank; ++i)
      identity[i] = i;
    return identity;
  }

  /// Adds two equally shaped tensors elementwise via %tensor.map_reduce.
  FailureOr<const mim::Def *> elementwiseAdd(Operation *op, Type elemType,
                                             ArrayRef<int64_t> shape,
                                             const mim::Def *lhs,
                                             const mim::Def *rhs) {
    const auto *type = moduleTranslation_.convertType(elemType);
    auto *fold =
        world_.mut_fun(mim::Defs{type, world_.arr(2, type)}, type);
    fold->set("add_init");
    const auto *ys = world_.extract(fold->var(mim::nat_t(0)), mim::u64(1));
    const auto *x = world_.extract(ys, mim::u64(0));
    const auto *y = world_.extract(ys, mim::u64(1));
    const mim::Def *sum;
    if (isa<FloatType>(elemType))
      sum = world_.call(math::arith::add, world_.lit_nat(0), mim::Defs{x, y});
    else if (elemType.isSignlessInteger())
      sum = world_.call(core::wrap::add, world_.lit_nat(0), mim::Defs{x, y});
    else
      return op->emitError("unsupported element type for accumulation");
    fold->app(true, fold->ret_var(), sum);

    uint64_t rank = shape.size();
    SmallVector<int64_t> identity = identityPerm(rank);
    const auto *shapeTuple = natTuple(shape);
    const auto *idMap = projMap(rank, identity);
    return mapReduce(type, rank, /*redRank=*/0, shape, shape, {type, type},
                     {int64_t(rank), int64_t(rank)},
                     {shapeTuple, shapeTuple}, fold, world_.bot(type), idMap,
                     {idMap, idMap}, {lhs, rhs});
  }

  /// Accumulating linalg ops add their result onto the `outs` operand: elide
  /// the addition when the initial value cannot contribute, otherwise add it.
  FailureOr<const mim::Def *> addInit(Operation *op, Type elemType,
                                      ArrayRef<int64_t> shape,
                                      const mim::Def *result,
                                      const mim::Def *init) {
    if (isZeroOrUndef(init))
      return result;
    return elementwiseAdd(op, elemType, shape, result, init);
  }

  /// Extracts the splat scalar of a (nested) pack or literal `outs` value.
  const mim::Def *splatScalar(const mim::Def *def) {
    while (auto pack = def->isa<mim::Pack>())
      def = pack->body();
    if (def->isa<mim::Lit>() || def->isa<mim::Bot>())
      return def;
    return nullptr;
  }

  /// Builds a `%tensor.Ring` value `(T, 0, add, mul)` for the given element
  /// type; fails for non-arithmetic element types.
  FailureOr<const mim::Def *> ringFor(Operation *op, Type elemType) {
    const auto *type = moduleTranslation_.convertType(elemType);
    const auto *zero = world_.lit(type, 0);
    const mim::Def *add, *mul;
    if (isa<FloatType>(elemType)) {
      add = binaryRingOp(type, math::arith::add, "ring_fadd");
      mul = binaryRingOp(type, math::arith::mul, "ring_fmul");
    } else if (elemType.isSignlessInteger()) {
      add = binaryRingOp(type, core::wrap::add, "ring_iadd");
      mul = binaryRingOp(type, core::wrap::mul, "ring_imul");
    } else {
      return op->emitError("unsupported ring element type");
    }
    return world_.tuple({type, zero, add, mul});
  }

  /// Converts a linalg body region (block arguments: one per input followed by
  /// one per output/accumulator) into a MimIR fold function
  /// `Fn [To, «nis; Ti»] → To` for %tensor.map_reduce. `indexVals` are the
  /// results of linalg.index ops in the body; they map to the trailing fold
  /// inputs (fed by iota tensors).
  FailureOr<const mim::Def *>
  convertBodyToFold(Block &body, const mim::Def *accType,
                    ArrayRef<const mim::Def *> inElemTypes,
                    ArrayRef<Value> indexVals = {}) {
    auto *fold = world_.mut_fun(
        mim::Defs{accType, world_.sigma(mim::Defs{inElemTypes})}, accType);
    fold->set("linalg_body");

    // The fold's domain is [[acc, ins], ret]: unpack accumulator and inputs.
    const auto *accAndIns = fold->var(mim::nat_t(0));
    const auto *acc = world_.extract(accAndIns, mim::u64(0));
    const auto *ins = world_.extract(accAndIns, mim::u64(1));

    size_t numIns = inElemTypes.size();
    size_t numBodyIns = numIns - indexVals.size();
    if (body.getNumArguments() != numBodyIns + 1)
      return failure();
    for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
      if (index < numBodyIns)
        moduleTranslation_.mapValue(arg, world_.extract(ins, numIns, index));
      else
        moduleTranslation_.mapValue(arg, acc);
    }
    for (auto [index, value] : llvm::enumerate(indexVals))
      moduleTranslation_.mapValue(
          value, world_.extract(ins, numIns, numBodyIns + index));

    // Convert the body ops into the fold lambda; linalg.yield applies the
    // return continuation. Save and restore the CPS insertion state of the
    // enclosing function.
    mim::Lam *savedLam = moduleTranslation_.currentLam();
    const mim::Def *savedMem = moduleTranslation_.currentMem();
    moduleTranslation_.mapBlock(&body, fold);
    LogicalResult converted =
        moduleTranslation_.convertBlock(body, /*ignoreArguments=*/true);
    moduleTranslation_.setCurrentLam(savedLam);
    moduleTranslation_.setCurrentMem(savedMem);
    if (failed(converted))
      return failure();
    return fold;
  }

  //===--------------------------------------------------------------------===//
  // Op visitors.
  //===--------------------------------------------------------------------===//

  // linalg.yield: apply the return continuation of the enclosing fold lambda.
  LogicalResult operator()(linalg::YieldOp op) {
    if (op.getNumOperands() != 1)
      return op.emitError("expected a single yielded value");
    const auto *value = moduleTranslation_.lookupValue(op.getOperand(0));
    if (!value)
      return op.emitError("failed to lookup yielded value");
    mim::Lam *lam = moduleTranslation_.currentLam();
    lam->app(true, lam->ret_var(), value);
    return success();
  }

  // linalg.fill: a pack `‹shape; value›`.
  LogicalResult operator()(linalg::FillOp op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!resultType || !resultType.hasStaticShape())
      return op.emitError("expected a statically shaped tensor result");
    const auto *value = moduleTranslation_.lookupValue(op.getInputs().front());
    if (!value)
      return op.emitError("failed to lookup fill value");
    mim::Vector<mim::u64> dims;
    for (int64_t dim : resultType.getShape())
      dims.push_back(dim);
    moduleTranslation_.mapValue(op.getResult(0), world_.pack(dims, value));
    return success();
  }

  // linalg.transpose: %tensor.map_reduce copying with a permuted read map.
  LogicalResult operator()(linalg::TransposeOp op) {
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().front().getType());
    if (!inType || !outType || !inType.hasStaticShape())
      return op.emitError("expected statically shaped tensors");
    const auto *input = moduleTranslation_.lookupValue(op.getInput());
    if (!input)
      return op.emitError("failed to lookup transpose input");

    // `dim(result, i) = dim(input, permutation[i])`, so reading the input for
    // output coordinates `o` uses `in[j] = o[perm^-1[j]]`.
    ArrayRef<int64_t> perm = op.getPermutation();
    uint64_t rank = perm.size();
    SmallVector<int64_t> invPerm(rank);
    for (auto [i, p] : llvm::enumerate(perm))
      invPerm[p] = i;

    const auto *elemType =
        moduleTranslation_.convertType(inType.getElementType());
    const auto *copy = copyFold(elemType, "transpose_copy");
    const auto *result = mapReduce(
        elemType, rank, /*redRank=*/0, outType.getShape(), outType.getShape(),
        {elemType}, {int64_t(rank)}, {natTuple(inType.getShape())}, copy,
        world_.bot(elemType), projMap(rank, identityPerm(rank)),
        {projMap(rank, invPerm)}, {input});
    moduleTranslation_.mapValue(op.getResult().front(), result);
    return success();
  }

  // linalg.broadcast: a map_reduce copy whose read map projects the preserved
  // dimensions (`dimensions` lists the added output dims).
  LogicalResult operator()(linalg::BroadcastOp op) {
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getInit().getType());
    if (!inType || !outType || !inType.hasStaticShape() ||
        !outType.hasStaticShape())
      return op.emitError("expected statically shaped tensors");
    const auto *input = moduleTranslation_.lookupValue(op.getInput());
    if (!input)
      return op.emitError("failed to lookup broadcast input");

    llvm::SmallDenseSet<int64_t> added(op.getDimensions().begin(),
                                       op.getDimensions().end());
    SmallVector<int64_t> subs;
    for (int64_t dim = 0; dim < outType.getRank(); ++dim)
      if (!added.contains(dim))
        subs.push_back(dim);
    if (int64_t(subs.size()) != inType.getRank())
      return op.emitError("unexpected broadcast dimensions");

    uint64_t rank = outType.getRank();
    const auto *elemType =
        moduleTranslation_.convertType(inType.getElementType());
    const auto *copy = copyFold(elemType, "broadcast_copy");
    const auto *result = mapReduce(
        elemType, rank, /*redRank=*/0, outType.getShape(), outType.getShape(),
        {elemType}, {inType.getRank()}, {natTuple(inType.getShape())}, copy,
        world_.bot(elemType), projMap(rank, identityPerm(rank)),
        {projMap(rank, subs)}, {input});
    moduleTranslation_.mapValue(op->getResult(0), result);
    return success();
  }

  // linalg.conv_2d_nchw_fchw: %tensor.conv over a ring, plus an elementwise
  // addition of the `outs` operand (the bias) when it contributes.
  LogicalResult operator()(linalg::Conv2DNchwFchwOp op) {
    auto inType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto wType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inType || !wType || !inType.hasStaticShape() ||
        !wType.hasStaticShape() || !outType.hasStaticShape())
      return op.emitError("expected statically shaped tensors");
    const auto *input = moduleTranslation_.lookupValue(op.getInputs()[0]);
    const auto *weight = moduleTranslation_.lookupValue(op.getInputs()[1]);
    const auto *init = moduleTranslation_.lookupValue(op.getOutputs()[0]);
    if (!input || !weight || !init)
      return op.emitError("failed to lookup convolution operands");

    auto ring = ringFor(op, inType.getElementType());
    if (failed(ring))
      return failure();

    SmallVector<int64_t> strides(op.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(op.getDilations().getValues<int64_t>());
    ArrayRef<int64_t> in = inType.getShape(), w = wType.getShape();

    const auto *conv = world_.annex<mim::plug::tensor::conv>();
    conv = world_.app(conv, *ring);
    conv = world_.app(conv, natTuple({in[0], in[1], w[0], in[2], in[3],
                                      w[2], w[3]})); // {n cin cout h w kh kw}
    conv = world_.app(conv, mim::Defs{natTuple(strides), natTuple(dilations),
                                      natTuple({0, 0})});
    const auto *result = world_.app(conv, mim::Defs{input, weight});

    auto sum = addInit(op, inType.getElementType(), outType.getShape(),
                       result, init);
    if (failed(sum))
      return failure();
    moduleTranslation_.mapValue(op->getResult(0), *sum);
    return success();
  }

  // linalg.pooling_nchw_{max,sum}: %tensor.pool. The second input only
  // carries the window shape; the fold is seeded from the splat `outs` value.
  template <class PoolOp, class Id>
  LogicalResult convertPooling(PoolOp op, Id mimId, std::string_view name) {
    auto inType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto windowType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inType || !windowType || !outType || !inType.hasStaticShape() ||
        !windowType.hasStaticShape() || !outType.hasStaticShape())
      return op.emitError("expected statically shaped tensors");
    const auto *input = moduleTranslation_.lookupValue(op.getInputs()[0]);
    const auto *outs = moduleTranslation_.lookupValue(op.getOutputs()[0]);
    if (!input || !outs)
      return op.emitError("failed to lookup pooling operands");
    const auto *init = splatScalar(outs);
    if (!init)
      return op.emitError("pooling requires a splat `outs` initial value");

    const auto *elemType =
        moduleTranslation_.convertType(inType.getElementType());
    const auto *fold = binaryRingOp(elemType, mimId, name);
    SmallVector<int64_t> strides(
        op.getStrides().template getValues<int64_t>());
    SmallVector<int64_t> dilations(
        op.getDilations().template getValues<int64_t>());
    ArrayRef<int64_t> in = inType.getShape(), out = outType.getShape();

    const auto *pool = world_.annex<mim::plug::tensor::pool>();
    pool = world_.app(pool, elemType);
    pool = world_.app(pool, mim::Defs{fold, init});
    pool = world_.app(pool, natTuple({in[0], in[1], in[2], in[3]}));
    pool = world_.app(pool, mim::Defs{natTuple(windowType.getShape()),
                                      natTuple(strides), natTuple(dilations),
                                      natTuple({0, 0}),
                                      natTuple({out[2], out[3]})});
    moduleTranslation_.mapValue(op->getResult(0), world_.app(pool, input));
    return success();
  }

  LogicalResult operator()(linalg::PoolingNchwMaxOp op) {
    return convertPooling(op, math::extrema::fmax, "pool_max");
  }
  LogicalResult operator()(linalg::PoolingNchwSumOp op) {
    return convertPooling(op, math::arith::add, "pool_sum");
  }

  // linalg.index: handled by the enclosing linalg.generic conversion, which
  // maps its result to an extra iota input of the fold.
  LogicalResult operator()(linalg::IndexOp op) {
    if (moduleTranslation_.lookupValue(op.getResult()))
      return success();
    return op.emitError("linalg.index outside a supported linalg.generic");
  }

  /// Builds a `«n; Idx rank»` tuple of index literals.
  const mim::Def *idxTuple(uint64_t rank, ArrayRef<int64_t> dims) {
    mim::DefVec idxs;
    for (int64_t d : dims)
      idxs.push_back(world_.lit_idx(rank, d));
    return world_.tuple(mim::Defs{idxs});
  }

  /// Translates a contraction (matmul, batch_matmul, matvec, dot) into
  /// %tensor.dot_product with the given contracting (`c1`/`c2`) and batching
  /// (`b1`/`b2`) dimensions of the two inputs.
  template <class ContractionOp>
  LogicalResult convertDotLike(ContractionOp op, ArrayRef<int64_t> c1,
                               ArrayRef<int64_t> c2, ArrayRef<int64_t> b1,
                               ArrayRef<int64_t> b2) {
    auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return op.emitError("expected statically shaped tensor operands");
    const auto *lhs = moduleTranslation_.lookupValue(op.getInputs()[0]);
    const auto *rhs = moduleTranslation_.lookupValue(op.getInputs()[1]);
    const auto *init = moduleTranslation_.lookupValue(op.getOutputs()[0]);
    if (!lhs || !rhs || !init)
      return op.emitError("failed to lookup contraction operands");

    auto ring = ringFor(op, lhsType.getElementType());
    if (failed(ring))
      return failure();

    uint64_t r1 = lhsType.getRank(), r2 = rhsType.getRank();
    const auto *dp = world_.annex<mim::plug::tensor::dot_product>();
    dp = world_.app(dp, *ring);
    dp = world_.app(dp, mim::Defs{world_.lit_nat(r1), world_.lit_nat(r2)});
    dp = world_.app(dp, mim::Defs{world_.lit_nat(c1.size()),
                                  world_.lit_nat(b1.size())});
    dp = world_.app(dp, mim::Defs{idxTuple(r1, c1), idxTuple(r2, c2),
                                  idxTuple(r1, b1), idxTuple(r2, b2)});
    dp = world_.app(dp, mim::Defs{natTuple(lhsType.getShape()),
                                  natTuple(rhsType.getShape())});
    const auto *product = world_.app(dp, mim::Defs{lhs, rhs});

    // The contraction accumulates onto its `outs` operand.
    auto outType = cast<RankedTensorType>(op->getResult(0).getType());
    auto sum = addInit(op, outType.getElementType(), outType.getShape(),
                       product, init);
    if (failed(sum))
      return failure();
    moduleTranslation_.mapValue(op.getResult(0), *sum);
    return success();
  }

  // linalg.matmul: contract dim 1 of the left with dim 0 of the right input.
  LogicalResult operator()(linalg::MatmulOp op) {
    return convertDotLike(op, {1}, {0}, {}, {});
  }
  // linalg.batch_matmul: additionally batch over dim 0 of both inputs.
  LogicalResult operator()(linalg::BatchMatmulOp op) {
    return convertDotLike(op, {2}, {1}, {0}, {0});
  }
  // linalg.matvec: contract dim 1 of the matrix with the vector.
  LogicalResult operator()(linalg::MatvecOp op) {
    return convertDotLike(op, {1}, {0}, {}, {});
  }
  // linalg.dot: contract the two vectors to a scalar.
  LogicalResult operator()(linalg::DotOp op) {
    return convertDotLike(op, {0}, {0}, {}, {});
  }

  // linalg.generic: %tensor.map_reduce with the body region as fold function.
  LogicalResult operator()(linalg::GenericOp op) {
    if (op.getNumResults() != 1 || op.getOutputs().size() != 1)
      return op.emitError("expected exactly one result");
    auto outType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!outType || !outType.hasStaticShape())
      return op.emitError("expected a statically shaped tensor result");

    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    auto iterators = op.getIteratorTypesArray();
    AffineMap outMap = indexingMaps.back();

    auto outDims = projectedDims(outMap);
    if (failed(outDims))
      return op.emitError("expected a projected permutation output map");

    // Order the loops as [output dims..., reduction dims...]: this makes the
    // output write map the identity, as required by %tensor.map_reduce (the
    // write position may only depend on the leading parallel loops).
    SmallVector<int64_t> loopOrder(*outDims);
    llvm::SmallDenseSet<int64_t> parallelSet(outDims->begin(), outDims->end());
    for (int64_t dim = 0; dim < int64_t(iterators.size()); ++dim) {
      bool isParallel = iterators[dim] == utils::IteratorType::parallel;
      if (parallelSet.contains(dim)) {
        if (!isParallel)
          return op.emitError("output map may only use parallel dimensions");
      } else {
        if (isParallel)
          return op.emitError(
              "parallel dimensions must appear in the output map");
        loopOrder.push_back(dim);
      }
    }
    SmallVector<int64_t> loopPos(iterators.size());
    for (auto [pos, dim] : llvm::enumerate(loopOrder))
      loopPos[dim] = pos;

    uint64_t outRank = outDims->size();
    uint64_t redRank = iterators.size() - outRank;

    // Loop bounds: a loop dimension is bounded by the static extent of any
    // operand axis that reads it as a plain dim. (Not getStaticLoopRanges():
    // that asserts on indexing maps with symbols instead of diagnosing.)
    SmallVector<int64_t> ranges(iterators.size(), ShapedType::kDynamic);
    for (auto [operand, map] :
         llvm::zip(op->getOperands(), indexingMaps)) {
      auto shapedType = dyn_cast<RankedTensorType>(operand.getType());
      if (!shapedType)
        continue;
      for (auto [pos, expr] : llvm::enumerate(map.getResults())) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr || shapedType.isDynamicDim(pos))
          continue;
        if (ShapedType::isDynamic(ranges[dimExpr.getPosition()]))
          ranges[dimExpr.getPosition()] = shapedType.getDimSize(pos);
      }
    }
    SmallVector<int64_t> loopBounds;
    for (int64_t dim : loopOrder) {
      if (ShapedType::isDynamic(ranges[dim]))
        return op.emitError("failed to derive a static bound for loop dim ")
               << dim;
      loopBounds.push_back(ranges[dim]);
    }

    // Per-input element types, shapes, and read maps.
    SmallVector<const mim::Def *> inputs, inElemTypes, inShapes, maps;
    SmallVector<int64_t> inRanks;
    for (auto [value, map] :
         llvm::zip(op.getInputs(), ArrayRef(indexingMaps).drop_back())) {
      auto inType = dyn_cast<RankedTensorType>(value.getType());
      if (!inType || !inType.hasStaticShape())
        return op.emitError("expected statically shaped tensor inputs");
      const auto *def = moduleTranslation_.lookupValue(value);
      if (!def)
        return op.emitError("failed to lookup input");
      auto readMap = affineIndexMap(op, map, iterators.size(), loopPos);
      if (failed(readMap))
        return failure();

      inputs.push_back(def);
      inElemTypes.push_back(
          moduleTranslation_.convertType(inType.getElementType()));
      inShapes.push_back(natTuple(inType.getShape()));
      inRanks.push_back(inType.getRank());
      maps.push_back(*readMap);
    }

    // linalg.index ops become extra iota inputs, read along their dimension:
    // the fold receives the current index as an ordinary input element.
    SmallVector<Value> indexVals;
    for (linalg::IndexOp indexOp : op.getBody()->getOps<linalg::IndexOp>()) {
      int64_t dim = indexOp.getDim();
      if (ShapedType::isDynamic(ranges[dim]))
        return op.emitError("failed to derive a bound for linalg.index");
      const auto *idxType = world_.type_int(64);
      mim::DefVec iota;
      for (int64_t i = 0; i < ranges[dim]; ++i)
        iota.push_back(world_.lit(idxType, i));
      inputs.push_back(world_.tuple(mim::Defs{iota}));
      inElemTypes.push_back(idxType);
      inShapes.push_back(natTuple(ranges[dim]));
      inRanks.push_back(1);
      maps.push_back(projMap(iterators.size(), loopPos[dim]));
      indexVals.push_back(indexOp.getResult());
    }

    const auto *accType =
        moduleTranslation_.convertType(outType.getElementType());

    // The fold's initial accumulator: irrelevant for purely parallel loops;
    // for reductions it must be a splat scalar from the `outs` operand.
    const mim::Def *init;
    if (redRank == 0) {
      init = world_.bot(accType);
    } else {
      const auto *outsDef =
          moduleTranslation_.lookupValue(op.getOutputs().front());
      init = outsDef ? splatScalar(outsDef) : nullptr;
      if (!init)
        return op.emitError(
            "reductions require a splat `outs` initial value");
    }

    auto fold =
        convertBodyToFold(*op.getBody(), accType, inElemTypes, indexVals);
    if (failed(fold))
      return op.emitError("failed to convert the body region");

    const auto *result = mapReduce(
        accType, outRank, redRank, outType.getShape(), loopBounds, inElemTypes,
        inRanks, inShapes, *fold, init,
        projMap(iterators.size(), identityPerm(outRank)), maps, inputs);
    moduleTranslation_.mapValue(op.getResult(0), result);
    return success();
  }

  LogicalResult operator()(Operation *op) {
    return op->emitError("linalg operation not supported yet");
  }

private:
  mim::World &world_;
  MimIR::ModuleTranslation &moduleTranslation_;
};

class LinalgDialectMimIRTranslationInterface
    : public MimIRTranslationDialectInterface {
public:
  using MimIRTranslationDialectInterface::MimIRTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const override {
    LinalgToMimIRVisitor visitor{world, moduleTranslation};
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::YieldOp, linalg::FillOp, linalg::TransposeOp,
              linalg::MatmulOp, linalg::BatchMatmulOp, linalg::MatvecOp,
              linalg::DotOp, linalg::GenericOp, linalg::BroadcastOp,
              linalg::Conv2DNchwFchwOp, linalg::PoolingNchwMaxOp,
              linalg::PoolingNchwSumOp, linalg::IndexOp>(
            [&](auto typedOp) { return visitor(typedOp); })
        .Default([&](Operation *) { return visitor(op); });
  }
};

} // namespace

void mlir::registerLinalgDialectTranslationMimIR(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    dialect->addInterfaces<LinalgDialectMimIRTranslationInterface>();
  });
}

void mlir::registerLinalgDialectTranslationMimIR(MLIRContext &context) {
  DialectRegistry registry;
  registerLinalgDialectTranslationMimIR(registry);
  context.appendDialectRegistry(registry);
}
