//===- ModuleTranslation.h - MLIR to MimIR conversion ------------*- C++
//-*-===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR MimIR dialect module and
// the corresponding MimIR module. It only handles core MimIR IR operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_MODULETRANSLATION_H
#define MLIR_TARGET_MIMIR_MODULETRANSLATION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/StateStack.h"
#include "mlir/Target/MimIR/Export.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/TypeToMimIR.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/FPEnv.h"

#include <mim/world.h>

namespace mlir {
class Attribute;
class Block;
class Location;

namespace MimIR {

/// Implementation class for module translation. Holds a reference to the module
/// being translated, and the mappings between the original and the translated
/// functions, basic blocks and values. It is practically easier to hold these
/// mappings in one class since the conversion of control flow operations
/// needs to look up block and function mappings.
class ModuleTranslation {
  friend std::unique_ptr<mim::World>
  mlir::translateModuleToMimIR(Operation *module, mim::Driver &driver,
                         llvm::StringRef name);

public:
  /// Stores the mapping between a function name and its MimIR IR
  /// representation.
  void mapFunction(StringRef name, mim::Lam *func) {
    auto result = functionMapping.try_emplace(name, func);
    (void)result;
    assert(result.second &&
           "attempting to map a function that is already mapped");
  }

  /// Finds an MimIR IR function by its name.
  mim::Lam *lookupFunction(StringRef name) const {
    return functionMapping.lookup(name);
  }

  /// Stores the mapping between an MLIR value and its MimIR counterpart.
  void mapValue(Value mlir, const mim::Def *m) { mapValue(mlir) = m; }

  /// Provides write-once access to store the MimIR IR value corresponding to
  /// the given MLIR value.
  const mim::Def *&mapValue(Value value) {
    const mim::Def *&m = valueMapping[value];
    assert(m == nullptr && "attempting to map a value that is already mapped");
    return m;
  }

  /// Finds an MimIR IR value corresponding to the given MLIR value.
  const mim::Def *lookupValue(Value value) const {
    return valueMapping.lookup(value);
  }

  /// Looks up remapped a list of remapped values.
  SmallVector<const mim::Def *> lookupValues(ValueRange values);

  /// Stores the mapping between an MLIR block and MimIR lambda.
  void mapBlock(Block *mlir, mim::Lam *lam) {
    auto result = blockMapping.try_emplace(mlir, lam);
    (void)result;
    assert(result.second && "attempting to map a block that is already mapped");
  }

  /// Finds an MimIR basic block that corresponds to the given MLIR block.
  const mim::Lam *lookupBlock(Block *block) const {
    return blockMapping.lookup(block);
  }

  /// Stores the mapping between an MLIR operation with successors and a
  /// corresponding MimIR instruction.
  void mapBranch(Operation *mlir, const mim::Def *m) {
    auto result = branchMapping.try_emplace(mlir, m);
    (void)result;
    assert(result.second &&
           "attempting to map a branch that is already mapped");
  }

  /// Finds an MimIR IR instruction that corresponds to the given MLIR operation
  /// with successors.
  const mim::Def *lookupBranch(Operation *op) const {
    return branchMapping.lookup(op);
  }

  /// Stores a mapping between an MLIR call operation and a corresponding MimIR
  /// call instruction.
  void mapCall(Operation *mlir, const mim::Def *m) {
    auto result = callMapping.try_emplace(mlir, m);
    (void)result;
    assert(result.second && "attempting to map a call that is already mapped");
  }

  /// Finds an MimIR call instruction that corresponds to the given MLIR call
  /// operation.
  const mim::Def *lookupCall(Operation *op) const {
    return callMapping.lookup(op);
  }

  /// Maps a blockaddress operation to its corresponding placeholder MimIR
  /// value.
  // void mapUnresolvedBlockAddress(BlockAddressOp op, const mim::Def *cst) {
  //   auto result = unresolvedBlockAddressMapping.try_emplace(op, cst);
  //   (void)result;
  //   assert(result.second &&
  //          "attempting to map a blockaddress operation that is already
  //          mapped");
  // }

  /// Maps a BlockAddressAttr to its corresponding MimIR basic block.
  // void mapBlockAddress(BlockAddressAttr attr, const mim::Def *block) {
  //   auto result = blockAddressToMimIRMapping.try_emplace(attr, block);
  //   (void)result;
  //   assert(result.second &&
  //          "attempting to map a blockaddress attribute that is already
  //          mapped");
  // }

  /// Finds the MimIR basic block that corresponds to the given
  /// BlockAddressAttr.
  // const mim::Def *lookupBlockAddress(BlockAddressAttr attr) const {
  //   return blockAddressToMimIRMapping.lookup(attr);
  // }

  /// Removes the mapping for blocks contained in the region and values defined
  /// in these blocks.
  void forgetMapping(Region &region);

  /// Converts the type from MLIR MimIR dialect to MimIR.
  const mim::Def *convertType(Type type);

  /// Returns the MLIR context of the module being translated.
  MLIRContext &getContext() { return *mlirModule->getContext(); }

  /// Returns the MimIR context in which the IR is being constructed.
  mim::World &world() const { return *world_; }

  /// Finds an MimIR IR global value that corresponds to the given MLIR
  /// operation defining a global value.
  const mim::Def *lookupGlobal(Operation *op) {
    return globalsMapping.lookup(op);
  }

  /// Finds an MimIR IR global value that corresponds to the given MLIR
  /// operation defining a global alias value.
  const mim::Def *lookupAlias(Operation *op) {
    return aliasesMapping.lookup(op);
  }

  /// Finds an MimIR IR global value that corresponds to the given MLIR
  /// operation defining an IFunc.
  const mim::Def *lookupIFunc(Operation *op) { return ifuncMapping.lookup(op); }

  /// Translates the given MimIR rounding mode metadata.
  // llvm::RoundingMode translateRoundingMode(LLVM::RoundingMode rounding);

  /// Translates the given MimIR FP exception behavior metadata.
  // llvm::fp::ExceptionBehavior
  // translateFPExceptionBehavior(MimIR::FPExceptionBehavior exceptionBehavior);

  /// Translates the contents of the given block to MimIR IR using this
  /// translator. The MimIR IR basic block corresponding to the given block is
  /// expected to exist in the mapping of this translator. Uses `builder` to
  /// translate the IR, leaving it at the end of the block. If `ignoreArguments`
  /// is set, does not produce PHI nodes for the block arguments. Otherwise, the
  /// PHI nodes are constructed for block arguments but are _not_ connected to
  /// the predecessors that may not exist yet.
  LogicalResult convertBlock(Block &bb, bool ignoreArguments) {
    return convertBlockImpl(bb, ignoreArguments,
                            /*recordInsertions=*/false);
  }

  /// Converts argument and result attributes from `attrsOp` to MimIR IR
  /// attributes on the `call` instruction. Returns failure if conversion fails.
  /// The `immArgPositions` parameter is only relevant for intrinsics. It
  /// specifies the positions of immediate arguments, which do not have
  /// associated argument attributes in MLIR and should be skipped during
  /// attribute mapping.
  // LogicalResult
  // convertArgAndResultAttrs(ArgAndResultAttrsOpInterface attrsOp,
  //                          const Def *call,
  //                          ArrayRef<unsigned> immArgPositions = {});

  /// Gets the named metadata in the MimIR IR module being constructed, creating
  /// it if it does not exist.
  // llvm::NamedMDNode *getOrInsertNamedModuleMetadata(StringRef name);

  /// Creates a stack frame of type `T` on ModuleTranslation stack. `T` must
  /// be derived from `StackFrameBase<T>` and constructible from the provided
  /// arguments. Doing this before entering the region of the op being
  /// translated makes the frame available when translating ops within that
  /// region.
  template <typename T, typename... Args>
  void stackPush(Args &&...args) {
    stack.stackPush<T>(std::forward<Args>(args)...);
  }

  /// Pops the last element from the ModuleTranslation stack.
  void stackPop() { stack.stackPop(); }

  /// Calls `callback` for every ModuleTranslation stack frame of type `T`
  /// starting from the top of the stack.
  template <typename T>
  WalkResult stackWalk(llvm::function_ref<WalkResult(T &)> callback) {
    return stack.stackWalk(callback);
  }

  /// RAII object calling stackPush/stackPop on construction/destruction.
  template <typename T>
  using SaveStack = SaveStateStack<T, ModuleTranslation>;

  SymbolTableCollection &symbolTable() { return symbolTableCollection; }

private:
  ModuleTranslation(Operation *module, mim::Driver &driver,
                    std::unique_ptr<mim::World> &&worldPtr);
  ~ModuleTranslation();

  /// Converts individual components.
  LogicalResult convertOperation(Operation &op, bool recordInsertions = false);
  LogicalResult convertFunctionSignatures();
  LogicalResult convertFunctions();
  LogicalResult convertIFuncs();
  LogicalResult convertComdats();

  LogicalResult convertUnresolvedBlockAddress();

  /// Handle conversion for both globals and global aliases.
  ///
  /// - Create named global variables that correspond to llvm.mlir.global
  /// definitions, similarly Convert llvm.global_ctors and global_dtors ops.
  /// - Create global alias that correspond to llvm.mlir.alias.
  LogicalResult convertGlobalsAndAliases();
  // LogicalResult convertOneFunction(MimIRFuncOp func);
  LogicalResult convertBlockImpl(Block &bb, bool ignoreArguments,
                                 bool recordInsertions);

  /// Translates dialect attributes attached to the given operation.
  LogicalResult
  convertDialectAttributes(Operation *op,
                           ArrayRef<const mim::Def *> instructions);

  /// Translates parameter attributes of a call and adds them to the returned
  /// AttrBuilder. Returns failure if any of the translations failed.
  // FailureOr<llvm::AttrBuilder> convertParameterAttrs(mlir::Location loc,
  //                                                    DictionaryAttr
  //                                                    paramAttrs);

  /// Translates parameter attributes of a function and adds them to the
  /// returned AttrBuilder. Returns failure if any of the translations failed.
  // FailureOr<llvm::AttrBuilder>
  // convertParameterAttrs(MimIRFuncOp func, int argIdx, DictionaryAttr
  // paramAttrs);

  /// Original and translated module.
  Operation *mlirModule;
  // std::unique_ptr<llvm::Module> llvmModule;
  std::unique_ptr<mim::World> world_;
  mim::Driver &driver_;

  /// A converter for translating debug information.

  /// Mappings between llvm.mlir.global definitions and corresponding globals.
  DenseMap<Operation *, const mim::Def *> globalsMapping;

  /// Mappings between llvm.mlir.alias definitions and corresponding global
  /// aliases.
  DenseMap<Operation *, const mim::Def *> aliasesMapping;

  /// Mappings between llvm.mlir.ifunc definitions and corresponding global
  /// ifuncs.
  DenseMap<Operation *, const mim::Def *> ifuncMapping;

  /// A stateful object used to translate types.
  TypeToMimIRTranslator typeTranslator;

  /// A dialect interface collection used for dispatching the translation to
  /// specific dialects.
  MimIRTranslationInterface iface;

  /// Mappings between original and translated values, used for lookups.
  llvm::StringMap<mim::Lam *> functionMapping;
  DenseMap<Value, const mim::Def *> valueMapping;
  DenseMap<Block *, mim::Lam *> blockMapping;

  /// A mapping between MLIR MimIR dialect terminators and MimIR IR terminators
  /// they are converted to. This allows for connecting PHI nodes to the source
  /// values after all operations are converted.
  DenseMap<Operation *, const mim::Def *> branchMapping;

  /// A mapping between MLIR MimIR dialect call operations and MimIR IR call
  /// instructions. This allows for adding branch weights after the operations
  /// have been converted.
  DenseMap<Operation *, const mim::Def *> callMapping;

  /// Stack of user-specified state elements, useful when translating operations
  /// with regions.
  StateStack stack;

  /// A cache for the symbol tables constructed during symbols lookup.
  SymbolTableCollection symbolTableCollection;
};

namespace detail {
/// For all blocks in the region that were converted to MimIR IR using the given
/// ModuleTranslation, connect the PHI nodes of the corresponding MimIR IR
/// blocks to the results of preceding blocks.
void connectPHINodes(Region &region, const ModuleTranslation &state);

/// Create an MimIR IR constant of `llvmType` from the MLIR attribute `attr`.
/// This currently supports integer, floating point, splat and dense element
/// attributes and combinations thereof. Also, an array attribute with two
/// elements is supported to represent a complex constant.  In case of error,
/// report it to `loc` and return nullptr.
const mim::Def *getMimIRConstant(const mim::Def *type, Attribute attr,
                                 Location loc,
                                 const ModuleTranslation &moduleTranslation);

/// Creates a call to an MimIR IR intrinsic function with the given arguments.
// const mim::Def *createIntrinsicCall(mim::World &w,
//                                     llvm::Intrinsic::ID intrinsic,
//                                     ArrayRef<llvm::Value *> args = {},
//                                     ArrayRef<llvm::Type *> tys = {});

/// Creates a call to a MimIR IR intrinsic defined by MimIR_IntrOpBase. This
/// resolves the overloads, and maps mixed MLIR value and attribute arguments to
/// MimIR values.
// llvm::CallInst *createIntrinsicCall(
//     llvm::IRBuilderBase &builder, ModuleTranslation &moduleTranslation,
//     Operation *intrOp, llvm::Intrinsic::ID intrinsic, unsigned numResults,
//     ArrayRef<unsigned> overloadedResults, ArrayRef<unsigned>
//     overloadedOperands, ArrayRef<unsigned> immArgPositions,
//     ArrayRef<StringLiteral> immArgAttrNames);

} // namespace detail

} // namespace MimIR
} // namespace mlir

#endif // MLIR_TARGET_MIMIR_MODULETRANSLATION_H
