//===- ModuleTranslation.cpp - MLIR to MimIR conversion
//--------------------===//
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

#include "mlir/Target/MimIR/ModuleTranslation.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Target/MimIR/MimIRTranslationInterface.h"
#include "mlir/Target/MimIR/TypeToMimIR.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>

#include "mim/driver.h"
#include "mim/world.h"
#include <mim/ast/ast.h>
#include <mim/ast/parser.h>
#include <mim/def.h>
#include <mim/plugin.h>

#include <string>

#define DEBUG_TYPE "llvm-dialect-to-llvm-ir"

using namespace mlir;
using namespace mlir::MimIR;
using namespace mlir::MimIR::detail;

/// Builds a constant of a sequential MimIR type `type`, potentially containing
/// other sequential types recursively, from the individual constant values
/// provided in `constants`. `shape` contains the number of elements in nested
/// sequential types. Reports errors at `loc` and returns nullptr on error.
// static const mim::Def *
// buildSequentialConstant(ArrayRef<const llvm::Def *> &constants,
//                         ArrayRef<int64_t> shape, const mim::Def *type,
//                         Location loc) {
//   if (shape.empty()) {
//     const mim::Def *result = constants.front();
//     constants = constants.drop_front();
//     return result;
//   }

//   const mim::Def *elementType;
//   if (auto *arrayTy = dyn_cast<llvm::ArrayType>(type)) {
//     elementType = arrayTy->getElementType();
//   } else if (auto *vectorTy = dyn_cast<llvm::VectorType>(type)) {
//     elementType = vectorTy->getElementType();
//   } else {
//     emitError(loc) << "expected sequential MimIR types wrapping a scalar";
//     return nullptr;
//   }

//   SmallVector<llvm::Constant *, 8> nested;
//   nested.reserve(shape.front());
//   for (int64_t i = 0; i < shape.front(); ++i) {
//     nested.push_back(buildSequentialConstant(constants, shape.drop_front(),
//                                              elementType, loc));
//     if (!nested.back())
//       return nullptr;
//   }

//   if (shape.size() == 1 && type->isVectorTy())
//     return llvm::ConstantVector::get(nested);
//   return llvm::ConstantArray::get(
//       llvm::ArrayType::get(elementType, shape.front()), nested);
// }

Attribute getConstantAttr(Operation *constantOp) {
  Attribute constant;
  matchPattern(constantOp, m_Constant(&constant));
  return constant;
}

/// Returns the first non-sequential type nested in sequential types.
static const mim::Def *getInnermostElementType(const mim::Def *type) {
  do {
    if (auto *arrayTy = type->isa<mim::Seq>()) {
      type = arrayTy->body();
    } else {
      return type;
    }
  } while (true);
}

ModuleTranslation::ModuleTranslation(Operation *module, mim::Driver &driver,
                                     std::unique_ptr<mim::World> &&worldPtr)
    : mlirModule(module), world_(std::move(worldPtr)), driver_(driver),
      typeTranslator(*world_), iface(module->getContext()) {}

ModuleTranslation::~ModuleTranslation() {}

void ModuleTranslation::forgetMapping(Region &region) {
  SmallVector<Region *> toProcess;
  toProcess.push_back(&region);
  while (!toProcess.empty()) {
    Region *current = toProcess.pop_back_val();
    for (Block &block : *current) {
      blockMapping.erase(&block);
      for (Value arg : block.getArguments())
        valueMapping.erase(arg);
      for (Operation &op : block) {
        for (Value value : op.getResults())
          valueMapping.erase(value);
        if (op.hasSuccessors())
          branchMapping.erase(&op);
        // if (isa<MimIR::GlobalOp>(op))
        //   globalsMapping.erase(&op);
        // if (isa<MimIR::AliasOp>(op))
        //   aliasesMapping.erase(&op);
        // if (isa<MimIR::IFuncOp>(op))
        //   ifuncMapping.erase(&op);
        // if (isa<MimIR::CallOp>(op))
        //   callMapping.erase(&op);
        llvm::append_range(
            toProcess,
            llvm::map_range(op.getRegions(), [](Region &r) { return &r; }));
      }
    }
  }
}

/// Given a single MLIR operation, create the corresponding MimIR IR operation
/// using the `builder`.
LogicalResult ModuleTranslation::convertOperation(Operation &op,
                                                  bool recordInsertions) {
  const MimIRTranslationDialectInterface *opIface = iface.getInterfaceFor(&op);
  if (!opIface)
    return op.emitError("cannot be converted to MimIR IR: missing "
                        "`MimIRTranslationDialectInterface` registration for "
                        "dialect for op: ")
           << op.getName();

  if (failed(opIface->convertOperation(&op, *world_, *this)))
    return op.emitError("MimIR Translation failed for operation: ")
           << op.getName();

  // return convertDialectAttributes(&op, scope.getCapturedInstructions());
  return LogicalResult::success();
}

/// Convert block to MimIR IR.  Unless `ignoreArguments` is set, emit PHI nodes
/// to define values corresponding to the MLIR block arguments.  These nodes
/// are not connected to the source basic blocks, which may not exist yet.  Uses
/// `builder` to construct the MimIR IR. Expects the MimIR IR basic block to
/// have been created for `bb` and included in the block mapping.  Inserts new
/// instructions at the end of the block and leaves `builder` in a state
/// suitable for further insertion into the end of the block.
LogicalResult ModuleTranslation::convertBlockImpl(Block &bb,
                                                  bool ignoreArguments,
                                                  bool recordInsertions) {
  // builder.SetInsertPoint(lookupBlock(&bb));
  // auto *subprogram = builder.GetInsertBlock()->getParent()->getSubprogram();
  mim::Lam *l = lookupBlock(&bb);

  // Before traversing operations, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first block have been already made available through the remapping of
  // MimIR function arguments.
  if (!ignoreArguments) {
    for (auto [arg, var] : llvm::zip(bb.getArguments(), l->vars())) {
      mapValue(arg, var);
    }
  }

  // Traverse operations.
  for (auto &op : bb) {
    if (failed(convertOperation(op, recordInsertions)))
      return failure();
  }

  return success();
}

/// A helper method to get the single Block in an operation honoring MimIR's
/// module requirements.
static Block &getModuleBody(Operation *module) {
  return module->getRegion(0).front();
}

/// A helper method to decide if a constant must not be set as a global variable
/// initializer. For an external linkage variable, the variable with an
/// initializer is considered externally visible and defined in this module, the
/// variable without an initializer is externally available and is defined
/// elsewhere.
// static bool shouldDropGlobalInitializer(llvm::GlobalValue::LinkageTypes
// linkage,
//                                         llvm::Constant *cst) {
//   return (linkage == llvm::GlobalVariable::ExternalLinkage && !cst) ||
//          linkage == llvm::GlobalVariable::ExternalWeakLinkage;
// }

/// Sets the runtime preemption specifier of `gv` to dso_local if
/// `dsoLocalRequested` is true, otherwise it is left unchanged.
// static void addRuntimePreemptionSpecifier(bool dsoLocalRequested,
//                                           llvm::GlobalValue *gv) {
//   if (dsoLocalRequested)
//     gv->setDSOLocal(true);
// }

/// Attempts to translate an MLIR attribute identified by `key`, optionally with
/// the given `value`, into an MimIR IR attribute. Reports errors at `loc` if
/// any. If the attribute name corresponds to a known MimIR IR attribute kind,
/// creates the MimIR attribute of that kind; otherwise, keeps it as a string
/// attribute. Performs additional checks for attributes known to have or not
/// have a value in order to avoid assertions inside MimIR upon construction.
// static FailureOr<llvm::Attribute>
// convertMLIRAttributeToMimIR(Location loc, llvm::MimIRContext &ctx,
//                             StringRef key, StringRef value = StringRef()) {
//   auto kind = llvm::Attribute::getAttrKindFromName(key);
//   if (kind == llvm::Attribute::None)
//     return llvm::Attribute::get(ctx, key, value);

//   if (llvm::Attribute::isIntAttrKind(kind)) {
//     if (value.empty())
//       return emitError(loc)
//              << "MimIR attribute '" << key << "' expects a value";

//     int64_t result;
//     if (!value.getAsInteger(/*Radix=*/0, result))
//       return llvm::Attribute::get(ctx, kind, result);
//     return llvm::Attribute::get(ctx, key, value);
//   }

//   if (!value.empty())
//     return emitError(loc) << "MimIR attribute '" << key
//                           << "' does not expect a value, found '" << value
//                           << "'";

//   return llvm::Attribute::get(ctx, kind);
// }

/// Converts the MLIR attributes listed in the given array attribute into MimIR
/// attributes. Returns an `AttrBuilder` containing the converted attributes.
/// Reports error to `loc` if any and returns immediately. Expects `arrayAttr`
/// to contain either string attributes, treated as value-less MimIR attributes,
/// or array attributes containing two string attributes, with the first string
/// being the name of the corresponding MimIR attribute and the second string
/// beings its value. Note that even integer attributes are expected to have
/// their values expressed as strings.
// static FailureOr<llvm::AttrBuilder>
// convertMLIRAttributesToMimIR(Location loc, llvm::MimIRContext &ctx,
//                              ArrayAttr arrayAttr, StringRef arrayAttrName) {
//   llvm::AttrBuilder attrBuilder(ctx);
//   if (!arrayAttr)
//     return attrBuilder;

//   for (Attribute attr : arrayAttr) {
//     if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
//       FailureOr<llvm::Attribute> llvmAttr =
//           convertMLIRAttributeToMimIR(loc, ctx, stringAttr.getValue());
//       if (failed(llvmAttr))
//         return failure();
//       attrBuilder.addAttribute(*llvmAttr);
//       continue;
//     }

//     auto arrayAttr = dyn_cast<ArrayAttr>(attr);
//     if (!arrayAttr || arrayAttr.size() != 2)
//       return emitError(loc) << "expected '" << arrayAttrName
//                             << "' to contain string or array attributes";

//     auto keyAttr = dyn_cast<StringAttr>(arrayAttr[0]);
//     auto valueAttr = dyn_cast<StringAttr>(arrayAttr[1]);
//     if (!keyAttr || !valueAttr)
//       return emitError(loc) << "expected arrays within '" << arrayAttrName
//                             << "' to contain two strings";

//     FailureOr<llvm::Attribute> llvmAttr = convertMLIRAttributeToMimIR(
//         loc, ctx, keyAttr.getValue(), valueAttr.getValue());
//     if (failed(llvmAttr))
//       return failure();
//     attrBuilder.addAttribute(*llvmAttr);
//   }

//   return attrBuilder;
// }

// LogicalResult ModuleTranslation::convertGlobalsAndAliases() {
//   // Mapping from compile unit to its respective set of global variables.
//   DenseMap<llvm::DICompileUnit *, SmallVector<llvm::Metadata *>> allGVars;

//   // First, create all global variables and global aliases in MimIR IR. A
//   global
//   // or alias body may refer to another global/alias or itself, so all the
//   // mapping needs to happen prior to body conversion.

//   // Create all llvm::GlobalVariable
//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::GlobalOp>()) {
//     llvm::Type *type = convertType(op.getType());
//     llvm::Constant *cst = nullptr;
//     if (op.getValueOrNull()) {
//       // String attributes are treated separately because they cannot appear
//       as
//       // in-function constants and are thus not supported by
//       getMimIRConstant. if (auto strAttr =
//       dyn_cast_or_null<StringAttr>(op.getValueOrNull())) {
//         cst = llvm::ConstantDataArray::getString(
//             llvmModule->getContext(), strAttr.getValue(), /*AddNull=*/false);
//         type = cst->getType();
//       } else if (!(cst = getMimIRConstant(type, op.getValueOrNull(),
//                                           op.getLoc(), *this))) {
//         return failure();
//       }
//     }

//     auto linkage = convertLinkageToMimIR(op.getLinkage());

//     // MimIR IR requires constant with linkage other than external or weak
//     // external to have initializers. If MLIR does not provide an
//     initializer,
//     // default to undef.
//     bool dropInitializer = shouldDropGlobalInitializer(linkage, cst);
//     if (!dropInitializer && !cst)
//       cst = llvm::UndefValue::get(type);
//     else if (dropInitializer && cst)
//       cst = nullptr;

//     auto *var = new llvm::GlobalVariable(
//         *llvmModule, type, op.getConstant(), linkage, cst, op.getSymName(),
//         /*InsertBefore=*/nullptr,
//         op.getThreadLocal_() ? llvm::GlobalValue::GeneralDynamicTLSModel
//                              : llvm::GlobalValue::NotThreadLocal,
//         op.getAddrSpace(), op.getExternallyInitialized());

//     if (std::optional<mlir::SymbolRefAttr> comdat = op.getComdat()) {
//       auto selectorOp = cast<ComdatSelectorOp>(
//           SymbolTable::lookupNearestSymbolFrom(op, *comdat));
//       var->setComdat(comdatMapping.lookup(selectorOp));
//     }

//     if (op.getUnnamedAddr().has_value())
//       var->setUnnamedAddr(convertUnnamedAddrToMimIR(*op.getUnnamedAddr()));

//     if (op.getSection().has_value())
//       var->setSection(*op.getSection());

//     addRuntimePreemptionSpecifier(op.getDsoLocal(), var);

//     std::optional<uint64_t> alignment = op.getAlignment();
//     if (alignment.has_value())
//       var->setAlignment(llvm::MaybeAlign(alignment.value()));

//     var->setVisibility(convertVisibilityToMimIR(op.getVisibility_()));

//     globalsMapping.try_emplace(op, var);

//     // Add debug information if present.
//     if (op.getDbgExprs()) {
//       for (auto exprAttr :
//            op.getDbgExprs()->getAsRange<DIGlobalVariableExpressionAttr>()) {
//         llvm::DIGlobalVariableExpression *diGlobalExpr =
//             debugTranslation->translateGlobalVariableExpression(exprAttr);
//         llvm::DIGlobalVariable *diGlobalVar = diGlobalExpr->getVariable();
//         var->addDebugInfo(diGlobalExpr);

//         // There is no `globals` field in DICompileUnitAttr which can be
//         // directly assigned to DICompileUnit. We have to build the list by
//         // looking at the dbgExpr of all the GlobalOps. The scope of the
//         // variable is used to get the DICompileUnit in which to add it. But
//         // there are cases where the scope of a global does not directly
//         point
//         // to the DICompileUnit and we have to do a bit more work to get to
//         // it. Some of those cases are:
//         //
//         // 1. For the languages that support modules, the scope hierarchy can
//         // be variable -> DIModule -> DICompileUnit
//         //
//         // 2. For the Fortran common block variable, the scope hierarchy can
//         // be variable -> DICommonBlock -> DISubprogram -> DICompileUnit
//         //
//         // 3. For entities like static local variables in C or variable with
//         // SAVE attribute in Fortran, the scope hierarchy can be
//         // variable -> DISubprogram -> DICompileUnit
//         llvm::DIScope *scope = diGlobalVar->getScope();
//         if (auto *mod = dyn_cast_if_present<llvm::DIModule>(scope))
//           scope = mod->getScope();
//         else if (auto *cb = dyn_cast_if_present<llvm::DICommonBlock>(scope))
//         {
//           if (auto *sp =
//                   dyn_cast_if_present<llvm::DISubprogram>(cb->getScope()))
//             scope = sp->getUnit();
//         } else if (auto *sp = dyn_cast_if_present<llvm::DISubprogram>(scope))
//           scope = sp->getUnit();

//         // Get the compile unit (scope) of the the global variable.
//         if (llvm::DICompileUnit *compileUnit =
//                 dyn_cast_if_present<llvm::DICompileUnit>(scope)) {
//           // Update the compile unit with this incoming global variable
//           // expression during the finalizing step later.
//           allGVars[compileUnit].push_back(diGlobalExpr);
//         }
//       }
//     }

//     // Forward the target-specific attributes to MimIR.
//     FailureOr<llvm::AttrBuilder> convertedTargetSpecificAttrs =
//         convertMLIRAttributesToMimIR(op.getLoc(), var->getContext(),
//                                      op.getTargetSpecificAttrsAttr(),
//                                      op.getTargetSpecificAttrsAttrName());
//     if (failed(convertedTargetSpecificAttrs))
//       return failure();
//     var->addAttributes(*convertedTargetSpecificAttrs);
//   }

//   // Create all llvm::GlobalAlias
//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::AliasOp>()) {
//     llvm::Type *type = convertType(op.getType());
//     llvm::Constant *cst = nullptr;
//     llvm::GlobalValue::LinkageTypes linkage =
//         convertLinkageToMimIR(op.getLinkage());
//     llvm::Module &llvmMod = *llvmModule;

//     // Note address space and aliasee info isn't set just yet.
//     llvm::GlobalAlias *var = llvm::GlobalAlias::create(
//         type, op.getAddrSpace(), linkage, op.getSymName(), /*placeholder*/
//         cst, &llvmMod);

//     var->setThreadLocalMode(op.getThreadLocal_()
//                                 ? llvm::GlobalAlias::GeneralDynamicTLSModel
//                                 : llvm::GlobalAlias::NotThreadLocal);

//     // Note there is no need to setup the comdat because GlobalAlias calls
//     into
//     // the aliasee comdat information automatically.

//     if (op.getUnnamedAddr().has_value())
//       var->setUnnamedAddr(convertUnnamedAddrToMimIR(*op.getUnnamedAddr()));

//     var->setVisibility(convertVisibilityToMimIR(op.getVisibility_()));

//     aliasesMapping.try_emplace(op, var);
//   }

//   // Convert global variable bodies.
//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::GlobalOp>()) {
//     if (Block *initializer = op.getInitializerBlock()) {
//       llvm::IRBuilder<llvm::TargetFolder> builder(
//           llvmModule->getContext(),
//           llvm::TargetFolder(llvmModule->getDataLayout()));

//       [[maybe_unused]] int numConstantsHit = 0;
//       [[maybe_unused]] int numConstantsErased = 0;
//       DenseMap<llvm::ConstantAggregate *, int> constantAggregateUseMap;

//       for (auto &op : initializer->without_terminator()) {
//         if (failed(convertOperation(op, builder)))
//           return emitError(op.getLoc(), "fail to convert global
//           initializer");
//         auto *cst = dyn_cast<llvm::Constant>(lookupValue(op.getResult(0)));
//         if (!cst)
//           return emitError(op.getLoc(), "unemittable constant value");

//         // When emitting an MimIR constant, a new constant is created and the
//         // old constant may become dangling and take space. We should remove
//         the
//         // dangling constants to avoid memory explosion especially for
//         constant
//         // arrays whose number of elements is large.
//         // Because multiple operations may refer to the same constant, we
//         need
//         // to count the number of uses of each constant array and remove it
//         only
//         // when the count becomes zero.
//         if (auto *agg = dyn_cast<llvm::ConstantAggregate>(cst)) {
//           numConstantsHit++;
//           Value result = op.getResult(0);
//           int numUsers = std::distance(result.use_begin(), result.use_end());
//           auto [iterator, inserted] =
//               constantAggregateUseMap.try_emplace(agg, numUsers);
//           if (!inserted) {
//             // Key already exists, update the value
//             iterator->second += numUsers;
//           }
//         }
//         // Scan the operands of the operation to decrement the use count of
//         // constants. Erase the constant if the use count becomes zero.
//         for (Value v : op.getOperands()) {
//           auto cst = dyn_cast<llvm::ConstantAggregate>(lookupValue(v));
//           if (!cst)
//             continue;
//           auto iter = constantAggregateUseMap.find(cst);
//           assert(iter != constantAggregateUseMap.end() && "constant not
//           found"); iter->second--; if (iter->second == 0) {
//             // NOTE: cannot call removeDeadConstantUsers() here because it
//             // may remove the constant which has uses not be converted yet.
//             if (cst->user_empty()) {
//               cst->destroyConstant();
//               numConstantsErased++;
//             }
//             constantAggregateUseMap.erase(iter);
//           }
//         }
//       }

//       ReturnOp ret = cast<ReturnOp>(initializer->getTerminator());
//       llvm::Constant *cst =
//           cast<llvm::Constant>(lookupValue(ret.getOperand(0)));
//       auto *global = cast<llvm::GlobalVariable>(lookupGlobal(op));
//       if (!shouldDropGlobalInitializer(global->getLinkage(), cst))
//         global->setInitializer(cst);

//       // Try to remove the dangling constants again after all operations are
//       // converted.
//       for (auto it : constantAggregateUseMap) {
//         auto cst = it.first;
//         cst->removeDeadConstantUsers();
//         if (cst->user_empty()) {
//           cst->destroyConstant();
//           numConstantsErased++;
//         }
//       }

//       MimIR_DEBUG(llvm::dbgs()
//                       << "Convert initializer for " << op.getName() << "\n";
//                   llvm::dbgs() << numConstantsHit << " new constants hit\n";
//                   llvm::dbgs()
//                   << numConstantsErased << " dangling constants erased\n";);
//     }
//   }

//   // Convert llvm.mlir.global_ctors and dtors.
//   for (Operation &op : getModuleBody(mlirModule)) {
//     auto ctorOp = dyn_cast<GlobalCtorsOp>(op);
//     auto dtorOp = dyn_cast<GlobalDtorsOp>(op);
//     if (!ctorOp && !dtorOp)
//       continue;

//     // The empty / zero initialized version of llvm.global_(c|d)tors cannot
//     be
//     // handled by appendGlobalFn logic below, which just ignores empty
//     (c|d)tor
//     // lists. Make sure it gets emitted.
//     if ((ctorOp && ctorOp.getCtors().empty()) ||
//         (dtorOp && dtorOp.getDtors().empty())) {
//       llvm::IRBuilder<llvm::TargetFolder> builder(
//           llvmModule->getContext(),
//           llvm::TargetFolder(llvmModule->getDataLayout()));
//       llvm::Type *eltTy = llvm::StructType::get(
//           builder.getInt32Ty(), builder.getPtrTy(), builder.getPtrTy());
//       llvm::ArrayType *at = llvm::ArrayType::get(eltTy, 0);
//       llvm::Constant *zeroInit = llvm::Constant::getNullValue(at);
//       (void)new llvm::GlobalVariable(
//           *llvmModule, zeroInit->getType(), false,
//           llvm::GlobalValue::AppendingLinkage, zeroInit,
//           ctorOp ? "llvm.global_ctors" : "llvm.global_dtors");
//     } else {
//       auto range = ctorOp
//                        ? llvm::zip(ctorOp.getCtors(), ctorOp.getPriorities())
//                        : llvm::zip(dtorOp.getDtors(),
//                        dtorOp.getPriorities());
//       auto appendGlobalFn =
//           ctorOp ? llvm::appendToGlobalCtors : llvm::appendToGlobalDtors;
//       for (const auto &[sym, prio] : range) {
//         llvm::Function *f =
//             lookupFunction(cast<FlatSymbolRefAttr>(sym).getValue());
//         appendGlobalFn(*llvmModule, f, cast<IntegerAttr>(prio).getInt(),
//                        /*Data=*/nullptr);
//       }
//     }
//   }

//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::GlobalOp>())
//     if (failed(convertDialectAttributes(op, {})))
//       return failure();

//   // Finally, update the compile units their respective sets of global
//   variables
//   // created earlier.
//   for (const auto &[compileUnit, globals] : allGVars) {
//     compileUnit->replaceGlobalVariables(
//         llvm::MDTuple::get(getMimIRContext(), globals));
//   }

//   // Convert global alias bodies.
//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::AliasOp>()) {
//     Block &initializer = op.getInitializerBlock();
//     llvm::IRBuilder<llvm::TargetFolder> builder(
//         llvmModule->getContext(),
//         llvm::TargetFolder(llvmModule->getDataLayout()));

//     for (mlir::Operation &op : initializer.without_terminator()) {
//       if (failed(convertOperation(op, builder)))
//         return emitError(op.getLoc(), "fail to convert alias initializer");
//       if (!isa<llvm::Constant>(lookupValue(op.getResult(0))))
//         return emitError(op.getLoc(), "unemittable constant value");
//     }

//     auto ret = cast<ReturnOp>(initializer.getTerminator());
//     auto *cst = cast<llvm::Constant>(lookupValue(ret.getOperand(0)));
//     assert(aliasesMapping.count(op));
//     auto *alias = cast<llvm::GlobalAlias>(aliasesMapping[op]);
//     alias->setAliasee(cst);
//   }

//   for (auto op : getModuleBody(mlirModule).getOps<MimIR::AliasOp>())
//     if (failed(convertDialectAttributes(op, {})))
//       return failure();

//   return success();
// }

/// Return a representation of `value` as metadata.
// static llvm::Metadata *convertIntegerToMetadata(llvm::MimIRContext &context,
//                                                 const llvm::APInt &value) {
//   llvm::Constant *constant = llvm::ConstantInt::get(context, value);
//   return llvm::ConstantAsMetadata::get(constant);
// }

// /// Return a representation of `value` as an MDNode.
// static llvm::MDNode *convertIntegerToMDNode(llvm::MimIRContext &context,
//                                             const llvm::APInt &value) {
//   return llvm::MDNode::get(context, convertIntegerToMetadata(context,
//   value));
// }

/// Return an MDNode encoding `vec_type_hint` metadata.
// static llvm::MDNode *convertVecTypeHintToMDNode(llvm::MimIRContext &context,
//                                                 llvm::Type *type,
//                                                 bool isSigned) {
//   llvm::Metadata *typeMD =
//       llvm::ConstantAsMetadata::get(llvm::UndefValue::get(type));
//   llvm::Metadata *isSignedMD =
//       convertIntegerToMetadata(context, llvm::APInt(32, isSigned ? 1 : 0));
//   return llvm::MDNode::get(context, {typeMD, isSignedMD});
// }

/// Return an MDNode with a tuple given by the values in `values`.
// static llvm::MDNode *convertIntegerArrayToMDNode(llvm::MimIRContext &context,
//                                                  ArrayRef<int32_t> values) {
//   SmallVector<llvm::Metadata *> mdValues;
//   llvm::transform(
//       values, std::back_inserter(mdValues), [&context](int32_t value) {
//         return convertIntegerToMetadata(context, llvm::APInt(32, value));
//       });
//   return llvm::MDNode::get(context, mdValues);
// }

LogicalResult ModuleTranslation::convertOneFunction(func::FuncOp &func) {
  // Clear the block, branch value mappings, they are only relevant within one
  // function.
  blockMapping.clear();
  valueMapping.clear();
  branchMapping.clear();
  mim::Lam *lam = lookupFunction(func.getName());

  // Add function arguments to the value remapping table.
  for (auto [mlirArg, mimVar] : llvm::zip(func.getArguments(), lam->vars()))
    mapValue(mlirArg, mimVar);

  // First, create all blocks so we can jump to them.
  for (auto &bb : func) {
    mim::DefVec blockArgTypes{bb.getArguments(), [this](Value v) {
                                return convertType(v.getType());
                              }};
    // todo: do we need a cn for the successors?
    auto *newBlock = world_->mut_con(blockArgTypes);
    mapBlock(&bb, newBlock);
  }

  // Then, convert blocks one by one in topological order to ensure defs are
  // converted before uses.
  auto blocks = getBlocksSortedByDominance(func.getBody());
  for (Block *bb : blocks) {
    if (failed(convertBlockImpl(*bb, bb->isEntryBlock(),
                                /*recordInsertions=*/true)))
      return failure();
  }

  // Finally, convert dialect attributes attached to the function.
  return convertDialectAttributes(func, {});
}

LogicalResult
ModuleTranslation::convertDialectAttributes(Operation *op,
                                            ArrayRef<const mim::Def *> nodes) {
  for (NamedAttribute attribute : op->getDialectAttrs())
    llvm::outs() << "Warning: skipping dialect attribute '"
                 << attribute.getName() << "=" << attribute.getValue() << "'\n";
  //   if (failed(iface.amendOperation(op, instructions, attribute, *this)))
  //     return failure();
  return success();
}

/*
/// Converts memory effect attributes from `func` and attaches them to
/// `llvmFunc`.
static void convertFunctionMemoryAttributes(MimIRFuncOp func,
                                            llvm::Function *llvmFunc) {
  if (!func.getMemoryEffects())
    return;

  MemoryEffectsAttr memEffects = func.getMemoryEffectsAttr();

  // Add memory effects incrementally.
  llvm::MemoryEffects newMemEffects =
      llvm::MemoryEffects(llvm::MemoryEffects::Location::ArgMem,
                          convertModRefInfoToMimIR(memEffects.getArgMem()));
  newMemEffects |= llvm::MemoryEffects(
      llvm::MemoryEffects::Location::InaccessibleMem,
      convertModRefInfoToMimIR(memEffects.getInaccessibleMem()));
  newMemEffects |=
      llvm::MemoryEffects(llvm::MemoryEffects::Location::Other,
                          convertModRefInfoToMimIR(memEffects.getOther()));
  llvmFunc->setMemoryEffects(newMemEffects);
}

/// Converts function attributes from `func` and attaches them to `llvmFunc`.
static void convertFunctionAttributes(MimIRFuncOp func,
                                      llvm::Function *llvmFunc) {
  if (func.getNoInlineAttr())
    llvmFunc->addFnAttr(llvm::Attribute::NoInline);
  if (func.getAlwaysInlineAttr())
    llvmFunc->addFnAttr(llvm::Attribute::AlwaysInline);
  if (func.getInlineHintAttr())
    llvmFunc->addFnAttr(llvm::Attribute::InlineHint);
  if (func.getOptimizeNoneAttr())
    llvmFunc->addFnAttr(llvm::Attribute::OptimizeNone);
  if (func.getConvergentAttr())
    llvmFunc->addFnAttr(llvm::Attribute::Convergent);
  if (func.getNoUnwindAttr())
    llvmFunc->addFnAttr(llvm::Attribute::NoUnwind);
  if (func.getWillReturnAttr())
    llvmFunc->addFnAttr(llvm::Attribute::WillReturn);
  if (TargetFeaturesAttr targetFeatAttr = func.getTargetFeaturesAttr())
    llvmFunc->addFnAttr("target-features", targetFeatAttr.getFeaturesString());
  if (FramePointerKindAttr fpAttr = func.getFramePointerAttr())
    llvmFunc->addFnAttr("frame-pointer", stringifyFramePointerKind(
                                             fpAttr.getFramePointerKind()));
  if (UWTableKindAttr uwTableKindAttr = func.getUwtableKindAttr())
    llvmFunc->setUWTableKind(
        convertUWTableKindToMimIR(uwTableKindAttr.getUwtableKind()));
  convertFunctionMemoryAttributes(func, llvmFunc);
}

/// Converts function attributes from `func` and attaches them to `llvmFunc`.
static void convertFunctionKernelAttributes(MimIRFuncOp func,
                                            llvm::Function *llvmFunc,
                                            ModuleTranslation &translation) {
  llvm::MimIRContext &llvmContext = llvmFunc->getContext();

  if (VecTypeHintAttr vecTypeHint = func.getVecTypeHintAttr()) {
    Type type = vecTypeHint.getHint().getValue();
    llvm::Type *llvmType = translation.convertType(type);
    bool isSigned = vecTypeHint.getIsSigned();
    llvmFunc->setMetadata(
        func.getVecTypeHintAttrName(),
        convertVecTypeHintToMDNode(llvmContext, llvmType, isSigned));
  }

  if (std::optional<ArrayRef<int32_t>> workGroupSizeHint =
          func.getWorkGroupSizeHint()) {
    llvmFunc->setMetadata(
        func.getWorkGroupSizeHintAttrName(),
        convertIntegerArrayToMDNode(llvmContext, *workGroupSizeHint));
  }

  if (std::optional<ArrayRef<int32_t>> reqdWorkGroupSize =
          func.getReqdWorkGroupSize()) {
    llvmFunc->setMetadata(
        func.getReqdWorkGroupSizeAttrName(),
        convertIntegerArrayToMDNode(llvmContext, *reqdWorkGroupSize));
  }

  if (std::optional<uint32_t> intelReqdSubGroupSize =
          func.getIntelReqdSubGroupSize()) {
    llvmFunc->setMetadata(
        func.getIntelReqdSubGroupSizeAttrName(),
        convertIntegerToMDNode(llvmContext,
                               llvm::APInt(32, *intelReqdSubGroupSize)));
  }
}
*/
// static LogicalResult convertParameterAttr(llvm::AttrBuilder &attrBuilder,
//                                           llvm::Attribute::AttrKind llvmKind,
//                                           NamedAttribute namedAttr,
//                                           ModuleTranslation
//                                           &moduleTranslation, Location loc) {
//   return llvm::TypeSwitch<Attribute, LogicalResult>(namedAttr.getValue())
//       .Case<TypeAttr>([&](auto typeAttr) {
//         attrBuilder.addTypeAttr(
//             llvmKind, moduleTranslation.convertType(typeAttr.getValue()));
//         return success();
//       })
//       .Case<IntegerAttr>([&](auto intAttr) {
//         attrBuilder.addRawIntAttr(llvmKind, intAttr.getInt());
//         return success();
//       })
//       .Case<UnitAttr>([&](auto) {
//         attrBuilder.addAttribute(llvmKind);
//         return success();
//       })
//       .Case<MimIR::ConstantRangeAttr>([&](auto rangeAttr) {
//         attrBuilder.addConstantRangeAttr(
//             llvmKind,
//             llvm::ConstantRange(rangeAttr.getLower(), rangeAttr.getUpper()));
//         return success();
//       })
//       .Default([loc](auto) {
//         return emitError(loc, "unsupported parameter attribute type");
//       });
// }

/*
FailureOr<llvm::AttrBuilder>
ModuleTranslation::convertParameterAttrs(MimIRFuncOp func, int argIdx,
                                         DictionaryAttr paramAttrs) {
  llvm::AttrBuilder attrBuilder(llvmModule->getContext());
  auto attrNameToKindMapping = getAttrNameToKindMapping();
  Location loc = func.getLoc();

  for (auto namedAttr : paramAttrs) {
    auto it = attrNameToKindMapping.find(namedAttr.getName());
    if (it != attrNameToKindMapping.end()) {
      llvm::Attribute::AttrKind llvmKind = it->second;
      if (failed(convertParameterAttr(attrBuilder, llvmKind, namedAttr, *this,
                                      loc)))
        return failure();
    } else if (namedAttr.getNameDialect()) {
      if (failed(iface.convertParameterAttr(func, argIdx, namedAttr, *this)))
        return failure();
    }
  }

  return attrBuilder;
}
*/

// LogicalResult ModuleTranslation::convertArgAndResultAttrs(
//     ArgAndResultAttrsOpInterface attrsOp, llvm::CallBase *call,
//     ArrayRef<unsigned> immArgPositions) {
//   // Convert the argument attributes.
//   if (ArrayAttr argAttrsArray = attrsOp.getArgAttrsAttr()) {
//     unsigned argAttrIdx = 0;
//     llvm::SmallDenseSet<unsigned> immArgPositionsSet(immArgPositions.begin(),
//                                                      immArgPositions.end());
//     for (unsigned argIdx : llvm::seq<unsigned>(call->arg_size())) {
//       if (argAttrIdx >= argAttrsArray.size())
//         break;
//       // Skip immediate arguments (they have no entries in argAttrsArray).
//       if (immArgPositionsSet.contains(argIdx))
//         continue;
//       // Skip empty argument attributes.
//       auto argAttrs = cast<DictionaryAttr>(argAttrsArray[argAttrIdx++]);
//       if (argAttrs.empty())
//         continue;
//       // Convert and add attributes to the call instruction.
//       FailureOr<llvm::AttrBuilder> attrBuilder =
//           convertParameterAttrs(attrsOp->getLoc(), argAttrs);
//       if (failed(attrBuilder))
//         return failure();
//       call->addParamAttrs(argIdx, *attrBuilder);
//     }
//   }

//   // Convert the result attributes.
//   if (ArrayAttr resAttrsArray = attrsOp.getResAttrsAttr()) {
//     if (!resAttrsArray.empty()) {
//       auto resAttrs = cast<DictionaryAttr>(resAttrsArray[0]);
//       FailureOr<llvm::AttrBuilder> attrBuilder =
//           convertParameterAttrs(attrsOp->getLoc(), resAttrs);
//       if (failed(attrBuilder))
//         return failure();
//       call->addRetAttrs(*attrBuilder);
//     }
//   }

//   return success();
// }

// FailureOr<llvm::AttrBuilder>
// ModuleTranslation::convertParameterAttrs(Location loc,
//                                          DictionaryAttr paramAttrs) {
//   llvm::AttrBuilder attrBuilder(llvmModule->getContext());
//   auto attrNameToKindMapping = getAttrNameToKindMapping();

//   for (auto namedAttr : paramAttrs) {
//     auto it = attrNameToKindMapping.find(namedAttr.getName());
//     if (it != attrNameToKindMapping.end()) {
//       llvm::Attribute::AttrKind llvmKind = it->second;
//       if (failed(convertParameterAttr(attrBuilder, llvmKind, namedAttr,
//       *this,
//                                       loc)))
//         return failure();
//     }
//   }

//   return attrBuilder;
// }

LogicalResult ModuleTranslation::convertFunctionSignatures() {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles, or global initializers that reference functions.
  for (auto function : getModuleBody(mlirModule).getOps<func::FuncOp>()) {
    mim::DefVec lamArgTypes{function.getArgumentTypes(),
                            [this](Type t) { return convertType(t); }};
    mim::DefVec conArgTypes{function.getFunctionType().getResults(),
                            [this](Type t) { return convertType(t); }};
    const auto *retCn = world_->cn(conArgTypes);
    lamArgTypes.push_back(retCn);
    auto *lam =
        world_->mut_con(lamArgTypes)->set(std::string(function.getName()));

    mapFunction(function.getName(), lam);

    // Convert function attributes.
    // convertFunctionAttributes(function, lam);

    // Convert function kernel attributes to metadata.
    // convertFunctionKernelAttributes(function, lam, *this);

    // Convert function_entry_count attribute to metadata.
    // if (std::optional<uint64_t> entryCount =
    // function.getFunctionEntryCount())
    //   llvmFunc->setEntryCount(entryCount.value());

    // // Convert result attributes.
    // if (ArrayAttr allResultAttrs = function.getAllResultAttrs()) {
    //   DictionaryAttr resultAttrs = cast<DictionaryAttr>(allResultAttrs[0]);
    //   FailureOr<llvm::AttrBuilder> attrBuilder =
    //       convertParameterAttrs(function, -1, resultAttrs);
    //   if (failed(attrBuilder))
    //     return failure();
    //   llvmFunc->addRetAttrs(*attrBuilder);
    // }

    // Convert argument attributes.
    // for (auto [argIdx, llvmArg] : llvm::enumerate(llvmFunc->args())) {
    //   if (DictionaryAttr argAttrs = function.getArgAttrDict(argIdx)) {
    //     FailureOr<llvm::AttrBuilder> attrBuilder =
    //         convertParameterAttrs(function, argIdx, argAttrs);
    //     if (failed(attrBuilder))
    //       return failure();
    //     llvmArg.addAttrs(*attrBuilder);
    //   }
    // }

    // Forward the pass-through attributes to MimIR.
    // FailureOr<llvm::AttrBuilder> convertedPassthroughAttrs =
    //     convertMLIRAttributesToMimIR(function.getLoc(),
    //     llvmFunc->getContext(),
    //                                  function.getPassthroughAttr(),
    //                                  function.getPassthroughAttrName());
    // if (failed(convertedPassthroughAttrs))
    //   return failure();
    // llvmFunc->addFnAttrs(*convertedPassthroughAttrs);

    // Convert visibility attribute.
    // llvmFunc->setVisibility(
    //     convertVisibilityToMimIR(function.getVisibility()));
    if (function.getVisibility() == SymbolTable::Visibility::Public)
      lam->externalize();
    else if (function.isExternal())
      lam->externalize();

    // // Convert the comdat attribute.
    // if (std::optional<mlir::SymbolRefAttr> comdat = function.getComdat()) {
    //   auto selectorOp = cast<ComdatSelectorOp>(
    //       SymbolTable::lookupNearestSymbolFrom(function, *comdat));
    //   llvmFunc->setComdat(comdatMapping.lookup(selectorOp));
    // }

    // if (auto gc = function.getGarbageCollector())
    //   llvmFunc->setGC(gc->str());

    // if (auto unnamedAddr = function.getUnnamedAddr())
    //   llvmFunc->setUnnamedAddr(convertUnnamedAddrToMimIR(*unnamedAddr));

    // if (auto alignment = function.getAlignment())
    //   llvmFunc->setAlignment(llvm::MaybeAlign(*alignment));

    // Translate the debug information for this function.
    // debugTranslation->translate(function, *llvmFunc);
  }

  return success();
}

LogicalResult ModuleTranslation::convertFunctions() {
  // Convert functions.
  for (auto function : getModuleBody(mlirModule).getOps<func::FuncOp>()) {
    // Do not convert external functions, but do process dialect attributes
    // attached to them.
    if (function.isExternal()) {
      if (failed(convertDialectAttributes(function, {})))
        return failure();
      continue;
    }

    if (failed(convertOneFunction(function)))
      return failure();
  }

  return success();
}
LogicalResult ModuleTranslation::convertIFuncs() {
  // for (auto op : getModuleBody(mlirModule).getOps<IFuncOp>()) {
  //   llvm::Type *type = convertType(op.getIFuncType());
  //   llvm::GlobalValue::LinkageTypes linkage =
  //       convertLinkageToMimIR(op.getLinkage());
  //   llvm::Constant *resolver;
  //   if (auto *resolverFn = lookupFunction(op.getResolver())) {
  //     resolver = cast<llvm::Constant>(resolverFn);
  //   } else {
  //     Operation *aliasOp =
  //     symbolTable().lookupSymbolIn(parentMimIRModule(op),
  //                                                       op.getResolverAttr());
  //     resolver = cast<llvm::Constant>(lookupAlias(aliasOp));
  //   }

  //   auto *ifunc =
  //       llvm::GlobalIFunc::create(type, op.getAddressSpace(), linkage,
  //                                 op.getSymName(), resolver,
  //                                 llvmModule.get());
  //   addRuntimePreemptionSpecifier(op.getDsoLocal(), ifunc);
  //   ifunc->setUnnamedAddr(convertUnnamedAddrToMimIR(op.getUnnamedAddr()));
  //   ifunc->setVisibility(convertVisibilityToMimIR(op.getVisibility_()));

  //   ifuncMapping.try_emplace(op, ifunc);
  // }

  return success();
}

LogicalResult ModuleTranslation::convertComdats() {
  // for (auto comdatOp : getModuleBody(mlirModule).getOps<ComdatOp>()) {
  //   for (auto selectorOp : comdatOp.getOps<ComdatSelectorOp>()) {
  //     llvm::Module *module = getMimIRModule();
  //     if
  //     (module->getComdatSymbolTable().contains(selectorOp.getSymName()))
  //       return emitError(selectorOp.getLoc())
  //              << "comdat selection symbols must be unique even in
  //              different "
  //                 "comdat regions";
  //     llvm::Comdat *comdat =
  //     module->getOrInsertComdat(selectorOp.getSymName());
  //     comdat->setSelectionKind(convertComdatToMimIR(selectorOp.getComdat()));
  //     comdatMapping.try_emplace(selectorOp, comdat);
  //   }
  // }
  return success();
}

LogicalResult ModuleTranslation::convertUnresolvedBlockAddress() {
  // for (auto &[blockAddressOp, llvmCst] : unresolvedBlockAddressMapping) {
  //   BlockAddressAttr blockAddressAttr = blockAddressOp.getBlockAddr();
  //   llvm::BasicBlock *llvmBlock = lookupBlockAddress(blockAddressAttr);
  //   assert(llvmBlock && "expected MimIR blocks to be already
  //   translated");

  //   // Update mapping with new block address constant.
  //   auto *llvmBlockAddr = llvm::BlockAddress::get(
  //       lookupFunction(blockAddressAttr.getFunction().getValue()),
  //       llvmBlock);
  //   llvmCst->replaceAllUsesWith(llvmBlockAddr);
  //   assert(llvmCst->use_empty() && "expected all uses to be replaced");
  //   cast<llvm::GlobalVariable>(llvmCst)->eraseFromParent();
  // }
  // unresolvedBlockAddressMapping.clear();
  return success();
}

// void ModuleTranslation::setAccessGroupsMetadata(AccessGroupOpInterface op,
//                                                 llvm::Instruction *inst) {
//   if (llvm::MDNode *node = loopAnnotationTranslation->getAccessGroups(op))
//     inst->setMetadata(llvm::MimIRContext::MD_access_group, node);
// }

// llvm::MDNode *
// ModuleTranslation::getOrCreateAliasScope(AliasScopeAttr aliasScopeAttr) {
//   auto [scopeIt, scopeInserted] =
//       aliasScopeMetadataMapping.try_emplace(aliasScopeAttr, nullptr);
//   if (!scopeInserted)
//     return scopeIt->second;
//   llvm::MimIRContext &ctx = llvmModule->getContext();
//   auto dummy = llvm::MDNode::getTemporary(ctx, {});
//   // Convert the domain metadata node if necessary.
//   auto [domainIt, insertedDomain] = aliasDomainMetadataMapping.try_emplace(
//       aliasScopeAttr.getDomain(), nullptr);
//   if (insertedDomain) {
//     llvm::SmallVector<llvm::Metadata *, 2> operands;
//     // Placeholder for potential self-reference.
//     operands.push_back(dummy.get());
//     if (StringAttr description = aliasScopeAttr.getDomain().getDescription())
//       operands.push_back(llvm::MDString::get(ctx, description));
//     domainIt->second = llvm::MDNode::get(ctx, operands);
//     // Self-reference for uniqueness.
//     llvm::Metadata *replacement;
//     if (auto stringAttr =
//             dyn_cast<StringAttr>(aliasScopeAttr.getDomain().getId()))
//       replacement = llvm::MDString::get(ctx, stringAttr.getValue());
//     else
//       replacement = domainIt->second;
//     domainIt->second->replaceOperandWith(0, replacement);
//   }
//   // Convert the scope metadata node.
//   assert(domainIt->second && "Scope's domain should already be valid");
//   llvm::SmallVector<llvm::Metadata *, 3> operands;
//   // Placeholder for potential self-reference.
//   operands.push_back(dummy.get());
//   operands.push_back(domainIt->second);
//   if (StringAttr description = aliasScopeAttr.getDescription())
//     operands.push_back(llvm::MDString::get(ctx, description));
//   scopeIt->second = llvm::MDNode::get(ctx, operands);
//   // Self-reference for uniqueness.
//   llvm::Metadata *replacement;
//   if (auto stringAttr = dyn_cast<StringAttr>(aliasScopeAttr.getId()))
//     replacement = llvm::MDString::get(ctx, stringAttr.getValue());
//   else
//     replacement = scopeIt->second;
//   scopeIt->second->replaceOperandWith(0, replacement);
//   return scopeIt->second;
// }

// llvm::MDNode *ModuleTranslation::getOrCreateAliasScopes(
//     ArrayRef<AliasScopeAttr> aliasScopeAttrs) {
//   SmallVector<llvm::Metadata *> nodes;
//   nodes.reserve(aliasScopeAttrs.size());
//   for (AliasScopeAttr aliasScopeAttr : aliasScopeAttrs)
//     nodes.push_back(getOrCreateAliasScope(aliasScopeAttr));
//   return llvm::MDNode::get(getMimIRContext(), nodes);
// }

// void ModuleTranslation::setAliasScopeMetadata(AliasAnalysisOpInterface op,
//                                               llvm::Instruction *inst) {
//   auto populateScopeMetadata = [&](ArrayAttr aliasScopeAttrs, unsigned kind)
//   {
//     if (!aliasScopeAttrs || aliasScopeAttrs.empty())
//       return;
//     llvm::MDNode *node = getOrCreateAliasScopes(
//         llvm::to_vector(aliasScopeAttrs.getAsRange<AliasScopeAttr>()));
//     inst->setMetadata(kind, node);
//   };

//   populateScopeMetadata(op.getAliasScopesOrNull(),
//                         llvm::MimIRContext::MD_alias_scope);
//   populateScopeMetadata(op.getNoAliasScopesOrNull(),
//                         llvm::MimIRContext::MD_noalias);
// }

// llvm::MDNode *ModuleTranslation::getTBAANode(TBAATagAttr tbaaAttr) const {
//   return tbaaMetadataMapping.lookup(tbaaAttr);
// }

// void ModuleTranslation::setTBAAMetadata(AliasAnalysisOpInterface op,
//                                         llvm::Instruction *inst) {
//   ArrayAttr tagRefs = op.getTBAATagsOrNull();
//   if (!tagRefs || tagRefs.empty())
//     return;

//   // MimIR IR currently does not support attaching more than one TBAA access
//   tag
//   // to a memory accessing instruction. It may be useful to support this in
//   // future, but for the time being just ignore the metadata if MLIR
//   operation
//   // has multiple access tags.
//   if (tagRefs.size() > 1) {
//     op.emitWarning() << "TBAA access tags were not translated, because MimIR
//     "
//                         "IR only supports a single tag per instruction";
//     return;
//   }

//   llvm::MDNode *node = getTBAANode(cast<TBAATagAttr>(tagRefs[0]));
//   inst->setMetadata(llvm::MimIRContext::MD_tbaa, node);
// }

// void ModuleTranslation::setDereferenceableMetadata(
//     DereferenceableOpInterface op, llvm::Instruction *inst) {
//   DereferenceableAttr derefAttr = op.getDereferenceableOrNull();
//   if (!derefAttr)
//     return;

//   llvm::MDNode *derefSizeNode = llvm::MDNode::get(
//       getMimIRContext(),
//       llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
//                              llvm::IntegerType::get(getMimIRContext(), 64),
//                              derefAttr.getBytes())));
//   unsigned kindId = derefAttr.getMayBeNull()
//                         ? llvm::MimIRContext::MD_dereferenceable_or_null
//                         : llvm::MimIRContext::MD_dereferenceable;
//   inst->setMetadata(kindId, derefSizeNode);
// }

// void ModuleTranslation::setBranchWeightsMetadata(WeightedBranchOpInterface
// op) {
//   SmallVector<uint32_t> weights;
//   llvm::transform(op.getWeights(), std::back_inserter(weights),
//                   [](int32_t value) { return static_cast<uint32_t>(value);
//                   });
//   if (weights.empty())
//     return;

//   llvm::Instruction *inst = isa<CallOp>(op) ? lookupCall(op) :
//   lookupBranch(op); assert(inst && "expected the operation to have a mapping
//   to an instruction"); inst->setMetadata(
//       llvm::MimIRContext::MD_prof,
//       llvm::MDBuilder(getMimIRContext()).createBranchWeights(weights));
// }

// LogicalResult ModuleTranslation::createTBAAMetadata() {
//   llvm::MimIRContext &ctx = llvmModule->getContext();
//   llvm::IntegerType *offsetTy = llvm::IntegerType::get(ctx, 64);

//   // Walk the entire module and create all metadata nodes for the TBAA
//   // attributes. The code below relies on two invariants of the
//   // `AttrTypeWalker`:
//   // 1. Attributes are visited in post-order: Since the attributes create a
//   DAG,
//   //    this ensures that any lookups into `tbaaMetadataMapping` for child
//   //    attributes succeed.
//   // 2. Attributes are only ever visited once: This way we don't leak any
//   //    MimIR metadata instances.
//   AttrTypeWalker walker;
//   walker.addWalk([&](TBAARootAttr root) {
//     tbaaMetadataMapping.insert(
//         {root, llvm::MDNode::get(ctx, llvm::MDString::get(ctx,
//         root.getId()))});
//   });

//   walker.addWalk([&](TBAATypeDescriptorAttr descriptor) {
//     SmallVector<llvm::Metadata *> operands;
//     operands.push_back(llvm::MDString::get(ctx, descriptor.getId()));
//     for (TBAAMemberAttr member : descriptor.getMembers()) {
//       operands.push_back(tbaaMetadataMapping.lookup(member.getTypeDesc()));
//       operands.push_back(llvm::ConstantAsMetadata::get(
//           llvm::ConstantInt::get(offsetTy, member.getOffset())));
//     }

//     tbaaMetadataMapping.insert({descriptor, llvm::MDNode::get(ctx,
//     operands)});
//   });

//   walker.addWalk([&](TBAATagAttr tag) {
//     SmallVector<llvm::Metadata *> operands;

//     operands.push_back(tbaaMetadataMapping.lookup(tag.getBaseType()));
//     operands.push_back(tbaaMetadataMapping.lookup(tag.getAccessType()));

//     operands.push_back(llvm::ConstantAsMetadata::get(
//         llvm::ConstantInt::get(offsetTy, tag.getOffset())));
//     if (tag.getConstant())
//       operands.push_back(
//           llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(offsetTy,
//           1)));

//     tbaaMetadataMapping.insert({tag, llvm::MDNode::get(ctx, operands)});
//   });

//   mlirModule->walk([&](AliasAnalysisOpInterface analysisOpInterface) {
//     if (auto attr = analysisOpInterface.getTBAATagsOrNull())
//       walker.walk(attr);
//   });

//   return success();
// }

// LogicalResult ModuleTranslation::createIdentMetadata() {
//   if (auto attr = mlirModule->getAttrOfType<StringAttr>(
//           MimIRDialect::getIdentAttrName())) {
//     StringRef ident = attr;
//     llvm::MimIRContext &ctx = llvmModule->getContext();
//     llvm::NamedMDNode *namedMd =
//         llvmModule->getOrInsertNamedMetadata(MimIRDialect::getIdentAttrName());
//     llvm::MDNode *md = llvm::MDNode::get(ctx, llvm::MDString::get(ctx,
//     ident)); namedMd->addOperand(md);
//   }

//   return success();
// }

// LogicalResult ModuleTranslation::createCommandlineMetadata() {
//   if (auto attr = mlirModule->getAttrOfType<StringAttr>(
//           MimIRDialect::getCommandlineAttrName())) {
//     StringRef cmdLine = attr;
//     llvm::MimIRContext &ctx = llvmModule->getContext();
//     llvm::NamedMDNode *nmd = llvmModule->getOrInsertNamedMetadata(
//         MimIRDialect::getCommandlineAttrName());
//     llvm::MDNode *md =
//         llvm::MDNode::get(ctx, llvm::MDString::get(ctx, cmdLine));
//     nmd->addOperand(md);
//   }

//   return success();
// }

// LogicalResult ModuleTranslation::createDependentLibrariesMetadata() {
//   if (auto dependentLibrariesAttr = mlirModule->getDiscardableAttr(
//           MimIR::MimIRDialect::getDependentLibrariesAttrName())) {
//     auto *nmd =
//         llvmModule->getOrInsertNamedMetadata("llvm.dependent-libraries");
//     llvm::MimIRContext &ctx = llvmModule->getContext();
//     for (auto libAttr :
//          cast<ArrayAttr>(dependentLibrariesAttr).getAsRange<StringAttr>()) {
//       auto *md =
//           llvm::MDNode::get(ctx, llvm::MDString::get(ctx,
//           libAttr.getValue()));
//       nmd->addOperand(md);
//     }
//   }
//   return success();
// }

// void ModuleTranslation::setLoopMetadata(Operation *op,
//                                         llvm::Instruction *inst) {
//   LoopAnnotationAttr attr =
//       TypeSwitch<Operation *, LoopAnnotationAttr>(op)
//           .Case<MimIR::BrOp, MimIR::CondBrOp>(
//               [](auto branchOp) { return branchOp.getLoopAnnotationAttr();
//               });
//   if (!attr)
//     return;
//   llvm::MDNode *loopMD =
//       loopAnnotationTranslation->translateLoopAnnotation(attr, op);
//   inst->setMetadata(llvm::MimIRContext::MD_loop, loopMD);
// }

// void ModuleTranslation::setDisjointFlag(Operation *op, llvm::Value *value) {
//   auto iface = cast<DisjointFlagInterface>(op);
//   // We do a dyn_cast here in case the value got folded into a constant.
//   if (auto disjointInst = dyn_cast<llvm::PossiblyDisjointInst>(value))
//     disjointInst->setIsDisjoint(iface.getIsDisjoint());
// }

const mim::Def *ModuleTranslation::convertType(Type type) {
  return typeTranslator.translateType(type);
}

/// A helper to look up remapped operands in the value remapping table.
SmallVector<const mim::Def *>
ModuleTranslation::lookupValues(ValueRange values) {
  SmallVector<const mim::Def *> remapped;
  remapped.reserve(values.size());
  for (Value v : values)
    remapped.push_back(lookupValue(v));
  return remapped;
}

// llvm::DILocation *ModuleTranslation::translateLoc(Location loc,
//                                                   llvm::DILocalScope *scope)
//                                                   {
//   return debugTranslation->translateLoc(loc, scope);
// }

// llvm::DIExpression *
// ModuleTranslation::translateExpression(MimIR::DIExpressionAttr attr) {
//   return debugTranslation->translateExpression(attr);
// }

// llvm::DIGlobalVariableExpression *
// ModuleTranslation::translateGlobalVariableExpression(
//     MimIR::DIGlobalVariableExpressionAttr attr) {
//   return debugTranslation->translateGlobalVariableExpression(attr);
// }

// llvm::Metadata *ModuleTranslation::translateDebugInfo(MimIR::DINodeAttr attr)
// {
//   return debugTranslation->translate(attr);
// }

// llvm::RoundingMode
// ModuleTranslation::translateRoundingMode(MimIR::RoundingMode rounding) {
//   return convertRoundingModeToMimIR(rounding);
// }

// llvm::fp::ExceptionBehavior ModuleTranslation::translateFPExceptionBehavior(
//     MimIR::FPExceptionBehavior exceptionBehavior) {
//   return convertFPExceptionBehaviorToMimIR(exceptionBehavior);
// }

// llvm::NamedMDNode *
// ModuleTranslation::getOrInsertNamedModuleMetadata(StringRef name) {
//   return llvmModule->getOrInsertNamedMetadata(name);
// }

std::unique_ptr<mim::World> mlir::translateModuleToMimIR(Operation *module,
                                                         mim::Driver &driver,
                                                         llvm::StringRef name) {
  // if (!satisfiesMimIRModule(module)) {
  //   module->emitOpError("can not be translated to an MimIR module");
  //   return nullptr;
  // }
  using namespace std::literals;

  auto world = std::make_unique<mim::World>(&driver);
  mim::ast::load_plugins(*world, {"compile"s, "mem"s, "core"s, "math"s,
                                  "affine"s, "vec"s, "tensor"s, "direct"s});

  ModuleTranslation translator(module, driver, std::move(world));
  std::string s;
  auto ost = llvm::raw_string_ostream(s);
  module->print(ost);
  std::cout << s << std::endl;
  s = "";
  auto M = llvm::dyn_cast<ModuleOp>(module);
  for (auto &Op : M.getOps()) {
    ost << "Op:\n";
    Op.print(ost);
    ost << "\n";
  }

  for (auto &Attr : M->getAttrs()) {
    ost << "Attr\n";
    ost << Attr.getName() << "\n"
        << Attr.getNameDialect()->getNamespace() << "\n"
        << Attr.getValue() << "\n";
  }
  for (auto &Attr : M->getDialectAttrs()) {
    ost << "Attr\n";
    ost << Attr.getName() << "\n"
        << Attr.getNameDialect()->getNamespace() << "\n"
        << Attr.getValue() << "\n";
  }

  for (auto &R : M->getRegions()) {
    ost << "Region\n";
    for (auto &B : R) {
      ost << "Block\n";
      for (auto &Op : B) {
        Op.print(ost);
      }
    }
  }
  std::cout << s << std::endl;

  // Convert module before functions and operations inside, so dialect
  // attributes can be used to change dialect-specific global configurations
  // via `amendOperation()`. These configurations can then influence the
  // translation of operations afterwards.
  if (failed(translator.convertOperation(*module)))
    return nullptr;

  if (failed(translator.convertFunctionSignatures()))
    return nullptr;
  // if (failed(translator.convertGlobalsAndAliases()))
  //   return nullptr;
  // if (failed(translator.convertIFuncs()))
  //   return nullptr;
  // if (failed(translator.createDependentLibrariesMetadata()))
  //   return nullptr;

  // Convert other top-level operations if possible.
  // for (Operation &o : getModuleBody(module).getOperations()) {
  // if (!isa<MimIR::MimIRFuncOp, MimIR::AliasOp, MimIR::GlobalOp,
  //          MimIR::GlobalCtorsOp, MimIR::GlobalDtorsOp, MimIR::ComdatOp,
  //          MimIR::IFuncOp>(&o) &&
  //     !o.hasTrait<OpTrait::IsTerminator>() &&
  //     failed(translator.convertOperation(o, llvmBuilder))) {
  //   return nullptr;
  // }
  // }

  // Operations in function bodies with symbolic references must be
  // converted after the top-level operations they refer to are declared, so
  // we do it last.
  if (failed(translator.convertFunctions()))
    return nullptr;

  // Now that all MLIR blocks are resolved into MimIR ones, patch block
  // address constants to point to the correct blocks. if
  // (failed(translator.convertUnresolvedBlockAddress()))
  //   return nullptr;

  // Add the necessary debug info module flags, if they were not encoded in
  // MLIR beforehand.
  // translator.debugTranslation->addModuleFlagsIfNotPresent();

  return std::move(translator.world_);
}
