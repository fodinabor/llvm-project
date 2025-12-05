//===- MimIRTranslationInterface.h - Translation to MimIR iface ---*- C++ -*-===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines dialect interfaces for translation to MimIR IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_MimIRTRANSLATIONINTERFACE_H
#define MLIR_TARGET_MIMIR_MimIRTRANSLATIONINTERFACE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"

#include <mim/world.h>

namespace mlir {
namespace MimIR {
class ModuleTranslation;
// class MimIRFuncOp;
} // namespace MimIR

/// Base class for dialect interfaces providing translation to MimIR IR.
/// Dialects that can be translated should provide an implementation of this
/// interface for the supported operations. The interface may be implemented in
/// a separate library to avoid the "main" dialect library depending on MimIR.
/// The interface can be attached using the delayed registration mechanism
/// available in DialectRegistry.
class MimIRTranslationDialectInterface
    : public DialectInterface::Base<MimIRTranslationDialectInterface> {
public:
  MimIRTranslationDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to provide translation of the
  /// operations to MimIR IR.
  virtual LogicalResult
  convertOperation(Operation *op, mim::World &world,
                   MimIR::ModuleTranslation &moduleTranslation) const {
    return failure();
  }

  /// Hook for derived dialect interface to act on an operation that has dialect
  /// attributes from the derived dialect (the operation itself may be from a
  /// different dialect). This gets called after the operation has been
  /// translated. The hook is expected to use moduleTranslation to look up the
  /// translation results and amend the corresponding IR constructs. Does
  /// nothing and succeeds by default.
  virtual LogicalResult
  amendOperation(Operation *op, ArrayRef<const mim::Def *> instructions,
                 NamedAttribute attribute,
                 MimIR::ModuleTranslation &moduleTranslation) const {
    return success();
  }

  /// Hook for derived dialect interface to translate or act on a derived
  /// dialect attribute that appears on a function parameter. This gets called
  /// after the function operation has been translated.
  // virtual LogicalResult
  // convertParameterAttr(MimIR::MimIRFuncOp function, int argIdx,
  //                      NamedAttribute attr,
  //                      MimIR::ModuleTranslation &moduleTranslation) const {
  //   return success();
  // }
};

/// Interface collection for translation to MimIR IR, dispatches to a concrete
/// interface implementation based on the dialect to which the given op belongs.
class MimIRTranslationInterface
    : public DialectInterfaceCollection<MimIRTranslationDialectInterface> {
public:
  using Base::Base;

  /// Translates the given operation to MimIR IR using the interface implemented
  /// by the op's dialect.
  virtual LogicalResult
  convertOperation(Operation *op, mim::World &w,
                   MimIR::ModuleTranslation &moduleTranslation) const {
    if (const MimIRTranslationDialectInterface *iface = getInterfaceFor(op))
      return iface->convertOperation(op, w, moduleTranslation);
    return failure();
  }

  /// Acts on the given operation using the interface implemented by the dialect
  /// of one of the operation's dialect attributes.
  virtual LogicalResult
  amendOperation(Operation *op, ArrayRef<const mim::Def *> instructions,
                 NamedAttribute attribute,
                 MimIR::ModuleTranslation &moduleTranslation) const {
    if (const MimIRTranslationDialectInterface *iface =
            getInterfaceFor(attribute.getNameDialect())) {
      return iface->amendOperation(op, instructions, attribute,
                                   moduleTranslation);
    }
    return success();
  }

  /// Acts on the given function operation using the interface implemented by
  /// the dialect of one of the function parameter attributes.
  // virtual LogicalResult
  // convertParameterAttr(MimIR::MimIRFuncOp function, int argIdx,
  //                      NamedAttribute attribute,
  //                      MimIR::ModuleTranslation &moduleTranslation) const {
  //   if (const MimIRTranslationDialectInterface *iface =
  //           getInterfaceFor(attribute.getNameDialect())) {
  //     return iface->convertParameterAttr(function, argIdx, attribute,
  //                                        moduleTranslation);
  //   }
  //   function.emitWarning("Unhandled parameter attribute '" +
  //                        attribute.getName().str() + "'");
  //   return success();
  // }
};

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_MimIRTRANSLATIONINTERFACE_H
