//===- TypeToMimIR.h - Translate types from MLIR to MimIR --*- C++ -*-===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type translation function going from MLIR
// dialect to MimIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_TYPETOMimIR_H
#define MLIR_TARGET_MIMIR_TYPETOMimIR_H

#include <memory>

namespace mim {
class Def;
class World;
} // namespace mim

namespace mlir {

class Type;
class MLIRContext;

namespace MimIR {

namespace detail {
class TypeToMimIRTranslatorImpl;
} // namespace detail

/// Utility class to translate MLIR MimIR dialect types to MimIR IR. Stores the
/// translation state, in particular any identified structure types that can be
/// reused in further translation.
class TypeToMimIRTranslator {
public:
  TypeToMimIRTranslator(mim::World &world);
  ~TypeToMimIRTranslator();

  /// Returns the preferred alignment for the type given the data layout. Note
  /// that this will perform type conversion and store its results for future
  /// uses.
  // TODO: this should be removed when MLIR has proper data layout.
  // unsigned getPreferredAlignment(Type type, const llvm::DataLayout &layout);

  /// Translates the given MLIR MimIR dialect type to MimIR IR.
  const mim::Def *translateType(Type type);

private:
  /// Private implementation.
  std::unique_ptr<detail::TypeToMimIRTranslatorImpl> impl;
};

} // namespace MimIR
} // namespace mlir

#endif // MLIR_TARGET_MIMIR_TYPETOMimIR_H
