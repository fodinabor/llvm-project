//===- Export.h - MLIR to MimIR IR translation entry point -------*- C++ -*-===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_EXPORT_H
#define MLIR_TARGET_MIMIR_EXPORT_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mim {
class World;
class Driver;
} // namespace mim

namespace mlir {
class Operation;

/// Translates a given `module` into an MimIR module living in
/// the given context. Returns nullptr when the translation fails.
std::unique_ptr<mim::World>
translateModuleToMimIR(Operation *module, mim::Driver &driver,
                       llvm::StringRef name = "MimIRDialectModule");
} // namespace mlir

#endif // MLIR_TARGET_MIMIR_EXPORT_H
