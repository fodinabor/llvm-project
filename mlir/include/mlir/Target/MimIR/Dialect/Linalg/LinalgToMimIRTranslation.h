//==- LinalgToMimIRTranslation.h - Linalg Dialect to MIM IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MIM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for Linalg dialect to MimIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_DIALECT_LINALG_LINALGTOMIMIRTRANSLATION_H
#define MLIR_TARGET_MIMIR_DIALECT_LINALG_LINALGTOMIMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the translation from the Linalg dialect to the MIM IR in the
/// given registry.
void registerLinalgDialectTranslationMimIR(DialectRegistry &registry);

/// Register the translation from the Linalg dialect in the registry
/// associated with the given context.
void registerLinalgDialectTranslationMimIR(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_DIALECT_LINALG_LINALGTOMIMIRTRANSLATION_H
