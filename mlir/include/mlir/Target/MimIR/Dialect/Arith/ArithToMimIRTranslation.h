//==- ArithToMIMIRTranslation.h - Arith Dialect to MIM IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MIM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for builtin dialect to MimIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_DIALECT_ARITH_ARITHTOMIMIRTRANSLATION_H
#define MLIR_TARGET_MIMIR_DIALECT_ARITH_ARITHTOMIMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the translation from the Arith dialect to the MIM IR in the
/// given registry.
void registerArithDialectTranslationMimIR(DialectRegistry &registry);

/// Register the translation from the Arith dialect in the registry associated
/// with the given context.
void registerArithDialectTranslationMimIR(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_DIALECT_ARITH_ARITHTOMIMIRTRANSLATION_H
