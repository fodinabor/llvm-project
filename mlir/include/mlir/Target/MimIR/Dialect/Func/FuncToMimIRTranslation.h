//==- FuncToMIMIRTranslation.h - Func Dialect to MIM IR -*- C++ -*-==//
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

#ifndef MLIR_TARGET_MIMIR_DIALECT_FUNC_FUNCTOMIMIRTRANSLATION_H
#define MLIR_TARGET_MIMIR_DIALECT_FUNC_FUNCTOMIMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the translation from the Func dialect to the MIM IR in the
/// given registry.
void registerFuncDialectTranslationMimIR(DialectRegistry &registry);

/// Register the translation from the Func dialect in the registry associated
/// with the given context.
void registerFuncDialectTranslationMimIR(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_DIALECT_FUNC_FUNCTOMIMIRTRANSLATION_H
