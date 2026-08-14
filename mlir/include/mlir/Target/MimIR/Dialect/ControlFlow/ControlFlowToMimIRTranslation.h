//==- ControlFlowToMimIRTranslation.h - ControlFlow to MIM IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MIM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for ControlFlow dialect to MimIR
// translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_DIALECT_CONTROLFLOW_CONTROLFLOWTOMIMIRTRANSLATION_H
#define MLIR_TARGET_MIMIR_DIALECT_CONTROLFLOW_CONTROLFLOWTOMIMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the translation from the ControlFlow dialect to the MIM IR in the
/// given registry.
void registerControlFlowDialectTranslationMimIR(DialectRegistry &registry);

/// Register the translation from the ControlFlow dialect in the registry
/// associated with the given context.
void registerControlFlowDialectTranslationMimIR(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_DIALECT_CONTROLFLOW_CONTROLFLOWTOMIMIRTRANSLATION_H
