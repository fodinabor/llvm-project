//===- All.h - MLIR To LLVM IR Translation Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the translations of all suitable
// dialects to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIMIR_DIALECT_ALL_H
#define MLIR_TARGET_MIMIR_DIALECT_ALL_H

#include "mlir/Target/MimIR/Dialect/Arith/ArithToMimIRTranslation.h"
#include "mlir/Target/MimIR/Dialect/Builtin/BuiltinToMimIRTranslation.h"
#include "mlir/Target/MimIR/Dialect/ControlFlow/ControlFlowToMimIRTranslation.h"
#include "mlir/Target/MimIR/Dialect/Func/FuncToMimIRTranslation.h"

namespace mlir {
class DialectRegistry;

/// Registers all dialects that can be translated to MIM IR and the
/// corresponding translation interfaces.
static inline void registerAllToMimIRTranslations(DialectRegistry &registry) {
  registerArithDialectTranslationMimIR(registry);
  registerBuiltinDialectTranslationMimIR(registry);
  registerControlFlowDialectTranslationMimIR(registry);
  registerFuncDialectTranslationMimIR(registry);
}

/// Registers all the translations to MIM IR required by GPU passes.
/// TODO: Remove this function when a safe dialect interface registration
/// mechanism is implemented, see D157703.
static inline void
registerAllGPUToMimIRTranslations(DialectRegistry &registry) {
  registerArithDialectTranslationMimIR(registry);
  registerBuiltinDialectTranslationMimIR(registry);
  registerControlFlowDialectTranslationMimIR(registry);
  registerFuncDialectTranslationMimIR(registry);
}

} // namespace mlir

#endif // MLIR_TARGET_MIMIR_DIALECT_ALL_H
