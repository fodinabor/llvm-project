//===- ConvertToMimIR.cpp - MLIR to MimIR IR conversion -------------------===//
//
// Part of the MimIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR MimIR dialect and MimIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/MimIR/Dialect/All.h"
#include "mlir/Target/MimIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <mim/def.h>
#include <mim/driver.h>
#include <mim/world.h>

#include <sstream>

using namespace mlir;

namespace mlir {
void registerToMimIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-mim", "Translate MLIR to MimIR",
      [](Operation *op, raw_ostream &output) {
        try {
          mim::Driver driver;
          driver.log().set(&std::cerr).set(mim::Log::Level::Error);

          auto world = translateModuleToMimIR(op, driver);
          if (!world)
            return failure();

          std::stringstream ss;
          world->dump(ss);
          output << ss.str();
        } catch (const mim::Error &e) {
          std::stringstream ess;
          ess << "MimIR translation error: " << e << "\n";
          output << ess.str();
          return failure();
        }
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect, func::FuncDialect>();
        registry.insert<affine::AffineDialect, arith::ArithDialect>();
        registry.insert<cf::ControlFlowDialect, scf::SCFDialect>();
        registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
        registerAllToMimIRTranslations(registry);
      });
}
} // namespace mlir
