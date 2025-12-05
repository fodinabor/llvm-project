//===- ConvertToMimIR.cpp - MLIR to MimIR IR conversion -------------------===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR MimIR dialect and MimIR IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/MimIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <mim/world.h>
#include <mim/driver.h>
#include <sstream>

using namespace mlir;

namespace mlir {
void registerToMimIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-mim", "Translate MLIR to MimIR",
      [](Operation *op, raw_ostream &output) {
        mim::Driver driver;
        auto world = translateModuleToMimIR(op, driver);
        if (!world)
          return failure();

        std::stringstream ss;
        world->dump(ss);
        output << ss.str();
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect, func::FuncDialect>();
        // registerAllToMimIRTranslations(registry);
      });
}
} // namespace mlir
