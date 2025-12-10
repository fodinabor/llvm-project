add_mlir_translation_library(MLIRBuiltinToMimIRTranslation
  BuiltinToMimIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTargetMimIRExport
  )
