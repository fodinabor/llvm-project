set(LLVM_OPTIONAL_SOURCES
  ConvertToMimIR.cpp
  ModuleTranslation.cpp
  TypeToMimIR.cpp
  )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-fexceptions -Wno-nullability-extension)
add_compile_options(-Wno-nullability-extension)

if(NOT LLVM_ENABLE_EH)
  message(FATAL_ERROR "MimIR target requires exception handling. Please enable LLVM_ENABLE_EH.")
endif()
if(NOT LLVM_ENABLE_RTTI)
  message(FATAL_ERROR "MimIR target requires RTTI. Please enable LLVM_ENABLE_RTTI.")
endif()

find_package(Mim CONFIG REQUIRED)

add_mlir_translation_library(MLIRTargetMimIRExport
  ModuleTranslation.cpp
  TypeToMimIR.cpp
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MimIR/ModuleTranslation.h
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MimIR/TypeToMimIR.h
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MimIR/Export.h
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MimIR/MimIRTranslationInterface.h

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MimIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  TransformUtils
  TargetParser

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRTranslateLib
  MLIRTransformUtils
  mim::libmim
  )

add_subdirectory(Dialect)

add_mlir_translation_library(MLIRToMimIRTranslationRegistration
  ConvertToMimIR.cpp

  # LINK_LIBS PUBLIC
  # MLIRArmNeonToMimIRTranslation
  # MLIRArmSMEToMimIRTranslation
  # MLIRArmSVEToMimIRTranslation
  # MLIRBuiltinToMimIRTranslation
  # MLIRGPUToMimIRTranslation
  # MLIRMimIRToMimIRTranslation
  # MLIRNVVMToMimIRTranslation
  # MLIROpenACCToMimIRTranslation
  # MLIROpenMPToMimIRTranslation
  # MLIRROCDLToMimIRTranslation
  # MLIRPtrToMimIRTranslation
  # MLIRSPIRVToMimIRTranslation
  # MLIRVCIXToMimIRTranslation
  # MLIRXeVMToMimIRTranslation
  LINK_LIBS PUBLIC 
  MLIRBuiltinToMimIRTranslation
  mim::libmim
  )

target_link_libraries(obj.MLIRTargetMimIRExport PUBLIC
  mim::libmim
  )

target_link_libraries(obj.MLIRToMimIRTranslationRegistration PUBLIC
  mim::libmim
  )

# target_compile_features(MLIRTargetMimIRExport PRIVATE cxx_std_20)
# target_compile_options(MLIRTargetMimIRExport PRIVATE -fexceptions -Wno-nullability-extension)
# target_compile_features(MLIRToMimIRTranslationRegistration PRIVATE cxx_std_20)
# target_compile_options(MLIRToMimIRTranslationRegistration PRIVATE -fexceptions -Wno-nullability-extension)

# target_link_libraries(obj.MLIRTargetMimIRExport PUBLIC mim::libmim)

