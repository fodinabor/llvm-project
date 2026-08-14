//===- TypeToMimIR.cpp - type translation from MLIR to MimIR IR -===//
//
// Part of the MimIR Project, under the Apache License v2.0 with MimIR
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH MimIR-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/MimIR/TypeToMimIR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"

#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/world.h>

using namespace mlir;

namespace mlir {
namespace MimIR {
namespace detail {
/// Support for translating MLIR MimIR dialect types to MimIR IR.
class TypeToMimIRTranslatorImpl {
public:
  /// Constructs a class creating types in the given MimIR context.
  TypeToMimIRTranslatorImpl(mim::World &world) : world(world) {}

  /// Translates a single type.
  const mim::Def *translateType(Type type) {
    // If the conversion is already known, just return it.
    if (knownTranslations.count(type))
      return knownTranslations.lookup(type);

    // Dispatch to an appropriate function.
    const mim::Def *translated =
        llvm::TypeSwitch<Type, const mim::Def *>(type)
            // .Case([this](MimIR::MimIRVoidType) {
            //   return llvm::Type::getVoidTy(context);
            // })
            .Case([this](Float16Type) {
              return world.annex<mim::plug::math::F16>();
            })
            .Case([this](BFloat16Type) {
              return world.annex<mim::plug::math::BF16>();
            })
            .Case([this](Float32Type) {
              return world.annex<mim::plug::math::F32>();
            })
            .Case([this](Float64Type) {
              return world.annex<mim::plug::math::F64>();
            })
            .Case<IntegerType, VectorType, RankedTensorType,
                  PtrLikeTypeInterface, IndexType>(
                [this](auto type) { return this->translate(type); })
            .Default([](Type type) {
              llvm::errs() << type << " cannot be converted to MimIR, yet";
              llvm::llvm_unreachable_internal("not convertible yet");
              return nullptr;
            });
    // .DefaultUnreachable("unknown MimIR dialect type");

    // Cache the result of the conversion and return.
    knownTranslations.try_emplace(type, translated);
    return translated;
  }

private:
  /// Translates the given array type.
  // llvm::Type *translate(MimIR::MimIRArrayType type) {
  //   return llvm::ArrayType::get(translateType(type.getElementType()),
  //                               type.getNumElements());
  // }

  // /// Translates the given function type.
  // llvm::Type *translate(MimIR::MimIRFunctionType type) {
  //   SmallVector<llvm::Type *, 8> paramTypes;
  //   translateTypes(type.getParams(), paramTypes);
  //   return llvm::FunctionType::get(translateType(type.getReturnType()),
  //                                  paramTypes, type.isVarArg());
  // }

  /// Translates the given integer type.
  const mim::Def *translate(IntegerType type) {
    assert(type.isSignless() &&
           "only signless integer types are supported in MimIR translation");
    return world.type_int(type.getWidth());
  }

  /// Translates the given Index type.
  const mim::Def *translate(IndexType type) { return world.type_int(64); }

  // /// Translates the given pointer type.
  // llvm::Type *translate(MimIR::MimIRPointerType type) {
  //   return llvm::PointerType::get(context, type.getAddressSpace());
  // }

  // /// Translates the given structure type, supports both identified and
  // literal
  // /// structs. This will _create_ a new identified structure every time, use
  // /// `convertType` if a structure with the same name must be looked up
  // instead. llvm::Type *translate(MimIR::MimIRStructType type) {
  //   SmallVector<llvm::Type *, 8> subtypes;
  //   if (!type.isIdentified()) {
  //     translateTypes(type.getBody(), subtypes);
  //     return llvm::StructType::get(context, subtypes, type.isPacked());
  //   }

  //   llvm::StructType *structType =
  //       llvm::StructType::create(context, type.getName());
  //   // Mark the type we just created as known so that recursive calls can
  //   pick
  //   // it up and use directly.
  //   knownTranslations.try_emplace(type, structType);
  //   if (type.isOpaque())
  //     return structType;

  //   translateTypes(type.getBody(), subtypes);
  //   structType->setBody(subtypes, type.isPacked());
  //   return structType;
  // }

  /// Translates the given built-in vector type compatible with MimIR.
  const mim::Def *translate(VectorType type) {
    if (type.isScalable()) {
      llvm::report_fatal_error(
          "scalable vector types are not yet supported in MimIR translation");
    }
    auto s = type.getShape();
    mim::Vector<mim::u64> dims;
    for (auto dim : s) {
      assert(dim >= 0 && "expected static vector dimension");
      dims.push_back(dim);
    }
    return world.arr(dims, translateType(type.getElementType()));
  }

  /// Translates the given ranked tensor type to a (nested) MimIR array.
  const mim::Def *translate(RankedTensorType type) {
    if (!type.hasStaticShape()) {
      llvm::report_fatal_error(
          "dynamic tensor shapes are not yet supported in MimIR translation");
    }
    mim::Vector<mim::u64> dims;
    for (auto dim : type.getShape())
      dims.push_back(dim);
    return world.arr(dims, translateType(type.getElementType()));
  }

  // /// Translates the given target extension type.
  // llvm::Type *translate(MimIR::MimIRTargetExtType type) {
  //   SmallVector<llvm::Type *> typeParams;
  //   translateTypes(type.getTypeParams(), typeParams);
  //   return llvm::TargetExtType::get(context, type.getExtTypeName(),
  //   typeParams,
  //                                   type.getIntParams());
  // }

  /// Translates the given ptr type.
  const mim::Def *translate(PtrLikeTypeInterface type) {
    // auto memSpace =
    //     dyn_cast<MimIR::MimIRAddrSpaceAttrInterface>(type.getMemorySpace());
    // assert(memSpace && "expected pointer with an MimIR address space");
    assert(!type.hasPtrMetadata() && "expected pointer without metadata");
    return world.call<mim::plug::mem::Ptr>(
        mim::Defs{translateType(type.getElementType()),
                  /* todo: translate addrspace! */ world.lit_nat_0()});
  }

  /// Translates a list of types.
  void translateTypes(ArrayRef<Type> types,
                      SmallVectorImpl<const mim::Def *> &result) {
    result.reserve(result.size() + types.size());
    for (auto type : types)
      result.push_back(translateType(type));
  }

  /// Reference to the context in which the MimIR IR types are created.
  mim::World &world;

  /// Map of known translation. This serves a double purpose: caches translation
  /// results to avoid repeated recursive calls and makes sure identified
  /// structs with the same name (that is, equal) are resolved to an existing
  /// type instead of creating a new type.
  llvm::DenseMap<Type, const mim::Def *> knownTranslations;
};
} // namespace detail
} // namespace MimIR
} // namespace mlir

MimIR::TypeToMimIRTranslator::TypeToMimIRTranslator(mim::World &world)
    : impl(new detail::TypeToMimIRTranslatorImpl(world)) {}

MimIR::TypeToMimIRTranslator::~TypeToMimIRTranslator() = default;

const mim::Def *MimIR::TypeToMimIRTranslator::translateType(Type type) {
  return impl->translateType(type);
}

// unsigned MimIR::TypeToMimIRTranslator::getPreferredAlignment(
//     Type type, const llvm::DataLayout &layout) {
//   return layout.getPrefTypeAlign(translateType(type)).value();
// }
