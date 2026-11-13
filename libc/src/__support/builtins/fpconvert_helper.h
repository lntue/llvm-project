//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared float-to-float extend/truncate conversions, rounding to nearest
/// (ties to even) when narrowing.  These mirror compiler-rt's __extend<a><b>2 /
/// __trunc<a><b>2 builtins, so they can be reused by compiler-rt's builtins.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_BUILTINS_FPCONVERT_HELPER_H
#define LLVM_LIBC_SRC___SUPPORT_BUILTINS_FPCONVERT_HELPER_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fp_convert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/cpu_features.h"

namespace LIBC_NAMESPACE_DECL {
namespace builtins {

// Convert a floating-point value from From to To (extend or truncate) using
// pure-integer RTNE emulation so soft-float builtins never recurse into
// compiler runtime routines.
template <typename To, typename From>
LIBC_ALWAYS_INLINE constexpr To fpconvert(From x) {
  return fputil::integer_eval::cast<To, From>(x);
}

// Same, for a source delivered as raw bits.  Keeps _Float16 out of the
// signature, which would otherwise lower to a circular __extendhfsf2.
template <typename To, fputil::FPType FromFPType>
LIBC_ALWAYS_INLINE constexpr To
fpconvert_from_bits(typename fputil::FPRep<FromFPType>::StorageType bits) {
  constexpr fputil::FPType TO_FP_TYPE = fputil::get_fp_type<To>();
  return fputil::internal::bits_to_val<To>(
      fputil::integer_eval::cast_bits<TO_FP_TYPE, FromFPType>(bits));
}

// Same, for a destination delivered as raw bits.  Keeps _Float16 out of the
// signature, which would otherwise lower to a circular __trunc*hf2.
template <fputil::FPType ToFPType, typename From>
LIBC_ALWAYS_INLINE constexpr typename fputil::FPRep<ToFPType>::StorageType
fpconvert_to_bits(From x) {
  constexpr fputil::FPType FROM_FP_TYPE = fputil::get_fp_type<From>();
  return fputil::integer_eval::cast_bits<ToFPType, FROM_FP_TYPE>(
      fputil::internal::raw_to_bits<From>(x));
}

} // namespace builtins
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_BUILTINS_FPCONVERT_HELPER_H
