//===-- Conversion between floating-point types -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_CAST_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_CAST_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fp_convert.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

namespace internal {

#ifdef LIBC_MATH_HAS_NO_ERRNO
LIBC_INLINE_VAR constexpr uint8_t ERRNO_EXCEPT_MASK = FP_EXCEPT_NONE;
#else
LIBC_INLINE_VAR constexpr uint8_t ERRNO_EXCEPT_MASK =
    FP_EXCEPT_OVERFLOW | FP_EXCEPT_UNDERFLOW;
#endif

LIBC_INLINE_VAR constexpr uint8_t ACTIVE_ENV_FLAGS_MASK =
    (HAS_FENV_EXCEPT ? uint8_t(0xffU) : FP_EXCEPT_NONE) | ERRNO_EXCEPT_MASK;

template <FPType DstType>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT void
signal_conversion_exceptions(uint8_t flags, uint8_t possible_mask = 0xffU) {
  flags &= possible_mask;
  if (flags == FP_EXCEPT_NONE || cpp::is_constant_evaluated())
    return;

  using ExceptT = cpp::conditional_t<
      DstType == FPType::IEEE754_Binary64 && HAS_HW_FPU<double>, double,
      cpp::conditional_t<HAS_HW_FPU<float>, float, void>>;

  constexpr uint8_t RARE_MASK =
      FP_EXCEPT_INVALID | FP_EXCEPT_OVERFLOW | FP_EXCEPT_UNDERFLOW;
  if (LIBC_UNLIKELY((flags & RARE_MASK) != 0)) {
    raise_rare_conversion_except<ExceptT>((flags & FP_EXCEPT_INVALID) != 0,
                                          (flags & FP_EXCEPT_OVERFLOW) != 0,
                                          (flags & FP_EXCEPT_UNDERFLOW) != 0);
  } else if constexpr (HAS_FENV_EXCEPT) {
    if ((flags & FP_EXCEPT_INEXACT) != 0)
      raise_inexact_except_if_required();
  }
}

template <typename T>
LIBC_INLINE_VAR constexpr bool IS_NATIVE_HW_CAST_TYPE =
    (cpp::is_same_v<T, float> || cpp::is_same_v<T, double>) && HAS_HW_FPU<T>;

} // namespace internal

namespace fenv_eval {

template <FPType DstType, FPType SrcType>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT typename FPRep<DstType>::StorageType
cast_bits(typename FPRep<SrcType>::StorageType src_bits) {
  using SrcRep = FPRep<SrcType>;
  using DstRep = FPRep<DstType>;
  constexpr bool EXACT_WIDENING =
      DstRep::FRACTION_LEN >= SrcRep::FRACTION_LEN &&
      DstRep::EXP_BIAS >= SrcRep::EXP_BIAS;
  constexpr bool STATIC_NEAREST =
      EXACT_WIDENING || !fputil::internal::HAS_DYNAMIC_ROUNDING;
  constexpr uint8_t POSSIBLE_FLAGS = static_cast<uint8_t>(
      (EXACT_WIDENING ? fputil::internal::FP_EXCEPT_INVALID : uint8_t(0xffU)) &
      fputil::internal::ACTIVE_ENV_FLAGS_MASK);

  if constexpr (STATIC_NEAREST && POSSIBLE_FLAGS == 0) {
    return integer_eval::cast_bits<DstType, SrcType>(src_bits);
  } else {
    int rounding = FE_TONEAREST;
    if constexpr (!STATIC_NEAREST) {
      if (!cpp::is_constant_evaluated())
        rounding = quick_get_round();
    }
    uint8_t flags = fputil::internal::FP_EXCEPT_NONE;
    auto bits =
        fputil::internal::cast_bits_impl<DstType, SrcType, STATIC_NEAREST,
                                         POSSIBLE_FLAGS>(src_bits, rounding,
                                                         &flags);
    fputil::internal::signal_conversion_exceptions<DstType>(flags,
                                                            POSSIBLE_FLAGS);
    return bits;
  }
}

template <typename OutType, typename InType>
LIBC_ALWAYS_INLINE LIBC_CONSTEXPR_DEFAULT cpp::enable_if_t<
    cpp::is_floating_point_v<OutType> && cpp::is_floating_point_v<InType>,
    OutType>
cast(InType x) {
  if constexpr (cpp::is_same_v<InType, OutType>) {
    return x;
  } else {
    constexpr FPType SRC_TYPE = get_fp_type<InType>();
    constexpr FPType DST_TYPE = get_fp_type<OutType>();
    return fputil::internal::bits_to_val<OutType>(cast_bits<DST_TYPE, SRC_TYPE>(
        fputil::internal::raw_to_bits<InType>(x)));
  }
}

} // namespace fenv_eval

namespace fp_eval {

template <typename OutType, typename InType>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<
    fputil::internal::IS_NATIVE_HW_CAST_TYPE<OutType> &&
        fputil::internal::IS_NATIVE_HW_CAST_TYPE<InType>,
    OutType>
cast(InType x) {
  return static_cast<OutType>(x);
}

} // namespace fp_eval

template <typename OutType, typename InType>
LIBC_ALWAYS_INLINE LIBC_CONSTEXPR_DEFAULT cpp::enable_if_t<
    cpp::is_floating_point_v<OutType> && cpp::is_floating_point_v<InType>,
    OutType>
cast(InType x) {
  if constexpr (cpp::is_same_v<InType, OutType>) {
    return x;
  } else {
#ifdef LIBC_FPUTIL_CAST_IMPL
#if LIBC_MATH_IMPL_IS(LIBC_FPUTIL_CAST_IMPL, integer_eval)
    return integer_eval::cast<OutType, InType>(x);
#elif LIBC_MATH_IMPL_IS(LIBC_FPUTIL_CAST_IMPL, fenv_eval)
    return fenv_eval::cast<OutType, InType>(x);
#elif LIBC_MATH_IMPL_IS(LIBC_FPUTIL_CAST_IMPL, fp_eval)
    return fp_eval::cast<OutType, InType>(x);
#else
#error "Unsupported LIBC_FPUTIL_CAST_IMPL"
#endif
#else // !LIBC_FPUTIL_CAST_IMPL
    // Only native float <-> double on targets with hardware FPU use hardware
    // static_cast; float16, bfloat16, Float80/long double, Float128/float128,
    // and soft-float targets always use full IEEE-754 software emulation.
    if constexpr (internal::IS_NATIVE_HW_CAST_TYPE<OutType> &&
                  internal::IS_NATIVE_HW_CAST_TYPE<InType>) {
      return fp_eval::cast<OutType, InType>(x);
    } else if constexpr (
#if defined(LIBC_MATH_HAS_NO_EXCEPT) && defined(LIBC_MATH_HAS_NO_ERRNO) &&     \
    defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)
        true
#else
        false
#endif
    ) {
      return integer_eval::cast<OutType, InType>(x);
    } else {
      return fenv_eval::cast<OutType, InType>(x);
    }
#endif // LIBC_FPUTIL_CAST_IMPL
  }
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_CAST_H
