//===---- Free-standing function to detect rounding mode --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_ROUNDING_MODE_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_ROUNDING_MODE_H

#include "hdr/fenv_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/type_traits.h"   // is_constant_evaluated
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/macros/properties/cpu_features.h"

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

namespace generic {

// Quick free-standing test whether fegetround() == FE_UPWARD.
// Using the following observation:
//   1.0f + 2^-25 = 1.0f        for FE_TONEAREST, FE_DOWNWARD, FE_TOWARDZERO
//                = 0x1.000002f for FE_UPWARD.
LIBC_INLINE bool fenv_is_round_up() {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
  return false;
#else
  static volatile float x = 0x1.0p-25f;
  return (1.0f + x != 1.0f);
#endif
}

// Quick free-standing test whether fegetround() == FE_DOWNWARD.
// Using the following observation:
//   -1.0f - 2^-25 = -1.0f        for FE_TONEAREST, FE_UPWARD, FE_TOWARDZERO
//                 = -0x1.000002f for FE_DOWNWARD.
LIBC_INLINE bool fenv_is_round_down() {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
  return false;
#else
  static volatile float x = 0x1.0p-25f;
  return (-1.0f - x != -1.0f);
#endif
}

// Quick free-standing test whether fegetround() == FE_TONEAREST.
// Using the following observation:
//   1.5f + 2^-24 = 1.5f           for FE_TONEAREST, FE_DOWNWARD, FE_TOWARDZERO
//                = 0x1.100002p0f  for FE_UPWARD,
//   1.5f - 2^-24 = 1.5f           for FE_TONEAREST, FE_UPWARD
//                = 0x1.0ffffep-1f for FE_DOWNWARD, FE_TOWARDZERO
LIBC_INLINE bool fenv_is_round_to_nearest() {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
  return true;
#else
  static volatile float x = 0x1.0p-24f;
  float y = 1.5f + x;
  return (y == 1.5f - x);
#endif
}

// Quick free-standing test whether fegetround() == FE_TOWARDZERO.
// Using the following observation:
//   1.0f + 2^-23 + 2^-24 = 0x1.000002p0f for FE_DOWNWARD, FE_TOWARDZERO
//                        = 0x1.000004p0f for FE_TONEAREST, FE_UPWARD,
//  -1.0f - 2^-24 = -1.0f          for FE_TONEAREST, FE_UPWARD, FE_TOWARDZERO
//                = -0x1.000002p0f for FE_DOWNWARD
// So:
// (0x1.000002p0f + 2^-24) + (-1.0f - 2^-24) = 2^-23 for FE_TOWARDZERO
//                                           = 2^-22 for FE_TONEAREST, FE_UPWARD
//                                           = 0 for FE_DOWNWARD
LIBC_INLINE bool fenv_is_round_to_zero() {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
  return false;
#else
  static volatile float x = 0x1.0p-24f;
  float y = x;
  return ((0x1.000002p0f + y) + (-1.0f - y) == 0x1.0p-23f);
#endif
}

// Quick free standing get rounding mode based on the above observations.
LIBC_INLINE int quick_get_round() {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
  return FE_TONEAREST;
#else
  static volatile float x = 0x1.0p-24f;
  float y = x;
  float z = (0x1.000002p0f + y) + (-1.0f - y);

  if (z == 0.0f)
    return FE_DOWNWARD;
  if (z == 0x1.0p-23f)
    return FE_TOWARDZERO;
  return (2.0f + y == 2.0f) ? FE_TONEAREST : FE_UPWARD;
#endif
}

} // namespace generic

namespace internal {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
LIBC_INLINE_VAR constexpr bool HAS_DYNAMIC_ROUNDING = false;
#else
LIBC_INLINE_VAR constexpr bool HAS_DYNAMIC_ROUNDING = true;
#endif
} // namespace internal

#if !defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) &&                       \
    defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT) &&                                  \
    !defined(LIBC_COMPILER_IS_MSVC) && defined(__ARM_FP) &&                    \
    (defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_AARCH64))
#define LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING 1
namespace internal {

// ARMv7 VFP, ARMv8 AArch32, and AArch64 all store IEEE-754 rounding control in
// FPSCR/FPCR bits [23:22]: 00 = Nearest, 01 = Upward, 10 = Downward, 11 = Zero.
LIBC_INLINE uint32_t arm_get_fpcr_rmode_bits() {
#ifdef LIBC_TARGET_ARCH_IS_AARCH64
#ifdef LIBC_COMPILER_IS_CLANG
  uint32_t cw = static_cast<uint32_t>(__builtin_arm_rsr("fpcr"));
#else
  uint32_t cw = __builtin_aarch64_get_fpcr();
#endif
#else
  uint32_t cw = __builtin_arm_get_fpscr();
#endif
  return cw & 0xc00000U;
}

} // namespace internal
#endif

LIBC_INLINE LIBC_CONSTEXPR_DEFAULT bool fenv_is_round_up() {
  if (cpp::is_constant_evaluated()) {
    return false;
  } else {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
    return false;
#elif defined(LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING)
    return internal::arm_get_fpcr_rmode_bits() == 0x400000U;
#else
    return generic::fenv_is_round_up();
#endif
  }
}

LIBC_INLINE LIBC_CONSTEXPR_DEFAULT bool fenv_is_round_down() {
  if (cpp::is_constant_evaluated()) {
    return false;
  } else {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
    return false;
#elif defined(LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING)
    return internal::arm_get_fpcr_rmode_bits() == 0x800000U;
#else
    return generic::fenv_is_round_down();
#endif
  }
}

LIBC_INLINE LIBC_CONSTEXPR_DEFAULT bool fenv_is_round_to_nearest() {
  if (cpp::is_constant_evaluated()) {
    return true;
  } else {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
    return true;
#elif defined(LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING)
    return internal::arm_get_fpcr_rmode_bits() == 0U;
#else
    return generic::fenv_is_round_to_nearest();
#endif
  }
}

LIBC_INLINE LIBC_CONSTEXPR_DEFAULT bool fenv_is_round_to_zero() {
  if (cpp::is_constant_evaluated()) {
    return false;
  } else {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
    return false;
#elif defined(LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING)
    return internal::arm_get_fpcr_rmode_bits() == 0xc00000U;
#else
    return generic::fenv_is_round_to_zero();
#endif
  }
}

// Quick free standing get rounding mode.
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT int quick_get_round() {
  if (cpp::is_constant_evaluated()) {
    return FE_TONEAREST;
  } else {
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    !defined(LIBC_TARGET_CPU_HAS_FPU_FLOAT)
    return FE_TONEAREST;
#elif defined(LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING)
    uint32_t rmode = internal::arm_get_fpcr_rmode_bits();
    if constexpr (FE_TONEAREST == 0 && FE_UPWARD == 0x400000 &&
                  FE_DOWNWARD == 0x800000 && FE_TOWARDZERO == 0xc00000) {
      return static_cast<int>(rmode);
    } else {
      constexpr int ROUND_MAP[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD,
                                    FE_TOWARDZERO};
      return ROUND_MAP[rmode >> 22];
    }
#else
    return generic::quick_get_round();
#endif
  }
}

#undef LIBC_FPUTIL_HAS_ARM_FPCR_ROUNDING

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_ROUNDING_MODE_H
