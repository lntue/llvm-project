//===-- Bit-level floating-point format conversions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_CONVERT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_CONVERT_H

#include "hdr/fenv_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/sign.h"
#include "src/__support/uint128.h"

#include <limits.h>
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

namespace internal {

// Internal exception status flags independent of target FE_* macro values
// (which may be defined to 0 in freestanding or non-fenv environments).
LIBC_INLINE_VAR constexpr uint8_t FP_EXCEPT_NONE = 0;
LIBC_INLINE_VAR constexpr uint8_t FP_EXCEPT_INVALID = 1 << 0;
LIBC_INLINE_VAR constexpr uint8_t FP_EXCEPT_OVERFLOW = 1 << 1;
LIBC_INLINE_VAR constexpr uint8_t FP_EXCEPT_UNDERFLOW = 1 << 2;
LIBC_INLINE_VAR constexpr uint8_t FP_EXCEPT_INEXACT = 1 << 3;

template <typename StorageType> struct ConversionResult {
  StorageType bits;
  uint8_t except_flags;
};

LIBC_ALWAYS_INLINE constexpr uint64_t u128_lo(const UInt128 &v) {
#ifdef LIBC_TYPES_HAS_INT128
  return static_cast<uint64_t>(v);
#else
  return v[0];
#endif
}

LIBC_ALWAYS_INLINE constexpr uint64_t u128_hi(const UInt128 &v) {
#ifdef LIBC_TYPES_HAS_INT128
  return static_cast<uint64_t>(v >> 64U);
#else
  return v[1];
#endif
}

LIBC_ALWAYS_INLINE constexpr UInt128 make_u128(uint64_t hi, uint64_t lo) {
#ifdef LIBC_TYPES_HAS_INT128
  return (static_cast<UInt128>(hi) << 64U) | static_cast<UInt128>(lo);
#else
  UInt128 r{};
  r[0] = lo;
  r[1] = hi;
  return r;
#endif
}

// Bit-cast helpers that bypass FPBits::FP_MASK BigInt<128>::operator& at -Oz
// when the floating-point type occupies its full storage width.
template <typename T>
LIBC_ALWAYS_INLINE constexpr typename FPBits<T>::StorageType raw_to_bits(T x) {
  using StorageType = typename FPBits<T>::StorageType;
  if constexpr (get_fp_type<T>() != FPType::X86_Binary80 &&
                sizeof(T) == sizeof(StorageType)) {
    return cpp::bit_cast<StorageType>(x);
  } else {
    return FPBits<T>(x).uintval();
  }
}

template <typename T>
LIBC_ALWAYS_INLINE constexpr T
bits_to_val(typename FPBits<T>::StorageType bits) {
  using StorageType = typename FPBits<T>::StorageType;
  if constexpr (get_fp_type<T>() != FPType::X86_Binary80 &&
                sizeof(T) == sizeof(StorageType)) {
    return cpp::bit_cast<T>(bits);
  } else {
    return FPBits<T>(bits).get_val();
  }
}

LIBC_INLINE constexpr bool should_round_up(Sign sign, bool lsb, bool round_bit,
                                           bool sticky_bit, int rounding) {
  switch (rounding) {
  case FE_TONEAREST:
    return round_bit && (lsb || sticky_bit);
  case FE_DOWNWARD:
    return sign.is_neg() && (round_bit || sticky_bit);
  case FE_UPWARD:
    return sign.is_pos() && (round_bit || sticky_bit);
  case FE_TOWARDZERO:
  default:
    return false;
  }
}

LIBC_INLINE constexpr bool overflow_rounds_to_inf(Sign sign, int rounding) {
  switch (rounding) {
  case FE_TONEAREST:
    return true;
  case FE_DOWNWARD:
    return sign.is_neg();
  case FE_UPWARD:
    return sign.is_pos();
  case FE_TOWARDZERO:
  default:
    return false;
  }
}

// Round a packed implicit-bit IEEE-754 value `out` given remainder word `rem`
// whose MSB holds `round_bit` and lower bits hold `sticky_bits`.
// Incrementing `out` naturally carries across subnormal->min_normal, normal
// binades, and max_normal->inf.
template <typename DstStorage, typename RemT, bool StaticNearest>
LIBC_ALWAYS_INLINE constexpr DstStorage
round_packed_ieee(bool is_neg, DstStorage out, RemT rem,
                  [[maybe_unused]] int rounding) {
  constexpr RemT HALF = RemT(1) << (sizeof(RemT) * CHAR_BIT - 1);
  if constexpr (StaticNearest) {
    return static_cast<DstStorage>(
        out +
        static_cast<DstStorage>((rem | RemT(out & DstStorage(1))) > HALF));
  } else {
    bool up = false;
    switch (rounding) {
    case FE_TONEAREST:
      up = (rem | RemT(out & DstStorage(1))) > HALF;
      break;
    case FE_DOWNWARD:
      up = is_neg && (rem != 0);
      break;
    case FE_UPWARD:
      up = !is_neg && (rem != 0);
      break;
    case FE_TOWARDZERO:
    default:
      break;
    }
    if (up)
      ++out;
    return out;
  }
}

// Increment a packed floating-point representation by 1 ULP, handling the
// explicit integer bit of X86_Binary80 across subnormal->normal, normal
// binade carries, and max_normal->inf transitions.
template <FPType DstType>
LIBC_INLINE constexpr typename FPRep<DstType>::StorageType
apply_rounding_increment(typename FPRep<DstType>::StorageType result) {
  using DstRep = FPRep<DstType>;
  using StorageType = typename DstRep::StorageType;
  ++result;
  if constexpr (DstType == FPType::X86_Binary80) {
    constexpr StorageType EXPLICIT_BIT = StorageType(1) << DstRep::FRACTION_LEN;
    if ((result & DstRep::EXP_MASK) == 0) {
      if ((result & EXPLICIT_BIT) != 0)
        result += StorageType(1) << DstRep::SIG_LEN;
    } else {
      result |= EXPLICIT_BIT;
    }
  }
  return result;
}

template <FPType DstType, uint8_t FlagsMask>
LIBC_ALWAYS_INLINE constexpr void
record_normal_narrow_flags(typename FPRep<DstType>::StorageType out,
                           bool is_inexact, uint8_t *out_flags) {
  using DstRep = FPRep<DstType>;
  if constexpr (FlagsMask != 0) {
    if constexpr ((FlagsMask & FP_EXCEPT_INEXACT) == 0) {
      if constexpr ((FlagsMask & FP_EXCEPT_OVERFLOW) != 0) {
        if (LIBC_UNLIKELY((out & DstRep::EXP_MASK) == DstRep::EXP_MASK))
          *out_flags = FP_EXCEPT_OVERFLOW;
      }
    } else if (is_inexact) {
      *out_flags =
          LIBC_UNLIKELY((out & DstRep::EXP_MASK) == DstRep::EXP_MASK)
              ? static_cast<uint8_t>((FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT) &
                                     FlagsMask)
              : static_cast<uint8_t>(FP_EXCEPT_INEXACT & FlagsMask);
    }
  }
}

template <uint8_t FlagsMask>
LIBC_ALWAYS_INLINE constexpr void set_masked_flags(uint8_t f,
                                                   uint8_t *out_flags) {
  if constexpr (FlagsMask != 0)
    *out_flags = static_cast<uint8_t>(f & FlagsMask);
}

template <FPType DstType, bool StaticNearest, uint8_t FlagsMask>
LIBC_ALWAYS_INLINE constexpr typename FPRep<DstType>::StorageType
overflow_result(Sign sign, [[maybe_unused]] int rounding, uint8_t *out_flags) {
  using DstRep = FPRep<DstType>;
  using DstStorage = typename DstRep::StorageType;
  set_masked_flags<FlagsMask>(FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT,
                              out_flags);
  bool to_inf = StaticNearest ? true : overflow_rounds_to_inf(sign, rounding);
  if constexpr (DstType != FPType::X86_Binary80) {
    DstStorage s_bit =
        sign.is_neg() ? (DstStorage(1) << (DstRep::STORAGE_LEN - 1)) : 0;
    return static_cast<DstStorage>(s_bit |
                                   (DstRep::EXP_MASK - DstStorage(!to_inf)));
  }
  return to_inf ? DstRep::inf(sign).uintval()
                : DstRep::max_normal(sign).uintval();
}

// Core bit-level IEEE-754 conversion returning DstStorage directly in registers
// (avoiding AAPCS sret stack spills on 32-bit targets).
template <FPType DstType, FPType SrcType, bool StaticNearest, uint8_t FlagsMask>
LIBC_ALWAYS_INLINE constexpr typename FPRep<DstType>::StorageType
cast_bits_impl(typename FPRep<SrcType>::StorageType src_bits,
               [[maybe_unused]] int rounding,
               [[maybe_unused]] uint8_t *out_flags) {
  using SrcRep = FPRep<SrcType>;
  using SrcStorage = typename SrcRep::StorageType;
  using DstRep = FPRep<DstType>;
  using DstStorage = typename DstRep::StorageType;

  if constexpr (FlagsMask != 0)
    *out_flags = FP_EXCEPT_NONE;

#define LIBC_FP_SET_FLAGS(f) set_masked_flags<FlagsMask>((f), out_flags)
#define LIBC_FP_OVF_RESULT(s)                                                  \
  overflow_result<DstType, StaticNearest, FlagsMask>((s), rounding, out_flags)

  if constexpr (DstType == SrcType) {
    SrcRep x_bits(src_bits);
    if (LIBC_UNLIKELY(x_bits.is_nan())) {
      if (x_bits.is_signaling_nan())
        LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
      return DstRep::quiet_nan(x_bits.sign(), x_bits.get_mantissa()).uintval();
    }
    return x_bits.uintval();
  } else if constexpr (SrcType == FPType::BFloat16 &&
                       DstType == FPType::IEEE754_Binary32) {
    uint32_t out = static_cast<uint32_t>(src_bits) << 16U;
    uint16_t abs_bits = static_cast<uint16_t>(src_bits & 0x7fffU);
    if (LIBC_UNLIKELY(abs_bits > 0x7f80U)) {
      if ((abs_bits & 0x0040U) == 0)
        LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
      return out | 0x0040'0000U;
    }
    return out;
  } else if constexpr (SrcType == FPType::IEEE754_Binary32 &&
                       DstType == FPType::BFloat16) {
    uint32_t abs32 = src_bits & 0x7fff'ffffU;
    if constexpr (FlagsMask == 0) {
      if (LIBC_UNLIKELY(abs32 > 0x7f80'0000U))
        return static_cast<uint16_t>((src_bits >> 16U) | 0x0040U);
      return round_packed_ieee<uint16_t, uint32_t, StaticNearest>(
          src_bits >= 0x8000'0000U, static_cast<uint16_t>(src_bits >> 16U),
          src_bits << 16U, rounding);
    } else {
      if (LIBC_UNLIKELY(abs32 >= 0x7f80'0000U)) {
        if (abs32 == 0x7f80'0000U)
          return static_cast<uint16_t>(src_bits >> 16U);
        if ((abs32 & 0x0040'0000U) == 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
        return static_cast<uint16_t>((src_bits >> 16U) | 0x0040U);
      }
      uint32_t rem = src_bits << 16U;
      uint16_t out = round_packed_ieee<uint16_t, uint32_t, StaticNearest>(
          src_bits >= 0x8000'0000U, static_cast<uint16_t>(src_bits >> 16U), rem,
          rounding);
      if (rem != 0) {
        if (LIBC_UNLIKELY((out & 0x7f80U) == 0x7f80U))
          LIBC_FP_SET_FLAGS(FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT);
        else if (LIBC_UNLIKELY(abs32 < 0x0080'0000U))
          LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
        else
          LIBC_FP_SET_FLAGS(FP_EXCEPT_INEXACT);
      }
      return out;
    }
  } else if constexpr (SrcType != FPType::X86_Binary80 &&
                       DstType != FPType::X86_Binary80 &&
                       DstRep::FRACTION_LEN >= SrcRep::FRACTION_LEN &&
                       DstRep::EXP_BIAS > SrcRep::EXP_BIAS) {
    // Fast packed-bit IEEE-754 exact widening (f16 -> f32/f64/f128,
    // bf16 -> f64/f128, f32 -> f64/f128, f64 -> f128).
    // Adjacent exponent bit transitions (x ^ (x << 1)) are all zero iff the
    // biased exponent is 0 or all-ones.
    if constexpr (SrcType == FPType::IEEE754_Binary64 &&
                  DstType == FPType::IEEE754_Binary128) {
      uint32_t lo32 = static_cast<uint32_t>(src_bits);
      uint32_t hi32 = static_cast<uint32_t>(src_bits >> 32U);
      uint32_t abs_hi32 = hi32 & 0x7fff'ffffU;
      uint64_t dst_sign_hi = static_cast<uint64_t>(hi32 & 0x8000'0000U) << 32U;

      if (LIBC_LIKELY(((hi32 ^ (hi32 << 1U)) & 0x7fe0'0000U) != 0)) {
        uint64_t abs64 = (static_cast<uint64_t>(abs_hi32) << 32U) | lo32;
        uint64_t hi64 =
            dst_sign_hi | ((abs64 >> 4U) + (uint64_t(16383 - 1023) << 48U));
        uint64_t lo64 = static_cast<uint64_t>(lo32 << 28U) << 32U;
        return make_u128(hi64, lo64);
      }

      if ((hi32 & 0x0010'0000U) != 0) {
        uint32_t frac_hi = abs_hi32 & 0x000f'ffffU;
        if ((frac_hi | lo32) == 0)
          return make_u128(dst_sign_hi | 0x7fff'0000'0000'0000ULL, 0);
        if ((frac_hi & 0x0008'0000U) == 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
        uint64_t frac64 = (static_cast<uint64_t>(frac_hi) << 32U) | lo32;
        return make_u128(dst_sign_hi | 0x7fff'8000'0000'0000ULL |
                             (frac64 >> 4U),
                         static_cast<uint64_t>(lo32 << 28U) << 32U);
      }

      if ((abs_hi32 | lo32) == 0)
        return make_u128(dst_sign_hi, 0);

      uint64_t raw_sig = (static_cast<uint64_t>(abs_hi32) << 32U) | lo32;
      int sub_shift = cpp::countl_zero(raw_sig) - 11;
      uint64_t norm = raw_sig << static_cast<size_t>(sub_shift);
      uint64_t bias_adj = static_cast<uint64_t>(16383 - 1023 - sub_shift)
                          << 48U;
      return make_u128(dst_sign_hi | ((norm >> 4U) + bias_adj), norm << 60U);
    } else {
      // SrcStorage is uint16_t or uint32_t: operate in a single 32-bit word.
      uint32_t u = static_cast<uint32_t>(src_bits);
      constexpr uint32_t SIGN_BIT = uint32_t(1) << (SrcRep::STORAGE_LEN - 1);
      constexpr uint32_t ABS_MASK = SIGN_BIT - 1U;
      constexpr uint32_t MIN_NORM = uint32_t(1) << SrcRep::FRACTION_LEN;
      constexpr uint32_t EXP_TRANS_MASK =
          ((uint32_t(1) << (SrcRep::EXP_LEN - 1)) - 1U)
          << (SrcRep::FRACTION_LEN + 1);

      uint32_t abs_u = u & ABS_MASK;
      uint32_t sign_u = u & SIGN_BIT;

      auto pack_wide = [&](uint32_t norm_u,
                           uint32_t bias_diff) constexpr -> DstStorage {
        if constexpr (DstType == FPType::IEEE754_Binary128) {
          constexpr int HI_SHIFT = 48 - SrcRep::FRACTION_LEN;
          uint64_t dst_sign_hi = static_cast<uint64_t>(sign_u)
                                 << (64 - SrcRep::STORAGE_LEN);
          uint64_t hi64 =
              dst_sign_hi | ((static_cast<uint64_t>(norm_u) << HI_SHIFT) +
                             (static_cast<uint64_t>(bias_diff) << 48U));
          return make_u128(hi64, 0);
        } else {
          constexpr int FRAC_DIFF = DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN;
          DstStorage dst_sign = static_cast<DstStorage>(sign_u)
                                << (DstRep::STORAGE_LEN - SrcRep::STORAGE_LEN);
          return static_cast<DstStorage>(
              dst_sign |
              ((static_cast<DstStorage>(norm_u) << FRAC_DIFF) +
               (static_cast<DstStorage>(bias_diff) << DstRep::FRACTION_LEN)));
        }
      };

      if (LIBC_LIKELY(((u ^ (u << 1U)) & EXP_TRANS_MASK) != 0)) {
        return pack_wide(
            abs_u, static_cast<uint32_t>(DstRep::EXP_BIAS - SrcRep::EXP_BIAS));
      }

      if ((u & MIN_NORM) != 0) {
        uint32_t frac_u = abs_u & (MIN_NORM - 1U);
        if constexpr (DstType == FPType::IEEE754_Binary128) {
          uint64_t dst_sign_hi = static_cast<uint64_t>(sign_u)
                                 << (64 - SrcRep::STORAGE_LEN);
          if (frac_u == 0)
            return make_u128(dst_sign_hi | 0x7fff'0000'0000'0000ULL, 0);
          if ((frac_u & (MIN_NORM >> 1U)) == 0)
            LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
          constexpr int HI_SHIFT = 48 - SrcRep::FRACTION_LEN;
          return make_u128(dst_sign_hi | 0x7fff'8000'0000'0000ULL |
                               (static_cast<uint64_t>(frac_u) << HI_SHIFT),
                           0);
        } else {
          constexpr int FRAC_DIFF = DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN;
          DstStorage dst_sign = static_cast<DstStorage>(sign_u)
                                << (DstRep::STORAGE_LEN - SrcRep::STORAGE_LEN);
          if (frac_u == 0)
            return static_cast<DstStorage>(dst_sign | DstRep::EXP_MASK);
          if ((frac_u & (MIN_NORM >> 1U)) == 0)
            LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
          constexpr DstStorage QNAN_BITS =
              DstRep::EXP_MASK | (DstStorage(1) << (DstRep::FRACTION_LEN - 1));
          return static_cast<DstStorage>(
              dst_sign | QNAN_BITS |
              (static_cast<DstStorage>(frac_u) << FRAC_DIFF));
        }
      }

      if (abs_u == 0) {
        if constexpr (DstType == FPType::IEEE754_Binary128) {
          return make_u128(
              static_cast<uint64_t>(sign_u) << (64 - SrcRep::STORAGE_LEN), 0);
        } else {
          return static_cast<DstStorage>(
              static_cast<DstStorage>(sign_u)
              << (DstRep::STORAGE_LEN - SrcRep::STORAGE_LEN));
        }
      }

      int sub_shift = cpp::countl_zero(abs_u) - (32 - 1 - SrcRep::FRACTION_LEN);
      uint32_t norm_u = abs_u << static_cast<size_t>(sub_shift);
      return pack_wide(norm_u,
                       static_cast<uint32_t>(DstRep::EXP_BIAS -
                                             SrcRep::EXP_BIAS - sub_shift));
    }
  } else if constexpr (SrcType != FPType::X86_Binary80 &&
                       DstType != FPType::X86_Binary80 &&
                       SrcRep::FRACTION_LEN > DstRep::FRACTION_LEN &&
                       SrcRep::EXP_BIAS > DstRep::EXP_BIAS) {
    // Fast packed-bit IEEE-754 strict narrowing (f32 -> f16,
    // f64 -> f32/f16/bf16, f128 -> f64/f32/f16/bf16).
    // For any normal destination exponent e = E - (SRC_BIAS - DST_BIAS), the
    // top two bits {sign, dst_exp_msb} match the top two bits of the source
    // word, allowing BFI/mask insertion without an exponent subtraction.
    if constexpr (SrcType == FPType::IEEE754_Binary32) {
      // f32 -> f16
      uint32_t src_exp = (src_bits >> 23U) & 0xffU;
      bool is_neg = (src_bits >= 0x8000'0000U);

      constexpr uint32_t MIN_NORM_EXP = uint32_t(127 - DstRep::EXP_BIAS + 1);
      constexpr uint32_t NORM_EXP_COUNT =
          uint32_t(DstRep::MAX_BIASED_EXPONENT - 1);
      constexpr int RSHIFT = 23 - DstRep::FRACTION_LEN;

      if (LIBC_LIKELY(src_exp - MIN_NORM_EXP < NORM_EXP_COUNT)) {
        DstStorage out = static_cast<DstStorage>(
            ((src_bits >> 16U) & 0xc000U) | ((src_bits >> RSHIFT) & 0x3fffU));
        uint32_t rem = src_bits << (32 - RSHIFT);
        out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
            is_neg, out, rem, rounding);
        record_normal_narrow_flags<DstType, FlagsMask>(out, rem != 0,
                                                       out_flags);
        return out;
      }

      DstStorage dst_sign = static_cast<DstStorage>((src_bits & 0x8000'0000U) >>
                                                    (32 - DstRep::STORAGE_LEN));
      constexpr uint32_t MIN_SUB_EXP =
          uint32_t(127 - DstRep::EXP_BIAS - DstRep::FRACTION_LEN);
      if constexpr (StaticNearest && FlagsMask == 0) {
        if (src_exp < MIN_SUB_EXP)
          return dst_sign;
      } else {
        if (src_exp == 0 && (src_bits << 9U) == 0)
          return dst_sign;
      }

      if (src_exp == 0xffU) {
        uint32_t frac_shifted = src_bits << 9U;
        if (frac_shifted == 0)
          return static_cast<DstStorage>(dst_sign | DstRep::EXP_MASK);
        if ((frac_shifted & 0x8000'0000U) == 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
        constexpr DstStorage QNAN_BITS =
            DstRep::EXP_MASK | (DstStorage(1) << (DstRep::FRACTION_LEN - 1));
        return static_cast<DstStorage>(
            dst_sign | QNAN_BITS |
            static_cast<DstStorage>(frac_shifted >> (9 + RSHIFT)));
      }

      constexpr uint32_t OVF_EXP =
          uint32_t(127 - DstRep::EXP_BIAS + DstRep::MAX_BIASED_EXPONENT);
      if (src_exp >= OVF_EXP)
        return LIBC_FP_OVF_RESULT(is_neg ? Sign::NEG : Sign::POS);

      uint32_t rem = 1U;
      DstStorage out = dst_sign;
      if (src_exp >= MIN_SUB_EXP) {
        uint32_t sig32 = 0x8000'0000U | (src_bits << 8U);
        uint32_t rshift =
            uint32_t(127 - DstRep::EXP_BIAS + 32 - DstRep::FRACTION_LEN) -
            src_exp;
        out = static_cast<DstStorage>(dst_sign |
                                      ((sig32 >> 1U) >> (rshift - 1U)));
        rem = sig32 << (32U - rshift);
      }
      out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
          is_neg, out, rem, rounding);
      if (rem != 0)
        LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
      return out;
    } else if constexpr (SrcType == FPType::IEEE754_Binary64) {
      // f64 -> f32, f16, bf16 (operate on 32-bit halves).
      uint32_t lo32 = static_cast<uint32_t>(src_bits);
      uint32_t hi32 = static_cast<uint32_t>(src_bits >> 32U);
      uint32_t src_exp = (hi32 >> 20U) & 0x7ffU;
      bool is_neg = (hi32 >= 0x8000'0000U);

      constexpr uint32_t MIN_NORM_EXP = uint32_t(1023 - DstRep::EXP_BIAS + 1);
      constexpr uint32_t NORM_EXP_COUNT =
          uint32_t(DstRep::MAX_BIASED_EXPONENT - 1);

      if (LIBC_LIKELY(src_exp - MIN_NORM_EXP < NORM_EXP_COUNT)) {
        if constexpr (DstRep::FRACTION_LEN == 23 && StaticNearest) {
          // Fused 64-bit exponent rebias + round-to-nearest-even (adds + adc on
          // ARM32/Thumb-2, single adcq on x86-64); decoupled shift/add on
          // carryless RISC-V.
          uint32_t out = 0;
#if defined(LIBC_TARGET_ARCH_IS_RISCV64)
          uint32_t u = static_cast<uint32_t>(src_bits >> 29U);
          out = (u + 0x4000'0000U) | (hi32 & 0x8000'0000U);
          uint32_t rem_lsb = (lo32 << 3U) | (u & 1U);
          out += (rem_lsb > 0x8000'0000U);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
          uint32_t mant_exp = ((hi32 << 3U) | (lo32 >> 29U)) & 0x3fff'ffffU;
          out = (hi32 & 0xc000'0000U) | mant_exp;
          uint32_t rem_lsb = (lo32 << 3U) | ((lo32 >> 29U) & 1U);
          out += (rem_lsb > 0x8000'0000U);
#else
          uint32_t bias_lo = 0x0fff'ffffU + ((lo32 >> 29U) & 1U);
          uint64_t bias = (uint64_t(0x0800'0000U) << 32U) | bias_lo;
          uint64_t r64 = src_bits + bias;
          out = static_cast<uint32_t>(r64 >> 29U) | (hi32 & 0x8000'0000U);
#endif
          if constexpr (FlagsMask != 0) {
            record_normal_narrow_flags<DstType, FlagsMask>(
                out, (lo32 << 3U) != 0, out_flags);
          }
          return out;
        } else {
          DstStorage out{};
          uint32_t rem = 0;
          if constexpr (DstRep::FRACTION_LEN == 23) {
            uint32_t mant_exp = ((hi32 << 3U) | (lo32 >> 29U)) & 0x3fff'ffffU;
            out = (hi32 & 0xc000'0000U) | mant_exp;
            rem = lo32 << 3U;
          } else {
            constexpr int HI_RSHIFT = 20 - DstRep::FRACTION_LEN;
            out = static_cast<DstStorage>(((hi32 >> 16U) & 0xc000U) |
                                          ((hi32 >> HI_RSHIFT) & 0x3fffU));
            rem = (hi32 << (32 - HI_RSHIFT)) | (lo32 != 0 ? 1U : 0U);
          }
          out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
              is_neg, out, rem, rounding);
          record_normal_narrow_flags<DstType, FlagsMask>(out, rem != 0,
                                                         out_flags);
          return out;
        }
      }

      constexpr int SIGN_RSHIFT = 32 - DstRep::STORAGE_LEN;
      constexpr uint32_t MIN_SUB_EXP =
          uint32_t(1023 - DstRep::EXP_BIAS - DstRep::FRACTION_LEN);
      constexpr uint32_t OVF_EXP =
          uint32_t(1023 - DstRep::EXP_BIAS + DstRep::MAX_BIASED_EXPONENT);
      constexpr DstStorage QNAN_BITS =
          DstRep::EXP_MASK | (DstStorage(1) << (DstRep::FRACTION_LEN - 1));

      if constexpr (StaticNearest && FlagsMask == 0) {
        if (src_exp < MIN_SUB_EXP)
          return static_cast<DstStorage>((hi32 & 0x8000'0000U) >> SIGN_RSHIFT);

        if (src_exp >= OVF_EXP) {
          DstStorage s_val =
              static_cast<DstStorage>((hi32 & 0x8000'0000U) >> SIGN_RSHIFT);
          uint32_t frac_hi_shifted = hi32 << 12U;
          if (src_exp == 0x7ffU && ((frac_hi_shifted | lo32) != 0)) {
            if constexpr (DstRep::FRACTION_LEN == 23) {
              return s_val | QNAN_BITS | (frac_hi_shifted >> 9U) |
                     (lo32 >> 29U);
            } else {
              return static_cast<DstStorage>(
                  s_val | QNAN_BITS |
                  static_cast<DstStorage>(frac_hi_shifted >>
                                          (32 - DstRep::FRACTION_LEN)));
            }
          }
          return static_cast<DstStorage>(s_val | DstRep::EXP_MASK);
        }

        uint32_t sig32 = 0x8000'0000U | (hi32 << 11U) | (lo32 >> 21U) |
                         ((lo32 << 11U) != 0 ? 1U : 0U);
        uint32_t rshift =
            uint32_t(1023 - DstRep::EXP_BIAS + 32 - DstRep::FRACTION_LEN) -
            src_exp;
        DstStorage s_val =
            static_cast<DstStorage>((hi32 & 0x8000'0000U) >> SIGN_RSHIFT);
        DstStorage out =
            static_cast<DstStorage>(s_val | ((sig32 >> 1U) >> (rshift - 1U)));
        uint32_t rem = sig32 << (32U - rshift);
        return round_packed_ieee<DstStorage, uint32_t, true>(is_neg, out, rem,
                                                             rounding);
      } else {
        DstStorage s_val =
            static_cast<DstStorage>((hi32 & 0x8000'0000U) >> SIGN_RSHIFT);
        if (src_exp == 0 && ((hi32 << 12U) | lo32) == 0)
          return s_val;

        if (src_exp == 0x7ffU) {
          uint32_t frac_hi_shifted = hi32 << 12U;
          if ((frac_hi_shifted | lo32) == 0)
            return static_cast<DstStorage>(s_val | DstRep::EXP_MASK);
          if ((frac_hi_shifted & 0x8000'0000U) == 0)
            LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
          if constexpr (DstRep::FRACTION_LEN == 23) {
            return s_val | QNAN_BITS | (frac_hi_shifted >> 9U) | (lo32 >> 29U);
          } else {
            return static_cast<DstStorage>(
                s_val | QNAN_BITS |
                static_cast<DstStorage>(frac_hi_shifted >>
                                        (32 - DstRep::FRACTION_LEN)));
          }
        }

        if (src_exp >= OVF_EXP)
          return LIBC_FP_OVF_RESULT(is_neg ? Sign::NEG : Sign::POS);

        uint32_t rem = 1U;
        DstStorage out = s_val;
        if (src_exp >= MIN_SUB_EXP) {
          uint32_t sig32 = 0x8000'0000U | (hi32 << 11U) | (lo32 >> 21U) |
                           ((lo32 << 11U) != 0 ? 1U : 0U);
          uint32_t rshift =
              uint32_t(1023 - DstRep::EXP_BIAS + 32 - DstRep::FRACTION_LEN) -
              src_exp;
          out =
              static_cast<DstStorage>(s_val | ((sig32 >> 1U) >> (rshift - 1U)));
          rem = sig32 << (32U - rshift);
        }
        out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
            is_neg, out, rem, rounding);
        if (rem != 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
        return out;
      }
    } else {
      // SrcType == IEEE754_Binary128 -> f64, f32, f16, bf16 (zero UInt128
      // shifts on 32-bit and 64-bit targets).
      uint64_t hi64 = u128_hi(src_bits);
      uint64_t lo64 = u128_lo(src_bits);
      uint32_t top32 = static_cast<uint32_t>(hi64 >> 32U);
      uint32_t mid32 = static_cast<uint32_t>(hi64);
      uint32_t src_exp = (top32 >> 16U) & 0x7fffU;
      bool is_neg = (top32 >= 0x8000'0000U);

      constexpr uint32_t MIN_NORM_EXP = uint32_t(16383 - DstRep::EXP_BIAS + 1);
      constexpr uint32_t NORM_EXP_COUNT =
          uint32_t(DstRep::MAX_BIASED_EXPONENT - 1);

      if (LIBC_LIKELY(src_exp - MIN_NORM_EXP < NORM_EXP_COUNT)) {
        if constexpr (DstType == FPType::IEEE754_Binary64) {
          if constexpr (sizeof(size_t) >= 8) {
            uint64_t out =
                (hi64 & 0xc000'0000'0000'0000ULL) |
                (((hi64 << 4U) | (lo64 >> 60U)) & 0x3fff'ffff'ffff'ffffULL);
            uint64_t rem64 = lo64 << 4U;
            out = round_packed_ieee<uint64_t, uint64_t, StaticNearest>(
                is_neg, out, rem64, rounding);
            record_normal_narrow_flags<DstType, FlagsMask>(out, rem64 != 0,
                                                           out_flags);
            return out;
          } else {
            uint32_t lo_hi32 = static_cast<uint32_t>(lo64 >> 32U);
            uint32_t lo_lo32 = static_cast<uint32_t>(lo64);
            uint32_t out_hi32 =
                (top32 & 0xc000'0000U) |
                (((top32 << 4U) | (mid32 >> 28U)) & 0x3fff'ffffU);
            uint32_t out_lo32 = (mid32 << 4U) | (lo_hi32 >> 28U);
            uint64_t out = (static_cast<uint64_t>(out_hi32) << 32U) | out_lo32;
            uint32_t rem = (lo_hi32 << 4U) | (lo_lo32 != 0 ? 1U : 0U);
            out = round_packed_ieee<uint64_t, uint32_t, StaticNearest>(
                is_neg, out, rem, rounding);
            record_normal_narrow_flags<DstType, FlagsMask>(out, rem != 0,
                                                           out_flags);
            return out;
          }
        } else {
          DstStorage out{};
          uint32_t rem = 0;
          if constexpr (DstType == FPType::IEEE754_Binary32) {
            out = (top32 & 0xc000'0000U) |
                  (((top32 << 7U) | (mid32 >> 25U)) & 0x3fff'ffffU);
            rem = (mid32 << 7U) | (lo64 != 0 ? 1U : 0U);
          } else {
            constexpr int TOP_RSHIFT = 16 - DstRep::FRACTION_LEN;
            out = static_cast<DstStorage>(((top32 >> 16U) & 0xc000U) |
                                          ((top32 >> TOP_RSHIFT) & 0x3fffU));
            rem = (top32 << (32 - TOP_RSHIFT)) |
                  (((mid32 != 0) || (lo64 != 0)) ? 1U : 0U);
          }
          out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
              is_neg, out, rem, rounding);
          record_normal_narrow_flags<DstType, FlagsMask>(out, rem != 0,
                                                         out_flags);
          return out;
        }
      }

      DstStorage dst_sign = 0;
      if constexpr (DstRep::STORAGE_LEN == 64) {
        dst_sign = static_cast<DstStorage>(
            static_cast<uint64_t>(top32 & 0x8000'0000U) << 32U);
      } else {
        dst_sign = static_cast<DstStorage>((top32 & 0x8000'0000U) >>
                                           (32 - DstRep::STORAGE_LEN));
      }

      constexpr uint32_t MIN_SUB_EXP =
          uint32_t(16383 - DstRep::EXP_BIAS - DstRep::FRACTION_LEN);
      if constexpr (StaticNearest && FlagsMask == 0) {
        if (src_exp < MIN_SUB_EXP)
          return dst_sign;
      } else {
        if (src_exp == 0 && ((top32 << 16U) | mid32) == 0 && lo64 == 0)
          return dst_sign;
      }

      if (src_exp == 0x7fffU) {
        uint32_t frac_top_shifted = top32 << 16U;
        if ((frac_top_shifted | mid32) == 0 && lo64 == 0)
          return static_cast<DstStorage>(dst_sign | DstRep::EXP_MASK);
        if ((frac_top_shifted & 0x8000'0000U) == 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
        constexpr DstStorage QNAN_BITS =
            DstRep::EXP_MASK | (DstStorage(1) << (DstRep::FRACTION_LEN - 1));
        if constexpr (DstType == FPType::IEEE754_Binary64) {
          uint64_t payload =
              (static_cast<uint64_t>(frac_top_shifted >> 16U) << 36U) |
              (static_cast<uint64_t>(mid32) << 4U) | (lo64 >> 60U);
          return dst_sign | QNAN_BITS | payload;
        } else if constexpr (DstType == FPType::IEEE754_Binary32) {
          return dst_sign | QNAN_BITS | (frac_top_shifted >> 9U) |
                 (mid32 >> 25U);
        } else {
          return static_cast<DstStorage>(
              dst_sign | QNAN_BITS |
              static_cast<DstStorage>(frac_top_shifted >>
                                      (32 - DstRep::FRACTION_LEN)));
        }
      }

      constexpr uint32_t OVF_EXP =
          uint32_t(16383 - DstRep::EXP_BIAS + DstRep::MAX_BIASED_EXPONENT);
      if (src_exp >= OVF_EXP)
        return LIBC_FP_OVF_RESULT(is_neg ? Sign::NEG : Sign::POS);

      if constexpr (DstType == FPType::IEEE754_Binary64) {
        uint64_t rem64 = 1ULL;
        uint64_t out = dst_sign;
        if (src_exp >= MIN_SUB_EXP) {
          uint64_t sig64 = (uint64_t(1) << 63U) |
                           (static_cast<uint64_t>(top32 & 0xffffU) << 47U) |
                           (static_cast<uint64_t>(mid32) << 15U) |
                           ((lo64 >> 49U) & ~uint64_t(1)) |
                           ((lo64 << 14U) != 0 ? 1ULL : 0ULL);
          uint32_t rshift = uint32_t(16383 - 1023 + 64 - 52) - src_exp;
          out = dst_sign | ((sig64 >> 1U) >> (rshift - 1U));
          rem64 = sig64 << (64U - rshift);
        }
        out = round_packed_ieee<uint64_t, uint64_t, StaticNearest>(
            is_neg, out, rem64, rounding);
        if (rem64 != 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
        return out;
      } else {
        uint32_t rem = 1U;
        DstStorage out = dst_sign;
        if (src_exp >= MIN_SUB_EXP) {
          uint32_t sig32 = 0x8000'0000U | (top32 << 15U) |
                           ((mid32 >> 17U) & ~1U) |
                           ((((mid32 << 14U) != 0) || (lo64 != 0)) ? 1U : 0U);
          uint32_t rshift =
              uint32_t(16383 - DstRep::EXP_BIAS + 32 - DstRep::FRACTION_LEN) -
              src_exp;
          out = static_cast<DstStorage>(dst_sign |
                                        ((sig32 >> 1U) >> (rshift - 1U)));
          rem = sig32 << (32U - rshift);
        }
        out = round_packed_ieee<DstStorage, uint32_t, StaticNearest>(
            is_neg, out, rem, rounding);
        if (rem != 0)
          LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
        return out;
      }
    }
  } else if constexpr (DstRep::FRACTION_LEN >= SrcRep::FRACTION_LEN &&
                       DstRep::EXP_BIAS >= SrcRep::EXP_BIAS) {
    // Exact widening involving X86_Binary80.
    SrcRep x_bits(src_bits);
    Sign sign = x_bits.sign();

    if (LIBC_UNLIKELY(!x_bits.is_finite())) {
      if (x_bits.is_inf())
        return DstRep::inf(sign).uintval();
      if (x_bits.is_signaling_nan())
        LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
      DstStorage payload = static_cast<DstStorage>(
          static_cast<DstStorage>(x_bits.get_mantissa())
          << (DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN));
      return DstRep::quiet_nan(sign, payload).uintval();
    }

    constexpr int FRAC_DIFF = DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN;
    if constexpr (DstRep::EXP_BIAS == SrcRep::EXP_BIAS) {
      DstStorage out_exp =
          static_cast<DstStorage>(x_bits.get_biased_exponent());
      DstStorage out_mant = static_cast<DstStorage>(
          static_cast<DstStorage>(x_bits.get_explicit_mantissa()) << FRAC_DIFF);
      return DstRep::create_value(sign, out_exp, out_mant).uintval();
    } else {
      uint32_t src_exp = x_bits.get_biased_exponent();
      if (LIBC_LIKELY(src_exp != 0)) {
        DstStorage out_exp = static_cast<DstStorage>(
            src_exp +
            static_cast<uint32_t>(DstRep::EXP_BIAS - SrcRep::EXP_BIAS));
        DstStorage out_mant = static_cast<DstStorage>(
            static_cast<DstStorage>(x_bits.get_explicit_mantissa())
            << FRAC_DIFF);
        return DstRep::create_value(sign, out_exp, out_mant).uintval();
      }

      SrcStorage raw_sig = x_bits.get_mantissa();
      if (raw_sig == 0)
        return DstRep::zero(sign).uintval();

      int lz = cpp::countl_zero(raw_sig);
      int sub_shift = lz - (SrcRep::STORAGE_LEN - 1 - SrcRep::FRACTION_LEN);
      SrcStorage norm_sig =
          static_cast<SrcStorage>(raw_sig << static_cast<size_t>(sub_shift));
      DstStorage out_exp = static_cast<DstStorage>(
          1 - SrcRep::EXP_BIAS - sub_shift + DstRep::EXP_BIAS);
      DstStorage out_mant = static_cast<DstStorage>(
          static_cast<DstStorage>(norm_sig) << FRAC_DIFF);
      return DstRep::create_value(sign, out_exp, out_mant).uintval();
    }
  } else {
    // Cross-range (Binary16 <-> BFloat16) or narrowing involving X86_Binary80.
    auto round_up = [&](Sign sign, bool lsb, bool round_bit,
                        bool sticky_bit) constexpr -> bool {
      if constexpr (StaticNearest)
        return round_bit && (lsb || sticky_bit);
      else
        return should_round_up(sign, lsb, round_bit, sticky_bit, rounding);
    };

    SrcRep x_bits(src_bits);
    Sign sign = x_bits.sign();

    if (LIBC_UNLIKELY(!x_bits.is_finite())) {
      if (x_bits.is_inf())
        return DstRep::inf(sign).uintval();
      if (x_bits.is_signaling_nan())
        LIBC_FP_SET_FLAGS(FP_EXCEPT_INVALID);
      DstStorage payload = 0;
      if constexpr (SrcRep::FRACTION_LEN >= DstRep::FRACTION_LEN) {
        payload = static_cast<DstStorage>(
            x_bits.get_mantissa() >>
            (SrcRep::FRACTION_LEN - DstRep::FRACTION_LEN));
      } else {
        payload = static_cast<DstStorage>(
            static_cast<DstStorage>(x_bits.get_mantissa())
            << (DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN));
      }
      return DstRep::quiet_nan(sign, payload).uintval();
    }

    uint32_t src_exp = x_bits.get_biased_exponent();

    if constexpr (DstRep::EXP_BIAS == SrcRep::EXP_BIAS) {
      SrcStorage mant = x_bits.get_explicit_mantissa();
      if (src_exp == 0 && mant == 0)
        return DstRep::zero(sign).uintval();

      constexpr int RSHIFT = SrcRep::FRACTION_LEN - DstRep::FRACTION_LEN;
      DstStorage out_exp = static_cast<DstStorage>(src_exp);
      DstStorage out_mant = static_cast<DstStorage>(mant >> RSHIFT);
      bool round_bit = ((mant >> (RSHIFT - 1)) & SrcStorage(1)) != 0;
      constexpr SrcStorage STICKY_MASK = static_cast<SrcStorage>(
          (SrcStorage(1) << (RSHIFT - 1)) - SrcStorage(1));
      bool sticky_bit = (mant & STICKY_MASK) != 0;
      bool lsb = (out_mant & DstStorage(1)) != 0;

      DstStorage result =
          DstRep::create_value(sign, out_exp, out_mant).uintval();
      if (round_up(sign, lsb, round_bit, sticky_bit))
        result = apply_rounding_increment<DstType>(result);

      if constexpr (FlagsMask != 0) {
        if (round_bit || sticky_bit) {
          if (LIBC_UNLIKELY(DstRep(result).is_inf()))
            LIBC_FP_SET_FLAGS(FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT);
          else if (LIBC_UNLIKELY(src_exp == 0))
            LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
          else
            LIBC_FP_SET_FLAGS(FP_EXCEPT_INEXACT);
        }
      }
      return result;
    } else {
      SrcStorage mant{};
      int unbiased_exp = 0;

      if (LIBC_LIKELY(src_exp != 0)) {
        unbiased_exp = static_cast<int>(src_exp) - SrcRep::EXP_BIAS;
        mant = x_bits.get_explicit_mantissa();
      } else {
        SrcStorage raw_sig = x_bits.get_mantissa();
        if (raw_sig == 0)
          return DstRep::zero(sign).uintval();

        if constexpr (DstRep::EXP_BIAS < SrcRep::EXP_BIAS) {
          LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
          return round_up(sign, false, false, true)
                     ? DstRep::min_subnormal(sign).uintval()
                     : DstRep::zero(sign).uintval();
        } else {
          int lz = cpp::countl_zero(raw_sig);
          int sub_shift = lz - (SrcRep::STORAGE_LEN - 1 - SrcRep::FRACTION_LEN);
          unbiased_exp = 1 - SrcRep::EXP_BIAS - sub_shift;
          mant = static_cast<SrcStorage>(raw_sig
                                         << static_cast<size_t>(sub_shift));
        }
      }

      int dst_biased_exp = unbiased_exp + DstRep::EXP_BIAS;
      if (LIBC_UNLIKELY(dst_biased_exp >= DstRep::MAX_BIASED_EXPONENT))
        return LIBC_FP_OVF_RESULT(sign);

      if (LIBC_LIKELY(dst_biased_exp >= 1)) {
        DstStorage out_exp = static_cast<DstStorage>(dst_biased_exp);
        if constexpr (SrcRep::FRACTION_LEN >= DstRep::FRACTION_LEN) {
          constexpr int RSHIFT = SrcRep::FRACTION_LEN - DstRep::FRACTION_LEN;
          DstStorage out_mant = static_cast<DstStorage>(mant >> RSHIFT);
          bool round_bit = false;
          bool sticky_bit = false;
          if constexpr (RSHIFT > 0) {
            round_bit = ((mant >> (RSHIFT - 1)) & SrcStorage(1)) != 0;
            constexpr SrcStorage STICKY_MASK = static_cast<SrcStorage>(
                (SrcStorage(1) << (RSHIFT - 1)) - SrcStorage(1));
            sticky_bit = (mant & STICKY_MASK) != 0;
          }
          bool lsb = (out_mant & DstStorage(1)) != 0;

          DstStorage result =
              DstRep::create_value(sign, out_exp, out_mant).uintval();
          if (round_up(sign, lsb, round_bit, sticky_bit))
            result = apply_rounding_increment<DstType>(result);

          if constexpr (FlagsMask != 0) {
            if (round_bit || sticky_bit) {
              LIBC_FP_SET_FLAGS(LIBC_UNLIKELY(DstRep(result).is_inf())
                                    ? (FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT)
                                    : FP_EXCEPT_INEXACT);
            }
          }
          return result;
        } else {
          DstStorage out_mant = static_cast<DstStorage>(
              static_cast<DstStorage>(mant)
              << (DstRep::FRACTION_LEN - SrcRep::FRACTION_LEN));
          return DstRep::create_value(sign, out_exp, out_mant).uintval();
        }
      }

      int rshift =
          (SrcRep::FRACTION_LEN - DstRep::FRACTION_LEN) + (1 - dst_biased_exp);
      if constexpr (SrcRep::FRACTION_LEN < DstRep::FRACTION_LEN) {
        if (rshift <= 0) {
          DstStorage out_mant = static_cast<DstStorage>(
              static_cast<DstStorage>(mant) << static_cast<size_t>(-rshift));
          return DstRep::create_value(sign, DstStorage(0), out_mant).uintval();
        }
      }

      DstStorage out_mant = 0;
      bool round_bit = false;
      bool sticky_bit = false;

      if (rshift > SrcRep::FRACTION_LEN + 1) {
        sticky_bit = true;
      } else {
        size_t shift_sz = static_cast<size_t>(rshift);
        if (rshift <= SrcRep::FRACTION_LEN)
          out_mant = static_cast<DstStorage>(mant >> shift_sz);
        round_bit = ((mant >> (shift_sz - 1)) & SrcStorage(1)) != 0;
        SrcStorage sticky_mask = static_cast<SrcStorage>(
            (SrcStorage(1) << (shift_sz - 1)) - SrcStorage(1));
        sticky_bit = (mant & sticky_mask) != 0;
      }

      bool lsb = (out_mant & DstStorage(1)) != 0;
      DstStorage result =
          DstRep::create_value(sign, DstStorage(0), out_mant).uintval();
      if (round_up(sign, lsb, round_bit, sticky_bit))
        result = apply_rounding_increment<DstType>(result);

      if (round_bit || sticky_bit)
        LIBC_FP_SET_FLAGS(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
      return result;
    }
  }
}

#undef LIBC_FP_OVF_RESULT
#undef LIBC_FP_SET_FLAGS

} // namespace internal

namespace static_rounding {

// Pure-integer IEEE-754 convertFormat kernel with an explicit rounding mode and
// returned exception status flags (no errno or FEnv side effects, and no native
// floating-point operations).
template <FPType DstType, FPType SrcType>
LIBC_INLINE constexpr internal::ConversionResult<
    typename FPRep<DstType>::StorageType>
cast_bits(typename FPRep<SrcType>::StorageType src_bits,
          int rounding = FE_TONEAREST) {
  uint8_t flags = internal::FP_EXCEPT_NONE;
  auto bits = internal::cast_bits_impl<DstType, SrcType,
                                       /*StaticNearest=*/false, 0xffU>(
      src_bits, rounding, &flags);
  return {bits, flags};
}

template <typename OutType, typename InType>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<
    cpp::is_floating_point_v<OutType> && cpp::is_floating_point_v<InType>,
    OutType>
cast(InType x, int rounding = FE_TONEAREST) {
  constexpr FPType SRC_TYPE = get_fp_type<InType>();
  constexpr FPType DST_TYPE = get_fp_type<OutType>();
  return internal::bits_to_val<OutType>(
      cast_bits<DST_TYPE, SRC_TYPE>(internal::raw_to_bits<InType>(x), rounding)
          .bits);
}

} // namespace static_rounding

namespace integer_eval {

template <FPType DstType, FPType SrcType>
LIBC_ALWAYS_INLINE constexpr typename FPRep<DstType>::StorageType
cast_bits(typename FPRep<SrcType>::StorageType src_bits) {
  return fputil::internal::cast_bits_impl<DstType, SrcType,
                                          /*StaticNearest=*/true, 0>(
      src_bits, FE_TONEAREST, nullptr);
}

template <typename OutType, typename InType>
LIBC_ALWAYS_INLINE constexpr cpp::enable_if_t<
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

} // namespace integer_eval

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FP_CONVERT_H
