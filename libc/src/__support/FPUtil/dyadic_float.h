//===-- A class to store high precision floating point numbers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fp_convert.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/macros/properties/types.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

// Decide whether to round a UInt up, down or not at all at a given bit
// position, based on the current rounding mode. The assumption is that the
// caller is going to make the integer `value >> rshift`, and then might need
// to round it up by 1 depending on the value of the bits shifted off the
// bottom.
//
// `logical_sign` causes the behavior of FE_DOWNWARD and FE_UPWARD to
// be reversed, which is what you'd want if this is the mantissa of a
// negative floating-point number.
//
// Return value is +1 if the value should be rounded up; -1 if it should be
// rounded down; 0 if it's exact and needs no rounding.
template <size_t Bits>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT int
rounding_direction(const LIBC_NAMESPACE::UInt<Bits> &value, size_t rshift,
                   Sign logical_sign) {
  if (rshift == 0 || (rshift < Bits && (value << (Bits - rshift)).is_zero()) ||
      (rshift >= Bits && value.is_zero()))
    return 0; // exact

  bool lsb = (rshift < Bits) ? value.get_bit(rshift) : false;
  bool round_bit = (rshift <= Bits) ? value.get_bit(rshift - 1) : false;
  bool sticky_bit =
      (rshift > Bits)
          ? true
          : ((rshift > 1) ? !(value << (Bits - rshift + 1)).is_zero() : false);

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  int rounding = FE_TONEAREST;
#else
  int rounding =
      cpp::is_constant_evaluated() ? FE_TONEAREST : quick_get_round();
#endif
  return internal::should_round_up(logical_sign, lsb, round_bit, sticky_bit,
                                   rounding)
             ? +1
             : -1;
}

// A generic class to perform computations of high precision floating points.
// We store the value in dyadic format, including 3 fields:
//   sign    : boolean value - false means positive, true means negative
//   exponent: the exponent value of the least significant bit of the mantissa.
//   mantissa: unsigned integer of length `Bits`.
// So the real value that is stored is:
//   real value = (-1)^sign * 2^exponent * (mantissa as unsigned integer)
// The stored data is normal if for non-zero mantissa, the leading bit is 1.
// The outputs of the constructors and most functions will be normalized.
// To simplify and improve the efficiency, many functions will assume that the
// inputs are normal.
template <size_t Bits> struct DyadicFloat {
  using MantissaType = LIBC_NAMESPACE::UInt<Bits>;

  Sign sign = Sign::POS;
  int exponent = 0;
  MantissaType mantissa = MantissaType(0);

  LIBC_INLINE constexpr DyadicFloat() = default;

  template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
  LIBC_INLINE LIBC_BIT_CAST_CONSTEXPR DyadicFloat(T x) {
    static_assert(FPBits<T>::FRACTION_LEN < Bits);
    using StorageType = typename FPBits<T>::StorageType;
    using WordType = typename MantissaType::word_type;

    FPBits<T> x_bits(x);
    sign = x_bits.sign();

    if constexpr (!is_big_int_v<StorageType> &&
                  sizeof(StorageType) <= sizeof(WordType)) {
      // Normalize in narrow StorageType/WordType first and write directly into
      // the top word of MantissaType, avoiding multi-word clz and shifts.
      constexpr size_t WORD_BITS = MantissaType::WORD_SIZE;
      uint32_t biased_exp = x_bits.get_biased_exponent();
      if (LIBC_LIKELY(biased_exp != 0)) {
        exponent = static_cast<int>(biased_exp) - FPBits<T>::EXP_BIAS -
                   static_cast<int>(Bits - 1);
        mantissa.val[MantissaType::WORD_COUNT - 1] = static_cast<WordType>(
            static_cast<WordType>(x_bits.get_explicit_mantissa())
            << (WORD_BITS - 1 - FPBits<T>::FRACTION_LEN));
      } else {
        StorageType raw_sig = x_bits.get_mantissa();
        if (raw_sig != 0) {
          int lz = cpp::countl_zero(raw_sig);
          constexpr int STORAGE_BITS = FPBits<T>::STORAGE_LEN;
          int sub_shift = lz - (STORAGE_BITS - 1 - FPBits<T>::FRACTION_LEN);
          exponent =
              1 - FPBits<T>::EXP_BIAS - sub_shift - static_cast<int>(Bits - 1);
          mantissa.val[MantissaType::WORD_COUNT - 1] = static_cast<WordType>(
              static_cast<WordType>(raw_sig)
              << (static_cast<int>(WORD_BITS) - STORAGE_BITS + lz));
        }
      }
    } else {
      exponent = x_bits.get_explicit_exponent() - FPBits<T>::FRACTION_LEN;
      mantissa = MantissaType(x_bits.get_explicit_mantissa());
      normalize();
    }
  }

  LIBC_INLINE constexpr DyadicFloat(Sign s, int e, const MantissaType &m)
      : sign(s), exponent(e), mantissa(m) {
    normalize();
  }

  // Normalizing the mantissa, bringing the leading 1 bit to the most
  // significant bit.
  LIBC_INLINE constexpr DyadicFloat &normalize() {
    if (!mantissa.is_zero()) {
      int shift_length = cpp::countl_zero(mantissa);
      exponent -= shift_length;
      mantissa <<= static_cast<size_t>(shift_length);
    }
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  LIBC_INLINE constexpr DyadicFloat &shift_left(unsigned shift_length) {
    exponent -= static_cast<int>(shift_length);
    if (shift_length < Bits)
      mantissa <<= shift_length;
    else
      mantissa = MantissaType(0);
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  LIBC_INLINE constexpr DyadicFloat &shift_right(unsigned shift_length) {
    exponent += static_cast<int>(shift_length);
    if (shift_length < Bits)
      mantissa >>= shift_length;
    else
      mantissa = MantissaType(0);
    return *this;
  }

  // Assume that it is already normalized.  Output the unbiased exponent.
  LIBC_INLINE constexpr int get_unbiased_exponent() const {
    return exponent + static_cast<int>(Bits - 1);
  }

  // Produce a correctly rounded DyadicFloat from a too-large mantissa,
  // by shifting it down and rounding if necessary.
  template <size_t MantissaBits>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT static DyadicFloat<Bits>
  round(Sign result_sign, int result_exponent,
        const LIBC_NAMESPACE::UInt<MantissaBits> &input_mantissa,
        size_t rshift) {
    MantissaType result_mantissa(input_mantissa >> rshift);
    if (rounding_direction(input_mantissa, rshift, result_sign) > 0) {
      ++result_mantissa;
      if (result_mantissa == 0) {
        // Rounding up made the mantissa integer wrap round to 0,
        // carrying a bit off the top. So we've rounded up to the next
        // exponent.
        result_mantissa.set_bit(Bits - 1);
        ++result_exponent;
      }
    }
    return DyadicFloat(result_sign, result_exponent, result_mantissa);
  }

  template <FPType DstType, bool ShouldSignalExceptions = true>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT typename FPRep<DstType>::StorageType
  generic_as_bits() const;

  template <typename T, bool ShouldSignalExceptions = true,
            cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                 (FPBits<T>::FRACTION_LEN < Bits),
                             int> = 0>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T generic_as() const;

  template <typename T, bool ShouldSignalExceptions = true,
            cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                 (FPBits<T>::FRACTION_LEN < Bits),
                             int> = 0>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T fast_as() const;

  // Assume that it is already normalized.
  // Output is rounded correctly with respect to the current rounding mode.
  template <typename T, bool ShouldSignalExceptions = true,
            cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                 (FPBits<T>::FRACTION_LEN < Bits),
                             int> = 0>
  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T as() const;

  template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                             (FPBits<T>::FRACTION_LEN < Bits),
                                         int> = 0>
  LIBC_INLINE explicit constexpr operator T() const;

  LIBC_INLINE constexpr MantissaType as_mantissa_type() const {
    if (mantissa.is_zero())
      return 0;

    MantissaType new_mant = mantissa;
    if (exponent > 0) {
      new_mant <<= exponent;
    } else {
      // Cast the exponent to size_t before negating it, rather than after,
      // to avoid undefined behavior negating INT_MIN as an integer (although
      // exponents coming in to this function _shouldn't_ be that large). The
      // result should always end up as a positive size_t.
      size_t shift = -static_cast<size_t>(exponent);
      size_t limit = cpp::numeric_limits<MantissaType>::digits;
      if (shift >= limit)
        new_mant = 0;
      else
        new_mant >>= shift;
    }

    if (sign.is_neg())
      new_mant = (~new_mant) + 1;

    return new_mant;
  }

  LIBC_INLINE LIBC_CONSTEXPR_DEFAULT MantissaType
  as_mantissa_type_rounded(int *round_dir_out = nullptr) const {
    int round_dir = 0;
    MantissaType new_mant(0);
    if (mantissa.is_zero()) {
      new_mant = 0;
    } else {
      new_mant = mantissa;
      if (exponent > 0) {
        new_mant <<= exponent;
      } else if (exponent < 0) {
        size_t shift = -static_cast<size_t>(exponent);
        if (shift >= Bits)
          new_mant = 0;
        else
          new_mant >>= shift;
        round_dir = rounding_direction(mantissa, shift, sign);
        if (round_dir > 0)
          ++new_mant;
      }

      if (sign.is_neg())
        new_mant = (~new_mant) + 1;
    }

    if (round_dir_out)
      *round_dir_out = round_dir;

    return new_mant;
  }

  LIBC_INLINE constexpr DyadicFloat operator-() const {
    return DyadicFloat(sign.negate(), exponent, mantissa);
  }
};

template <typename T> struct is_dyadic_float : cpp::false_type {};
template <size_t Bits>
struct is_dyadic_float<DyadicFloat<Bits>> : cpp::true_type {};
template <typename T>
LIBC_INLINE_VAR constexpr bool is_dyadic_float_v =
    is_dyadic_float<cpp::remove_cv_t<T>>::value;

namespace internal {

template <FPType DstType, bool StaticNearest, uint8_t FlagsMask, size_t Bits>
LIBC_INLINE constexpr typename FPRep<DstType>::StorageType
dyadic_as_bits_impl(const DyadicFloat<Bits> &xd, [[maybe_unused]] int rounding,
                    [[maybe_unused]] uint8_t *out_flags) {
  using DstRep = FPRep<DstType>;
  using StorageType = typename DstRep::StorageType;
  using MantissaType = typename DyadicFloat<Bits>::MantissaType;
  static_assert(DstRep::FRACTION_LEN < Bits);

  if constexpr (FlagsMask != 0)
    *out_flags = FP_EXCEPT_NONE;

  auto set_flags = [&](uint8_t f) constexpr {
    if constexpr (FlagsMask != 0)
      *out_flags = static_cast<uint8_t>(f & FlagsMask);
  };

  auto round_up = [&](bool lsb, bool round_bit,
                      bool sticky_bit) constexpr -> bool {
    if constexpr (StaticNearest)
      return round_bit && (lsb || sticky_bit);
    else
      return should_round_up(xd.sign, lsb, round_bit, sticky_bit, rounding);
  };

  if (LIBC_UNLIKELY(xd.mantissa.is_zero()))
    return DstRep::zero(xd.sign).uintval();

  int unbiased_exp = xd.get_unbiased_exponent();
  int dst_biased_exp = unbiased_exp + DstRep::EXP_BIAS;

  if (LIBC_UNLIKELY(dst_biased_exp >= DstRep::MAX_BIASED_EXPONENT)) {
    set_flags(FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT);
    bool to_inf =
        StaticNearest ? true : overflow_rounds_to_inf(xd.sign, rounding);
    return to_inf ? DstRep::inf(xd.sign).uintval()
                  : DstRep::max_normal(xd.sign).uintval();
  }

  constexpr int EXTRA_FRACTION_LEN =
      static_cast<int>(Bits - 1) - DstRep::FRACTION_LEN;

  // Normal destination path: shift and masks are compile-time constants,
  // avoiding runtime multi-word shifts on 32-bit targets.
  if (LIBC_LIKELY(dst_biased_exp >= 1)) {
    StorageType out_biased_exp = static_cast<StorageType>(dst_biased_exp);
    StorageType out_mantissa = static_cast<StorageType>(
        xd.mantissa >> static_cast<size_t>(EXTRA_FRACTION_LEN));
    bool round_bit = false;
    bool sticky_bit = false;
    if constexpr (EXTRA_FRACTION_LEN > 0) {
      constexpr MantissaType ROUND_MASK =
          MantissaType(1) << static_cast<size_t>(EXTRA_FRACTION_LEN - 1);
      constexpr MantissaType STICKY_MASK = ROUND_MASK - MantissaType(1);
      round_bit = !(xd.mantissa & ROUND_MASK).is_zero();
      sticky_bit = !(xd.mantissa & STICKY_MASK).is_zero();
    }

    bool lsb = (out_mantissa & StorageType(1)) != 0;
    StorageType result =
        DstRep::create_value(xd.sign, out_biased_exp, out_mantissa).uintval();
    if (round_up(lsb, round_bit, sticky_bit))
      result = apply_rounding_increment<DstType>(result);

    if constexpr (FlagsMask != 0) {
      if (round_bit || sticky_bit) {
        set_flags(LIBC_UNLIKELY(DstRep(result).is_inf())
                      ? (FP_EXCEPT_OVERFLOW | FP_EXCEPT_INEXACT)
                      : FP_EXCEPT_INEXACT);
      }
    }
    return result;
  }

  // Subnormal or extreme-underflow path (dst_biased_exp <= 0).
  StorageType out_mantissa = 0;
  bool round_bit = false;
  bool sticky_bit = false;

  if (unbiased_exp < -DstRep::EXP_BIAS - DstRep::FRACTION_LEN) {
    sticky_bit = true;
  } else if (unbiased_exp == -DstRep::EXP_BIAS - DstRep::FRACTION_LEN) {
    round_bit = true;
    constexpr MantissaType STICKY_MASK =
        (MantissaType(1) << (Bits - 1)) - MantissaType(1);
    sticky_bit = !(xd.mantissa & STICKY_MASK).is_zero();
  } else {
    size_t rshift =
        static_cast<size_t>(EXTRA_FRACTION_LEN + 1 - dst_biased_exp);
    constexpr size_t WORD_BITS = MantissaType::WORD_SIZE;
    using WordType = typename MantissaType::word_type;
    if constexpr (!is_big_int_v<StorageType> &&
                  sizeof(StorageType) <= sizeof(WordType)) {
      constexpr size_t TOP = MantissaType::WORD_COUNT - 1;
      WordType top_word = xd.mantissa[TOP];
      size_t top_rshift = rshift - TOP * WORD_BITS;
      out_mantissa = static_cast<StorageType>(top_word >> top_rshift);
      round_bit = ((top_word >> (top_rshift - 1)) & WordType(1)) != 0;
      WordType low_mask = static_cast<WordType>(
          static_cast<WordType>(WordType(1) << (top_rshift - 1)) - WordType(1));
      sticky_bit = (top_word & low_mask) != 0;
      for (size_t i = 0; i < TOP; ++i)
        sticky_bit = sticky_bit || (xd.mantissa[i] != 0);
    } else {
      size_t rb_idx = rshift - 1;
      size_t rb_word = rb_idx / WORD_BITS;
      size_t rb_off = rb_idx % WORD_BITS;
      round_bit = ((xd.mantissa[rb_word] >> rb_off) & WordType(1)) != 0;
      for (size_t i = 0; i < rb_word; ++i)
        sticky_bit = sticky_bit || (xd.mantissa[i] != 0);
      if (rb_off > 0) {
        WordType low_mask = static_cast<WordType>(
            static_cast<WordType>(WordType(1) << rb_off) - WordType(1));
        sticky_bit = sticky_bit || ((xd.mantissa[rb_word] & low_mask) != 0);
      }
      out_mantissa = static_cast<StorageType>(xd.mantissa >> rshift);
    }
  }

  bool lsb = (out_mantissa & StorageType(1)) != 0;
  StorageType result =
      DstRep::create_value(xd.sign, StorageType(0), out_mantissa).uintval();
  if (round_up(lsb, round_bit, sticky_bit))
    result = apply_rounding_increment<DstType>(result);

  // Underflow is detected pre-rounding; FE_UNDERFLOW may be raised even if
  // rounding produces min_normal.
  if (round_bit || sticky_bit)
    set_flags(FP_EXCEPT_UNDERFLOW | FP_EXCEPT_INEXACT);
  return result;
}

} // namespace internal

namespace static_rounding {

// Pure-integer conversion from a normalized DyadicFloat<Bits> to DstType bits
// under an explicit rounding mode, returning {bits, except_flags} with no FEnv
// or errno side effects and no native floating-point operations.
template <FPType DstType, size_t Bits>
LIBC_INLINE constexpr internal::ConversionResult<
    typename FPRep<DstType>::StorageType>
dyadic_as_bits(const DyadicFloat<Bits> &xd, int rounding = FE_TONEAREST) {
  uint8_t flags = internal::FP_EXCEPT_NONE;
  auto bits =
      internal::dyadic_as_bits_impl<DstType, /*StaticNearest=*/false, 0xffU>(
          xd, rounding, &flags);
  return {bits, flags};
}

template <typename T, size_t Bits>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), T>
dyadic_as(const DyadicFloat<Bits> &xd, int rounding = FE_TONEAREST) {
  return FPBits<T>(dyadic_as_bits<get_fp_type<T>()>(xd, rounding).bits)
      .get_val();
}

} // namespace static_rounding

namespace integer_eval {

template <FPType DstType, size_t Bits>
LIBC_INLINE constexpr typename FPRep<DstType>::StorageType
dyadic_as_bits(const DyadicFloat<Bits> &xd) {
  return fputil::internal::dyadic_as_bits_impl<DstType, /*StaticNearest=*/true,
                                               0>(xd, FE_TONEAREST, nullptr);
}

template <typename T, size_t Bits>
LIBC_INLINE constexpr cpp::enable_if_t<
    cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), T>
dyadic_as(const DyadicFloat<Bits> &xd) {
  return FPBits<T>(dyadic_as_bits<get_fp_type<T>()>(xd)).get_val();
}

} // namespace integer_eval

namespace fenv_eval {

template <FPType DstType, bool ShouldSignalExceptions = true, size_t Bits>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT typename FPRep<DstType>::StorageType
dyadic_as_bits(const DyadicFloat<Bits> &xd) {
  constexpr bool STATIC_NEAREST = !fputil::internal::HAS_DYNAMIC_ROUNDING;
#ifdef LIBC_MATH_HAS_NO_ERRNO
  constexpr uint8_t ERRNO_MASK = fputil::internal::FP_EXCEPT_NONE;
#else
  constexpr uint8_t ERRNO_MASK = fputil::internal::FP_EXCEPT_OVERFLOW |
                                 fputil::internal::FP_EXCEPT_UNDERFLOW;
#endif
  constexpr uint8_t FLAGS_MASK =
      ShouldSignalExceptions
          ? static_cast<uint8_t>((fputil::internal::HAS_FENV_EXCEPT
                                      ? uint8_t(0xffU)
                                      : uint8_t(0)) |
                                 ERRNO_MASK)
          : fputil::internal::FP_EXCEPT_NONE;

  if constexpr (STATIC_NEAREST && FLAGS_MASK == 0) {
    return integer_eval::dyadic_as_bits<DstType>(xd);
  } else {
    int rounding = FE_TONEAREST;
    if constexpr (!STATIC_NEAREST) {
      if (!cpp::is_constant_evaluated())
        rounding = quick_get_round();
    }
    uint8_t except_flags = fputil::internal::FP_EXCEPT_NONE;
    auto bits = fputil::internal::dyadic_as_bits_impl<DstType, STATIC_NEAREST,
                                                      FLAGS_MASK>(
        xd, rounding, &except_flags);

    if constexpr (FLAGS_MASK != 0) {
      if (except_flags != fputil::internal::FP_EXCEPT_NONE &&
          !cpp::is_constant_evaluated()) {
        using ExceptT = cpp::conditional_t<
            DstType == FPType::IEEE754_Binary64 &&
                fputil::internal::HAS_HW_FPU<double>,
            double,
            cpp::conditional_t<fputil::internal::HAS_HW_FPU<float>, float,
                               void>>;
        constexpr uint8_t RARE_MASK = fputil::internal::FP_EXCEPT_INVALID |
                                      fputil::internal::FP_EXCEPT_OVERFLOW |
                                      fputil::internal::FP_EXCEPT_UNDERFLOW;
        if (LIBC_UNLIKELY((except_flags & RARE_MASK) != 0)) {
          fputil::internal::raise_rare_conversion_except<ExceptT>(
              false, (except_flags & fputil::internal::FP_EXCEPT_OVERFLOW) != 0,
              (except_flags & fputil::internal::FP_EXCEPT_UNDERFLOW) != 0);
        } else if constexpr (fputil::internal::HAS_FENV_EXCEPT) {
          if ((except_flags & fputil::internal::FP_EXCEPT_INEXACT) != 0)
            raise_inexact_except_if_required();
        }
      }
    }

    return bits;
  }
}

template <typename T, bool ShouldSignalExceptions = true, size_t Bits>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT cpp::enable_if_t<
    cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), T>
dyadic_as(const DyadicFloat<Bits> &xd) {
  return FPBits<T>(dyadic_as_bits<get_fp_type<T>(), ShouldSignalExceptions>(xd))
      .get_val();
}

namespace internal {

template <typename T, bool ShouldSignalExceptions, size_t Bits>
LIBC_NOINLINE LIBC_COLD T dyadic_as_cold(const DyadicFloat<Bits> &xd) {
  return dyadic_as<T, ShouldSignalExceptions>(xd);
}

} // namespace internal
} // namespace fenv_eval

namespace fp_eval {

namespace internal {

#if !defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) &&                       \
    !defined(LIBC_TARGET_ARCH_IS_GPU)
template <typename T>
LIBC_INLINE_VAR const volatile T DYADIC_RS_TABLE[4] = {T(0), T(1), T(2), T(3)};
#endif

template <typename T>
LIBC_INLINE_VAR constexpr bool IS_NATIVE_HW_DYADIC_TYPE =
    (cpp::is_same_v<T, float> || cpp::is_same_v<T, double>) &&
    fputil::internal::HAS_HW_FPU<T>;

} // namespace internal

// Fast hardware FMA/add path for float and double when HAS_HW_FPU<T> is true.
// Inlines only the normal, non-overflowing, normal-d_lo exponent range via a
// single unsigned range check, delegating all edge cases (subnormals,
// exp_lo <= 0, top-binade carry, and overflow) to outlined fenv_eval.
template <typename T, bool ShouldSignalExceptions = true, size_t Bits>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT cpp::enable_if_t<
    internal::IS_NATIVE_HW_DYADIC_TYPE<T> && (FPBits<T>::FRACTION_LEN < Bits),
    T>
dyadic_as(const DyadicFloat<Bits> &xd) {
  if constexpr (!ShouldSignalExceptions) {
    return fenv_eval::dyadic_as<T, /*ShouldSignalExceptions=*/false>(xd);
  } else {
    if (LIBC_UNLIKELY(xd.mantissa.is_zero()))
      return FPBits<T>::zero(xd.sign).get_val();

    constexpr uint32_t PRECISION = FPBits<T>::FRACTION_LEN + 1;
    using OutputBitsT = typename FPBits<T>::StorageType;
    constexpr OutputBitsT IMPLICIT_MASK =
        FPBits<T>::SIG_MASK - FPBits<T>::FRACTION_MASK;

    int exp_hi =
        xd.exponent + static_cast<int>((Bits - 1) + FPBits<T>::EXP_BIAS);

    // Hot path: exp_lo = exp_hi - PRECISION - 1 >= 1 AND exp_hi < 2 * EXP_BIAS
    // (excluding the top binade exp_hi == 2 * EXP_BIAS so rounding carry into
    // infinity always sets ERANGE via the cold path).
    constexpr int MIN_HOT_EXP = static_cast<int>(PRECISION) + 2;
    constexpr int MAX_HOT_EXP = 2 * FPBits<T>::EXP_BIAS - 1;
    if (LIBC_UNLIKELY(static_cast<uint32_t>(exp_hi - MIN_HOT_EXP) >
                      static_cast<uint32_t>(MAX_HOT_EXP - MIN_HOT_EXP))) {
      if (cpp::is_constant_evaluated())
        return fenv_eval::dyadic_as<T, ShouldSignalExceptions>(xd);
      return fenv_eval::internal::dyadic_as_cold<T, ShouldSignalExceptions>(xd);
    }

    constexpr uint32_t SHIFT = Bits - PRECISION;
    OutputBitsT m_hi = static_cast<OutputBitsT>(xd.mantissa >> SHIFT);
    T d_hi =
        FPBits<T>::create_value(xd.sign, static_cast<OutputBitsT>(exp_hi),
                                (m_hi & FPBits<T>::SIG_MASK) | IMPLICIT_MASK)
            .get_val();

    if constexpr (SHIFT == 0) {
      return d_hi;
    } else {
      using MantissaType = typename DyadicFloat<Bits>::MantissaType;
      constexpr MantissaType ROUND_MASK = MantissaType(1) << (SHIFT - 1);
      constexpr MantissaType STICKY_MASK = ROUND_MASK - MantissaType(1);

      bool round_bit = !(xd.mantissa & ROUND_MASK).is_zero();
      bool sticky_bit = !(xd.mantissa & STICKY_MASK).is_zero();
      int rs_idx =
          static_cast<int>(round_bit) * 2 + static_cast<int>(sticky_bit);

      int exp_lo = exp_hi - static_cast<int>(PRECISION) - 1;
      T d_lo = FPBits<T>::create_value(
                   xd.sign, static_cast<OutputBitsT>(exp_lo), IMPLICIT_MASK)
                   .get_val();

      T rs = T(0);
#if defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) ||                        \
    defined(LIBC_TARGET_ARCH_IS_GPU)
      rs = static_cast<T>(rs_idx);
#else
      if (cpp::is_constant_evaluated())
        rs = static_cast<T>(rs_idx);
      else
        rs = internal::DYADIC_RS_TABLE<T>[rs_idx];
#endif

      // Still correct without FMA instructions because d_lo is guaranteed
      // normal on this path.
      return multiply_add(d_lo, rs, d_hi);
    }
  }
}

} // namespace fp_eval

template <size_t Bits>
template <FPType DstType, bool ShouldSignalExceptions>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT typename FPRep<DstType>::StorageType
DyadicFloat<Bits>::generic_as_bits() const {
#if defined(LIBC_MATH_HAS_NO_EXCEPT) && defined(LIBC_MATH_HAS_NO_ERRNO) &&     \
    defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)
  return integer_eval::dyadic_as_bits<DstType>(*this);
#else
  return fenv_eval::dyadic_as_bits<DstType, ShouldSignalExceptions>(*this);
#endif
}

template <size_t Bits>
template <
    typename T, bool ShouldSignalExceptions,
    cpp::enable_if_t<
        cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), int>>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T DyadicFloat<Bits>::generic_as() const {
  return FPBits<T>(generic_as_bits<get_fp_type<T>(), ShouldSignalExceptions>())
      .get_val();
}

template <size_t Bits>
template <
    typename T, bool ShouldSignalExceptions,
    cpp::enable_if_t<
        cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), int>>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T DyadicFloat<Bits>::fast_as() const {
  if constexpr (fp_eval::internal::IS_NATIVE_HW_DYADIC_TYPE<T>)
    return fp_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
  else
    return fenv_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
}

template <size_t Bits>
template <
    typename T, bool ShouldSignalExceptions,
    cpp::enable_if_t<
        cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), int>>
LIBC_INLINE LIBC_CONSTEXPR_DEFAULT T DyadicFloat<Bits>::as() const {
#ifdef LIBC_FPUTIL_DYADIC_AS_IMPL
#if LIBC_MATH_IMPL_IS(LIBC_FPUTIL_DYADIC_AS_IMPL, integer_eval)
  return integer_eval::dyadic_as<T>(*this);
#elif LIBC_MATH_IMPL_IS(LIBC_FPUTIL_DYADIC_AS_IMPL, fenv_eval)
  return fenv_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
#elif LIBC_MATH_IMPL_IS(LIBC_FPUTIL_DYADIC_AS_IMPL, fp_eval)
  return fp_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
#else
#error "Unsupported LIBC_FPUTIL_DYADIC_AS_IMPL"
#endif
#else // !LIBC_FPUTIL_DYADIC_AS_IMPL
#if defined(LIBC_MATH_HAS_NO_EXCEPT) && defined(LIBC_MATH_HAS_NO_ERRNO) &&     \
    defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)
  if constexpr (!fp_eval::internal::IS_NATIVE_HW_DYADIC_TYPE<T> ||
                !ShouldSignalExceptions)
    return integer_eval::dyadic_as<T>(*this);
#endif
  if constexpr (fp_eval::internal::IS_NATIVE_HW_DYADIC_TYPE<T> &&
                ShouldSignalExceptions) {
    return fp_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
  } else {
    return fenv_eval::dyadic_as<T, ShouldSignalExceptions>(*this);
  }
#endif // LIBC_FPUTIL_DYADIC_AS_IMPL
}

template <size_t Bits>
template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                           (FPBits<T>::FRACTION_LEN < Bits),
                                       int>>
LIBC_INLINE constexpr DyadicFloat<Bits>::operator T() const {
  return as<T, /*ShouldSignalExceptions=*/true>();
}

// Quick add - Add 2 dyadic floats with rounding toward 0 and then normalize
// the output:
//   - Align the exponents so that:
//     new a.exponent = new b.exponent = max(a.exponent, b.exponent)
//   - Add or subtract the mantissas depending on the signs.
//   - Normalize the result.
// The absolute errors compared to the mathematical sum is bounded by:
//   | quick_add(a, b) - (a + b) | < MSB(a + b) * 2^(-Bits + 2),
// i.e., errors are up to 2 ULPs.
// Assume inputs are normalized (by constructors or other functions) so that
// we don't need to normalize the inputs again in this function.  If the
// inputs are not normalized, the results might lose precision significantly.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> quick_add(DyadicFloat<Bits> a,
                                                  DyadicFloat<Bits> b) {
  if (LIBC_UNLIKELY(a.mantissa.is_zero()))
    return b;
  if (LIBC_UNLIKELY(b.mantissa.is_zero()))
    return a;

  // Align exponents
  if (a.exponent > b.exponent)
    b.shift_right(static_cast<unsigned>(a.exponent - b.exponent));
  else if (b.exponent > a.exponent)
    a.shift_right(static_cast<unsigned>(b.exponent - a.exponent));

  DyadicFloat<Bits> result;

  if (a.sign == b.sign) {
    // Addition
    result.sign = a.sign;
    result.exponent = a.exponent;
    result.mantissa = a.mantissa;
    if (result.mantissa.add_overflow(b.mantissa)) {
      // Mantissa addition overflow.
      result.shift_right(1);
      result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORD_COUNT - 1] |=
          (uint64_t(1) << 63);
    }
    // Result is already normalized.
    return result;
  }

  // Subtraction
  if (a.mantissa >= b.mantissa) {
    result.sign = a.sign;
    result.exponent = a.exponent;
    result.mantissa = a.mantissa - b.mantissa;
  } else {
    result.sign = b.sign;
    result.exponent = b.exponent;
    result.mantissa = b.mantissa - a.mantissa;
  }

  return result.normalize();
}

template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> quick_sub(DyadicFloat<Bits> a,
                                                  DyadicFloat<Bits> b) {
  return quick_add(a, -b);
}

// Quick Mul - Slightly less accurate but efficient multiplication of 2 dyadic
// floats with rounding toward 0 and then normalize the output:
//   result.exponent = a.exponent + b.exponent + Bits,
//   result.mantissa = quick_mul_hi(a.mantissa + b.mantissa)
//                   ~ (full product a.mantissa * b.mantissa) >> Bits.
// The errors compared to the mathematical product is bounded by:
//   2 * errors of quick_mul_hi = 2 * (UInt<Bits>::WORD_COUNT - 1) in ULPs.
// Assume inputs are normalized (by constructors or other functions) so that
// we don't need to normalize the inputs again in this function.  If the
// inputs are not normalized, the results might lose precision significantly.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> quick_mul(const DyadicFloat<Bits> &a,
                                                  const DyadicFloat<Bits> &b) {
  DyadicFloat<Bits> result;
  result.sign = (a.sign != b.sign) ? Sign::NEG : Sign::POS;
  result.exponent = a.exponent + b.exponent + static_cast<int>(Bits);

  if (!(a.mantissa.is_zero() || b.mantissa.is_zero())) {
    result.mantissa = a.mantissa.quick_mul_hi(b.mantissa);
    // Check the leading bit directly, should be faster than using clz in
    // normalize().
    if (result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORD_COUNT - 1] >>
            (DyadicFloat<Bits>::MantissaType::WORD_SIZE - 1) ==
        0)
      result.shift_left(1);
  } else {
    result.mantissa = (typename DyadicFloat<Bits>::MantissaType)(0);
  }
  return result;
}

// Correctly rounded multiplication of 2 dyadic floats, assuming the
// exponent remains within range.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits>
rounded_mul(const DyadicFloat<Bits> &a, const DyadicFloat<Bits> &b) {
  using DblMant = LIBC_NAMESPACE::UInt<(2 * Bits)>;
  Sign result_sign = (a.sign != b.sign) ? Sign::NEG : Sign::POS;
  int result_exponent = a.exponent + b.exponent + static_cast<int>(Bits);
  auto product = DblMant(a.mantissa) * DblMant(b.mantissa);
  // As in quick_mul(), renormalize by 1 bit manually rather than countl_zero
  if (product.get_bit(2 * Bits - 1) == 0) {
    product <<= 1;
    result_exponent -= 1;
  }

  return DyadicFloat<Bits>::round(result_sign, result_exponent, product, Bits);
}

// Approximate reciprocal - given a nonzero a, make a good approximation to
// 1/a. The method is Newton-Raphson iteration, based on quick_mul.
template <size_t Bits, typename = cpp::enable_if_t<(Bits >= 32)>>
LIBC_INLINE constexpr DyadicFloat<Bits>
approx_reciprocal(const DyadicFloat<Bits> &a) {
  // Given an approximation x to 1/a, a better one is x' = x(2-ax).
  //
  // You can derive this by using the Newton-Raphson formula with the function
  // f(x) = 1/x - a. But another way to see that it works is to say: suppose
  // that ax = 1-e for some small error e. Then ax' = ax(2-ax) = (1-e)(1+e) =
  // 1-e^2. So the error in x' is the square of the error in x, i.e. the
  // number of correct bits in x' is double the number in x.

  // An initial approximation to the reciprocal
  DyadicFloat<Bits> x(Sign::POS, -32 - a.exponent - int(Bits),
                      uint64_t(0xFFFFFFFFFFFFFFFF) /
                          static_cast<uint64_t>(a.mantissa >> (Bits - 32)));

  // The constant 2, which we'll need in every iteration
  DyadicFloat<Bits> two(Sign::POS, 1, 1);

  // We expect at least 31 correct bits from our 32-bit starting approximation
  size_t ok_bits = 31;

  // The number of good bits doubles in each iteration, except that rounding
  // errors introduce a little extra each time. Subtract a bit from our
  // accuracy assessment to account for that.
  while (ok_bits < Bits) {
    x = quick_mul(x, quick_sub(two, quick_mul(a, x)));
    ok_bits = 2 * ok_bits - 1;
  }

  return x;
}

// Correctly rounded division of 2 dyadic floats, assuming the
// exponent remains within range.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits>
rounded_div(const DyadicFloat<Bits> &af, const DyadicFloat<Bits> &bf) {
  using DblMant = LIBC_NAMESPACE::UInt<(Bits * 2 + 64)>;

  // Make an approximation to the quotient as a * (1/b). Both the
  // multiplication and the reciprocal are a bit sloppy, which doesn't
  // matter, because we're going to correct for that below.
  auto qf = fputil::quick_mul(af, fputil::approx_reciprocal(bf));

  // Switch to BigInt and stop using quick_add and quick_mul: now
  // we're working in exact integers so as to get the true remainder.
  DblMant a = af.mantissa, b = bf.mantissa, q = qf.mantissa;
  q <<= 2; // leave room for a round bit, even if exponent decreases
  a <<= af.exponent - bf.exponent - qf.exponent + 2;
  DblMant qb = q * b;
  if (qb < a) {
    DblMant too_small = a - b;
    while (qb <= too_small) {
      qb += b;
      ++q;
    }
  } else {
    while (qb > a) {
      qb -= b;
      --q;
    }
  }

  DyadicFloat<(Bits * 2)> qbig(qf.sign, qf.exponent - 2, q);
  return DyadicFloat<Bits>::round(qbig.sign, qbig.exponent + Bits,
                                  qbig.mantissa, Bits);
}

// Simple polynomial approximation.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits>
multiply_add(const DyadicFloat<Bits> &a, const DyadicFloat<Bits> &b,
             const DyadicFloat<Bits> &c) {
  return quick_add(c, quick_mul(a, b));
}

// Simple exponentiation implementation for printf. Only handles positive
// exponents, since division isn't implemented.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> pow_n(const DyadicFloat<Bits> &a,
                                              uint32_t power) {
  DyadicFloat<Bits> result = 1.0;
  DyadicFloat<Bits> cur_power = a;

  while (power > 0) {
    if ((power % 2) > 0)
      result = quick_mul(result, cur_power);
    power = power >> 1;
    cur_power = quick_mul(cur_power, cur_power);
  }
  return result;
}

template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> mul_pow_2(const DyadicFloat<Bits> &a,
                                                  int32_t pow_2) {
  DyadicFloat<Bits> result = a;
  result.exponent += pow_2;
  return result;
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
