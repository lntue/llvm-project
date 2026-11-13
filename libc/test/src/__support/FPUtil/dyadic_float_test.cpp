//===-- Unittests for the DyadicFloat class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/FPUtil/float80.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using DFloat128 = LIBC_NAMESPACE::fputil::DyadicFloat<128>;
using Float192 = LIBC_NAMESPACE::fputil::DyadicFloat<192>;
using Float256 = LIBC_NAMESPACE::fputil::DyadicFloat<256>;
using LIBC_NAMESPACE::Sign;

TEST(LlvmLibcDyadicFloatTest, BasicConversions) {
  DFloat128 x(Sign::POS, /*exponent*/ 0,
              /*mantissa*/ DFloat128::MantissaType(1));
  ASSERT_FP_EQ(1.0f, float(x));
  ASSERT_FP_EQ(1.0, double(x));

  DFloat128 y(0x1.0p-53);
  ASSERT_FP_EQ(0x1.0p-53f, float(y));
  ASSERT_FP_EQ(0x1.0p-53, double(y));

  DFloat128 z = quick_add(x, y);

  EXPECT_FP_EQ_ALL_ROUNDING(float(x) + float(y), float(z));
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) + double(y), double(z));
}

TEST(LlvmLibcDyadicFloatTest, QuickAdd) {
  Float192 x(Sign::POS, /*exponent*/ 0,
             /*mantissa*/ Float192::MantissaType(0x123456));
  ASSERT_FP_EQ(0x1.23456p20, double(x));

  Float192 y(0x1.abcdefp-20);
  ASSERT_FP_EQ(0x1.abcdefp-20, double(y));

  Float192 z = quick_add(x, y);
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) + double(y), double(z));

  DFloat128 a(0x1.0p0);
  ASSERT_FP_EQ(0x1.0p0, double(a));

  DFloat128 b(0x1.0p128);
  ASSERT_FP_EQ(0x1.0p128, double(b));

  DFloat128 c1 = quick_add(a, b);
  EXPECT_FP_EQ(double(a) + double(b), double(c1));

  DFloat128 c2 = quick_add(b, a);
  EXPECT_FP_EQ(double(b) + double(a), double(c2));
}

TEST(LlvmLibcDyadicFloatTest, QuickMul) {
  Float256 x(Sign::POS, /*exponent*/ 0,
             /*mantissa*/ Float256::MantissaType(0x123456));
  ASSERT_FP_EQ(0x1.23456p20, double(x));

  Float256 y(0x1.abcdefp-25);
  ASSERT_FP_EQ(0x1.abcdefp-25, double(y));

  Float256 z = quick_mul(x, y);
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) * double(y), double(z));
}

#define TEST_EDGE_RANGES(Name, Type)                                           \
  TEST(LlvmLibcDyadicFloatTest, EdgeRanges##Name) {                            \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    using DFType = LIBC_NAMESPACE::fputil::DyadicFloat<Bits::STORAGE_LEN>;     \
    Type max_normal = Bits::max_normal().get_val();                            \
    Type min_normal = Bits::min_normal().get_val();                            \
    Type min_subnormal = Bits::min_subnormal().get_val();                      \
    Type two(2);                                                               \
                                                                               \
    DFType x(min_normal);                                                      \
    EXPECT_FP_EQ_ALL_ROUNDING(min_normal, static_cast<Type>(x));               \
    --x.exponent;                                                              \
    EXPECT_FP_EQ(min_normal / two, static_cast<Type>(x));                      \
                                                                               \
    DFType y(two * min_normal - min_subnormal);                                \
    --y.exponent;                                                              \
    EXPECT_FP_EQ(min_normal, static_cast<Type>(y));                            \
                                                                               \
    DFType z(min_subnormal);                                                   \
    EXPECT_FP_EQ_ALL_ROUNDING(min_subnormal, static_cast<Type>(z));            \
    --z.exponent;                                                              \
    EXPECT_FP_EQ(Bits::zero().get_val(), static_cast<Type>(z));                \
                                                                               \
    DFType t(max_normal);                                                      \
    EXPECT_FP_EQ_ALL_ROUNDING(max_normal, static_cast<Type>(t));               \
    ++t.exponent;                                                              \
    EXPECT_FP_EQ(Bits::inf().get_val(), static_cast<Type>(t));                 \
  }                                                                            \
  static_assert(true, "Require semicolon.")

TEST_EDGE_RANGES(Float, float);
TEST_EDGE_RANGES(Double, double);
TEST_EDGE_RANGES(LongDouble, long double);
TEST_EDGE_RANGES(Float80, LIBC_NAMESPACE::fputil::Float80);
TEST_EDGE_RANGES(Float128, LIBC_NAMESPACE::fputil::Float128);
TEST_EDGE_RANGES(BFloat16, LIBC_NAMESPACE::fputil::BFloat16);
#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_EDGE_RANGES(Float16, float16);
#endif

TEST(LlvmLibcDyadicFloatTest, EvalNamespacesConsistency) {
  using namespace LIBC_NAMESPACE::fputil::testing;
  // 1/3 in DyadicFloat<128>: inexact in all binary formats.
  DFloat128 third(Sign::POS, -129,
                  LIBC_NAMESPACE::UInt<128>(
                      {0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL}));

  double d_int = LIBC_NAMESPACE::fputil::integer_eval::dyadic_as<double>(third);
  ASSERT_FP_EQ(1.0 / 3.0, d_int);

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
  for (RoundingMode mode : MODES) {
    ForceRoundingMode r(mode);
    if (!r.success)
      continue;
    int fenv_mode = LIBC_NAMESPACE::fputil::quick_get_round();

    double d_default = static_cast<double>(third);
    double d_fenv = LIBC_NAMESPACE::fputil::fenv_eval::dyadic_as<double>(third);
    double d_static =
        LIBC_NAMESPACE::fputil::static_rounding::dyadic_as<double>(third,
                                                                   fenv_mode);
    EXPECT_FP_EQ(d_fenv, d_default);
    EXPECT_FP_EQ(d_static, d_default);

    // Also test extreme underflow (shift >= Bits) across all rounding modes.
    DFloat128 tiny(Sign::POS, -2000, DFloat128::MantissaType(1));
    double tiny_default = static_cast<double>(tiny);
    double tiny_fenv =
        LIBC_NAMESPACE::fputil::fenv_eval::dyadic_as<double>(tiny);
    EXPECT_FP_EQ(tiny_fenv, tiny_default);
  }
#endif
}
