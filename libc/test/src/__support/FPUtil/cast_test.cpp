//===-- Unittests for fputil::cast and conversion namespaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/FPUtil/float80.h"
#include "src/__support/FPUtil/fp_convert.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;
using LIBC_NAMESPACE::fputil::BFloat16;
using LIBC_NAMESPACE::fputil::Float128;
using LIBC_NAMESPACE::fputil::Float80;
using LIBC_NAMESPACE::fputil::FPBits;

using LlvmLibcCastTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

// Verify constexpr usability of all software conversion namespaces.
static_assert(FPBits<BFloat16>(LIBC_NAMESPACE::fputil::cast<BFloat16>(1.5f))
                  .uintval() == 0x3fc0U);
static_assert(FPBits<float>(LIBC_NAMESPACE::fputil::integer_eval::cast<float>(
                                FPBits<BFloat16>(uint16_t(0x3fc0U)).get_val()))
                  .uintval() == 0x3fc0'0000U);
static_assert(LIBC_NAMESPACE::fputil::static_rounding::cast_bits<
                  LIBC_NAMESPACE::fputil::FPType::BFloat16,
                  LIBC_NAMESPACE::fputil::FPType::IEEE754_Binary32>(
                  0x3f80'0001U, FE_UPWARD)
                  .bits == 0x3f81U);

TEST_F(LlvmLibcCastTest, BFloat16ExhaustiveRoundTripAndNamespaces) {
  using BF16Bits = FPBits<BFloat16>;
  using F32Bits = FPBits<float>;

  for (uint32_t i = 0; i <= 0xffffU; ++i) {
    uint16_t raw = static_cast<uint16_t>(i);
    BFloat16 bf = BF16Bits(raw).get_val();
    BF16Bits bf_bits(bf);

    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    float f_default = LIBC_NAMESPACE::fputil::cast<float>(bf);
    int except_default = LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT);

    float f_int = LIBC_NAMESPACE::fputil::integer_eval::cast<float>(bf);
    ASSERT_EQ(F32Bits(f_default).uintval(), F32Bits(f_int).uintval());

    if (bf_bits.is_nan()) {
      ASSERT_TRUE(F32Bits(f_default).is_quiet_nan());
      if (bf_bits.is_signaling_nan()) {
        EXPECT_EQ(except_default & FE_INVALID, FE_INVALID);
      } else {
        EXPECT_EQ(except_default, 0);
      }
      continue;
    }

    EXPECT_EQ(except_default, 0);
    ASSERT_EQ(F32Bits(f_default).uintval(), static_cast<uint32_t>(raw) << 16U);

    // Exact round-trip back to BFloat16 across all namespaces.
    ASSERT_EQ(
        BF16Bits(LIBC_NAMESPACE::fputil::cast<BFloat16>(f_default)).uintval(),
        raw);
    ASSERT_EQ(
        BF16Bits(LIBC_NAMESPACE::fputil::fenv_eval::cast<BFloat16>(f_default))
            .uintval(),
        raw);
    ASSERT_EQ(BF16Bits(LIBC_NAMESPACE::fputil::integer_eval::cast<BFloat16>(
                           f_default))
                  .uintval(),
              raw);

    // Also widen through double and Float128 and narrow back.
    double d = LIBC_NAMESPACE::fputil::cast<double>(bf);
    Float128 q = LIBC_NAMESPACE::fputil::cast<Float128>(bf);
    ASSERT_EQ(BF16Bits(LIBC_NAMESPACE::fputil::cast<BFloat16>(d)).uintval(),
              raw);
    ASSERT_EQ(BF16Bits(LIBC_NAMESPACE::fputil::cast<BFloat16>(q)).uintval(),
              raw);
  }
}

#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_F(LlvmLibcCastTest, Float16ExhaustiveRoundTripAndCrossBFloat16) {
  using F16Bits = FPBits<float16>;
  using BF16Bits = FPBits<BFloat16>;
  using F32Bits = FPBits<float>;

  for (uint32_t i = 0; i <= 0xffffU; ++i) {
    uint16_t raw = static_cast<uint16_t>(i);
    float16 h = F16Bits(raw).get_val();
    F16Bits h_bits(h);

    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    float f = LIBC_NAMESPACE::fputil::cast<float>(h);
    float f_int = LIBC_NAMESPACE::fputil::integer_eval::cast<float>(h);
    ASSERT_EQ(F32Bits(f).uintval(), F32Bits(f_int).uintval());

    if (h_bits.is_nan()) {
      ASSERT_TRUE(F32Bits(f).is_quiet_nan());
      if (h_bits.is_signaling_nan())
        EXPECT_FP_EXCEPTION(FE_INVALID);
      continue;
    }

    EXPECT_FP_EXCEPTION(0);
    ASSERT_EQ(F16Bits(LIBC_NAMESPACE::fputil::cast<float16>(f)).uintval(), raw);

    double d = LIBC_NAMESPACE::fputil::cast<double>(h);
    Float128 q = LIBC_NAMESPACE::fputil::cast<Float128>(h);
    ASSERT_EQ(F16Bits(LIBC_NAMESPACE::fputil::cast<float16>(d)).uintval(), raw);
    ASSERT_EQ(F16Bits(LIBC_NAMESPACE::fputil::cast<float16>(q)).uintval(), raw);

    // Cross-range float16 -> bfloat16 must agree with float16 -> float ->
    // bfloat16 because float16 -> float is exact.
    BFloat16 bf_direct = LIBC_NAMESPACE::fputil::cast<BFloat16>(h);
    BFloat16 bf_via_f32 = LIBC_NAMESPACE::fputil::cast<BFloat16>(f);
    ASSERT_EQ(BF16Bits(bf_direct).uintval(), BF16Bits(bf_via_f32).uintval());
  }

  // Cross-range bfloat16 -> float16 must agree with bfloat16 -> float ->
  // float16 across all finite/Inf bfloat16 values because bfloat16 -> float is
  // exact.
  for (uint32_t i = 0; i <= 0xffffU; ++i) {
    uint16_t raw = static_cast<uint16_t>(i);
    BFloat16 bf = BF16Bits(raw).get_val();
    if (BF16Bits(bf).is_nan())
      continue;
    float f = LIBC_NAMESPACE::fputil::cast<float>(bf);
    float16 h_direct = LIBC_NAMESPACE::fputil::cast<float16>(bf);
    float16 h_via_f32 = LIBC_NAMESPACE::fputil::cast<float16>(f);
    ASSERT_EQ(F16Bits(h_direct).uintval(), F16Bits(h_via_f32).uintval());
  }
}
#endif // LIBC_TYPES_HAS_FLOAT16

TEST_F(LlvmLibcCastTest, DirectedRoundingOverflowAndUnderflow) {
  using namespace LIBC_NAMESPACE::fputil::testing;
  using BF16Bits = FPBits<BFloat16>;

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
  for (RoundingMode mode : MODES) {
    ForceRoundingMode r(mode);
    if (!r.success)
      continue;
    int fenv_mode = LIBC_NAMESPACE::fputil::quick_get_round();

    // Exact tie in normal range: 1.0 + 2^-8 (0x3f808000) -> 0x3f80 except RU.
    float tie_even = FPBits<float>(0x3f80'8000U).get_val();
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    libc_errno = 0;
    BFloat16 res_tie = LIBC_NAMESPACE::fputil::cast<BFloat16>(tie_even);
    BFloat16 res_static =
        LIBC_NAMESPACE::fputil::static_rounding::cast<BFloat16>(tie_even,
                                                                fenv_mode);
    uint16_t expected_tie = (mode == RoundingMode::Upward) ? 0x3f81U : 0x3f80U;
    EXPECT_EQ(BF16Bits(res_tie).uintval(), expected_tie);
    EXPECT_EQ(BF16Bits(res_static).uintval(), expected_tie);
    EXPECT_FP_EXCEPTION(FE_INEXACT);
    EXPECT_MATH_ERRNO(0);

    // Overflow from double(1e50) -> BFloat16: sets ERANGE and
    // FE_OVERFLOW|FE_INEXACT.
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    libc_errno = 0;
    BFloat16 res_ovf = LIBC_NAMESPACE::fputil::cast<BFloat16>(1e50);
    uint16_t expected_ovf =
        (mode == RoundingMode::Nearest || mode == RoundingMode::Upward)
            ? BF16Bits::inf().uintval()
            : BF16Bits::max_normal().uintval();
    EXPECT_EQ(BF16Bits(res_ovf).uintval(), expected_ovf);
    EXPECT_FP_EXCEPTION(FE_OVERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);

    // Subnormal underflow from double(1e-50) -> BFloat16.
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    libc_errno = 0;
    BFloat16 res_udf = LIBC_NAMESPACE::fputil::cast<BFloat16>(1e-50);
    uint16_t expected_udf = (mode == RoundingMode::Upward)
                                ? BF16Bits::min_subnormal().uintval()
                                : BF16Bits::zero().uintval();
    EXPECT_EQ(BF16Bits(res_udf).uintval(), expected_udf);
    EXPECT_FP_EXCEPTION(FE_UNDERFLOW | FE_INEXACT);
    EXPECT_MATH_ERRNO(ERANGE);
  }
#endif // !LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}

TEST_F(LlvmLibcCastTest, Float80AndFloat128Edges) {
  using F80Bits = FPBits<Float80>;
  using F128Bits = FPBits<Float128>;

  // Subnormal Float80 -> Float128 -> Float80 exact preservation.
  Float80 min_sub_80 = F80Bits::min_subnormal().get_val();
  Float80 max_sub_80 = F80Bits::max_subnormal().get_val();
  Float128 wide_min = LIBC_NAMESPACE::fputil::cast<Float128>(min_sub_80);
  Float128 wide_max = LIBC_NAMESPACE::fputil::cast<Float128>(max_sub_80);
  EXPECT_TRUE(
      F80Bits(LIBC_NAMESPACE::fputil::cast<Float80>(wide_min)).uintval() ==
      F80Bits(min_sub_80).uintval());
  EXPECT_TRUE(
      F80Bits(LIBC_NAMESPACE::fputil::cast<Float80>(wide_max)).uintval() ==
      F80Bits(max_sub_80).uintval());

  // Float128 just below Float80 min_normal (max_subnormal + 0.5 ULP_80)
  // rounding up to Float80 min_normal must set the explicit bit of X86_Binary80
  // properly! In Float128, 0.5 ULP of Float80 subnormal is also subnormal:
  // 2^(-16382 - 64) = 2^(1 - 16383 - 112) * 2^48.
  Float128 half_sub_ulp_80 =
      F128Bits::create_value(
          Sign::POS, F128Bits::StorageType(0),
          F128Bits::StorageType(1)
              << (F128Bits::FRACTION_LEN - F80Bits::FRACTION_LEN - 1))
          .get_val();
  Float128 almost_min_norm = wide_max + half_sub_ulp_80;
  Float80 rounded_up = LIBC_NAMESPACE::fputil::cast<Float80>(almost_min_norm);
  EXPECT_TRUE(F80Bits(rounded_up).uintval() == F80Bits::min_normal().uintval());

  // Float128 exceeding Float80 max_normal (max_normal + 0.5 ULP_80) rounding
  // to Float80 inf must set the explicit bit of X86_Binary80 inf properly (not
  // pseudo-infinity)!
  Float80 max_norm_80 = F80Bits::max_normal().get_val();
  Float128 wide_max_norm = LIBC_NAMESPACE::fputil::cast<Float128>(max_norm_80);
  Float128 half_ulp_80 =
      F128Bits::create_value(
          Sign::POS,
          static_cast<F128Bits::StorageType>(F128Bits::EXP_BIAS + 16383 - 64),
          F128Bits::StorageType(0))
          .get_val();
  Float80 rounded_inf =
      LIBC_NAMESPACE::fputil::cast<Float80>(wide_max_norm + half_ulp_80);
  EXPECT_TRUE(F80Bits(rounded_inf).is_inf());
}
