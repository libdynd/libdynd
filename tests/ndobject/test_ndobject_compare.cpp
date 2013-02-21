//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(NDObjectCompare, Bool) {
    // Equality
    EXPECT_TRUE(ndobject(true) == ndobject(true));
    EXPECT_TRUE(ndobject(false) == ndobject(false));
    EXPECT_FALSE(ndobject(true) == ndobject(false));
    EXPECT_FALSE(ndobject(false) == ndobject(true));
    // Inequality
    EXPECT_FALSE(ndobject(true) != ndobject(true));
    EXPECT_FALSE(ndobject(false) != ndobject(false));
    EXPECT_TRUE(ndobject(true) != ndobject(false));
    EXPECT_TRUE(ndobject(false) != ndobject(true));
    // Comparison for sorting
    EXPECT_TRUE(ndobject(false).op_sorting_less(ndobject(true)));
    EXPECT_FALSE(ndobject(false).op_sorting_less(ndobject(false)));
    EXPECT_FALSE(ndobject(true).op_sorting_less(ndobject(true)));
    EXPECT_FALSE(ndobject(true).op_sorting_less(ndobject(false)));
    // Other comparisons are not permitted
    EXPECT_THROW((ndobject(false) < ndobject(true)), runtime_error);
    EXPECT_THROW((ndobject(false) <= ndobject(true)), runtime_error);
    EXPECT_THROW((ndobject(false) >= ndobject(true)), runtime_error);
    EXPECT_THROW((ndobject(false) > ndobject(true)), runtime_error);
    // Compare Bool with other types
    EXPECT_TRUE(ndobject(true) == ndobject(1));
    EXPECT_TRUE(ndobject(true) == ndobject(1.f));
    EXPECT_TRUE(ndobject(true) == ndobject(1.0));
    EXPECT_TRUE(ndobject(true) == ndobject(complex<double>(1.0)));
    EXPECT_TRUE(ndobject(false) == ndobject(0));
    EXPECT_TRUE(ndobject(false) == ndobject(0.f));
    EXPECT_TRUE(ndobject(false) == ndobject(0.0));
    EXPECT_TRUE(ndobject(false) == ndobject(complex<double>()));
    EXPECT_TRUE(ndobject(true) != ndobject(2));
    EXPECT_TRUE(ndobject(true) != ndobject(2.f));
    EXPECT_TRUE(ndobject(true) != ndobject(2.0));
    EXPECT_TRUE(ndobject(true) != ndobject(complex<double>(1,1)));
    EXPECT_TRUE(ndobject(false) != ndobject(-1));
    EXPECT_TRUE(ndobject(false) != ndobject(-1.f));
    EXPECT_TRUE(ndobject(false) != ndobject(-1.0));
    EXPECT_TRUE(ndobject(false) != ndobject(complex<double>(0,1)));
}

TEST(NDObjectCompare, EqualityIntUInt) {
    // Equality
    EXPECT_TRUE(ndobject((uint8_t)127) == ndobject((int8_t)127));
    EXPECT_TRUE(ndobject((int8_t)127) == ndobject((uint8_t)127));
    EXPECT_TRUE(ndobject((uint16_t)32767) == ndobject((int16_t)32767));
    EXPECT_TRUE(ndobject((int16_t)32767) == ndobject((uint16_t)32767));
    EXPECT_TRUE(ndobject((uint32_t)2147483647) == ndobject((int32_t)2147483647));
    EXPECT_TRUE(ndobject((int32_t)2147483647) == ndobject((uint32_t)2147483647));
    EXPECT_TRUE(ndobject((uint64_t)9223372036854775807LL) == ndobject((int64_t)9223372036854775807LL));
    EXPECT_TRUE(ndobject((int64_t)9223372036854775807LL) == ndobject((uint64_t)9223372036854775807LL));
    // Inequality
    EXPECT_FALSE(ndobject((uint8_t)127) != ndobject((int16_t)127));
    EXPECT_FALSE(ndobject((int8_t)127) != ndobject((uint16_t)127));
    EXPECT_FALSE(ndobject((uint16_t)32767) != ndobject((int16_t)32767));
    EXPECT_FALSE(ndobject((int16_t)32767) != ndobject((uint16_t)32767));
    EXPECT_FALSE(ndobject((uint32_t)2147483647) != ndobject((int32_t)2147483647));
    EXPECT_FALSE(ndobject((int32_t)2147483647) != ndobject((uint32_t)2147483647));
    EXPECT_FALSE(ndobject((uint64_t)9223372036854775807LL) != ndobject((int64_t)9223372036854775807LL));
    EXPECT_FALSE(ndobject((int64_t)9223372036854775807LL) != ndobject((uint64_t)9223372036854775807LL));
    // Equality with same bits
    EXPECT_FALSE(ndobject((uint8_t)255) == ndobject((int8_t)-1));
    EXPECT_FALSE(ndobject((int8_t)-1) == ndobject((uint8_t)255));
    EXPECT_FALSE(ndobject((uint16_t)65535) == ndobject((int16_t)-1));
    EXPECT_FALSE(ndobject((int16_t)-1) == ndobject((uint16_t)65535));
    EXPECT_FALSE(ndobject((uint32_t)4294967295u) == ndobject((int32_t)-1));
    EXPECT_FALSE(ndobject((int32_t)-1) == ndobject((uint32_t)4294967295u));
    EXPECT_FALSE(ndobject((uint64_t)18446744073709551615ULL) == ndobject((int64_t)-1));
    EXPECT_FALSE(ndobject((int64_t)-1) == ndobject((uint64_t)18446744073709551615ULL));
    // Inequality with same bits
    EXPECT_TRUE(ndobject((uint8_t)255) != ndobject((int8_t)-1));
    EXPECT_TRUE(ndobject((int8_t)-1) != ndobject((uint8_t)255));
    EXPECT_TRUE(ndobject((uint16_t)65535) != ndobject((int16_t)-1));
    EXPECT_TRUE(ndobject((int16_t)-1) != ndobject((uint16_t)65535));
    EXPECT_TRUE(ndobject((uint32_t)4294967295u) != ndobject((int32_t)-1));
    EXPECT_TRUE(ndobject((int32_t)-1) != ndobject((uint32_t)4294967295u));
    EXPECT_TRUE(ndobject((uint64_t)18446744073709551615ULL) != ndobject((int64_t)-1));
    EXPECT_TRUE(ndobject((int64_t)-1) != ndobject((uint64_t)18446744073709551615ULL));
}

TEST(NDObjectCompare, InequalityInt8UInt8) {
    ndobject a, b;

    a = (int8_t)-1;
    b = (uint8_t)0;
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);

    a = (int8_t)0;
    b = (uint8_t)0;
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int8_t)1;
    b = (uint8_t)0;
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int8_t)127;
    b = (uint8_t)126;
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int8_t)127;
    b = (uint8_t)127;
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int8_t)127;
    b = (uint8_t)128;
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);
}

TEST(NDObjectCompare, InequalityInt64UInt64) {
    ndobject a, b;

    a = (int64_t)-1;
    b = (uint64_t)0;
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);

    a = (int64_t)0;
    b = (uint64_t)0;
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int64_t)1;
    b = (uint64_t)0;
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int64_t)9223372036854775807LL;
    b = (uint64_t)9223372036854775806LL;
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a > b);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int64_t)9223372036854775807LL;
    b = (uint64_t)9223372036854775807LL;
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    a = (int64_t)9223372036854775807LL;
    b = (uint64_t)9223372036854775808ULL;
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);

    // Smallest int64_t vs largest uint64_t
    a = numeric_limits<int64_t>::min();
    b = numeric_limits<uint64_t>::max();
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);
}

TEST(NDObjectCompare, EqualityIntFloat) {
    ndobject a, b;

    // 2**24 is the end of the consecutive float32 integers
    a = 16777216;
    b = 16777216.f;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    // Some of the following tests are excluded
    // because they return the wrong answer.
    // THIS IS COMMON TO C/C++/NumPy/etc, TOO.
    // TODO: Implement rigorous comparisons between types.
    a = 16777217;
    b = 16777216.f;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    //EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a >= b);
    //EXPECT_TRUE(a > b);
    //EXPECT_TRUE(b.op_sorting_less(a));
    //EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    //EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    // The following work because the platforms tested on
    // convert 16777217 -> 16777216.f before doing
    // the operations.
    a = 16777217;
    b = 16777218.f;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);
}

TEST(NDObjectCompare, EqualityUIntFloat) {
    ndobject a, b;

    // 2**24 is the end of the consecutive float32 integers
    a = (uint32_t)16777216;
    b = 16777216.f;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b > a);

    // Some of the following tests are excluded
    // because they return the wrong answer.
    // THIS IS COMMON TO C/C++/NumPy/etc, TOO.
    // TODO: Implement rigorous comparisons between types.
    a = (uint32_t)16777217;
    b = 16777216.f;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    //EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a >= b);
    //EXPECT_TRUE(a > b);
    //EXPECT_TRUE(b.op_sorting_less(a));
    //EXPECT_TRUE(b < a);
    EXPECT_TRUE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    //EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    // The following work because the platforms tested on
    // convert 16777217 -> 16777216.f before doing
    // the operations.
    a = (uint32_t)16777217;
    b = 16777218.f;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b > a);
}

TEST(NDObjectCompare, NaNFloat32) {
    ndobject a, b;
    ndobject nan = ndobject("nan").cast_scalars<float>().eval();
    ndobject pinf = ndobject("inf").cast_scalars<float>().eval();

    // A NaN, compared against itself
    a = nan;
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_FALSE(a < a);
    EXPECT_FALSE(a <= a);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_FALSE(a >= a);
    EXPECT_FALSE(a > a);

    // Compare NaN against zero.
    // Special "sorting less than" orders them, other
    // comparisons do not.
    a = nan;
    b = 0.f;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    // Compare NaN against inf.
    // Special "sorting less than" orders them, other
    // comparisons do not.
    a = nan;
    b = pinf;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);
}

TEST(NDObjectCompare, NaNFloat64) {
    ndobject a, b;
    ndobject nan = ndobject("nan").cast_scalars<double>().eval();
    ndobject pinf = ndobject("inf").cast_scalars<double>().eval();

    // A NaN, compared against itself
    a = nan;
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_FALSE(a < a);
    EXPECT_FALSE(a <= a);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_FALSE(a >= a);
    EXPECT_FALSE(a > a);

    // Compare NaN against zero.
    // Special "sorting less than" orders them, other
    // comparisons do not.
    a = nan;
    b = 0.0;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);

    // Compare NaN against inf.
    // Special "sorting less than" orders them, other
    // comparisons do not.
    a = nan;
    b = pinf;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a <= b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a >= b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(b <= a);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(b >= a);
    EXPECT_FALSE(b > a);
}

TEST(NDObjectCompare, ComplexFloat32) {
    ndobject a, b;
    // For complex, op_sorting_less is lexicographic,
    // and other inequalities raise exceptions.

    // Compare 0 with 0
    a = complex<float>();
    b = complex<float>();
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    // Compare 0+1j with 1+1j
    a = complex<float>(0, 1);
    b = complex<float>(1, 1);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    // Compare 0+1j with 0+2j
    a = complex<float>(0, 1);
    b = complex<float>(0, 2);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);
}

TEST(NDObjectCompare, ComplexFloat64) {
    ndobject a, b;
    // For complex, op_sorting_less is lexicographic,
    // and other inequalities raise exceptions.

    // Compare 0 with 0
    a = complex<double>();
    b = complex<double>();
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    // Compare 0+1j with 1+1j
    a = complex<double>(0, 1);
    b = complex<double>(1, 1);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    // Compare 0+1j with 0+2j
    a = complex<double>(0, 1);
    b = complex<double>(0, 2);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);
}

TEST(NDObjectCompare, NaNComplexFloat32) {
    // The strange way to assign complex values is because
    // on clang, complex<float>(0.f, nan) was creating (nan, nan)
    // instead of the requested complex.
    complex<float> cval;
    ndobject a, b;
    float nan = ndobject("nan").cast_scalars<float>().as<float>();

    // real component NaN, compared against itself
    cval.real(nan);
    cval.imag(0.f);
    a = cval;
    EXPECT_TRUE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_FALSE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), runtime_error);
    EXPECT_THROW((a <= a), runtime_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), runtime_error);
    EXPECT_THROW((a > a), runtime_error);

    // imaginary component NaN, compared against itself
    cval.real(0.f);
    cval.imag(nan);
    a = cval;
    EXPECT_FALSE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), runtime_error);
    EXPECT_THROW((a <= a), runtime_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), runtime_error);
    EXPECT_THROW((a > a), runtime_error);

    // both components NaN, compared against itself
    cval.real(nan);
    cval.imag(nan);
    a = cval;
    EXPECT_TRUE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), runtime_error);
    EXPECT_THROW((a <= a), runtime_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), runtime_error);
    EXPECT_THROW((a > a), runtime_error);

    // NaNs compared against non-NaNs
    cval.real(nan);
    cval.imag(0.f);
    a = cval;
    b = complex<float>(0.f, 1.f);
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(0.f);
    cval.imag(nan);
    a = cval;
    b = complex<float>(1.f, 1.f);
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(nan);
    cval.imag(nan);
    a = cval;
    b = complex<float>(1.f, 1.f);
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    // NaNs compared against NaNs
    cval.real(nan);
    cval.imag(0.f);
    a = cval;
    cval.real(nan);
    cval.imag(1.f);
    b = cval;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(0.f);
    cval.imag(nan);
    a = cval;
    cval.real(nan);
    cval.imag(1.f);
    b = cval;
    EXPECT_FALSE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(nan);
    cval.imag(nan);
    a = cval;
    cval.real(nan);
    cval.imag(1.f);
    b = cval;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(0.f);
    cval.imag(nan);
    a = cval;
    cval.real(1.f);
    cval.imag(nan);
    b = cval;
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);

    cval.real(nan);
    cval.imag(nan);
    a = cval;
    cval.real(0.f);
    cval.imag(nan);
    b = cval;
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), runtime_error);
    EXPECT_THROW((a <= b), runtime_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), runtime_error);
    EXPECT_THROW((a > b), runtime_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), runtime_error);
    EXPECT_THROW((b <= a), runtime_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), runtime_error);
    EXPECT_THROW((b > a), runtime_error);
}

