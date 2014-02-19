//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/date_type.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayCompare, Bool) {
    // Equality
    EXPECT_TRUE(nd::array(true) == nd::array(true));
    EXPECT_TRUE(nd::array(false) == nd::array(false));
    EXPECT_FALSE(nd::array(true) == nd::array(false));
    EXPECT_FALSE(nd::array(false) == nd::array(true));
    // Inequality
    EXPECT_FALSE(nd::array(true) != nd::array(true));
    EXPECT_FALSE(nd::array(false) != nd::array(false));
    EXPECT_TRUE(nd::array(true) != nd::array(false));
    EXPECT_TRUE(nd::array(false) != nd::array(true));
    // Comparison for sorting
    EXPECT_TRUE(nd::array(false).op_sorting_less(nd::array(true)));
    EXPECT_FALSE(nd::array(false).op_sorting_less(nd::array(false)));
    EXPECT_FALSE(nd::array(true).op_sorting_less(nd::array(true)));
    EXPECT_FALSE(nd::array(true).op_sorting_less(nd::array(false)));
    // Other comparisons are not permitted
    EXPECT_THROW((nd::array(false) < nd::array(true)), not_comparable_error);
    EXPECT_THROW((nd::array(false) <= nd::array(true)), not_comparable_error);
    EXPECT_THROW((nd::array(false) >= nd::array(true)), not_comparable_error);
    EXPECT_THROW((nd::array(false) > nd::array(true)), not_comparable_error);
    // Compare Bool with other types
    EXPECT_TRUE(nd::array(true) == nd::array(1));
    EXPECT_TRUE(nd::array(true) == nd::array(1.f));
    EXPECT_TRUE(nd::array(true) == nd::array(1.0));
    EXPECT_TRUE(nd::array(true) == nd::array(dynd_complex<double>(1.0)));
    EXPECT_TRUE(nd::array(false) == nd::array(0));
    EXPECT_TRUE(nd::array(false) == nd::array(0.f));
    EXPECT_TRUE(nd::array(false) == nd::array(0.0));
    EXPECT_TRUE(nd::array(false) == nd::array(dynd_complex<double>()));
    EXPECT_TRUE(nd::array(true) != nd::array(2));
    EXPECT_TRUE(nd::array(true) != nd::array(2.f));
    EXPECT_TRUE(nd::array(true) != nd::array(2.0));
    EXPECT_TRUE(nd::array(true) != nd::array(dynd_complex<double>(1,1)));
    EXPECT_TRUE(nd::array(false) != nd::array(-1));
    EXPECT_TRUE(nd::array(false) != nd::array(-1.f));
    EXPECT_TRUE(nd::array(false) != nd::array(-1.0));
    EXPECT_TRUE(nd::array(false) != nd::array(dynd_complex<double>(0,1)));
}

TEST(ArrayCompare, EqualityIntUInt) {
    // Equality
    EXPECT_TRUE(nd::array((uint8_t)127) == nd::array((int8_t)127));
    EXPECT_TRUE(nd::array((int8_t)127) == nd::array((uint8_t)127));
    EXPECT_TRUE(nd::array((uint16_t)32767) == nd::array((int16_t)32767));
    EXPECT_TRUE(nd::array((int16_t)32767) == nd::array((uint16_t)32767));
    EXPECT_TRUE(nd::array((uint32_t)2147483647) == nd::array((int32_t)2147483647));
    EXPECT_TRUE(nd::array((int32_t)2147483647) == nd::array((uint32_t)2147483647));
    EXPECT_TRUE(nd::array((uint64_t)9223372036854775807LL) == nd::array((int64_t)9223372036854775807LL));
    EXPECT_TRUE(nd::array((int64_t)9223372036854775807LL) == nd::array((uint64_t)9223372036854775807LL));
    // Inequality
    EXPECT_FALSE(nd::array((uint8_t)127) != nd::array((int16_t)127));
    EXPECT_FALSE(nd::array((int8_t)127) != nd::array((uint16_t)127));
    EXPECT_FALSE(nd::array((uint16_t)32767) != nd::array((int16_t)32767));
    EXPECT_FALSE(nd::array((int16_t)32767) != nd::array((uint16_t)32767));
    EXPECT_FALSE(nd::array((uint32_t)2147483647) != nd::array((int32_t)2147483647));
    EXPECT_FALSE(nd::array((int32_t)2147483647) != nd::array((uint32_t)2147483647));
    EXPECT_FALSE(nd::array((uint64_t)9223372036854775807LL) != nd::array((int64_t)9223372036854775807LL));
    EXPECT_FALSE(nd::array((int64_t)9223372036854775807LL) != nd::array((uint64_t)9223372036854775807LL));
    // Equality with same bits
    EXPECT_FALSE(nd::array((uint8_t)255) == nd::array((int8_t)-1));
    EXPECT_FALSE(nd::array((int8_t)-1) == nd::array((uint8_t)255));
    EXPECT_FALSE(nd::array((uint16_t)65535) == nd::array((int16_t)-1));
    EXPECT_FALSE(nd::array((int16_t)-1) == nd::array((uint16_t)65535));
    EXPECT_FALSE(nd::array((uint32_t)4294967295u) == nd::array((int32_t)-1));
    EXPECT_FALSE(nd::array((int32_t)-1) == nd::array((uint32_t)4294967295u));
    EXPECT_FALSE(nd::array((uint64_t)18446744073709551615ULL) == nd::array((int64_t)-1));
    EXPECT_FALSE(nd::array((int64_t)-1) == nd::array((uint64_t)18446744073709551615ULL));
    // Inequality with same bits
    EXPECT_TRUE(nd::array((uint8_t)255) != nd::array((int8_t)-1));
    EXPECT_TRUE(nd::array((int8_t)-1) != nd::array((uint8_t)255));
    EXPECT_TRUE(nd::array((uint16_t)65535) != nd::array((int16_t)-1));
    EXPECT_TRUE(nd::array((int16_t)-1) != nd::array((uint16_t)65535));
    EXPECT_TRUE(nd::array((uint32_t)4294967295u) != nd::array((int32_t)-1));
    EXPECT_TRUE(nd::array((int32_t)-1) != nd::array((uint32_t)4294967295u));
    EXPECT_TRUE(nd::array((uint64_t)18446744073709551615ULL) != nd::array((int64_t)-1));
    EXPECT_TRUE(nd::array((int64_t)-1) != nd::array((uint64_t)18446744073709551615ULL));
}

TEST(ArrayCompare, InequalityInt8UInt8) {
    nd::array a, b;

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

TEST(ArrayCompare, InequalityInt64UInt64) {
    nd::array a, b;

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

TEST(ArrayCompare, EqualityIntFloat) {
    nd::array a, b;

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

TEST(ArrayCompare, EqualityUIntFloat) {
    nd::array a, b;

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

TEST(ArrayCompare, NaNFloat32) {
    nd::array a, b;
    nd::array nan = nd::array("nan").ucast<float>().eval();
    nd::array pinf = nd::array("inf").ucast<float>().eval();

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

TEST(ArrayCompare, NaNFloat64) {
    nd::array a, b;
    nd::array nan = nd::array("nan").ucast<double>().eval();
    nd::array pinf = nd::array("inf").ucast<double>().eval();

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

TEST(ArrayCompare, ComplexFloat32) {
    nd::array a, b;
    // For complex, op_sorting_less is lexicographic,
    // and other inequalities raise exceptions.

    // Compare 0 with 0
    a = dynd_complex<float>();
    b = dynd_complex<float>();
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Compare 0+1j with 1+1j
    a = dynd_complex<float>(0, 1);
    b = dynd_complex<float>(1, 1);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Compare 0+1j with 0+2j
    a = dynd_complex<float>(0, 1);
    b = dynd_complex<float>(0, 2);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);
}

TEST(ArrayCompare, ComplexFloat64) {
    nd::array a, b;
    // For complex, op_sorting_less is lexicographic,
    // and other inequalities raise exceptions.

    // Compare 0 with 0
    a = dynd_complex<double>();
    b = dynd_complex<double>();
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Compare 0+1j with 1+1j
    a = dynd_complex<double>(0, 1);
    b = dynd_complex<double>(1, 1);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // Compare 0+1j with 0+2j
    a = dynd_complex<double>(0, 1);
    b = dynd_complex<double>(0, 2);
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);
}

TEST(ArrayCompare, NaNComplexFloat32) {
    // The strange way to assign complex values is because
    // on clang, dynd_complex<float>(0.f, nan) was creating (nan, nan)
    // instead of the requested complex.
    float cval[2];
    nd::array a, b;
    a = nd::empty(ndt::make_type<dynd_complex<float> >());
    b = nd::empty(ndt::make_type<dynd_complex<float> >());
    float nan = nd::array("nan").ucast<float>().as<float>();

    // real component NaN, compared against itself
    cval[0] = nan;
    cval[1] = 0.f;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_TRUE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_FALSE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), not_comparable_error);
    EXPECT_THROW((a <= a), not_comparable_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), not_comparable_error);
    EXPECT_THROW((a > a), not_comparable_error);

    // imaginary component NaN, compared against itself
    cval[0] = 0.f;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), not_comparable_error);
    EXPECT_THROW((a <= a), not_comparable_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), not_comparable_error);
    EXPECT_THROW((a > a), not_comparable_error);

    // both components NaN, compared against itself
    cval[0] = nan;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_TRUE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_FALSE(a.op_sorting_less(a));
    EXPECT_THROW((a < a), not_comparable_error);
    EXPECT_THROW((a <= a), not_comparable_error);
    EXPECT_FALSE(a == a);
    EXPECT_TRUE(a != a);
    EXPECT_THROW((a >= a), not_comparable_error);
    EXPECT_THROW((a > a), not_comparable_error);

    // NaNs compared against non-NaNs
    cval[0] = nan;
    cval[1] = 0.f;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = 0.f;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = 0.f;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = 1.f;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = nan;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = 1.f;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    // NaNs compared against NaNs
    cval[0] = nan;
    cval[1] = 0.f;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = nan;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = 0.f;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = nan;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(DYND_ISNAN(a.p("real").as<float>()));
    EXPECT_TRUE(DYND_ISNAN(a.p("imag").as<float>()));
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = nan;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = nan;
    cval[1] = 1.f;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = 0.f;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = 1.f;
    cval[1] = nan;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_TRUE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_FALSE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);

    cval[0] = nan;
    cval[1] = nan;
    a.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    cval[0] = 0.f;
    cval[1] = nan;
    b.val_assign(ndt::make_type<dynd_complex<float> >(), NULL, reinterpret_cast<const char *>(&cval[0]));
    EXPECT_FALSE(a.op_sorting_less(b));
    EXPECT_THROW((a < b), not_comparable_error);
    EXPECT_THROW((a <= b), not_comparable_error);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_THROW((a >= b), not_comparable_error);
    EXPECT_THROW((a > b), not_comparable_error);
    EXPECT_TRUE(b.op_sorting_less(a));
    EXPECT_THROW((b < a), not_comparable_error);
    EXPECT_THROW((b <= a), not_comparable_error);
    EXPECT_FALSE(b == a);
    EXPECT_TRUE(b != a);
    EXPECT_THROW((b >= a), not_comparable_error);
    EXPECT_THROW((b > a), not_comparable_error);
}

TEST(ArrayCompare, ExpressionDType) {
    nd::array a, b;
    // One expression operand
    a = nd::array(3).ucast<float>();
    b = nd::array(4.0);
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

    // Two expression operand
    a = nd::array(3).ucast<float>();
    b = nd::array("2012-03-04").ucast(ndt::make_date()).p("day");
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

    // Non-comparable operands should raise
    a = nd::array(3).ucast<dynd_complex<float> >();
    b = nd::array(4.0);
    EXPECT_THROW((a < b), not_comparable_error);
}
