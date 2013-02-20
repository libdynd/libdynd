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