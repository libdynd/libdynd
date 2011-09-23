#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/dtype_promotion.hpp"

using namespace std;
using namespace dnd;

template<class S, class T, class U>
void dtype_promotion_matches_cxx_test(S, T, U) {
    //cout << "S: " << make_dtype<S>() << ", T: " << make_dtype<T>() << ", U: " << make_dtype<U>() << "\n";
    EXPECT_EQ(promote_dtypes_arithmetic(make_dtype<S>(), make_dtype<T>()), make_dtype<U>());
}

template<class S, class T>
void dtype_promotion_matches_cxx() {
    S a = S();
    T b = T();
    dtype_promotion_matches_cxx_test(a, b, a + b);
}

#define TEST_ALL_SECOND(first) \
    dtype_promotion_matches_cxx<first, dnd_bool>(); \
    dtype_promotion_matches_cxx<first, int8_t>(); \
    dtype_promotion_matches_cxx<first, int16_t>(); \
    dtype_promotion_matches_cxx<first, int32_t>(); \
    dtype_promotion_matches_cxx<first, int64_t>(); \
    dtype_promotion_matches_cxx<first, uint8_t>(); \
    dtype_promotion_matches_cxx<first, uint16_t>(); \
    dtype_promotion_matches_cxx<first, uint32_t>(); \
    dtype_promotion_matches_cxx<first, uint64_t>(); \
    dtype_promotion_matches_cxx<first, float>(); \
    dtype_promotion_matches_cxx<first, double>()

#define TEST_ALL_FIRST() \
    TEST_ALL_SECOND(dnd_bool); \
    TEST_ALL_SECOND(int8_t); \
    TEST_ALL_SECOND(int16_t); \
    TEST_ALL_SECOND(int32_t); \
    TEST_ALL_SECOND(int64_t); \
    TEST_ALL_SECOND(uint8_t); \
    TEST_ALL_SECOND(uint16_t); \
    TEST_ALL_SECOND(uint32_t); \
    TEST_ALL_SECOND(uint64_t); \
    TEST_ALL_SECOND(float); \
    TEST_ALL_SECOND(double)

TEST(DTypePromotion, MatchesCxx) {
    TEST_ALL_FIRST();
}

#undef TEST_ALL_FIRST
#undef TEST_ALL_SECOND
