//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "inc_gtest.hpp"

#include "dynd/type_promotion.hpp"

using namespace std;
using namespace dynd;

template<class S, class T, class U>
void type_promotion_matches_cxx_test(S, T, U) {
    EXPECT_EQ(ndt::make_type<U>(), promote_types_arithmetic(ndt::make_type<S>(), ndt::make_type<T>()));
    if (ndt::make_type<U>() != promote_types_arithmetic(ndt::make_type<S>(), ndt::make_type<T>()))
        cout << "S: " << ndt::make_type<S>() << ", T: " << ndt::make_type<T>() << ", U: " << ndt::make_type<U>() << "\n";
}

template<class S, class T>
void type_promotion_matches_cxx() {
    S a = S();
    T b = T();
    type_promotion_matches_cxx_test(a, b, a + b);
}

#define TEST_ALL_SECOND(first) \
    type_promotion_matches_cxx<first, dynd_bool>(); \
    type_promotion_matches_cxx<first, int8_t>(); \
    type_promotion_matches_cxx<first, int16_t>(); \
    type_promotion_matches_cxx<first, int32_t>(); \
    type_promotion_matches_cxx<first, int64_t>(); \
    type_promotion_matches_cxx<first, uint8_t>(); \
    type_promotion_matches_cxx<first, uint16_t>(); \
    type_promotion_matches_cxx<first, uint32_t>(); \
    type_promotion_matches_cxx<first, uint64_t>(); \
    type_promotion_matches_cxx<first, float>(); \
    type_promotion_matches_cxx<first, double>()

#define TEST_ALL_FIRST() \
    TEST_ALL_SECOND(dynd_bool); \
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

TEST(DTypePromotion, IntWithComplex) {
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int8_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int16_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int32_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int64_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint8_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint16_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint32_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint64_t>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());

    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int8_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int16_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int32_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<int64_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint8_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint16_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint32_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<uint64_t>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
}

TEST(DTypePromotion, FloatWithComplex) {
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<float>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<float>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<double>(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<double>(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<float> >(), ndt::make_type<float>()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<double> >(), ndt::make_type<float>()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<float> >(), ndt::make_type<double>()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<double> >(), ndt::make_type<double>()), ndt::make_type<dynd_complex<double> >());
}

TEST(DTypePromotion, ComplexWithComplex) {
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<float> >(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<float> >(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<double> >(), ndt::make_type<dynd_complex<float> >()), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(promote_types_arithmetic(ndt::make_type<dynd_complex<double> >(), ndt::make_type<dynd_complex<double> >()), ndt::make_type<dynd_complex<double> >());
}
