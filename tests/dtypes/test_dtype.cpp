//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DType, BasicConstructor) {
    dtype d;

    // Default-constructed dtype properties
    EXPECT_EQ(uninitialized_type_id, d.get_type_id());
    EXPECT_EQ(void_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(0u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // void dtype
    d = dtype(void_type_id);
    EXPECT_EQ(void_type_id, d.get_type_id());
    EXPECT_EQ(void_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(0u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // bool dtype
    d = dtype(bool_type_id);
    EXPECT_EQ(bool_type_id, d.get_type_id());
    EXPECT_EQ(bool_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(1u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // int8 dtype
    d = dtype(int8_type_id);
    EXPECT_EQ(int8_type_id, d.get_type_id());
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(1u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // int16 dtype
    d = dtype(int16_type_id);
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_alignment());
    EXPECT_EQ(2u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // int32 dtype
    d = dtype(int32_type_id);
    EXPECT_EQ(int32_type_id, d.get_type_id());
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // int
    d = make_dtype<int>();
    EXPECT_EQ(int32_type_id, d.get_type_id());
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(sizeof(int), d.get_alignment());
    EXPECT_EQ(sizeof(int), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // long
    d = make_dtype<long>();
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(sizeof(long), d.get_alignment());
    EXPECT_EQ(sizeof(long), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // long long
    d = make_dtype<long long>();
    EXPECT_EQ(int64_type_id, d.get_type_id());
    EXPECT_EQ(int_kind, d.get_kind());
    EXPECT_EQ(sizeof(long long), d.get_alignment());
    EXPECT_EQ(sizeof(long long), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // unsigned int
    d = make_dtype<unsigned int>();
    EXPECT_EQ(uint32_type_id, d.get_type_id());
    EXPECT_EQ(uint_kind, d.get_kind());
    EXPECT_EQ(sizeof(unsigned int), d.get_alignment());
    EXPECT_EQ(sizeof(unsigned int), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // unsigned long
    d = make_dtype<unsigned long>();
    EXPECT_EQ(uint_kind, d.get_kind());
    EXPECT_EQ(sizeof(unsigned long), d.get_alignment());
    EXPECT_EQ(sizeof(unsigned long), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // unsigned long long
    d = make_dtype<unsigned long long>();
    EXPECT_EQ(uint64_type_id, d.get_type_id());
    EXPECT_EQ(uint_kind, d.get_kind());
    EXPECT_EQ(sizeof(unsigned long long), d.get_alignment());
    EXPECT_EQ(sizeof(unsigned long long), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // float
    d = make_dtype<float>();
    EXPECT_EQ(float32_type_id, d.get_type_id());
    EXPECT_EQ(real_kind, d.get_kind());
    EXPECT_EQ(sizeof(float), d.get_alignment());
    EXPECT_EQ(sizeof(float), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());

    // double
    d = make_dtype<double>();
    EXPECT_EQ(float64_type_id, d.get_type_id());
    EXPECT_EQ(real_kind, d.get_kind());
    EXPECT_EQ(sizeof(double), d.get_alignment());
    EXPECT_EQ(sizeof(double), d.get_data_size());
    EXPECT_TRUE(d.is_builtin());
}

#define TEST_COMPARISONS(type_id, type, lhs, rhs) \
    { \
        kernel_instance<compare_operations_t> k; \
        dtype d = dtype(type_id); \
        d.get_single_compare_kernel(k); \
        type v1 = lhs; type v2 = rhs; \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::less_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs < (type)rhs); \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::less_equal_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs <= (type)rhs); \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::equal_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs == (type)rhs); \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::not_equal_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs != (type)rhs); \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::greater_equal_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs >= (type)rhs); \
        EXPECT_EQ(k.kernel.ops[compare_operations_t::greater_id]((char *)&v1, (char *)&v2, &k.extra), (type)lhs > (type)rhs); \
    }


TEST(DType, SingleCompareBool) {
    TEST_COMPARISONS(bool_type_id, bool, 0, 1)
    TEST_COMPARISONS(bool_type_id, bool, 0, 0)
    TEST_COMPARISONS(bool_type_id, bool, 1, 0)
    TEST_COMPARISONS(bool_type_id, bool, 1, 1)
}


TEST(DType, SingleCompareInt) {
    TEST_COMPARISONS(int8_type_id, int8_t, 1, 2)
    TEST_COMPARISONS(int8_type_id, int8_t, 2, 2)
    TEST_COMPARISONS(int8_type_id, int8_t, 1, 0)
    TEST_COMPARISONS(int8_type_id, int8_t, -1, 0)
    TEST_COMPARISONS(int8_type_id, int8_t, -1, -1)
    TEST_COMPARISONS(int8_type_id, int8_t, -1, -2)

    TEST_COMPARISONS(int16_type_id, int16_t, 1, 2)
    TEST_COMPARISONS(int16_type_id, int16_t, 2, 2)
    TEST_COMPARISONS(int16_type_id, int16_t, 1, 0)
    TEST_COMPARISONS(int16_type_id, int16_t, -1, 0)
    TEST_COMPARISONS(int16_type_id, int16_t, -1, -1)
    TEST_COMPARISONS(int16_type_id, int16_t, -1, -2)

    TEST_COMPARISONS(int32_type_id, int32_t, 1, 2)
    TEST_COMPARISONS(int32_type_id, int32_t, 2, 2)
    TEST_COMPARISONS(int32_type_id, int32_t, 1, 0)
    TEST_COMPARISONS(int32_type_id, int32_t, -1, 0)
    TEST_COMPARISONS(int32_type_id, int32_t, -1, -1)
    TEST_COMPARISONS(int32_type_id, int32_t, -1, -2)

    TEST_COMPARISONS(int64_type_id, int64_t, 1, 2)
    TEST_COMPARISONS(int64_type_id, int64_t, 2, 2)
    TEST_COMPARISONS(int64_type_id, int64_t, 1, 0)
    TEST_COMPARISONS(int64_type_id, int64_t, -1, 0)
    TEST_COMPARISONS(int64_type_id, int64_t, -1, -1)
    TEST_COMPARISONS(int64_type_id, int64_t, -1, -2)
}

TEST(DType, SingleCompareUInt) {
    TEST_COMPARISONS(uint8_type_id, uint8_t, 1, 2)
    TEST_COMPARISONS(uint8_type_id, uint8_t, 2, 2)
    TEST_COMPARISONS(uint8_type_id, uint8_t, 1, 0)

    TEST_COMPARISONS(uint16_type_id, uint16_t, 1, 2)
    TEST_COMPARISONS(uint16_type_id, uint16_t, 2, 2)
    TEST_COMPARISONS(uint16_type_id, uint16_t, 1, 0)

    TEST_COMPARISONS(uint32_type_id, uint32_t, 1, 2)
    TEST_COMPARISONS(uint32_type_id, uint32_t, 2, 2)
    TEST_COMPARISONS(uint32_type_id, uint32_t, 1, 0)

    TEST_COMPARISONS(uint64_type_id, uint64_t, 1, 2)
    TEST_COMPARISONS(uint64_type_id, uint64_t, 2, 2)
    TEST_COMPARISONS(uint64_type_id, uint64_t, 1, 0)
}

TEST(DType, SingleCompareFloat) {
    TEST_COMPARISONS(float32_type_id, float, 1.0, 2.0)
    TEST_COMPARISONS(float32_type_id, float, 2.0, 2.0)
    TEST_COMPARISONS(float32_type_id, float, 1.0, 0.0)
    TEST_COMPARISONS(float32_type_id, float, -1.0, 0.0)
    TEST_COMPARISONS(float32_type_id, float, -1.0, -1.0)
    TEST_COMPARISONS(float32_type_id, float, -1.0, -2.0)

    TEST_COMPARISONS(float64_type_id, double, 1.0, 2.0)
    TEST_COMPARISONS(float64_type_id, double, 2.0, 2.0)
    TEST_COMPARISONS(float64_type_id, double, 1.0, 0.0)
    TEST_COMPARISONS(float64_type_id, double, -1.0, 0.0)
    TEST_COMPARISONS(float64_type_id, double, -1.0, -1.0)
    TEST_COMPARISONS(float64_type_id, double, -1.0, -2.0)
}
#undef TEST_COMPARISONS

// TODO: ordered comparisons for complex numbers
#define TEST_COMPLEX_COMPARISONS(type_id, type, lhs, rhs) \
    { \
        kernel_instance<compare_operations_t> k; \
        dtype d = dtype(type_id); \
        d.get_single_compare_kernel(k); \
        type v1 = lhs; type v2 = rhs; \
        EXPECT_EQ(lhs == rhs, k.kernel.ops[compare_operations_t::equal_id]((char *)&v1, (char *)&v2, &k.extra)); \
        EXPECT_EQ(lhs != rhs, k.kernel.ops[compare_operations_t::not_equal_id]((char *)&v1, (char *)&v2, &k.extra)); \
    }

TEST(DType, SingleCompareComplex) {
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(1.0), complex<float>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(2.0), complex<float>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(1.0), complex<float>(0.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(-1.0), complex<float>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(-2.0), complex<float>(-2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(-1.0), complex<float>(0.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, 1.0), complex<float>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, 2.0), complex<float>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, 1.0), complex<float>(0.0, 0.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, -1.0), complex<float>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, -2.0), complex<float>(0.0, -2.0))
    TEST_COMPLEX_COMPARISONS(complex_float32_type_id, complex<float>, complex<float>(0.0, -1.0), complex<float>(0.0, 0.0))

    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(1.0), complex<double>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(2.0), complex<double>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(1.0), complex<double>(0.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(-1.0), complex<double>(2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(-2.0), complex<double>(-2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(-1.0), complex<double>(0.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, 1.0), complex<double>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, 2.0), complex<double>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, 1.0), complex<double>(0.0, 0.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, -1.0), complex<double>(0.0, 2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, -2.0), complex<double>(0.0, -2.0))
    TEST_COMPLEX_COMPARISONS(complex_float64_type_id, complex<double>, complex<double>(0.0, -1.0), complex<double>(0.0, 0.0))
}
#undef TEST_COMPLEX_COMPARISONS

