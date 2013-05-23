//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
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

