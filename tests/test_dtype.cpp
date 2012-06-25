//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include <dnd/dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dnd;

TEST(DType, BasicConstructor) {
    dtype d;

    // Default-constructed dtype properties
    EXPECT_EQ(pattern_type_id, d.type_id());
    EXPECT_EQ(pattern_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(0u, d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // bool dtype
    d = dtype(bool_type_id);
    EXPECT_EQ(bool_type_id, d.type_id());
    EXPECT_EQ(bool_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1u, d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // int8 dtype
    d = dtype(int8_type_id);
    EXPECT_EQ(int8_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1u, d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // int16 dtype
    d = dtype(int16_type_id);
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(2, d.alignment());
    EXPECT_EQ(2u, d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // int32 dtype
    d = dtype(int32_type_id);
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(4, d.alignment());
    EXPECT_EQ(4u, d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // int
    d = make_dtype<int>();
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(int), d.alignment());
    EXPECT_EQ(sizeof(int), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // long
    d = make_dtype<long>();
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(long), d.alignment());
    EXPECT_EQ(sizeof(long), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // long long
    d = make_dtype<long long>();
    EXPECT_EQ(int64_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(long long), d.alignment());
    EXPECT_EQ(sizeof(long long), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // unsigned int
    d = make_dtype<unsigned int>();
    EXPECT_EQ(uint32_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned int), d.alignment());
    EXPECT_EQ(sizeof(unsigned int), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long
    d = make_dtype<unsigned long>();
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long long
    d = make_dtype<unsigned long long>();
    EXPECT_EQ(uint64_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned long long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long long), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // float
    d = make_dtype<float>();
    EXPECT_EQ(float32_type_id, d.type_id());
    EXPECT_EQ(real_kind, d.kind());
    EXPECT_EQ((int)sizeof(float), d.alignment());
    EXPECT_EQ(sizeof(float), d.element_size());
    EXPECT_EQ(NULL, d.extended());

    // double
    d = make_dtype<double>();
    EXPECT_EQ(float64_type_id, d.type_id());
    EXPECT_EQ(real_kind, d.kind());
    EXPECT_EQ((int)sizeof(double), d.alignment());
    EXPECT_EQ(sizeof(double), d.element_size());
    EXPECT_EQ(NULL, d.extended());
}


TEST(DType, BasicComparison) {
    dtype d;
    float f1, f2;
    double d1, d2;
    int32_t i32_1, i32_2;
    int64_t i64_1, i64_2;

    // float32 dtype
    d = dtype(float32_type_id);
    f1 = 1.0; f2 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&f1, (char *)&f2, NULL), false);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&f1, (char *)&f2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&f1, (char *)&f2, NULL), false);
    f1 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&f1, (char *)&f2, NULL), false);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&f1, (char *)&f2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&f1, (char *)&f2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&f1, (char *)&f2, NULL), false);

    // float64 dtype
    d = dtype(float64_type_id);
    d1 = 1.0; d2 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&d1, (char *)&d2, NULL), false);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&d1, (char *)&d2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&d1, (char *)&d2, NULL), false);
    d1 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&d1, (char *)&d2, NULL), false);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&d1, (char *)&d2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&d1, (char *)&d2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&d1, (char *)&d2, NULL), false);

    // int32 dtype
    d = dtype(int32_type_id);
    i32_1 = 1.0; i32_2 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&i32_1, (char *)&i32_2, NULL), false);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&i32_1, (char *)&i32_2, NULL), false);
    i32_1 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&i32_1, (char *)&i32_2, NULL), false);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&i32_1, (char *)&i32_2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&i32_1, (char *)&i32_2, NULL), false);

    // int64 dtype
    d = dtype(int64_type_id);
    i64_1 = 1.0; i64_2 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
    i64_1 = 2.0;
    EXPECT_EQ(d.get_comparison(less_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
    EXPECT_EQ(d.get_comparison(less_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(equal_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(not_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
    EXPECT_EQ(d.get_comparison(greater_equal_id)((char *)&i64_1, (char *)&i64_2, NULL), true);
    EXPECT_EQ(d.get_comparison(greater_id)((char *)&i64_1, (char *)&i64_2, NULL), false);
}




