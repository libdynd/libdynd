//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DTypeDType, Create) {
    dtype d;

    // Strings with various encodings and sizes
    d = make_dtype_dtype();
    EXPECT_EQ(dtype_type_id, d.get_type_id());
    EXPECT_EQ(custom_kind, d.get_kind());
    EXPECT_EQ(sizeof(const base_dtype *), d.get_alignment());
    EXPECT_EQ(sizeof(const base_dtype *), d.get_data_size());
    EXPECT_FALSE(d.is_expression());
}

TEST(DTypeDType, BasicNDobject) {
    ndobject a;

    a = dtype("int32");
    EXPECT_EQ(dtype_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(make_dtype<int32_t>(), a.as<dtype>());
}

TEST(DTypeDType, BasicRefCount) {
    ndobject a;
    dtype d, d2;

    a = empty(make_dtype_dtype());
    d = make_strided_dim_dtype(make_dtype<int>());
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    d2 = a.as<dtype>();
    EXPECT_EQ(3, d.extended()->get_use_count());
    d2 = dtype();
    EXPECT_EQ(2, d.extended()->get_use_count());
    // Assigning a new value in the ndobject should free the reference in 'a'
    a.vals() = dtype();
    EXPECT_EQ(1, d.extended()->get_use_count());
    a.vals() = d;
    EXPECT_EQ(2, d.extended()->get_use_count());
    // Assigning a new reference to 'a' should free the reference when
    // destructing the existing 'a'
    a = 1.0;
    EXPECT_EQ(1, d.extended()->get_use_count());
}