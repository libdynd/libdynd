//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/tuple_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

TEST(StridedArrayDType, ReplaceScalarTypes) {
    dtype dafloat, dadouble, daint32;
    dafloat = make_strided_array_dtype(make_dtype<float>());
    dadouble = make_strided_array_dtype(make_dtype<double>());

    EXPECT_EQ(make_strided_array_dtype(make_convert_dtype<float, double>()),
            dadouble.with_replaced_scalar_types(make_dtype<float>()));

    // Two dimensional array
    dafloat = make_strided_array_dtype(dafloat);
    dadouble = make_strided_array_dtype(dadouble);

    EXPECT_EQ(make_strided_array_dtype(make_strided_array_dtype(make_convert_dtype<double, float>())),
            dafloat.with_replaced_scalar_types(make_dtype<double>()));
}

TEST(StridedArrayDType, DTypeAt) {
    dtype dfloat = make_dtype<float>();
    dtype darr1 = make_strided_array_dtype(dfloat);
    dtype darr2 = make_strided_array_dtype(darr1);
    dtype dtest;

    // indexing into a dtype with a slice produces another
    // strided array, so the dtype is unchanged.
    EXPECT_EQ(darr1, darr1.at(1 <= irange() < 3));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() < 3));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() < 3, irange() < 2));

    // Even if it's just one element, a slice still produces another array
    EXPECT_EQ(darr1, darr1.at(1 <= irange() <= 1));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() <= 1));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() <= 1, 2 <= irange() <= 2));

    // Indexing with an integer collapses a dimension
    EXPECT_EQ(dfloat, darr1.at(1));
    EXPECT_EQ(darr1, darr2.at(1));
    EXPECT_EQ(darr1, darr2.at(1 <= irange() <= 1, 1));
    EXPECT_EQ(dfloat, darr2.at(2, 1));

    // Should get an exception with too many indices
    EXPECT_THROW(dfloat.at(1), too_many_indices);
    EXPECT_THROW(darr1.at(1, 2), too_many_indices);
    EXPECT_THROW(darr2.at(1, 2, 3), too_many_indices);
}

TEST(StridedArrayDType, IsExpression) {
    dtype dfloat = make_dtype<float>();
    dtype darr1 = make_strided_array_dtype(dfloat);
    dtype darr2 = make_strided_array_dtype(darr1);

    EXPECT_FALSE(darr1.is_expression());
    EXPECT_FALSE(darr2.is_expression());

    dfloat = make_convert_dtype(make_dtype<double>(), dfloat);
    darr1 = make_strided_array_dtype(dfloat);
    darr2 = make_strided_array_dtype(darr1);

    EXPECT_TRUE(darr1.is_expression());
    EXPECT_TRUE(darr2.is_expression());
}