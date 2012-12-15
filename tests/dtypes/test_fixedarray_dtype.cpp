//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(FixedArrayDType, Create) {
    dtype d;
    const fixedarray_dtype *fad;

    // Strings with various encodings and sizes
    d = make_fixedarray_dtype(make_dtype<int32_t>(), 3);
    EXPECT_EQ(fixedarray_type_id, d.get_type_id());
    EXPECT_EQ(uniform_array_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(12u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-3));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-2));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-1));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(0));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(1));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(2));
    EXPECT_THROW(d.at(-4), index_out_of_bounds);
    EXPECT_THROW(d.at(3), index_out_of_bounds);
    fad = static_cast<const fixedarray_dtype *>(d.extended());
    EXPECT_EQ(4, fad->get_fixed_stride());
    EXPECT_EQ(3u, fad->get_fixed_dim_size());

    d = make_fixedarray_dtype(make_dtype<int32_t>(), 1);
    EXPECT_EQ(fixedarray_type_id, d.get_type_id());
    EXPECT_EQ(uniform_array_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    fad = static_cast<const fixedarray_dtype *>(d.extended());
    EXPECT_EQ(0, fad->get_fixed_stride());
    EXPECT_EQ(1u, fad->get_fixed_dim_size());
}

TEST(FixedArrayDType, CreateCOrder) {
    intptr_t shape[3] = {2, 3, 4};
    dtype d = make_fixedarray_dtype(make_dtype<int16_t>(), 3, shape, NULL);
    EXPECT_EQ(fixedarray_type_id, d.get_type_id());
    EXPECT_EQ(make_fixedarray_dtype(make_dtype<int16_t>(), 2, shape+1, NULL), d.at(0));
    EXPECT_EQ(make_fixedarray_dtype(make_dtype<int16_t>(), 1, shape+2, NULL), d.at(0,0));
    EXPECT_EQ(make_dtype<int16_t>(), d.at(0,0,0));
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2u, static_cast<const fixedarray_dtype *>(d.extended())->get_fixed_dim_size());
    EXPECT_EQ(24, static_cast<const fixedarray_dtype *>(d.extended())->get_fixed_stride());
    EXPECT_EQ(3u, static_cast<const fixedarray_dtype *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(8, static_cast<const fixedarray_dtype *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4u, static_cast<const fixedarray_dtype *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(2, static_cast<const fixedarray_dtype *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(FixedArrayDType, CreateFOrder) {
    int axis_perm[3] = {0, 1, 2};
    intptr_t shape[3] = {2, 3, 4};
    dtype d = make_fixedarray_dtype(make_dtype<int16_t>(), 3, shape, axis_perm);
    EXPECT_EQ(fixedarray_type_id, d.get_type_id());
    EXPECT_EQ(fixedarray_type_id, d.at(0).get_type_id());
    EXPECT_EQ(fixedarray_type_id, d.at(0,0).get_type_id());
    EXPECT_EQ(int16_type_id, d.at(0,0,0).get_type_id());
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2u, static_cast<const fixedarray_dtype *>(d.extended())->get_fixed_dim_size());
    EXPECT_EQ(2, static_cast<const fixedarray_dtype *>(d.extended())->get_fixed_stride());
    EXPECT_EQ(3u, static_cast<const fixedarray_dtype *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(4, static_cast<const fixedarray_dtype *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4u, static_cast<const fixedarray_dtype *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(12, static_cast<const fixedarray_dtype *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(FixedArrayDType, Basic) {
    ndobject a;
    float vals[3] = {1.5f, 2.5f, -1.5f};

    a = ndobject(make_fixedarray_dtype(make_dtype<float>(), 3));
    a.vals() = vals;

    EXPECT_EQ(make_fixedarray_dtype(make_dtype<float>(), 3), a.get_dtype());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(4, a.get_strides()[0]);
    EXPECT_EQ(1.5f, a.at(-3).as<float>());
    EXPECT_EQ(2.5f, a.at(-2).as<float>());
    EXPECT_EQ(-1.5f, a.at(-1).as<float>());
    EXPECT_EQ(1.5f, a.at(0).as<float>());
    EXPECT_EQ(2.5f, a.at(1).as<float>());
    EXPECT_EQ(-1.5f, a.at(2).as<float>());
    EXPECT_THROW(a.at(-4), index_out_of_bounds);
    EXPECT_THROW(a.at(3), index_out_of_bounds);
}
