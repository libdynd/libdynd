//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(FixedDimDType, Create) {
    dtype d;
    const fixed_dim_dtype *fad;

    // Strings with various encodings and sizes
    d = make_fixed_dim_dtype(3, make_dtype<int32_t>());
    EXPECT_EQ(fixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(uniform_dim_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(12u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ(make_dtype<int32_t>(), d.p("element_dtype").as<dtype>());
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-3));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-2));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(-1));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(0));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(1));
    EXPECT_EQ(make_dtype<int32_t>(), d.at(2));
    fad = static_cast<const fixed_dim_dtype *>(d.extended());
    EXPECT_EQ(4, fad->get_fixed_stride());
    EXPECT_EQ(3u, fad->get_fixed_dim_size());

    d = make_fixed_dim_dtype(1, make_dtype<int32_t>());
    EXPECT_EQ(fixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(uniform_dim_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    fad = static_cast<const fixed_dim_dtype *>(d.extended());
    EXPECT_EQ(0, fad->get_fixed_stride());
    EXPECT_EQ(1u, fad->get_fixed_dim_size());
}

TEST(FixedDimDType, CreateCOrder) {
    intptr_t shape[3] = {2, 3, 4};
    dtype d = make_fixed_dim_dtype(3, shape, make_dtype<int16_t>(), NULL);
    EXPECT_EQ(fixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(make_fixed_dim_dtype(2, shape+1, make_dtype<int16_t>(), NULL), d.at(0));
    EXPECT_EQ(make_fixed_dim_dtype(1, shape+2, make_dtype<int16_t>(), NULL), d.at(0,0));
    EXPECT_EQ(make_dtype<int16_t>(), d.at(0,0,0));
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2u, static_cast<const fixed_dim_dtype *>(d.extended())->get_fixed_dim_size());
    EXPECT_EQ(24, static_cast<const fixed_dim_dtype *>(d.extended())->get_fixed_stride());
    EXPECT_EQ(3u, static_cast<const fixed_dim_dtype *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(8, static_cast<const fixed_dim_dtype *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4u, static_cast<const fixed_dim_dtype *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(2, static_cast<const fixed_dim_dtype *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(FixedDimDType, CreateFOrder) {
    int axis_perm[3] = {0, 1, 2};
    intptr_t shape[3] = {2, 3, 4};
    dtype d = make_fixed_dim_dtype(3, shape, make_dtype<int16_t>(), axis_perm);
    EXPECT_EQ(48u, d.get_data_size());
    EXPECT_EQ(fixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(fixed_dim_type_id, d.at(0).get_type_id());
    EXPECT_EQ(fixed_dim_type_id, d.at(0,0).get_type_id());
    EXPECT_EQ(int16_type_id, d.at(0,0,0).get_type_id());
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2u, static_cast<const fixed_dim_dtype *>(d.extended())->get_fixed_dim_size());
    EXPECT_EQ(2, static_cast<const fixed_dim_dtype *>(d.extended())->get_fixed_stride());
    EXPECT_EQ(3u, static_cast<const fixed_dim_dtype *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(4, static_cast<const fixed_dim_dtype *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4u, static_cast<const fixed_dim_dtype *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(12, static_cast<const fixed_dim_dtype *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(FixedDimDType, Basic) {
    ndobject a;
    float vals[3] = {1.5f, 2.5f, -1.5f};

    a = empty(make_fixed_dim_dtype(3, make_dtype<float>()));
    a.vals() = vals;

    EXPECT_EQ(make_fixed_dim_dtype(3, make_dtype<float>()), a.get_dtype());
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


TEST(FixedDimDType, AssignKernel_ScalarToFixed) {
    ndobject a, b;
    assignment_kernel k;

    // Assignment scalar -> fixed array
    a = empty(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = 9.0;
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a.at(0).as<int>());
    EXPECT_EQ(9, a.at(1).as<int>());
    EXPECT_EQ(9, a.at(2).as<int>());
}

TEST(FixedDimDType, AssignKernel_FixedToFixed) {
    ndobject a, b;
    assignment_kernel k;

    // Assignment fixed array -> fixed array
    a = empty(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
}

TEST(FixedDimDType, AssignKernel_FixedToScalarError) {
    ndobject a, b;
    assignment_kernel k;

    // Assignment fixed array -> scalar
    a = 9.0;
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    EXPECT_THROW(make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context),
                broadcast_error);
}

TEST(FixedDimDType, AssignFixedStridedKernel) {
    ndobject a, b;
    assignment_kernel k;
    int vals_int[] = {3,5,7};
    int vals_int_single[] = {9};

    // Assignment strided array -> fixed array
    a = empty(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = vals_int;
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Broadcasting assignment strided array -> fixed array
    a = empty(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = vals_int_single;
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a.at(0).as<int>());
    EXPECT_EQ(9, a.at(1).as<int>());
    EXPECT_EQ(9, a.at(2).as<int>());
    k.reset();

    // Assignment fixed array -> strided array
    a = make_strided_ndobject(3, make_dtype<float>());
    a.vals() = 0;
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Broadcasting assignment fixed array -> strided array
    a = make_strided_ndobject(3, make_dtype<float>());
    a.vals() = 0;
    b = parse_json("1, int32", "[9]");
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(),
                    kernel_request_single, assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a.at(0).as<int>());
    EXPECT_EQ(9, a.at(1).as<int>());
    EXPECT_EQ(9, a.at(2).as<int>());
    k.reset();
}

