//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(CFixedDimType, Create) {
    ndt::type d;
    const cfixed_dim_type *fad;

    // Strings with various encodings and sizes
    d = ndt::make_cfixed_dim(3, ndt::make_type<int32_t>());
    EXPECT_EQ(cfixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(dim_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_EQ(12u, d.get_data_size());
    EXPECT_EQ(1, d.get_ndim());
    EXPECT_EQ(1, d.get_strided_ndim());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ(ndt::make_type<int32_t>(), d.p("element_type").as<ndt::type>());
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-3));
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-2));
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(-1));
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(0));
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(1));
    EXPECT_EQ(ndt::make_type<int32_t>(), d.at(2));
    fad = d.tcast<cfixed_dim_type>();
    EXPECT_EQ(4, fad->get_fixed_stride());
    EXPECT_EQ(3, fad->get_fixed_dim_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_cfixed_dim(1, ndt::make_type<int32_t>());
    EXPECT_EQ(cfixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(dim_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    fad = d.tcast<cfixed_dim_type>();
    EXPECT_EQ(0, fad->get_fixed_stride());
    EXPECT_EQ(1, fad->get_fixed_dim_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    // With the stride != element type size
    d = ndt::make_cfixed_dim(3, ndt::make_type<int32_t>(), 8);
    EXPECT_EQ(cfixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(dim_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_EQ(20u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    fad = d.tcast<cfixed_dim_type>();
    EXPECT_EQ(8, fad->get_fixed_stride());
    EXPECT_EQ(3, fad->get_fixed_dim_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(CFixedDimType, CreateCOrder) {
    intptr_t shape[3] = {2, 3, 4};
    ndt::type d = ndt::make_cfixed_dim(3, shape, ndt::make_type<int16_t>(), NULL);
    EXPECT_EQ(3, d.get_ndim());
    EXPECT_EQ(3, d.get_strided_ndim());
    EXPECT_EQ(cfixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(ndt::make_cfixed_dim(2, shape+1, ndt::make_type<int16_t>(), NULL), d.at(0));
    EXPECT_EQ(ndt::make_cfixed_dim(1, shape+2, ndt::make_type<int16_t>(), NULL), d.at(0,0));
    EXPECT_EQ(ndt::make_type<int16_t>(), d.at(0,0,0));
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2, d.tcast<cfixed_dim_type>()->get_fixed_dim_size());
    EXPECT_EQ(24, d.tcast<cfixed_dim_type>()->get_fixed_stride());
    EXPECT_EQ(3, static_cast<const cfixed_dim_type *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(8, static_cast<const cfixed_dim_type *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4, static_cast<const cfixed_dim_type *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(2, static_cast<const cfixed_dim_type *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(CFixedDimType, CreateFOrder) {
    int axis_perm[3] = {0, 1, 2};
    intptr_t shape[3] = {2, 3, 4};
    ndt::type d = ndt::make_cfixed_dim(3, shape, ndt::make_type<int16_t>(), axis_perm);
    EXPECT_EQ(3, d.get_ndim());
    EXPECT_EQ(3, d.get_strided_ndim());
    EXPECT_EQ(48u, d.get_data_size());
    EXPECT_EQ(cfixed_dim_type_id, d.get_type_id());
    EXPECT_EQ(cfixed_dim_type_id, d.at(0).get_type_id());
    EXPECT_EQ(cfixed_dim_type_id, d.at(0,0).get_type_id());
    EXPECT_EQ(int16_type_id, d.at(0,0,0).get_type_id());
    // Check that the shape is right and the strides are in F-order
    EXPECT_EQ(2, d.tcast<cfixed_dim_type>()->get_fixed_dim_size());
    EXPECT_EQ(2, d.tcast<cfixed_dim_type>()->get_fixed_stride());
    EXPECT_EQ(3, static_cast<const cfixed_dim_type *>(d.at(0).extended())->get_fixed_dim_size());
    EXPECT_EQ(4, static_cast<const cfixed_dim_type *>(d.at(0).extended())->get_fixed_stride());
    EXPECT_EQ(4, static_cast<const cfixed_dim_type *>(d.at(0,0).extended())->get_fixed_dim_size());
    EXPECT_EQ(12, static_cast<const cfixed_dim_type *>(d.at(0,0).extended())->get_fixed_stride());
}

TEST(CFixedDimType, Basic) {
  nd::array a;
  float vals[3] = {1.5f, 2.5f, -1.5f};

  a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<float>()));
  a.vals() = vals;

  EXPECT_EQ(ndt::make_cfixed_dim(3, ndt::make_type<float>()), a.get_type());
  EXPECT_EQ(1, a.get_type().get_ndim());
  EXPECT_EQ(1, a.get_type().get_strided_ndim());
  EXPECT_EQ(1u, a.get_shape().size());
  EXPECT_EQ(3, a.get_shape()[0]);
  EXPECT_EQ(1u, a.get_strides().size());
  EXPECT_EQ(4, a.get_strides()[0]);
  EXPECT_EQ(1.5f, a(-3).as<float>());
  EXPECT_EQ(2.5f, a(-2).as<float>());
  EXPECT_EQ(-1.5f, a(-1).as<float>());
  EXPECT_EQ(1.5f, a(0).as<float>());
  EXPECT_EQ(2.5f, a(1).as<float>());
  EXPECT_EQ(-1.5f, a(2).as<float>());
  EXPECT_THROW(a(-4), index_out_of_bounds);
  EXPECT_THROW(a(3), index_out_of_bounds);
}

TEST(CFixedDimType, SimpleIndex) {
    nd::array a = parse_json("cfixed[2] * cfixed[3] * int16", "[[1, 2, 3], [4, 5, 6]]");
    ASSERT_EQ(ndt::make_cfixed_dim(2,
                    ndt::make_cfixed_dim(3, ndt::make_type<int16_t>())),
                a.get_type());

    nd::array b;

    b = a(0);
    ASSERT_EQ(ndt::make_cfixed_dim(3, ndt::make_type<int16_t>()),
                b.get_type());
    EXPECT_EQ(1, b(0).as<int16_t>());
    EXPECT_EQ(2, b(1).as<int16_t>());
    EXPECT_EQ(3, b(2).as<int16_t>());

    b = a(1);
    ASSERT_EQ(ndt::make_cfixed_dim(3, ndt::make_type<int16_t>()),
                b.get_type());
    EXPECT_EQ(4, b(0).as<int16_t>());
    EXPECT_EQ(5, b(1).as<int16_t>());
    EXPECT_EQ(6, b(2).as<int16_t>());

    EXPECT_THROW(a(2), index_out_of_bounds);
    EXPECT_THROW(a(-3), index_out_of_bounds);
}

TEST(CFixedDimType, AssignKernel_ScalarToFixed) {
    nd::array a, b;
    unary_ckernel_builder k;

    // Assignment scalar -> fixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = 9.0;
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
}

TEST(CFixedDimType, AssignKernel_FixedToFixed) {
    nd::array a, b;
    unary_ckernel_builder k;

    // Assignment fixed array -> fixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = parse_json("cfixed[3] * int32", "[3, 5, 7]");
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(cfixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
}

TEST(CFixedDimType, AssignKernel_FixedToScalarError) {
    nd::array a, b;
    unary_ckernel_builder k;

    // Assignment fixed array -> scalar
    a = 9.0;
    b = parse_json("cfixed[3] * int32", "[3, 5, 7]");
    EXPECT_EQ(cfixed_dim_type_id, b.get_type().get_type_id());
    EXPECT_THROW(make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(),
                                        b.get_type(), b.get_arrmeta(),
                                        kernel_request_single,
                                        &eval::default_eval_context),
                 broadcast_error);
}

TEST(CFixedDimType, AssignFixedStridedKernel) {
    nd::array a, b;
    unary_ckernel_builder k;
    int vals_int[] = {3,5,7};
    int vals_int_single[] = {9};

    // Assignment strided array -> fixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = vals_int;
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Broadcasting assignment strided array -> fixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = vals_int_single;
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
    k.reset();

    // Assignment fixed array -> strided array
    a = nd::empty<float[3]>();
    a.vals() = 0;
    b = parse_json("cfixed[3] * int32", "[3, 5, 7]");
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(cfixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Broadcasting assignment fixed array -> strided array
    a = nd::empty<float[3]>();
    a.vals() = 0;
    b = parse_json("cfixed[1] * int32", "[9]");
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(cfixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
    k.reset();
}

TEST(CFixedDimType, IsTypeSubarray) {
    EXPECT_TRUE(ndt::type("cfixed[3] * int32")
                    .is_type_subarray(ndt::type("cfixed[3] * int32")));
    EXPECT_TRUE(ndt::type("cfixed[10] * int32")
                    .is_type_subarray(ndt::type("cfixed[10] * int32")));
    EXPECT_TRUE(ndt::type("cfixed[3] * cfixed[10] * int32")
                    .is_type_subarray(ndt::type("cfixed[10] * int32")));
    EXPECT_TRUE(ndt::type("cfixed[3] * cfixed[10] * int32")
                    .is_type_subarray(ndt::type("int32")));
    EXPECT_TRUE(ndt::type("cfixed[5] * int32")
                    .is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(
        ndt::type("cfixed[5] * int32")));
    EXPECT_FALSE(ndt::type("cfixed[10] * int32").is_type_subarray(
        ndt::type("cfixed[3] * cfixed[10] * int32")));
    EXPECT_FALSE(ndt::type("cfixed[3] * int32")
                     .is_type_subarray(ndt::type("strided * int32")));
    EXPECT_FALSE(ndt::type("cfixed[3] * int32")
                     .is_type_subarray(ndt::type("var * int32")));
    EXPECT_FALSE(ndt::type("strided * int32")
                     .is_type_subarray(ndt::type("cfixed[3] * int32")));
    EXPECT_FALSE(ndt::type("var * int32")
                     .is_type_subarray(ndt::type("cfixed[3] * int32")));
}

TEST(CFixedDimType, FromCArray) {
    EXPECT_EQ(ndt::cfixed_dim_from_array<int>::make(), ndt::type("int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[1]>::make(),
              ndt::type("cfixed[1] * int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[2]>::make(),
              ndt::type("cfixed[2] * int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[3]>::make(),
              ndt::type("cfixed[3] * int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[2][1]>::make(),
              ndt::type("cfixed[2] * cfixed[1] * int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[1][2]>::make(),
              ndt::type("cfixed[1] * cfixed[2] * int32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<int[3][3]>::make(),
              ndt::type("cfixed[3] * cfixed[3] * int32"));
    EXPECT_EQ(
        ndt::cfixed_dim_from_array<int[3][5][8][10]>::make(),
        ndt::type("cfixed[3] * cfixed[5] * cfixed[8] * cfixed[10] * int32"));

    EXPECT_EQ(ndt::cfixed_dim_from_array<float>::make(), ndt::type("float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[1]>::make(), ndt::type("cfixed[1] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[2]>::make(), ndt::type("cfixed[2] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[3]>::make(), ndt::type("cfixed[3] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[2][1]>::make(), ndt::type("cfixed[2] * cfixed[1] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[1][2]>::make(), ndt::type("cfixed[1] * cfixed[2] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[3][3]>::make(), ndt::type("cfixed[3] * cfixed[3] * float32"));
    EXPECT_EQ(ndt::cfixed_dim_from_array<float[3][5][8][10]>::make(), ndt::type("cfixed[3] * cfixed[5] * cfixed[8] * cfixed[10] * float32"));
}
