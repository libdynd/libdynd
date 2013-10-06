//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/lift_ckernel_deferred.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(CKernelDeferred, Assignment) {
    ckernel_deferred ckd;
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(ndt::make_type<int>(), ndt::make_fixedstring(16),
                    unary_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(unary_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(2u, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_fixedstring(16), ckd.data_dynd_types[1]);

    const char *dynd_metadata[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_single);
    int int_out = 0;
    char str_in[16] = "3251";
    unary_single_operation_t usngo = ckb.get()->get_function<unary_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), str_in, ckb.get());
    EXPECT_EQ(3251, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_strided);
    int ints_out[3] = {0, 0, 0};
    char strs_in[3][16] = {"123", "4567", "891029"};
    unary_strided_operation_t ustro = ckb.get()->get_function<unary_strided_operation_t>();
    ustro(reinterpret_cast<char *>(&ints_out), sizeof(int), strs_in[0], 16, 3, ckb.get());
    EXPECT_EQ(123, ints_out[0]);
    EXPECT_EQ(4567, ints_out[1]);
    EXPECT_EQ(891029, ints_out[2]);
}


TEST(CKernelDeferred, AssignmentAsExpr) {
    ckernel_deferred ckd;
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(ndt::make_type<int>(), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(2u, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_fixedstring(16), ckd.data_dynd_types[1]);

    const char *dynd_metadata[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_single);
    int int_out = 0;
    char str_in[16] = "3251";
    char *str_in_ptr = str_in;
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), &str_in_ptr, ckb.get());
    EXPECT_EQ(3251, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_strided);
    int ints_out[3] = {0, 0, 0};
    char strs_in[3][16] = {"123", "4567", "891029"};
    char *strs_in_ptr = strs_in[0];
    intptr_t strs_in_stride = 16;
    expr_strided_operation_t ustro = ckb.get()->get_function<expr_strided_operation_t>();
    ustro(reinterpret_cast<char *>(&ints_out), sizeof(int), &strs_in_ptr, &strs_in_stride, 3, ckb.get());
    EXPECT_EQ(123, ints_out[0]);
    EXPECT_EQ(4567, ints_out[1]);
    EXPECT_EQ(891029, ints_out[2]);
}

TEST(CKernelDeferred, Expr) {
    ckernel_deferred ckd;
    // Create a deferred ckernel for adding two ints
    ndt::type add_ints_type = (nd::array((int)0) + nd::array((int)0)).get_type();
    make_ckernel_deferred_from_assignment(ndt::make_type<int>(), add_ints_type,
                    expr_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(3u, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[1]);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[2]);

    const char *dynd_metadata[3] = {NULL, NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_single);
    int int_out = 0;
    int int_in1 = 1, int_in2 = 3;
    char *int_in_ptr[2] = {reinterpret_cast<char *>(&int_in1),
                        reinterpret_cast<char *>(&int_in2)};
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), int_in_ptr, ckb.get());
    EXPECT_EQ(4, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_strided);
    int ints_out[3] = {0, 0, 0};
    int ints_in1[3] = {1,2,3}, ints_in2[3] = {5,-210,1234};
    char *ints_in_ptr[2] = {reinterpret_cast<char *>(&ints_in1),
                        reinterpret_cast<char *>(&ints_in2)};
    intptr_t ints_in_strides[2] = {sizeof(int), sizeof(int)};
    expr_strided_operation_t ustro = ckb.get()->get_function<expr_strided_operation_t>();
    ustro(reinterpret_cast<char *>(ints_out), sizeof(int),
                    ints_in_ptr, ints_in_strides, 3, ckb.get());
    EXPECT_EQ(6, ints_out[0]);
    EXPECT_EQ(-208, ints_out[1]);
    EXPECT_EQ(1237, ints_out[2]);
}


TEST(CKernelDeferred, LiftUnary) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(ndt::make_type<int>(), ndt::make_fixedstring(16),
                    unary_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular fixed dim arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("3, int32"));
    lifted_types.push_back(ndt::type("3, string(16)"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    ckernel_builder ckb;
    const char *dynd_metadata[2] = {NULL, NULL};
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata, kernel_request_single);
    int out[3] = {0, 0, 0};
    char in[3][16] = {"172", "-139", "12345"};
    unary_single_operation_t usngo = ckb.get()->get_function<unary_single_operation_t>();
    usngo(reinterpret_cast<char *>(&out), reinterpret_cast<const char *>(in), ckb.get());
    EXPECT_EQ(172, out[0]);
    EXPECT_EQ(-139, out[1]);
    EXPECT_EQ(12345, out[2]);
}