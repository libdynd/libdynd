//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/lift_ckernel_deferred.hpp>
#include <dynd/kernels/take_ckernel_deferred.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(CKernelDeferred, Assignment) {
    ckernel_deferred ckd;
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    unary_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(unary_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(2, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_fixedstring(16), ckd.data_dynd_types[1]);

    const char *dynd_metadata[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    int int_out = 0;
    char str_in[16] = "3251";
    unary_single_operation_t usngo = ckb.get()->get_function<unary_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), str_in, ckb.get());
    EXPECT_EQ(3251, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_strided, &eval::default_eval_context);
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
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(2, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_fixedstring(16), ckd.data_dynd_types[1]);

    const char *dynd_metadata[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    int int_out = 0;
    char str_in[16] = "3251";
    char *str_in_ptr = str_in;
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), &str_in_ptr, ckb.get());
    EXPECT_EQ(3251, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_strided, &eval::default_eval_context);
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
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), add_ints_type, add_ints_type,
                    expr_operation_funcproto, assign_error_default, ckd);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (deferred_ckernel_funcproto_t)ckd.ckernel_funcproto);
    ASSERT_EQ(3, ckd.data_types_size);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[0]);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[1]);
    ASSERT_EQ(ndt::make_type<int>(), ckd.data_dynd_types[2]);

    const char *dynd_metadata[3] = {NULL, NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    int int_out = 0;
    int int_in1 = 1, int_in2 = 3;
    char *int_in_ptr[2] = {reinterpret_cast<char *>(&int_in1),
                        reinterpret_cast<char *>(&int_in2)};
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(reinterpret_cast<char *>(&int_out), int_in_ptr, ckb.get());
    EXPECT_EQ(4, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_strided, &eval::default_eval_context);
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


TEST(CKernelDeferred, LiftUnaryExpr_FixedDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular fixed dim arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("cfixed[3] * int32"));
    lifted_types.push_back(ndt::type("cfixed[3] * string[16]"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    ckernel_builder ckb;
    const char *dynd_metadata[2] = {NULL, NULL};
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    int out[3] = {0, 0, 0};
    char in[3][16] = {"172", "-139", "12345"};
    const char *in_ptr = reinterpret_cast<const char *>(in);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(reinterpret_cast<char *>(&out), &in_ptr, ckb.get());
    EXPECT_EQ(172, out[0]);
    EXPECT_EQ(-139, out[1]);
    EXPECT_EQ(12345, out[2]);
}

TEST(CKernelDeferred, LiftUnaryExpr_StridedDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular fixed dim arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("strided * int32"));
    lifted_types.push_back(ndt::type("strided * string[16]"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    ckernel_builder ckb;
    nd::array in = nd::empty(3, ndt::type("strided * string[16]"));
    nd::array out = nd::empty(3, ndt::type("strided * int32"));
    in(0).vals() = "172";
    in(1).vals() = "-139";
    in(2).vals() = "12345";
    const char *in_ptr = in.get_readonly_originptr();
    const char *dynd_metadata[2] = {NULL, NULL};
    dynd_metadata[0] = out.get_arrmeta();
    dynd_metadata[1] = in.get_arrmeta();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(out.get_readwrite_originptr(), &in_ptr, ckb.get());
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
}

TEST(CKernelDeferred, LiftUnaryExpr_StridedToVarDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular fixed dim arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("var * int32"));
    lifted_types.push_back(ndt::type("strided * string[16]"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    ckernel_builder ckb;
    nd::array in = nd::empty(5, ndt::type("strided * string[16]"));
    nd::array out = nd::empty(ndt::type("var * int32"));
    in(0).vals() = "172";
    in(1).vals() = "-139";
    in(2).vals() = "12345";
    in(3).vals() = "-1111";
    in(4).vals() = "284";
    const char *in_ptr = in.get_readonly_originptr();
    const char *dynd_metadata[2] = {NULL, NULL};
    dynd_metadata[0] = out.get_arrmeta();
    dynd_metadata[1] = in.get_arrmeta();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(out.get_readwrite_originptr(), &in_ptr, ckb.get());
    EXPECT_EQ(5, out.get_shape()[0]);
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
    EXPECT_EQ(-1111, out(3).as<int>());
    EXPECT_EQ(284, out(4).as<int>());
}


TEST(CKernelDeferred, LiftUnaryExpr_VarToVarDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular fixed dim arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("var * int32"));
    lifted_types.push_back(ndt::type("var * string[16]"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    ckernel_builder ckb;
    nd::array in = nd::empty(ndt::type("var * string[16]"));
    nd::array out = nd::empty(ndt::type("var * int32"));
    const char *in_vals[] = {"172", "-139", "12345", "-1111", "284"};
    in.vals() = in_vals;
    const char *in_ptr = in.get_readonly_originptr();
    const char *dynd_metadata[2] = {NULL, NULL};
    dynd_metadata[0] = out.get_arrmeta();
    dynd_metadata[1] = in.get_arrmeta();
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(out.get_readwrite_originptr(), &in_ptr, ckb.get());
    EXPECT_EQ(5, out.get_shape()[0]);
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
    EXPECT_EQ(-1111, out(3).as<int>());
    EXPECT_EQ(284, out(4).as<int>());
}


TEST(CKernelDeferred, LiftUnaryExpr_MultiDimVarToVarDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for converting string to int
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int>(), ndt::make_fixedstring(16), ndt::make_fixedstring(16),
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular arrays
    ckernel_deferred ckd;
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("strided * var * int32"));
    lifted_types.push_back(ndt::type("3 * var * string[16]"));
    lift_ckernel_deferred(&ckd, ckd_base, lifted_types);

    // Test it on some data
    nd::array in = nd::empty(ndt::type("3 * var * string[16]"));
    nd::array out = nd::empty(3, ndt::type("strided * var * int32"));
    const char *in_vals0[] = {"172", "-139", "12345", "-1111", "284"};
    const char *in_vals1[] = {"989767"};
    const char *in_vals2[] = {"1", "2", "4"};
    in(0).vals() = in_vals0;
    in(1).vals() = in_vals1;
    in(2).vals() = in_vals2;

    const char *in_ptr = in.get_readonly_originptr();
    const char *dynd_metadata[2] = {NULL, NULL};
    dynd_metadata[0] = out.get_arrmeta();
    dynd_metadata[1] = in.get_arrmeta();
    ckernel_builder ckb;
    ckd.instantiate_func(ckd.data_ptr, &ckb, 0, dynd_metadata,
                         kernel_request_single, &eval::default_eval_context);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(out.get_readwrite_originptr(), &in_ptr, ckb.get());
    ASSERT_EQ(3, out.get_shape()[0]);
    ASSERT_EQ(5, out(0).get_shape()[0]);
    ASSERT_EQ(1, out(1).get_shape()[0]);
    ASSERT_EQ(3, out(2).get_shape()[0]);
    EXPECT_EQ(172, out(0, 0).as<int>());
    EXPECT_EQ(-139, out(0, 1).as<int>());
    EXPECT_EQ(12345, out(0, 2).as<int>());
    EXPECT_EQ(-1111, out(0, 3).as<int>());
    EXPECT_EQ(284, out(0, 4).as<int>());
    EXPECT_EQ(989767, out(1, 0).as<int>());
    EXPECT_EQ(1, out(2, 0).as<int>());
    EXPECT_EQ(2, out(2, 1).as<int>());
    EXPECT_EQ(4, out(2, 2).as<int>());
}

TEST(CKernelDeferred, LiftExpr_MultiDimVarToVarDim) {
    nd::array ckd_base = nd::empty(ndt::make_ckernel_deferred());
    // Create a deferred ckernel for adding two ints
    ndt::type add_ints_type = (nd::array((int32_t)0) + nd::array((int32_t)0)).get_type();
    make_ckernel_deferred_from_assignment(
                    ndt::make_type<int32_t>(), add_ints_type, add_ints_type,
                    expr_operation_funcproto, assign_error_default,
                    *reinterpret_cast<ckernel_deferred *>(ckd_base.get_readwrite_originptr()));

    // Lift the kernel to particular arrays
    nd::array ckd_lifted = nd::empty(ndt::make_ckernel_deferred());
    ckernel_deferred *ckd = reinterpret_cast<ckernel_deferred *>(ckd_lifted.get_readwrite_originptr());
    vector<ndt::type> lifted_types;
    lifted_types.push_back(ndt::type("strided * var * int32"));
    lifted_types.push_back(ndt::type("3 * var * int32"));
    lifted_types.push_back(ndt::type("strided * int32"));
    lift_ckernel_deferred(ckd, ckd_base, lifted_types);

    // Create some compatible values
    nd::array out = nd::empty(3, lifted_types[0]);
    nd::array in0 = nd::empty(lifted_types[1]);
    nd::array in1 = nd::empty(3, lifted_types[2]);
    int32_t in0_vals0[] = {1, 2, 3};
    int32_t in0_vals1[] = {4};
    int32_t in0_vals2[] = {-1, 10, 2};
    in0(0).vals() = in0_vals0;
    in0(1).vals() = in0_vals1;
    in0(2).vals() = in0_vals2;
    int32_t in1_vals[] = {2, 4, 10};
    in1.vals() = in1_vals;

    const char *dynd_metadata[3] = {NULL, NULL, NULL};
    dynd_metadata[0] = out.get_arrmeta();
    dynd_metadata[1] = in0.get_arrmeta();
    dynd_metadata[2] = in1.get_arrmeta();
    const char *const in_ptrs[2] = {in0.get_readonly_originptr(), in1.get_readonly_originptr()};
    ckernel_builder ckb;
    ckd->instantiate_func(ckd->data_ptr, &ckb, 0, dynd_metadata,
                          kernel_request_single, &eval::default_eval_context);
    expr_single_operation_t usngo = ckb.get()->get_function<expr_single_operation_t>();
    usngo(out.get_readwrite_originptr(), in_ptrs, ckb.get());
    ASSERT_EQ(3, out.get_shape()[0]);
    ASSERT_EQ(3, out(0).get_shape()[0]);
    ASSERT_EQ(3, out(1).get_shape()[0]);
    ASSERT_EQ(3, out(2).get_shape()[0]);
    EXPECT_EQ(3, out(0, 0).as<int>());
    EXPECT_EQ(6, out(0, 1).as<int>());
    EXPECT_EQ(13, out(0, 2).as<int>());
    EXPECT_EQ(6, out(1, 0).as<int>());
    EXPECT_EQ(8, out(1, 1).as<int>());
    EXPECT_EQ(14, out(1, 2).as<int>());
    EXPECT_EQ(1, out(2, 0).as<int>());
    EXPECT_EQ(14, out(2, 1).as<int>());
    EXPECT_EQ(12, out(2, 2).as<int>());

    // Do it again with the __call__ function
    out = nd::empty(3, lifted_types[0]);
    ckd_lifted.f("__call__", out, in0, in1);
    ASSERT_EQ(3, out.get_shape()[0]);
    ASSERT_EQ(3, out(0).get_shape()[0]);
    ASSERT_EQ(3, out(1).get_shape()[0]);
    ASSERT_EQ(3, out(2).get_shape()[0]);
    EXPECT_EQ(3, out(0, 0).as<int>());
    EXPECT_EQ(6, out(0, 1).as<int>());
    EXPECT_EQ(13, out(0, 2).as<int>());
    EXPECT_EQ(6, out(1, 0).as<int>());
    EXPECT_EQ(8, out(1, 1).as<int>());
    EXPECT_EQ(14, out(1, 2).as<int>());
    EXPECT_EQ(1, out(2, 0).as<int>());
    EXPECT_EQ(14, out(2, 1).as<int>());
    EXPECT_EQ(12, out(2, 2).as<int>());
}

TEST(CKernelDeferred, Take) {
    nd::array a, b, c;
    nd::array take;

    int avals[5] = {1, 2, 3, 4, 5};
    dynd_bool bvals[5] = {false, true, false, true, true};
    a = avals;
    b = bvals;

    c = nd::empty("var * int");
    take = kernels::make_take_ckernel_deferred(c.get_type(), a.get_type(), b.get_type());
    take.f("__call__", c, a, b);
    EXPECT_EQ(3, c.get_dim_size());
    EXPECT_EQ(2, c(0).as<int>());
    EXPECT_EQ(4, c(1).as<int>());
    EXPECT_EQ(5, c(2).as<int>());

    intptr_t bvals2[4] = {3, 0, -1, 4};
    b = bvals2;

    c = nd::empty("4 * int");
    take = kernels::make_take_ckernel_deferred(c.get_type(), a.get_type(), b.get_type());
    take.f("__call__", c, a, b);
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(5, c(2).as<int>());
    EXPECT_EQ(5, c(3).as<int>());
}
