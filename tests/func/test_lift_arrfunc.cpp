//
// Copyright (C) 2011-14 DyND Developers
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
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/take_arrfunc.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(LiftArrFunc, UnaryExpr_FixedDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_string(), assign_error_default);
    // Lift the arrfunc
    nd::arrfunc af = lift_arrfunc(af_base);

    // Test it on some data
    const char *in[3] = {"172", "-139", "12345"};
    nd::array a = nd::empty("3 * string");
    a.vals() = in;
    nd::array b = af(a);
    EXPECT_EQ(ndt::type("3 * int32"), b.get_type());
    EXPECT_EQ(172, b(0).as<int>());
    EXPECT_EQ(-139, b(1).as<int>());
    EXPECT_EQ(12345, b(2).as<int>());
}

TEST(LiftArrFunc, UnaryExpr_StridedDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);
    // Lift the arrfunc
    nd::arrfunc af = lift_arrfunc(af_base);

    // Test it on some data
    ckernel_builder ckb;
    nd::array in = nd::empty(3, "string[16]");
    in(0).vals() = "172";
    in(1).vals() = "-139";
    in(2).vals() = "12345";
    nd::array out = af(in);
    EXPECT_EQ(ndt::type("3 * int32"), out.get_type());
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
}

TEST(LiftArrFunc, UnaryExpr_VarDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);
    // Lift the arrfunc
    nd::arrfunc af = lift_arrfunc(af_base);

    const char *in[5] = {"172", "-139", "12345", "-1111", "284"};
    nd::array a = nd::empty("var * string[16]");
    a.vals() = in;
    nd::array out = af(a);
    EXPECT_EQ(ndt::type("var * string[16]"), a.get_type());
    EXPECT_EQ(ndt::type("var * int32"), out.get_type());
    EXPECT_EQ(5, out.get_shape()[0]);
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
    EXPECT_EQ(-1111, out(3).as<int>());
    EXPECT_EQ(284, out(4).as<int>());
}

TEST(LiftArrFunc, UnaryExpr_StridedToVarDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);

    // Lift the kernel to particular fixed dim arrays
    arrfunc_type_data af;
    lift_arrfunc(&af, af_base);

    // Test it on some data
    ndt::type dst_tp("var * int32");
    ndt::type src_tp("5 * string[16]");
    ckernel_builder ckb;
    nd::array in = nd::empty(src_tp);
    nd::array out = nd::empty(dst_tp);
    in(0).vals() = "172";
    in(1).vals() = "-139";
    in(2).vals() = "12345";
    in(3).vals() = "-1111";
    in(4).vals() = "284";
    const char *in_ptr = in.get_readonly_originptr();
    const char *src_arrmeta[1] = {in.get_arrmeta()};
    af.instantiate(&af, &ckb, 0, dst_tp,
                        out.get_arrmeta(), &src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), nd::array());
    expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
    usngo(out.get_readwrite_originptr(), const_cast<char **>(&in_ptr), ckb.get());
    EXPECT_EQ(5, out.get_shape()[0]);
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
    EXPECT_EQ(-1111, out(3).as<int>());
    EXPECT_EQ(284, out(4).as<int>());
}

TEST(LiftArrFunc, UnaryExpr_VarToVarDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);

    // Lift the kernel to particular fixed dim arrays
    arrfunc_type_data af;
    lift_arrfunc(&af, af_base);

    // Test it on some data
    ckernel_builder ckb;
    nd::array in = nd::empty("var * string[16]");
    nd::array out = nd::empty("var * int32");
    const char *in_vals[] = {"172", "-139", "12345", "-1111", "284"};
    in.vals() = in_vals;
    const char *in_ptr = in.get_readonly_originptr();
    const char *src_arrmeta[1] = {in.get_arrmeta()};
    af.instantiate(&af, &ckb, 0, out.get_type(),
                        out.get_arrmeta(), &in.get_type(), src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), nd::array());
    expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
    usngo(out.get_readwrite_originptr(), const_cast<char **>(&in_ptr), ckb.get());
    EXPECT_EQ(5, out.get_shape()[0]);
    EXPECT_EQ(172, out(0).as<int>());
    EXPECT_EQ(-139, out(1).as<int>());
    EXPECT_EQ(12345, out(2).as<int>());
    EXPECT_EQ(-1111, out(3).as<int>());
    EXPECT_EQ(284, out(4).as<int>());
}

TEST(LiftArrFunc, UnaryExpr_MultiDimVarToVarDim) {
    // Create an arrfunc for converting string to int
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);
    // Lift the arrfunc
    nd::arrfunc af = lift_arrfunc(af_base);

    // Test it on some data
    nd::array in = nd::empty("3 * var * string[16]");
    const char *in_vals0[] = {"172", "-139", "12345", "-1111", "284"};
    const char *in_vals1[] = {"989767"};
    const char *in_vals2[] = {"1", "2", "4"};
    in(0).vals() = in_vals0;
    in(1).vals() = in_vals1;
    in(2).vals() = in_vals2;

    nd::array out = af(in);
    EXPECT_EQ(ndt::type("3 * var * int"), out.get_type());
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

/*
// TODO Reenable once there's a convenient way to make the binary arrfunc
TEST(LiftArrFunc, Expr_MultiDimVarToVarDim) {
    // Create an arrfunc for adding two ints
    ndt::type add_ints_type = (nd::array((int32_t)0) + nd::array((int32_t)0)).get_type();
    nd::arrfunc af_base = make_arrfunc_from_assignment(
        ndt::make_type<int32_t>(), add_ints_type,
        assign_error_default);
    // Lift the arrfunc
    nd::arrfunc af = lift_arrfunc(af_base);

    // Lift the kernel to particular arrays
    nd::array af_lifted = nd::empty(ndt::make_arrfunc());
    ndt::type src0_tp("3 * var * int32");
    ndt::type src1_tp("fixed * int32");

    // Create some compatible values
    nd::array in0 = nd::empty(src0_tp);
    nd::array in1 = nd::empty(3, src1_tp);
    int32_t in0_vals0[] = {1, 2, 3};
    int32_t in0_vals1[] = {4};
    int32_t in0_vals2[] = {-1, 10, 2};
    in0(0).vals() = in0_vals0;
    in0(1).vals() = in0_vals1;
    in0(2).vals() = in0_vals2;
    int32_t in1_vals[] = {2, 4, 10};
    in1.vals() = in1_vals;

    nd::array out = af(in0, in1);
    EXPECT_EQ(ndt::type("3 * fixed * int32"), out.get_type());
    ASSERT_EQ(3, out.get_shape()[0]);
    ASSERT_EQ(3, out.get_shape()[1]);
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
*/
