//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/index.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include "../dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Elwise, UnaryFixedDim)
{
  nd::callable f = nd::functional::elwise(nd::functional::apply([](dynd::string s) { return s.size(); }));
  EXPECT_ARRAY_EQ((nd::array{static_cast<size_t>(5), static_cast<size_t>(2), static_cast<size_t>(6)}),
                  f({{"Hello", ", ", "world!"}}, {}));
}

TEST(Elwise, UnaryExpr_VarDim)
{
  // Create an callable for converting string to int
  nd::callable af_base =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Lift the callable
  nd::callable af = nd::functional::elwise(af_base);

  const char *in[5] = {"172", "-139", "12345", "-1111", "284"};
  nd::array a = nd::empty("var * fixed_string[16]");
  a.vals() = in;
  nd::array out = af(a);
  EXPECT_EQ(ndt::type("var * fixed_string[16]"), a.get_type());
  EXPECT_EQ(ndt::type("var * int32"), out.get_type());
  EXPECT_EQ(5, out.get_shape()[0]);
  EXPECT_EQ(172, out(0).as<int>());
  EXPECT_EQ(-139, out(1).as<int>());
  EXPECT_EQ(12345, out(2).as<int>());
  EXPECT_EQ(-1111, out(3).as<int>());
  EXPECT_EQ(284, out(4).as<int>());
}

TEST(Elwise, UnaryExpr_StridedToVarDim)
{
  nd::callable f = nd::functional::elwise(nd::assign.specialize(ndt::make_type<int>(), {ndt::type("string")}));
  EXPECT_ARRAY_EQ(nd::array({172, -139, 12345, -1111, 284}).cast(ndt::type("var * int32")),
                  f({{"172", "-139", "12345", "-1111", "284"}}, {{"dst_tp", ndt::type("var * int32")}}));
}

TEST(Elwise, UnaryExpr_VarToVarDim)
{
  // Create an callable for converting string to int
  nd::callable af_base =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::fixed_string_type::make(16), assign_error_default);

  // Lift the kernel to particular fixed dim arrays
  nd::callable af = nd::functional::elwise(af_base);

  // Test it on some data
  nd::kernel_builder ckb;
  nd::array in = nd::empty("var * fixed_string[16]");
  nd::array out = nd::empty("var * int32");
  const char *in_vals[] = {"172", "-139", "12345", "-1111", "284"};
  in.vals() = in_vals;
  const char *in_ptr = in.cdata();
  const char *src_arrmeta[1] = {in.get()->metadata()};
  af.get()->instantiate(NULL, &ckb, out.get_type(), out.get()->metadata(), af.get_type()->get_npos(), &in.get_type(),
                        src_arrmeta, kernel_request_single, 0, NULL, std::map<std::string, ndt::type>());
  kernel_single_t usngo = ckb.get()->get_function<kernel_single_t>();
  usngo(ckb.get(), out.data(), const_cast<char **>(&in_ptr));
  EXPECT_EQ(5, out.get_shape()[0]);
  EXPECT_EQ(172, out(0).as<int>());
  EXPECT_EQ(-139, out(1).as<int>());
  EXPECT_EQ(12345, out(2).as<int>());
  EXPECT_EQ(-1111, out(3).as<int>());
  EXPECT_EQ(284, out(4).as<int>());
}

TEST(Elwise, UnaryExpr_MultiDimVarToVarDim)
{
  // Create an callable for converting string to int
  nd::callable af_base =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Lift the callable
  nd::callable af = nd::functional::elwise(af_base);

  // Test it on some data
  nd::array in = nd::empty("3 * var * fixed_string[16]");
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

TEST(Elwise, Binary_FixedDim)
{
  nd::callable f = nd::functional::elwise(nd::functional::apply([](int x, int y) { return x + y; }));
  EXPECT_ARRAY_EQ((nd::array{3, 5, 7}), f({{0, 1, 2}, {3, 4, 5}}, {}));
}

/*
// TODO Reenable once there's a convenient way to make the binary callable
TEST(LiftCallable, Expr_MultiDimVarToVarDim) {
    // Create an callable for adding two ints
    ndt::type add_ints_type = (nd::array((int32_t)0) +
nd::array((int32_t)0)).get_type();
    nd::callable af_base = make_callable_from_assignment(
        ndt::make_type<int32_t>(), add_ints_type,
        assign_error_default);
    // Lift the callable
    nd::callable af = lift_callable(af_base);

    // Lift the kernel to particular arrays
    nd::array af_lifted = nd::empty(ndt::make_callable());
    ndt::type src0_tp("3 * var * int32");
    ndt::type src1_tp("Fixed * int32");

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
    EXPECT_EQ(ndt::type("3 * Fixed * int32"), out.get_type());
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
