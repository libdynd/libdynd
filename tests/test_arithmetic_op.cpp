//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(ArithmeticOp, SimpleBroadcast) {
    nd::array a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {1,2,3};
    const int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;

    c = (a + b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(1, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(-1, c(1,0).as<int>());
    EXPECT_EQ(-3, c(1,1).as<int>());
    EXPECT_EQ(13, c(1,2).as<int>());
    c = (b * a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(2, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(10, c(1,1).as<int>());
    EXPECT_EQ(-30, c(1,2).as<int>());
    c = (b / a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(0, c(0,1).as<int>());
    EXPECT_EQ(0, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(2, c(1,1).as<int>());
    EXPECT_EQ(-3, c(1,2).as<int>());
}

TEST(ArithmeticOp, StridedScalarBroadcast) {
    nd::array a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {2,4,6};
    a = v0;
    b = 2;

    c = (a + b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(8, c(2).as<int>());
    c = (b + a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(8, c(2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(2, c(1).as<int>());
    EXPECT_EQ(4, c(2).as<int>());
    c = (b - a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(-2, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
    c = (a * b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(8, c(1).as<int>());
    EXPECT_EQ(12, c(2).as<int>());
    c = (b * a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(8, c(1).as<int>());
    EXPECT_EQ(12, c(2).as<int>());
    c = (a / b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0).as<int>());
    EXPECT_EQ(2, c(1).as<int>());
    EXPECT_EQ(3, c(2).as<int>());
}

TEST(ArithmeticOp, VarToStridedBroadcast) {
    nd::array a, b, c;

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * 3 * int32",
                    "[[5, 6, 7], [8, 9, 10]]");

    // VarDim in the first operand
    c = (a + b).eval();
    ASSERT_EQ(ndt::type("M * N * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(8, c(0,1).as<int>());
    EXPECT_EQ(10, c(0,2).as<int>());
    EXPECT_EQ(12, c(1,0).as<int>());
    EXPECT_EQ(13, c(1,1).as<int>());
    EXPECT_EQ(14, c(1,2).as<int>());

    // VarDim in the second operand
    c = (b - a).eval();
    ASSERT_EQ(ndt::type("M * N * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(4, c(0,0).as<int>());
    EXPECT_EQ(4, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(4, c(1,0).as<int>());
    EXPECT_EQ(5, c(1,1).as<int>());
    EXPECT_EQ(6, c(1,2).as<int>());
}

TEST(ArithmeticOp, VarToVarBroadcast) {
    nd::array a, b, c;

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * var * int32",
                    "[[5], [6, 7]]");

    // VarDim in both operands, produces VarDim
    c = (a + b).eval();
    ASSERT_EQ(ndt::type("M * var * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(7, c(0,1).as<int>());
    EXPECT_EQ(8, c(0,2).as<int>());
    EXPECT_EQ(10, c(1,0).as<int>());
    EXPECT_EQ(11, c(1,1).as<int>());

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * 1 * int32",
                    "[[5], [6]]");

    // VarDim in first operand, strided of size 1 in the second
    ASSERT_EQ(ndt::type("M * var * int32"), c.get_type());
    c = (a + b).eval();
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(7, c(0,1).as<int>());
    EXPECT_EQ(8, c(0,2).as<int>());
    EXPECT_EQ(10, c(1,0).as<int>());

    // Strided of size 1 in the first operand, VarDim in second
    c = (b - a).eval();
    ASSERT_EQ(ndt::type("M * var * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(4, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
}

TEST(ArithmeticOp, ScalarOnTheRight) {
    nd::array a, b, c;

    const int v0[] = {1,2,3};
    a = v0;

    // A scalar on the right
    c = (a + 12).eval();
    EXPECT_EQ(13, c(0).as<int>());
    EXPECT_EQ(14, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (a - 12).eval();
    EXPECT_EQ(-11, c(0).as<int>());
    EXPECT_EQ(-10, c(1).as<int>());
    EXPECT_EQ(-9, c(2).as<int>());
    c = (a * 3).eval();
    EXPECT_EQ(3, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(9, c(2).as<int>());
    c = (a / 2).eval();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(1, c(2).as<int>());
}

TEST(ArithmeticOp, ScalarOnTheLeft) {
    nd::array a, b, c;

    const int v0[] = {1,2,3};
    a = v0;

    // A scalar on the left
    c = ((-1) + a).eval();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(2, c(2).as<int>());
    c = ((-1) - a).eval();
    EXPECT_EQ(-2, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
    c = (5 * a).eval();
    EXPECT_EQ(5, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (-6 / a).eval();
    EXPECT_EQ(-6, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-2, c(2).as<int>());
}

TEST(ArithmeticOp, ComplexScalar) {
    return;

    nd::array a, c;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    a = v0;

    // A complex scalar
    (a + dynd_complex<float>(1, 2)).debug_print(cout);
    c = (a + dynd_complex<float>(1, 2)).eval();
    EXPECT_EQ(dynd_complex<float>(2,2), c(0).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(3,2), c(1).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(4,2), c(2).as<dynd_complex<float> >());
    c = (dynd_complex<float>(0, -1) * a).eval();
    EXPECT_EQ(dynd_complex<float>(0,-1), c(0).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(0,-2), c(1).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(0,-3), c(2).as<dynd_complex<float> >());
}

TEST(ArithmeticOp, MatchingDTypes_View) {
    nd::array a, b, c, d;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;

    c = a + b;
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());

    // Note: 'c' contains an expression tree with an 'add' node,
    // so editing the values of 'a' or 'b' changes the values of 'c'
    int v2[] = {6,4,2};
    a.val_assign(nd::array(v2));
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(5, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(8, c(1,0).as<int>());
    EXPECT_EQ(9, c(1,1).as<int>());
    EXPECT_EQ(-8, c(1,2).as<int>());

    // Check also partial indexing through another temporary
    d = c(0);
    EXPECT_EQ(1, d.get_ndim());
    EXPECT_EQ(3, d.get_shape()[0]);
    a.val_assign(nd::array(v0));
    EXPECT_EQ(1, d(0).as<int>());
    EXPECT_EQ(3, d(1).as<int>());
    EXPECT_EQ(4, d(2).as<int>());
    d = c(1);
    a.val_assign(nd::array(v2));
    EXPECT_EQ(8, d(0).as<int>());
    EXPECT_EQ(9, d(1).as<int>());
    EXPECT_EQ(-8, d(2).as<int>());
}

/*
TEST(ArithmeticOp, Buffered) {
    nd::array a;

    // Basic case with no buffering
    a = nd::array(2) * nd::array(3);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(ndt::make_type<int>(), a.get_node()->get_type());
    EXPECT_EQ(ndt::make_type<int>(), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ(ndt::make_type<int>(), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<int>());

    // Buffering the first operand
    a = nd::array(2) * nd::array(3.f);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(ndt::make_type<float>(), a.get_node()->get_type());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ(ndt::make_type<float>(), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the second operand
    a = nd::array(2.) * nd::array(3);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(ndt::make_type<double>(), a.get_node()->get_type());
    EXPECT_EQ(ndt::make_type<double>(), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ((ndt::make_convert<double, int>()), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the output
    a = (nd::array(2) * nd::array(3)).as<float>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_type());
    EXPECT_EQ(ndt::make_type<int>(), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ(ndt::make_type<int>(), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<float>());

    // Buffering both operands
    a = nd::array(2) * nd::array(3u).as<float>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(ndt::make_type<float>(), a.get_node()->get_type());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ((ndt::make_convert<float, unsigned int>()), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the first operand and the output
    a = (nd::array(2) * nd::array(3.f)).as<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((ndt::make_convert<double, float>()), a.get_node()->get_type());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ(ndt::make_type<float>(), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<double>());

    // Buffering the second operand and the output
    a = (nd::array(2.f) * nd::array(3)).as<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((ndt::make_convert<double, float>()), a.get_node()->get_type());
    EXPECT_EQ(ndt::make_type<float>(), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<double>());

    // Buffering both operands and the output
    a = (nd::array(2) * nd::array(3u).as<float>()).as<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((ndt::make_convert<double, float>()), a.get_node()->get_type());
    EXPECT_EQ((ndt::make_convert<float, int>()), a.get_node()->get_opnode(0)->get_type());
    EXPECT_EQ((ndt::make_convert<float, unsigned int>()), a.get_node()->get_opnode(1)->get_type());
    EXPECT_EQ(6, a.as<double>());
}
*/
