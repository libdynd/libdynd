//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/ndobject.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(ArithmeticOp, SimpleBroadcast) {
    ndobject a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {1,2,3};
    const int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;

    c = (a + b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(1, c.at(0,0).as<int>());
    EXPECT_EQ(3, c.at(0,1).as<int>());
    EXPECT_EQ(4, c.at(0,2).as<int>());
    EXPECT_EQ(3, c.at(1,0).as<int>());
    EXPECT_EQ(7, c.at(1,1).as<int>());
    EXPECT_EQ(-7, c.at(1,2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(1, c.at(0,0).as<int>());
    EXPECT_EQ(1, c.at(0,1).as<int>());
    EXPECT_EQ(2, c.at(0,2).as<int>());
    EXPECT_EQ(-1, c.at(1,0).as<int>());
    EXPECT_EQ(-3, c.at(1,1).as<int>());
    EXPECT_EQ(13, c.at(1,2).as<int>());
    c = (b * a).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(0, c.at(0,0).as<int>());
    EXPECT_EQ(2, c.at(0,1).as<int>());
    EXPECT_EQ(3, c.at(0,2).as<int>());
    EXPECT_EQ(2, c.at(1,0).as<int>());
    EXPECT_EQ(10, c.at(1,1).as<int>());
    EXPECT_EQ(-30, c.at(1,2).as<int>());
    c = (b / a).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(0, c.at(0,0).as<int>());
    EXPECT_EQ(0, c.at(0,1).as<int>());
    EXPECT_EQ(0, c.at(0,2).as<int>());
    EXPECT_EQ(2, c.at(1,0).as<int>());
    EXPECT_EQ(2, c.at(1,1).as<int>());
    EXPECT_EQ(-3, c.at(1,2).as<int>());
}

TEST(ArithmeticOp, StridedScalarBroadcast) {
    ndobject a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {2,4,6};
    a = v0;
    b = 2;

    c = (a + b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(4, c.at(0).as<int>());
    EXPECT_EQ(6, c.at(1).as<int>());
    EXPECT_EQ(8, c.at(2).as<int>());
    c = (b + a).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(4, c.at(0).as<int>());
    EXPECT_EQ(6, c.at(1).as<int>());
    EXPECT_EQ(8, c.at(2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(0, c.at(0).as<int>());
    EXPECT_EQ(2, c.at(1).as<int>());
    EXPECT_EQ(4, c.at(2).as<int>());
    c = (b - a).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(0, c.at(0).as<int>());
    EXPECT_EQ(-2, c.at(1).as<int>());
    EXPECT_EQ(-4, c.at(2).as<int>());
    c = (a * b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(4, c.at(0).as<int>());
    EXPECT_EQ(8, c.at(1).as<int>());
    EXPECT_EQ(12, c.at(2).as<int>());
    c = (b * a).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(4, c.at(0).as<int>());
    EXPECT_EQ(8, c.at(1).as<int>());
    EXPECT_EQ(12, c.at(2).as<int>());
    c = (a / b).eval();
    EXPECT_EQ(make_dtype<int>(), c.get_udtype());
    EXPECT_EQ(1, c.at(0).as<int>());
    EXPECT_EQ(2, c.at(1).as<int>());
    EXPECT_EQ(3, c.at(2).as<int>());
}

TEST(ArithmeticOp, VarToStridedBroadcast) {
    ndobject a, b, c;

    a = parse_json("2, VarDim, int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2, 3, int32",
                    "[[5, 6, 7], [8, 9, 10]]");

    // VarDim in the first operand
    c = (a + b).eval();
    ASSERT_EQ(dtype("M, N, int32"), c.get_dtype());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(6, c.at(0,0).as<int>());
    EXPECT_EQ(8, c.at(0,1).as<int>());
    EXPECT_EQ(10, c.at(0,2).as<int>());
    EXPECT_EQ(12, c.at(1,0).as<int>());
    EXPECT_EQ(13, c.at(1,1).as<int>());
    EXPECT_EQ(14, c.at(1,2).as<int>());

    // VarDim in the second operand
    c = (b - a).eval();
    ASSERT_EQ(dtype("M, N, int32"), c.get_dtype());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(4, c.at(0,0).as<int>());
    EXPECT_EQ(4, c.at(0,1).as<int>());
    EXPECT_EQ(4, c.at(0,2).as<int>());
    EXPECT_EQ(4, c.at(1,0).as<int>());
    EXPECT_EQ(5, c.at(1,1).as<int>());
    EXPECT_EQ(6, c.at(1,2).as<int>());
}

TEST(ArithmeticOp, VarToVarBroadcast) {
    ndobject a, b, c;

    a = parse_json("2, VarDim, int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2, VarDim, int32",
                    "[[5], [6, 7]]");

    // VarDim in both operands, produces VarDim
    c = (a + b).eval();
    ASSERT_EQ(dtype("M, VarDim, int32"), c.get_dtype());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c.at(0,0).as<int>());
    EXPECT_EQ(7, c.at(0,1).as<int>());
    EXPECT_EQ(8, c.at(0,2).as<int>());
    EXPECT_EQ(10, c.at(1,0).as<int>());
    EXPECT_EQ(11, c.at(1,1).as<int>());

    a = parse_json("2, VarDim, int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2, 1, int32",
                    "[[5], [6]]");

    // VarDim in first operand, strided of size 1 in the second
    ASSERT_EQ(dtype("M, VarDim, int32"), c.get_dtype());
    c = (a + b).eval();
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c.at(0,0).as<int>());
    EXPECT_EQ(7, c.at(0,1).as<int>());
    EXPECT_EQ(8, c.at(0,2).as<int>());
    EXPECT_EQ(10, c.at(1,0).as<int>());

    // Strided of size 1 in the first operand, VarDim in second
    c = (b - a).eval();
    ASSERT_EQ(dtype("M, VarDim, int32"), c.get_dtype());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(4, c.at(0,0).as<int>());
    EXPECT_EQ(3, c.at(0,1).as<int>());
    EXPECT_EQ(2, c.at(0,2).as<int>());
    EXPECT_EQ(2, c.at(1,0).as<int>());
}

TEST(ArithmeticOp, ScalarOnTheRight) {
    ndobject a, b, c;

    const int v0[] = {1,2,3};
    a = v0;

    // A scalar on the right
    c = (a + 12).eval();
    EXPECT_EQ(13, c.at(0).as<int>());
    EXPECT_EQ(14, c.at(1).as<int>());
    EXPECT_EQ(15, c.at(2).as<int>());
    c = (a - 12).eval();
    EXPECT_EQ(-11, c.at(0).as<int>());
    EXPECT_EQ(-10, c.at(1).as<int>());
    EXPECT_EQ(-9, c.at(2).as<int>());
    c = (a * 3).eval();
    EXPECT_EQ(3, c.at(0).as<int>());
    EXPECT_EQ(6, c.at(1).as<int>());
    EXPECT_EQ(9, c.at(2).as<int>());
    c = (a / 2).eval();
    EXPECT_EQ(0, c.at(0).as<int>());
    EXPECT_EQ(1, c.at(1).as<int>());
    EXPECT_EQ(1, c.at(2).as<int>());
}

TEST(ArithmeticOp, ScalarOnTheLeft) {
    ndobject a, b, c;

    const int v0[] = {1,2,3};
    a = v0;

    // A scalar on the left
    c = ((-1) + a).eval();
    EXPECT_EQ(0, c.at(0).as<int>());
    EXPECT_EQ(1, c.at(1).as<int>());
    EXPECT_EQ(2, c.at(2).as<int>());
    c = ((-1) - a).eval();
    EXPECT_EQ(-2, c.at(0).as<int>());
    EXPECT_EQ(-3, c.at(1).as<int>());
    EXPECT_EQ(-4, c.at(2).as<int>());
    c = (5 * a).eval();
    EXPECT_EQ(5, c.at(0).as<int>());
    EXPECT_EQ(10, c.at(1).as<int>());
    EXPECT_EQ(15, c.at(2).as<int>());
    c = (-6 / a).eval();
    EXPECT_EQ(-6, c.at(0).as<int>());
    EXPECT_EQ(-3, c.at(1).as<int>());
    EXPECT_EQ(-2, c.at(2).as<int>());
}

TEST(ArithmeticOp, ComplexScalar) {
    return;

    ndobject a, c;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    a = v0;

    // A complex scalar
    (a + complex<float>(1, 2)).debug_print(cout);
    c = (a + complex<float>(1, 2)).eval();
    EXPECT_EQ(complex<float>(2,2), c.at(0).as<complex<float> >());
    EXPECT_EQ(complex<float>(3,2), c.at(1).as<complex<float> >());
    EXPECT_EQ(complex<float>(4,2), c.at(2).as<complex<float> >());
    c = (complex<float>(0, -1) * a).eval();
    EXPECT_EQ(complex<float>(0,-1), c.at(0).as<complex<float> >());
    EXPECT_EQ(complex<float>(0,-2), c.at(1).as<complex<float> >());
    EXPECT_EQ(complex<float>(0,-3), c.at(2).as<complex<float> >());
}

TEST(ArithmeticOp, MatchingDTypes_View) {
    ndobject a, b, c, d;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;

    c = a + b;
    EXPECT_EQ(1, c.at(0,0).as<int>());
    EXPECT_EQ(3, c.at(0,1).as<int>());
    EXPECT_EQ(4, c.at(0,2).as<int>());
    EXPECT_EQ(3, c.at(1,0).as<int>());
    EXPECT_EQ(7, c.at(1,1).as<int>());
    EXPECT_EQ(-7, c.at(1,2).as<int>());

    // Note: 'c' contains an expression tree with an 'add' node,
    // so editing the values of 'a' or 'b' changes the values of 'c'
    int v2[] = {6,4,2};
    a.val_assign(ndobject(v2));
    EXPECT_EQ(6, c.at(0,0).as<int>());
    EXPECT_EQ(5, c.at(0,1).as<int>());
    EXPECT_EQ(3, c.at(0,2).as<int>());
    EXPECT_EQ(8, c.at(1,0).as<int>());
    EXPECT_EQ(9, c.at(1,1).as<int>());
    EXPECT_EQ(-8, c.at(1,2).as<int>());

    // Check also partial indexing through another temporary
    d = c.at(0);
    EXPECT_EQ(1u, d.get_undim());
    EXPECT_EQ(3, d.get_shape()[0]);
    a.val_assign(ndobject(v0));
    EXPECT_EQ(1, d.at(0).as<int>());
    EXPECT_EQ(3, d.at(1).as<int>());
    EXPECT_EQ(4, d.at(2).as<int>());
    d = c.at(1);
    a.val_assign(ndobject(v2));
    EXPECT_EQ(8, d.at(0).as<int>());
    EXPECT_EQ(9, d.at(1).as<int>());
    EXPECT_EQ(-8, d.at(2).as<int>());
}

/*
TEST(ArithmeticOp, Buffered) {
    ndobject a;

    // Basic case with no buffering
    a = ndobject(2) * ndobject(3);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(make_dtype<int>(), a.get_node()->get_dtype());
    EXPECT_EQ(make_dtype<int>(), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ(make_dtype<int>(), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<int>());

    // Buffering the first operand
    a = ndobject(2) * ndobject(3.f);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(make_dtype<float>(), a.get_node()->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ(make_dtype<float>(), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the second operand
    a = ndobject(2.) * ndobject(3);
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(make_dtype<double>(), a.get_node()->get_dtype());
    EXPECT_EQ(make_dtype<double>(), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ((make_convert_dtype<double, int>()), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the output
    a = (ndobject(2) * ndobject(3)).as_dtype<float>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_dtype());
    EXPECT_EQ(make_dtype<int>(), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ(make_dtype<int>(), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<float>());

    // Buffering both operands
    a = ndobject(2) * ndobject(3u).as_dtype<float>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ(make_dtype<float>(), a.get_node()->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, unsigned int>()), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<float>());

    // Buffering the first operand and the output
    a = (ndobject(2) * ndobject(3.f)).as_dtype<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((make_convert_dtype<double, float>()), a.get_node()->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ(make_dtype<float>(), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<double>());

    // Buffering the second operand and the output
    a = (ndobject(2.f) * ndobject(3)).as_dtype<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((make_convert_dtype<double, float>()), a.get_node()->get_dtype());
    EXPECT_EQ(make_dtype<float>(), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<double>());

    // Buffering both operands and the output
    a = (ndobject(2) * ndobject(3u).as_dtype<float>()).as_dtype<double>();
    EXPECT_EQ(elwise_node_category, a.get_node()->get_category());
    EXPECT_EQ((make_convert_dtype<double, float>()), a.get_node()->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_node()->get_opnode(0)->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, unsigned int>()), a.get_node()->get_opnode(1)->get_dtype());
    EXPECT_EQ(6, a.as<double>());
}
*/
