//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

/*
TODO: This example exposes a bug with the lifted reduction code.

static void elwise_func(float& out, float in) {
    cout << out << " OP " << in;
    out += in;
    cout << " ==> " << out << endl;
}

TEST(Neighborhood, Sum) {
    // Start with a float32 reduction arrfunc
    nd::arrfunc reduction_kernel = nd::make_functor_arrfunc(elwise_func);

    // Lift it to a two-dimensional strided float32 reduction arrfunc
    bool reduction_dimflags[2] = {true, true};
    nd::arrfunc nh_op = lift_reduction_arrfunc(
        reduction_kernel, ndt::type("strided * strided * float32"), nd::array(),
        false, 2, reduction_dimflags, true, true, false, 0.f);

    intptr_t nh_shape[2] = {3, 3};
    intptr_t nh_centre[2] = {1, 1};
    nd::arrfunc naf = make_neighborhood2d_arrfunc(nh_op, nh_shape, nh_centre);

    nd::array a =
        parse_json("4 * 4 * float32",
                 "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]");
    //nd::array mask =
    //    parse_json("3 * 3 * float32", "[[0, 1, 0], [1, 1, 1], [0, 1, 0]]");
    nd::array b = nd::empty<float[4][4]>();
    b.vals() = 0;

    naf.call_out(a, b);
}
*/

template <int N>
void sum(float &dst, const nd::strided_vals<float, N> &src) {
    dst = 0.0;
    for (typename nd::strided_vals<float, N>::iterator it = src.begin(); it != src.end(); it++) {
        dst += **it;
    }
}

TEST(Neighborhood, Reduction2D) {
    nd::arrfunc af = nd::make_functor_arrfunc(sum<2>);

    intptr_t nh_shape[2] = {3, 3};
    intptr_t nh_centre[2] = {1, 1};
    nd::arrfunc naf = make_neighborhood_arrfunc(af, 2, nh_shape, nh_centre);

    nd::array a =
        parse_json("4 * 4 * float32",
                   "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]");
    nd::array b = nd::empty<float[4][4]>();
    b.vals() = 0;

    naf.call_out(a, b);

    EXPECT_EQ(b(1, 1).as<int>(), 54);
    EXPECT_EQ(b(1, 2).as<int>(), 63);
    EXPECT_EQ(b(2, 1).as<int>(), 90);
    EXPECT_EQ(b(2, 2).as<int>(), 99);
}

TEST(Neighborhood, Reduction3D) {
    nd::arrfunc af = nd::make_functor_arrfunc(sum<3>);

    intptr_t nh_shape[3] = {3, 3, 3};
    intptr_t nh_centre[3] = {1, 1, 1};
    nd::arrfunc naf = make_neighborhood_arrfunc(af, 3, nh_shape, nh_centre);

    nd::array a =
        parse_json("4 * 4 * 4 * float32",
                   "[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],"
                   "[[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],"
                   "[[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]],"
                   "[[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64]]]");
    nd::array b = nd::empty<float[4][4][4]>();
    b.vals() = 0;

    naf.call_out(a, b);

    EXPECT_EQ(b(1, 1, 1).as<int>(), 594);
    EXPECT_EQ(b(1, 1, 2).as<int>(), 621);
    EXPECT_EQ(b(1, 2, 1).as<int>(), 702);
    EXPECT_EQ(b(1, 2, 2).as<int>(), 729);
    EXPECT_EQ(b(2, 1, 1).as<int>(), 1026);
    EXPECT_EQ(b(2, 1, 2).as<int>(), 1053);
    EXPECT_EQ(b(2, 2, 1).as<int>(), 1134);
    EXPECT_EQ(b(2, 2, 2).as<int>(), 1161);
}
