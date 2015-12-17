//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/take_by_pointer.hpp>

using namespace std;
using namespace dynd;

/**
TODO: This test broken when the order of resolve_option_values and resolve_dst_type changed.
      It needs to be fixed.

TEST(TakeByPointer, Simple) {
    nd::arrfunc af = make_take_by_pointer_arrfunc();
    nd::array a, idx, res;

    a = parse_json("4 * int", "[0, 1, 2, 3]");
    idx = parse_json("4 * intptr", "[2, 1, 0, 3]");
    res = af(a, idx);
    EXPECT_EQ(4, res.get_dim_size());
    EXPECT_EQ(ndt::make_type<int *>(), res.get_dtype());
    EXPECT_EQ(2, *res(0).as<int *>());
    EXPECT_EQ(1, *res(1).as<int *>());
    EXPECT_EQ(0, *res(2).as<int *>());
    EXPECT_EQ(3, *res(3).as<int *>());

    a = parse_json("2 * 4 * float64",
        "[[-4.5, 1, 2.1, 3.5], [-32.7, 15.3, 6.9, 7]]");
    idx = parse_json("3 * 2 * intptr",
        "[[0, 2], [1, 0], [1, 1]]");
    res = af(a, idx);
    EXPECT_EQ(3, res.get_dim_size());
    EXPECT_EQ(ndt::make_type<double *>(), res.get_dtype());
    EXPECT_EQ(2.1, *res(0).as<double *>());
    EXPECT_EQ(-32.7, *res(1).as<double *>());
    EXPECT_EQ(15.3, *res(2).as<double *>());

    a = parse_json("4 * 4 * 4 * int",
        "[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],"
        "[[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]],"
        "[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]],"
        "[[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]");
    idx = parse_json("5 * 3 * intptr",
        "[[3, 2, 1], [1, 0, 1], [2, 1, 2], [3, 3, 3], [0, 2, 1]]");
    res = af(a, idx);
    EXPECT_EQ(5, res.get_dim_size());
    EXPECT_EQ(ndt::make_type<int *>(), res.get_dtype());
    EXPECT_EQ(57, *res(0).as<int *>());
    EXPECT_EQ(17, *res(1).as<int *>());
    EXPECT_EQ(38, *res(2).as<int *>());
    EXPECT_EQ(63, *res(3).as<int *>());
    EXPECT_EQ(9, *res(4).as<int *>());
}
*/
