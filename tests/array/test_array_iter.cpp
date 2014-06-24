//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array_iter.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayNeighborhoodIter, Simple) {
    intptr_t neighborhood_shape[2] = {2, 2};
    intptr_t neighborhood_offset[2] = {0, 0};

    int vals[2][3] = {{0, 1, 2}, {3, 4, 5}};

    nd::array arg = nd::empty<int[2][3]>();
    arg.vals() = vals;

    nd::array res = nd::empty<int[2][3]>();
    res.vals() = 0;

    array_neighborhood_iter<1, 1> iter(res, arg, neighborhood_shape, neighborhood_offset);
    do {
        int *res_data = reinterpret_cast<int *>(iter.data<0>());
        do {
            if (iter.neighbor_within_bounds()) {
                const int *neighbor_data = reinterpret_cast<const int *>(iter.neighbor_data<1>());
                *res_data += *neighbor_data;
            }
        } while (iter.next_neighbor());
    } while(iter.next());

    EXPECT_EQ(8, res(0, 0).as<int>());
    EXPECT_EQ(12, res(0, 1).as<int>());
    EXPECT_EQ(7, res(0, 2).as<int>());
    EXPECT_EQ(7, res(1, 0).as<int>());
    EXPECT_EQ(9, res(1, 1).as<int>());
    EXPECT_EQ(5, res(1, 2).as<int>());
}

TEST(ArrayNeighborhoodIter, Offset) {
    intptr_t neighborhood_shape[2] = {2, 2};
    intptr_t neighborhood_offset[2] = {-1, -1};

    int vals[2][3] = {{0, 1, 2}, {3, 4, 5}};

    nd::array arg = nd::empty<int[2][3]>();
    arg.vals() = vals;

    nd::array res = nd::empty<int[2][3]>();
    res.vals() = 0;

    array_neighborhood_iter<1, 1> iter(res, arg, neighborhood_shape, neighborhood_offset);
    do {
        int *res_data = reinterpret_cast<int *>(iter.data<0>());
        do {
            if (iter.neighbor_within_bounds()) {
                const int *neighbor_data = reinterpret_cast<const int *>(iter.neighbor_data<1>());
                *res_data += *neighbor_data;
            }
        } while (iter.next_neighbor());
    } while(iter.next());

    EXPECT_EQ(0, res(0, 0).as<int>());
    EXPECT_EQ(1, res(0, 1).as<int>());
    EXPECT_EQ(3, res(0, 2).as<int>());
    EXPECT_EQ(3, res(1, 0).as<int>());
    EXPECT_EQ(8, res(1, 1).as<int>());
    EXPECT_EQ(12, res(1, 2).as<int>());
}
