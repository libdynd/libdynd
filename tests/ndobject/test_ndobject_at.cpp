//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "inc_gtest.hpp"

#include "dynd/ndobject.hpp"
#include "dynd/exceptions.hpp"

using namespace std;
using namespace dynd;

TEST(NDObjectIndex, BasicInteger) {
    int i0[3][2] = {{1,2},{3,4},{5,6}};
    ndobject a = i0;
    ndobject b, c;

    // Indexing in two steps
    b = a.at(0);
    EXPECT_EQ(1, b.at(0).as<int>());
    EXPECT_EQ(2, b.at(1).as<int>());
    b = a.at(1);
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(4, b.at(1).as<int>());
    b = a.at(2);
    EXPECT_EQ(5, b.at(0).as<int>());
    EXPECT_EQ(6, b.at(1).as<int>());
    // Python style negative index
    EXPECT_EQ(5, b.at(-2).as<int>());
    EXPECT_EQ(6, b.at(-1).as<int>());
    EXPECT_THROW(b.at(-3), index_out_of_bounds);
    EXPECT_THROW(b.at(2), index_out_of_bounds);
    EXPECT_THROW(b.at(0,0), too_many_indices);

    // Indexing in one step
    EXPECT_EQ(1, a.at(0,0).as<int>());
    EXPECT_EQ(2, a.at(0,1).as<int>());
    EXPECT_EQ(3, a.at(1,0).as<int>());
    EXPECT_EQ(4, a.at(1,1).as<int>());
    EXPECT_EQ(5, a.at(2,0).as<int>());
    EXPECT_EQ(6, a.at(2,1).as<int>());
    // Indexing with negative Python style
    EXPECT_EQ(1, a.at(-3,-2).as<int>());
    EXPECT_EQ(2, a.at(-3,-1).as<int>());
    EXPECT_EQ(3, a.at(-2,-2).as<int>());
    EXPECT_EQ(4, a.at(-2,-1).as<int>());
    EXPECT_EQ(5, a.at(-1,-2).as<int>());
    EXPECT_EQ(6, a.at(-1,-1).as<int>());
    EXPECT_THROW(a.at(-4,0), index_out_of_bounds);
    EXPECT_THROW(a.at(3,0), index_out_of_bounds);
    EXPECT_THROW(a.at(0,-3), index_out_of_bounds);
    EXPECT_THROW(a.at(0,2), index_out_of_bounds);
    EXPECT_THROW(a.at(0,0,0), too_many_indices);

    int i1[2][2][2] = {{{1,2}, {3,4}}, {{5,6}, {7,8}}};
    a = i1;

    // Indexing in two steps
    b = a.at(0,0);
    EXPECT_EQ(1, b.at(0).as<int>());
    EXPECT_EQ(2, b.at(1).as<int>());
    b = a.at(0,1);
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(4, b.at(1).as<int>());
    b = a.at(1,0);
    EXPECT_EQ(5, b.at(0).as<int>());
    EXPECT_EQ(6, b.at(1).as<int>());
    b = a.at(1,1);
    EXPECT_EQ(7, b.at(0).as<int>());
    EXPECT_EQ(8, b.at(1).as<int>());

    // Indexing in one step
    EXPECT_EQ(1, a.at(0,0,0).as<int>());
    EXPECT_EQ(2, a.at(0,0,1).as<int>());
    EXPECT_EQ(3, a.at(0,1,0).as<int>());
    EXPECT_EQ(4, a.at(0,1,1).as<int>());
    EXPECT_EQ(5, a.at(1,0,0).as<int>());
    EXPECT_EQ(6, a.at(1,0,1).as<int>());
    EXPECT_EQ(7, a.at(1,1,0).as<int>());
    EXPECT_EQ(8, a.at(1,1,1).as<int>());
    // Indexing with negative Python style
    EXPECT_EQ(1, a.at(-2,-2,-2).as<int>());
    EXPECT_EQ(2, a.at(-2,-2,-1).as<int>());
    EXPECT_EQ(3, a.at(-2,-1,-2).as<int>());
    EXPECT_EQ(4, a.at(-2,-1,-1).as<int>());
    EXPECT_EQ(5, a.at(-1,-2,-2).as<int>());
    EXPECT_EQ(6, a.at(-1,-2,-1).as<int>());
    EXPECT_EQ(7, a.at(-1,-1,-2).as<int>());
    EXPECT_EQ(8, a.at(-1,-1,-1).as<int>());
    // Out of bounds
    EXPECT_THROW(a.at(-3,0,0), index_out_of_bounds);
    EXPECT_THROW(a.at(2,0,0), index_out_of_bounds);
    EXPECT_THROW(a.at(0,-3,0), index_out_of_bounds);
    EXPECT_THROW(a.at(0,2,0), index_out_of_bounds);
    EXPECT_THROW(a.at(0,0,-3), index_out_of_bounds);
    EXPECT_THROW(a.at(0,0,2), index_out_of_bounds);
    EXPECT_THROW(a.at(0,0,0,0), too_many_indices);
}

TEST(NDObjectIndex, SimpleOneDimensionalRange) {
    int i0[] = {1,2,3,4,5,6};
    ndobject a = i0, b;

    // full range
    b = a.at(irange());
    EXPECT_EQ(6, b.get_shape()[0]);
    EXPECT_EQ(1, b.at(0).as<int>());
    EXPECT_EQ(2, b.at(1).as<int>());
    EXPECT_EQ(3, b.at(2).as<int>());
    EXPECT_EQ(4, b.at(3).as<int>());
    EXPECT_EQ(5, b.at(4).as<int>());
    EXPECT_EQ(6, b.at(5).as<int>());

    // selected range
    b = a.at(1 <= irange() < 3);
    EXPECT_EQ(1u, b.get_shape().size());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(2, b.at(0).as<int>());
    EXPECT_EQ(3, b.at(1).as<int>());

    // lower-bound only
    b = a.at(3 <= irange());
    EXPECT_EQ(3, b.get_shape()[0]);
    EXPECT_EQ(4, b.at(0).as<int>());
    EXPECT_EQ(5, b.at(1).as<int>());
    EXPECT_EQ(6, b.at(2).as<int>());

    // upper-bound only
    b = a.at(irange() < 3);
    EXPECT_EQ(3, b.get_shape()[0]);
    EXPECT_EQ(1, b.at(0).as<int>());
    EXPECT_EQ(2, b.at(1).as<int>());
    EXPECT_EQ(3, b.at(2).as<int>());
}

TEST(NDObjectIndex, SteppedOneDimensionalRange) {
    int i0[] = {1,2,3,4,5,6};
    ndobject a = i0, b;

    // different step
    b = a.at(irange() / 2);
    EXPECT_EQ(3, b.get_shape()[0]);
    EXPECT_EQ(1, b.at(0).as<int>());
    EXPECT_EQ(3, b.at(1).as<int>());
    EXPECT_EQ(5, b.at(2).as<int>());

    // full reversed range
    b = a.at(irange() / -1);
    EXPECT_EQ(6, b.get_shape()[0]);
    EXPECT_EQ(6, b.at(0).as<int>());
    EXPECT_EQ(5, b.at(1).as<int>());
    EXPECT_EQ(4, b.at(2).as<int>());
    EXPECT_EQ(3, b.at(3).as<int>());
    EXPECT_EQ(2, b.at(4).as<int>());
    EXPECT_EQ(1, b.at(5).as<int>());

    // partial reversed range
    b = a.at(3 >= irange() / -1 >= 0);
    EXPECT_EQ(4, b.get_shape()[0]);
    EXPECT_EQ(4, b.at(0).as<int>());
    EXPECT_EQ(3, b.at(1).as<int>());
    EXPECT_EQ(2, b.at(2).as<int>());
    EXPECT_EQ(1, b.at(3).as<int>());

    // reversed range with different step
    b = a.at(irange() / -3);
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(6, b.at(0).as<int>());
    EXPECT_EQ(3, b.at(1).as<int>());

    // partial reversed range with different step
    b = a.at(2 >= irange() / -2);
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(1, b.at(1).as<int>());

    // empty range
    b = a.at(2 <= irange() < 2);
    EXPECT_EQ(0, b.get_shape()[0]);

    // applying two ranges, one after another
    b = a.at(1 <= irange() <= 5).at(irange() / -2);
    EXPECT_EQ(3, b.get_shape()[0]);
    EXPECT_EQ(6, b.at(0).as<int>());
    EXPECT_EQ(4, b.at(1).as<int>());
    EXPECT_EQ(2, b.at(2).as<int>());
}

TEST(NDObjectIndex, ExceptionsOneDimensionalRange) {
    int i0[] = {1,2,3,4,5,6};
    ndobject a = i0, b;

    // exceptions for out-of-bounds ranges
    EXPECT_THROW(a.at(-7 <= irange()), irange_out_of_bounds);
    EXPECT_THROW(a.at(0 <= irange() < 7), irange_out_of_bounds);
    EXPECT_THROW(a.at(0 <= irange() <= 6), irange_out_of_bounds);
    EXPECT_THROW(a.at(0,irange()), too_many_indices);
    EXPECT_THROW(a.at(0).at(irange()), too_many_indices);
}
