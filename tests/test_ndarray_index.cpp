#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"
#include "dnd/exceptions.hpp"

using namespace std;
using namespace dnd;

TEST(NDArrayIndex, BasicInteger) {
    int i0[3][2] = {{1,2},{3,4},{5,6}};
    ndarray a = i0;
    ndarray b;

    // Indexing in two steps
    b = a(0);
    EXPECT_EQ(1, b(0).as_scalar<int>());
    EXPECT_EQ(2, b(1).as_scalar<int>());
    b = a(1);
    EXPECT_EQ(3, b(0).as_scalar<int>());
    EXPECT_EQ(4, b(1).as_scalar<int>());
    b = a(2);
    EXPECT_EQ(5, b(0).as_scalar<int>());
    EXPECT_EQ(6, b(1).as_scalar<int>());
    EXPECT_THROW(b(-1), index_out_of_bounds);
    EXPECT_THROW(b(2), index_out_of_bounds);
    EXPECT_THROW(b(0,0), too_many_indices);

    // Indexing in one step
    EXPECT_EQ(1, a(0,0).as_scalar<int>());
    EXPECT_EQ(2, a(0,1).as_scalar<int>());
    EXPECT_EQ(3, a(1,0).as_scalar<int>());
    EXPECT_EQ(4, a(1,1).as_scalar<int>());
    EXPECT_EQ(5, a(2,0).as_scalar<int>());
    EXPECT_EQ(6, a(2,1).as_scalar<int>());
    EXPECT_THROW(a(-1,0), index_out_of_bounds);
    EXPECT_THROW(a(3,0), index_out_of_bounds);
    EXPECT_THROW(a(0,-1), index_out_of_bounds);
    EXPECT_THROW(a(0,2), index_out_of_bounds);
    EXPECT_THROW(a(0,0,0), too_many_indices);

    int i1[2][2][2] = {{{1,2}, {3,4}}, {{5,6}, {7,8}}};
    a = i1;

    // Indexing in two steps
    b = a(0,0);
    EXPECT_EQ(1, b(0).as_scalar<int>());
    EXPECT_EQ(2, b(1).as_scalar<int>());
    b = a(0,1);
    EXPECT_EQ(3, b(0).as_scalar<int>());
    EXPECT_EQ(4, b(1).as_scalar<int>());
    b = a(1,0);
    EXPECT_EQ(5, b(0).as_scalar<int>());
    EXPECT_EQ(6, b(1).as_scalar<int>());
    b = a(1,1);
    EXPECT_EQ(7, b(0).as_scalar<int>());
    EXPECT_EQ(8, b(1).as_scalar<int>());

    // Indexing in one step
    EXPECT_EQ(1, a(0,0,0).as_scalar<int>());
    EXPECT_EQ(2, a(0,0,1).as_scalar<int>());
    EXPECT_EQ(3, a(0,1,0).as_scalar<int>());
    EXPECT_EQ(4, a(0,1,1).as_scalar<int>());
    EXPECT_EQ(5, a(1,0,0).as_scalar<int>());
    EXPECT_EQ(6, a(1,0,1).as_scalar<int>());
    EXPECT_EQ(7, a(1,1,0).as_scalar<int>());
    EXPECT_EQ(8, a(1,1,1).as_scalar<int>());
    EXPECT_THROW(a(-1,0,0), index_out_of_bounds);
    EXPECT_THROW(a(2,0,0), index_out_of_bounds);
    EXPECT_THROW(a(0,-1,0), index_out_of_bounds);
    EXPECT_THROW(a(0,2,0), index_out_of_bounds);
    EXPECT_THROW(a(0,0,-1), index_out_of_bounds);
    EXPECT_THROW(a(0,0,2), index_out_of_bounds);
    EXPECT_THROW(a(0,0,0,0), too_many_indices);
}

TEST(NDArrayIndex, OneDimensionalRange) {
    int i0[] = {1,2,3,4,5,6};
    ndarray a = i0, b;

    // full range
    b = a(irange());
    EXPECT_EQ(6, b.shape(0));
    EXPECT_EQ(1, b(0).as_scalar<int>());
    EXPECT_EQ(2, b(1).as_scalar<int>());
    EXPECT_EQ(3, b(2).as_scalar<int>());
    EXPECT_EQ(4, b(3).as_scalar<int>());
    EXPECT_EQ(5, b(4).as_scalar<int>());
    EXPECT_EQ(6, b(5).as_scalar<int>());

    // selected range
    b = a(1 <= irange() < 3);
    EXPECT_EQ(2, b.shape(0));
    EXPECT_EQ(2, b(0).as_scalar<int>());
    EXPECT_EQ(3, b(1).as_scalar<int>());

    // lower-bound only
    b = a(3 <= irange());
    EXPECT_EQ(3, b.shape(0));
    EXPECT_EQ(4, b(0).as_scalar<int>());
    EXPECT_EQ(5, b(1).as_scalar<int>());
    EXPECT_EQ(6, b(2).as_scalar<int>());

    // upper-bound only
    b = a(irange() < 3);
    EXPECT_EQ(3, b.shape(0));
    EXPECT_EQ(1, b(0).as_scalar<int>());
    EXPECT_EQ(2, b(1).as_scalar<int>());
    EXPECT_EQ(3, b(2).as_scalar<int>());

    // different step
    b = a(irange() / 2);
    EXPECT_EQ(3, b.shape(0));
    EXPECT_EQ(1, b(0).as_scalar<int>());
    EXPECT_EQ(3, b(1).as_scalar<int>());
    EXPECT_EQ(5, b(2).as_scalar<int>());

    // full reversed range
    b = a(irange() / -1);
    EXPECT_EQ(6, b.shape(0));
    EXPECT_EQ(6, b(0).as_scalar<int>());
    EXPECT_EQ(5, b(1).as_scalar<int>());
    EXPECT_EQ(4, b(2).as_scalar<int>());
    EXPECT_EQ(3, b(3).as_scalar<int>());
    EXPECT_EQ(2, b(4).as_scalar<int>());
    EXPECT_EQ(1, b(5).as_scalar<int>());

    // partial reversed range
    b = a(3 >= irange() / -1 >= 0);
    EXPECT_EQ(4, b.shape(0));
    EXPECT_EQ(4, b(0).as_scalar<int>());
    EXPECT_EQ(3, b(1).as_scalar<int>());
    EXPECT_EQ(2, b(2).as_scalar<int>());
    EXPECT_EQ(1, b(3).as_scalar<int>());

    // reversed range with different step
    b = a(irange() / -3);
    EXPECT_EQ(2, b.shape(0));
    EXPECT_EQ(6, b(0).as_scalar<int>());
    EXPECT_EQ(3, b(1).as_scalar<int>());

    // partial reversed range with different step
    b = a(2 >= irange() / -2);
    EXPECT_EQ(2, b.shape(0));
    EXPECT_EQ(3, b(0).as_scalar<int>());
    EXPECT_EQ(1, b(1).as_scalar<int>());

    // empty range
    b = a(2 <= irange() < 2);
    EXPECT_EQ(0, b.shape(0));

    // applying two ranges, one after another
    b = a(1 <= irange() <= 5)(irange() / -2);
    EXPECT_EQ(3, b.shape(0));
    EXPECT_EQ(6, b(0).as_scalar<int>());
    EXPECT_EQ(4, b(1).as_scalar<int>());
    EXPECT_EQ(2, b(2).as_scalar<int>());

    // exceptions for out-of-bounds ranges
    EXPECT_THROW(a(-1 <= irange()), irange_out_of_bounds);
    EXPECT_THROW(a(0 <= irange() < 7), irange_out_of_bounds);
    EXPECT_THROW(a(0 <= irange() <= 6), irange_out_of_bounds);
    EXPECT_THROW(a(0,irange()), too_many_indices);
    EXPECT_THROW(a(0)(irange()), too_many_indices);
}
