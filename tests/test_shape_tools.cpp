//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "inc_gtest.hpp"

#include "dynd/shape_tools.hpp"
#include "dynd/exceptions.hpp"

using namespace std;
using namespace dynd;

TEST(ShapeTools, BroadcastToShape) {
    intptr_t shape0[] = {3,2,5}, shape1[] = {1,3,2,5};
    intptr_t strides0[] = {1,3,6}, strides1[] = {0,1,3,6}, strides_out[10];

    broadcast_to_shape(4, shape1, 3, shape0, strides0, strides_out);
    EXPECT_EQ(0, strides_out[0]);
    EXPECT_EQ(1, strides_out[1]);
    EXPECT_EQ(3, strides_out[2]);
    EXPECT_EQ(6, strides_out[3]);

    // Cannot broadcast from more to fewer ndim
    EXPECT_THROW(broadcast_to_shape(3, shape0, 4, shape1, strides1, strides_out),
                    broadcast_error);

    intptr_t shape2[] = {3,1}, shape3[] = {4,3};
    intptr_t strides2[] = {1,0}, strides3[] = {3,1};
    // Cannot broadcast size 3 to size 4
    EXPECT_THROW(broadcast_to_shape(2, shape3, 2, shape2, strides2, strides_out),
                    broadcast_error);
    // Cannot broadcast size 3 to size 1
    shape3[0] = 3;
    EXPECT_THROW(broadcast_to_shape(2, shape2, 2, shape3, strides3, strides_out),
                    broadcast_error);
}

TEST(ShapeTools, BroadcastInputShapes) {
    ndarray a(4, make_dtype<int>());
    ndarray b(3, 1, 1, make_dtype<float>());
    ndarray c(3, 2, 1, make_dtype<double>());
    ndarray d(5, 1, make_dtype<char>());

    ndarray_node_ptr operands[] = {a.get_node(), b.get_node(),
                                    c.get_node(), d.get_node()};

    // Broadcast the first three shapes together
    int ndim = 0;
    dimvector shape;
    broadcast_input_shapes(3, operands, &ndim, &shape);
    EXPECT_EQ(3, ndim);
    EXPECT_EQ(3, shape[0]);
    EXPECT_EQ(2, shape[1]);
    EXPECT_EQ(4, shape[2]);

    // Also broadcasting the fourth one should cause an error
    EXPECT_THROW(broadcast_input_shapes(4, operands, &ndim, &shape),
                    broadcast_error);
}

TEST(ShapeTools, CopyInputStrides) {
    ndarray a(2,3,2,1,make_dtype<int16_t>());
    intptr_t strides[6];

    EXPECT_EQ(4, a.get_ndim());
    EXPECT_EQ(12, a.get_strides()[0]);
    EXPECT_EQ(4, a.get_strides()[1]);
    EXPECT_EQ(2, a.get_strides()[2]);
    EXPECT_EQ(0, a.get_strides()[3]);

    copy_input_strides(a, 6, strides);
    EXPECT_EQ(0, strides[0]);
    EXPECT_EQ(0, strides[1]);
    EXPECT_EQ(12, strides[2]);
    EXPECT_EQ(4, strides[3]);
    EXPECT_EQ(2, strides[4]);
    EXPECT_EQ(0, strides[5]);
}

TEST(ShapeTools, MultiStridesToAxisPerm_OneOp) {
    // Basic test that a single C/F-order array works
    intptr_t strides_f[] = {1,2,4,8,16,32};
    intptr_t strides_c[] = {32,16,8,4,2,1};
    intptr_t *stridesptr;
    int axis_perm[6];

    // F-Order
    stridesptr = strides_f;
    multistrides_to_axis_perm(6, 1, &stridesptr, axis_perm);
    EXPECT_EQ(0, axis_perm[0]);
    EXPECT_EQ(1, axis_perm[1]);
    EXPECT_EQ(2, axis_perm[2]);
    EXPECT_EQ(3, axis_perm[3]);
    EXPECT_EQ(4, axis_perm[4]);
    EXPECT_EQ(5, axis_perm[5]);

    // Add some zero-strides in, which the sort tries to keep
    // in the default position while shuffling things around.
    strides_f[0] = 0;
    strides_f[3] = 0;
    strides_f[5] = 0;
    multistrides_to_axis_perm(6, 1, &stridesptr, axis_perm);
    EXPECT_EQ(5, axis_perm[0]);
    EXPECT_EQ(1, axis_perm[1]);
    EXPECT_EQ(2, axis_perm[2]);
    EXPECT_EQ(4, axis_perm[3]);
    EXPECT_EQ(3, axis_perm[4]);
    EXPECT_EQ(0, axis_perm[5]);

    // C-Order
    stridesptr = strides_c;
    multistrides_to_axis_perm(6, 1, &stridesptr, axis_perm);
    EXPECT_EQ(5, axis_perm[0]);
    EXPECT_EQ(4, axis_perm[1]);
    EXPECT_EQ(3, axis_perm[2]);
    EXPECT_EQ(2, axis_perm[3]);
    EXPECT_EQ(1, axis_perm[4]);
    EXPECT_EQ(0, axis_perm[5]);

    // Add some zero-strides in, which the sort keeps in place
    // while shuffling things around
    strides_c[0] = 0;
    strides_c[3] = 0;
    strides_c[5] = 0;
    multistrides_to_axis_perm(6, 1, &stridesptr, axis_perm);
    EXPECT_EQ(5, axis_perm[0]);
    EXPECT_EQ(4, axis_perm[1]);
    EXPECT_EQ(3, axis_perm[2]);
    EXPECT_EQ(2, axis_perm[3]);
    EXPECT_EQ(1, axis_perm[4]);
    EXPECT_EQ(0, axis_perm[5]);
}

TEST(ShapeTools, MultiStridesToAxisPerm_TwoOps) {
    // Basic test that a single C/F-order array works
    intptr_t strides_a[] = {1,2,4,8,0,0};
    intptr_t strides_b[] = {0,0,8,4,2,1};
    intptr_t *stridesptr[] = {strides_a, strides_b};
    int axis_perm[6];

    // C-order wins the conflicts between different strides
    multistrides_to_axis_perm(6, 2, stridesptr, axis_perm);
    EXPECT_EQ(5, axis_perm[0]);
    EXPECT_EQ(4, axis_perm[1]);
    EXPECT_EQ(0, axis_perm[2]);
    EXPECT_EQ(1, axis_perm[3]);
    EXPECT_EQ(3, axis_perm[4]);
    EXPECT_EQ(2, axis_perm[5]);

    // two mostly F-order arrays put together make the result F-order
    // as long as there are no ambiguities - well, they should, but
    // clearly the insertion short doesn't work properly...
    /*
    intptr_t strides_f1[] = {1,2,4,0};
    intptr_t strides_f2[] = {1,0,4,8};
    stridesptr[0] = strides_f1;
    stridesptr[0] = strides_f2;
    multistrides_to_axis_perm(4, 2, stridesptr, axis_perm);
    EXPECT_EQ(0, axis_perm[0]);
    EXPECT_EQ(1, axis_perm[1]);
    EXPECT_EQ(2, axis_perm[2]);
    EXPECT_EQ(3, axis_perm[3]);
    */

    intptr_t strides_a1[] = {1,2,0,0};
    intptr_t strides_a2[] = {0,0,2,1};
    stridesptr[0] = strides_a1;
    stridesptr[1] = strides_a2;
    multistrides_to_axis_perm(4, 2, stridesptr, axis_perm);
    EXPECT_EQ(3, axis_perm[0]);
    EXPECT_EQ(2, axis_perm[1]);
    EXPECT_EQ(0, axis_perm[2]);
    EXPECT_EQ(1, axis_perm[3]);
}
