#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArrayAssign, ScalarAssignment_Bool) {
    ndarray a;
    dnd_bool *ptr_b;

    // assignment to a bool scalar
    a = ndarray(make_dtype<dnd_bool>());
    ptr_b = (dnd_bool *)a.get_originptr();
    a.vassign(true);
    EXPECT_TRUE(*ptr_b);
    a.vassign(false);
    EXPECT_FALSE( *ptr_b);
    a.vassign(1);
    EXPECT_TRUE(*ptr_b);
    a.vassign(0);
    EXPECT_FALSE(*ptr_b);
    a.vassign(1.0);
    EXPECT_TRUE(*ptr_b);
    a.vassign(0.0);
    EXPECT_FALSE(*ptr_b);
    a.vassign(1.5, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    a.vassign(-3.5f, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    a.vassign(22, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    EXPECT_THROW(a.vassign(2), runtime_error);
    EXPECT_THROW(a.vassign(-1), runtime_error);
    EXPECT_THROW(a.vassign(1.5), runtime_error);
    EXPECT_THROW(a.vassign(1.5, assign_error_overflow), runtime_error);
    EXPECT_THROW(a.vassign(1.5, assign_error_fractional), runtime_error);
    EXPECT_THROW(a.vassign(1.5, assign_error_inexact), runtime_error);
}

TEST(NDArrayAssign, ScalarAssignment_Int8) {
    ndarray a;
    int8_t *ptr_i8;

    // Assignment to an int8_t scalar
    a = ndarray(make_dtype<int8_t>());
    ptr_i8 = (int8_t *)a.get_originptr();
    a.vassign(true);
    EXPECT_EQ(1, *ptr_i8);
    a.vassign(false);
    EXPECT_EQ(0, *ptr_i8);
    a.vassign(-10);
    EXPECT_EQ(-10, *ptr_i8);
    a.vassign(-128);
    EXPECT_EQ(-128, *ptr_i8);
    a.vassign(127);
    EXPECT_EQ(127, *ptr_i8);
    EXPECT_THROW(a.vassign(-129), runtime_error);
    EXPECT_THROW(a.vassign(128), runtime_error);
    a.vassign(5.0);
    EXPECT_EQ(5, *ptr_i8);
    a.vassign(-100.0f);
    EXPECT_EQ(-100, *ptr_i8);
    EXPECT_THROW(a.vassign(1.25), runtime_error);
    EXPECT_THROW(a.vassign(128.0), runtime_error);
    EXPECT_THROW(a.vassign(128.0, assign_error_inexact), runtime_error);
    EXPECT_THROW(a.vassign(1e30), runtime_error);
    a.vassign(1.25, assign_error_overflow);
    EXPECT_EQ(1, *ptr_i8);
    EXPECT_THROW(a.vassign(-129.0, assign_error_overflow), runtime_error);
    a.vassign(1.25, assign_error_none);
    EXPECT_EQ(1, *ptr_i8);
    a.vassign(-129.0, assign_error_none);
    //EXPECT_EQ((int8_t)-129.0, *ptr_i8); // < this is undefined behavior
}

TEST(NDArrayAssign, ScalarAssignment_UInt16) {
    ndarray a;
    uint16_t *ptr_u16;

    // Assignment to a uint16_t scalar
    a = ndarray(make_dtype<uint16_t>());
    ptr_u16 = (uint16_t *)a.get_originptr();
    a.vassign(true);
    EXPECT_EQ(1, *ptr_u16);
    a.vassign(false);
    EXPECT_EQ(0, *ptr_u16);
    EXPECT_THROW(a.vassign(-1), runtime_error);
    EXPECT_THROW(a.vassign(-1, assign_error_overflow), runtime_error);
    a.vassign(-1, assign_error_none);
    EXPECT_EQ(65535, *ptr_u16);
    a.vassign(1234);
    EXPECT_EQ(1234, *ptr_u16);
    a.vassign(65535.0f);
    EXPECT_EQ(65535, *ptr_u16);
}

TEST(NDArrayAssign, ScalarAssignment_Float32) {
    ndarray a;
    float *ptr_f32;

    // Assignment to a float scalar
    a = ndarray(make_dtype<float>());
    ptr_f32 = (float *)a.get_originptr();
    a.vassign(true);
    EXPECT_EQ(1, *ptr_f32);
    a.vassign(false);
    EXPECT_EQ(0, *ptr_f32);
    a.vassign(-10);
    EXPECT_EQ(-10, *ptr_f32);
    a.vassign((char)30);
    EXPECT_EQ(30, *ptr_f32);
    a.vassign((uint16_t)58000);
    EXPECT_EQ(58000, *ptr_f32);
    a.vassign(1.25);
    EXPECT_EQ(1.25, *ptr_f32);
    a.vassign(1/3.0);
    EXPECT_EQ((float)(1/3.0), *ptr_f32);
    EXPECT_THROW(a.vassign(1/3.0, assign_error_inexact), runtime_error);
    a.vassign(33554433);
    EXPECT_EQ(33554432, *ptr_f32);
    EXPECT_THROW(a.vassign(33554433, assign_error_inexact), runtime_error);
}

TEST(NDArrayAssign, ScalarAssignment_Float64) {
    ndarray a;
    double *ptr_f64;

    // Assignment to a double scalar
    a = ndarray(make_dtype<double>());
    ptr_f64 = (double *)a.get_originptr();
    a.vassign(true);
    EXPECT_EQ(1, *ptr_f64);
    a.vassign(false);
    EXPECT_EQ(0, *ptr_f64);
    a.vassign(1/3.0f);
    EXPECT_EQ(1/3.0f, *ptr_f64);
    a.vassign(1/3.0);
    EXPECT_EQ(1/3.0, *ptr_f64);
    a.vassign(33554433, assign_error_inexact);
    EXPECT_EQ(33554433, *ptr_f64);
    a.vassign(36028797018963969LL);
    EXPECT_EQ(36028797018963968LL, *ptr_f64);
    EXPECT_THROW(a.vassign(36028797018963969LL, assign_error_inexact), runtime_error);
}

TEST(DTypeAssign, BroadcastAssign) {
    ndarray a(2,3,4,make_dtype<float>());
    int v0[4] = {3,4,5,6};
    ndarray b = v0;

    // Broadcasts the 4-vector by a factor of 6,
    // converting the dtype
    a.vassign(b);
    float *ptr_f = (float *)a.get_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(3, *ptr_f++);
        EXPECT_EQ(4, *ptr_f++);
        EXPECT_EQ(5, *ptr_f++);
        EXPECT_EQ(6, *ptr_f++);
    }

    float v1[4] = {1.5, 2.5, 1.25, 2.25};
    b = v1;

    // Broadcasts the 4-vector by a factor of 6,
    // doesn't convert the dtype
    a.vassign(b);
    ptr_f = (float *)a.get_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(1.5, *ptr_f++);
        EXPECT_EQ(2.5, *ptr_f++);
        EXPECT_EQ(1.25, *ptr_f++);
        EXPECT_EQ(2.25, *ptr_f++);
    }

    double v2[3][1] = {{1.5}, {3.125}, {7.5}};
    b = v2;
    // Broadcasts the (3,1)-array by a factor of 8,
    // converting the dtype
    a.vassign(b);
    ptr_f = (float *)a.get_originptr();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(1.5, *ptr_f++);
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(3.125, *ptr_f++);
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(7.5, *ptr_f++);
    }

}
