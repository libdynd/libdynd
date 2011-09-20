#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArrayAssign, ScalarAssignment) {
    ndarray a;
    dnd_bool v_b;
    int8_t v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;
    uint8_t v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;
    float v_f32, *ptr_f32;
    double v_f64;

    // assignment to a float scalar
    a = ndarray(make_dtype<float>());
    ptr_f32 = (float *)a.data();
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
}
