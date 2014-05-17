// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/dynd_math.hpp>

using namespace std;
using namespace dynd;

TEST(Math, Constants) {
    EXPECT_EQ(DYND_E, dynd_e<double>());
    EXPECT_EQ(DYND_LOG2_E, dynd_log2_e<double>());
    EXPECT_EQ(DYND_LOG10_E, dynd_log10_e<double>());
    EXPECT_EQ(DYND_LOG_2, dynd_log_2<double>());
    EXPECT_EQ(DYND_LOG_10, dynd_log_10<double>());
    EXPECT_EQ(DYND_PI, dynd_pi<double>());
    EXPECT_EQ(DYND_2_MUL_PI, dynd_2_mul_pi<double>());
    EXPECT_EQ(DYND_PI_DIV_2, dynd_pi_div_2<double>());
    EXPECT_EQ(DYND_PI_DIV_4, dynd_pi_div_4<double>());
    EXPECT_EQ(DYND_1_DIV_PI, dynd_1_div_pi<double>());
    EXPECT_EQ(DYND_2_DIV_PI, dynd_2_div_pi<double>());
    EXPECT_EQ(DYND_SQRT_PI, dynd_sqrt_pi<double>());
    EXPECT_EQ(DYND_1_DIV_SQRT_2, dynd_1_div_sqrt_2<double>());
}
