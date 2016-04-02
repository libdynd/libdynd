//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

/*
TEST(Float16, CommonType)
{
  EXPECT_TRUE((std::is_same<typename std::common_type<float16, float32>::type,
                            float32>::value));
  EXPECT_TRUE((std::is_same<typename std::common_type<float16, float64>::type,
                            float64>::value));
}
*/
