//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include "../dynd_assertions.hpp"
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

TEST(Where, Untitled) {
  nd::callable f = nd::functional::where([](int x) { return x < 11; });
  nd::array res = f(nd::array{9, 34, 1, -7, 23});
  EXPECT_ARRAY_EQ(nd::array({static_cast<intptr_t>(0)}), res(0));
  EXPECT_ARRAY_EQ(nd::array({static_cast<intptr_t>(2)}), res(1));
  EXPECT_ARRAY_EQ(nd::array({static_cast<intptr_t>(3)}), res(2));
}
