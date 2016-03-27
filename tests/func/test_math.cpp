//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/math.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

TEST(Math, Sin)
{
  nd::array x = nd::random::uniform({}, {{"dst_tp", ndt::type("100 * float64")}});
  nd::sin(x);
}
