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

TEST(NDObjectCast, BasicCast) {
    ndobject n = 3.5;
    EXPECT_EQ(3.5f, n.cast(make_dtype<float>()).as<float>());
}
