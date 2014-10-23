//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "inc_gtest.hpp"

#include <dynd/func/functor_arrfunc.hpp>

using namespace std;
using namespace dynd;

double func(int x, double y) {
    return x + y;
}

TEST(Aux, Simple) {
    nd::arrfunc af = nd::make_functor_arrfunc<1>(func);
    EXPECT_EQ(3.5, af(1, 2.5).as<double>());
}
