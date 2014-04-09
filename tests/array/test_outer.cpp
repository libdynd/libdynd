//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/outer.hpp>

using namespace std;
using namespace dynd;

void func(int &res, int x, int y) {
    res = x + y;
}

TEST(Outer, IntFuncRefRes) {
    nd::array res, a, b;

    int aval[3] = {0, 1, 2};
    int bval[3] = {5, 2, 4};

    a = aval;
    b = bval;
    res = nd::outer(func, a, b);
}
