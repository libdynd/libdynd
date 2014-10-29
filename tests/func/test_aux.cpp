//
// Copyright (C) 2011-14 DyND Developers
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

int func(int x, int y, int z) {
    return x * y - z;
}

struct cls {
    int m_x, m_y;

    cls(int x, int y) : m_x(x + 2), m_y(y + 3) {
    }

    int operator ()(int z) {
        return m_x * m_y - z;
    }
};

TEST(Aux, Simple) {
    nd::arrfunc af;

    af = nd::make_functor_arrfunc<2>(func);
    EXPECT_EQ(1, af.get()->get_nsrc());
    EXPECT_EQ(2, af.get()->get_naux());
    EXPECT_EQ(8, af(3, 5, 7).as<int>());

    af = nd::make_functor_arrfunc<int, int, cls>();
    EXPECT_EQ(1, af.get()->get_nsrc());
    EXPECT_EQ(2, af.get()->get_naux());
    EXPECT_EQ(8, af(7, 1, 2).as<int>());
}
