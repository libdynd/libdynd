// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/special.hpp>

using namespace std;
using namespace dynd;

#define REL_ERROR 1E-6

TEST(Special, BesselJ0) {
    bessel_y0(3);
}
