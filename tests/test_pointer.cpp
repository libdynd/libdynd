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

#include <dynd/pointer.hpp>

using namespace std;
using namespace dynd;

TEST(Pointer, AddressOf)
{

}

/*
TEST(Pointer, Dereference)
{
  int vals[4] = {5, -1, 7, 3};

  nd::array a(vals);
  a = combine_into_tuple(1, &a);
  a = a(0);

  EXPECT_ARRAY_EQ(nd::array({5, -1, 7, 3}), nd::dereference(a));
}
*/
