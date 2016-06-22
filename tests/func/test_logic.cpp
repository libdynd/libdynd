//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/logic.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;


TEST(All, FixedDim)
{
  EXPECT_ARRAY_EQ(true, nd::all(nd::array{true, true, true, true}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{true, false, true, false}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{false, false, false, false}));
}

TEST(All, FixedDimFixedDim)
{
  EXPECT_ARRAY_EQ(true, nd::all(nd::array{{true, true}, {true, true}}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{{true, false}, {true, false}}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{{true, false}, {true, false}, {false, true}}));
}

/*
TEST(All, FixedDimVarDim)
{
  EXPECT_ARRAY_EQ(true, nd::all(nd::array{{true}, {true, true}}));
  EXPECT_ARRAY_EQ(true, nd::all(nd::array{{true, true}, {true}}));
  EXPECT_ARRAY_EQ(true, nd::all(nd::array{{true}, {true, true}, {true, true, true}}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{{false}, {true, true}, {true, true, true}}));
  EXPECT_ARRAY_EQ(false, nd::all(nd::array{{false}, {true, false}, {true, false, true}}));
}
*/
