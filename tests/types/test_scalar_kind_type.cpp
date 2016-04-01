//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "dynd_assertions.hpp"

#include "inc_gtest.hpp"

#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(ScalarKindType, Basic)
{
  ndt::type tp = ndt::scalar_kind_type::make();
  EXPECT_EQ(scalar_kind_id, tp.get_id());
  EXPECT_TRUE(tp.is_scalar());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_EQ(0, tp.get_ndim());
  EXPECT_EQ("Scalar", tp.str());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(ScalarKindType, Match)
{
  ndt::type tp = ndt::scalar_kind_type::make();
  EXPECT_TRUE(tp.match(ndt::type("Scalar")));
  EXPECT_TRUE(tp.match(ndt::type("int32")));
  EXPECT_TRUE(tp.match(ndt::type("float64")));
  EXPECT_TRUE(tp.match(ndt::type("(int32, float64)")));
  EXPECT_TRUE(tp.match(ndt::type("(int32, float64, ...)")));
  EXPECT_FALSE(tp.match(ndt::type("Any")));
  EXPECT_FALSE(tp.match(ndt::type("4 * int32")));
  EXPECT_FALSE(tp.match(ndt::type("Fixed * Any")));
  EXPECT_FALSE(tp.match(ndt::type("Dims... * float64")));
}
