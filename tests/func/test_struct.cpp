//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../dynd_assertions.hpp"

#include <dynd/struct.hpp>

using namespace std;
using namespace dynd;


TEST(Struct, FieldAccess)
{
  nd::array s1 = nd::as_struct({{"x", 7}, {"y", 0.5}});
  EXPECT_EQ(s1, s1);

  EXPECT_EQ(7, nd::field_access(s1, "x"));
  EXPECT_EQ(0.5, nd::field_access(s1, "y"));

  nd::array a = nd::array({1,2,3,4,5,6,7,8,9,10,11,12});
  nd::array s2 = nd::as_struct({{"a", a}, {"s1", s1}});
  EXPECT_EQ(a, nd::field_access(s2, "a"));
  //EXPECT_EQ(s1, nd::field_access(s2, "s1"));
  EXPECT_EQ(12, nd::field_access(s2, "a")(11));
  EXPECT_EQ(0.5, nd::field_access(nd::field_access(s2, "s1"), "y"));

  nd::array b = nd::array({{1,2,3,4}, {5,6,7,8}, {9,10,11,12}});
  nd::array s3 = nd::as_struct({{"b", b}, {"s2", s2}});
  EXPECT_EQ(b, nd::field_access(s3, "b"));
  //EXPECT_EQ(s2, nd::field_access(s3, "s2"));
  EXPECT_EQ(7, nd::field_access(s3, "b")(1)(2));
  EXPECT_EQ(10, nd::field_access(nd::field_access(s3, "s2"), "a")(9));
}


