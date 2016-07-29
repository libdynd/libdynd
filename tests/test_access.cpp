//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/access.hpp>
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

TEST(Struct, FieldAccess) {
  nd::array s1 = nd::as_struct({{"x", 7}, {"y", 0.5}});
  EXPECT_ARRAY_EQ(7, nd::field_access(s1, "x"));
  EXPECT_ARRAY_EQ(0.5, nd::field_access(s1, "y"));
  //  EXPECT_THROW(nd::field_access(s1, "z"), std::invalid_argument);

  nd::array a = nd::array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  nd::array s2 = nd::as_struct({{"a", a}, {"s1", s1}});
  EXPECT_ARRAY_EQ(a, nd::field_access(s2, "a"));
  EXPECT_ARRAY_EQ(s1, nd::field_access(s2, "s1"));
  EXPECT_ARRAY_EQ(12, nd::field_access(s2, "a")(11));
  EXPECT_ARRAY_EQ(0.5, nd::field_access(nd::field_access(s2, "s1"), "y"));

  nd::array b = nd::array({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  nd::array s3 = nd::as_struct({{"b", b}, {"s2", s2}});
  EXPECT_ARRAY_EQ(b, nd::field_access(s3, "b"));
  // EXPECT_ARRAY_EQ(s2, nd::field_access(s3, "s2"));

  EXPECT_ARRAY_EQ(7, nd::field_access(s3, "b")(1)(2));
  EXPECT_ARRAY_EQ(10, nd::field_access(nd::field_access(s3, "s2"), "a")(9));
}
