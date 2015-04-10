//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/types/c_contiguous_type.hpp>

using namespace std;
using namespace dynd;

TEST(CContiguousType, Basic)
{
  ndt::type tp = ndt::make_c_contiguous(ndt::type("10 * int32"));

  EXPECT_EQ(c_contiguous_type_id, tp.get_type_id());
  EXPECT_EQ(dim_kind, tp.get_kind());
  EXPECT_FALSE(tp.is_expression());
  EXPECT_EQ(tp, ndt::type("c_contiguous[10 * int32]"));
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));

  EXPECT_THROW(ndt::make_c_contiguous(ndt::type("int32")), invalid_argument);
}

TEST(CContiguousType, PatternMatch)
{
  EXPECT_TRUE(ndt::type("c_contiguous[10 * int32]")
                  .match(ndt::type("c_contiguous[10 * int32]")));
  EXPECT_TRUE(ndt::type("c_contiguous[10 * T]")
                  .match(ndt::type("c_contiguous[10 * int32]")));
  EXPECT_FALSE(ndt::type("c_contiguous[10 * int32]")
                   .match(ndt::type("c_contiguous[10 * T]")));
  EXPECT_TRUE(ndt::type("c_contiguous[Fixed * int32]")
                  .match(ndt::type("c_contiguous[10 * int32]")));
  EXPECT_FALSE(ndt::type("c_contiguous[10 * int32]")
                   .match(ndt::type("c_contiguous[Fixed * int32]")));

  EXPECT_FALSE(
      ndt::type("c_contiguous[Fixed * int32]").match(ndt::type("10 *int32")));
  EXPECT_FALSE(
      ndt::type("Fixed * int32").match(ndt::type("c_contiguous[10 *int32]")));

  nd::array a = nd::empty(ndt::type("c_contiguous[10 * int32]"));
  nd::array b = nd::empty(ndt::type("10 * int32"));
  EXPECT_TRUE(
      a.get_type().match(a.get_arrmeta(), b.get_type(), b.get_arrmeta()));
  EXPECT_TRUE(
      b.get_type().match(b.get_arrmeta(), a.get_type(), a.get_arrmeta()));
}