//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/ref_type.hpp>

using namespace std;
using namespace dynd;

TEST(RefType, Constructor)
{
  ndt::type ref_tp = ndt::ref_type::make(ndt::type::make<int32>());
  EXPECT_EQ(ref_type_id, ref_tp.get_type_id());
  EXPECT_EQ(expr_kind, ref_tp.get_kind());
  EXPECT_EQ(sizeof(nd::array), ref_tp.get_data_size());
  EXPECT_EQ(sizeof(nd::array), ref_tp.get_data_alignment());
  EXPECT_FALSE(ref_tp.is_expression());
  EXPECT_EQ(ref_tp, ndt::type(ref_tp.str())); // Round trip through a string
}

TEST(RefType, Null)
{
  nd::array a = nd::empty(ndt::type("ref[int32]"));
  EXPECT_FALSE(a.is_null());
  EXPECT_TRUE(a.underlying().is_null());
}
