//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/array_type.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayType, Constructor)
{
  ndt::type array_tp = ndt::array_type::make(ndt::type::make<int32>());
  EXPECT_EQ(array_type_id, array_tp.get_type_id());
  EXPECT_EQ(expr_kind, array_tp.get_kind());
  EXPECT_EQ(sizeof(nd::array), array_tp.get_data_size());
  EXPECT_EQ(sizeof(nd::array), array_tp.get_data_alignment());
  EXPECT_FALSE(array_tp.is_expression());
  EXPECT_EQ(array_tp, ndt::type(array_tp.str())); // Round trip through a string
}

TEST(ArrayType, Null)
{
  nd::array a = nd::empty(ndt::type("array[int32]"));
  EXPECT_FALSE(a.is_null());
  EXPECT_TRUE(a.underlying().is_null());
}
