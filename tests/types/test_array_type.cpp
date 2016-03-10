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
  ndt::type array_tp = ndt::array_type::make();
  EXPECT_EQ(array_id, array_tp.get_id());
  EXPECT_EQ(scalar_kind_id, array_tp.get_base_id());
  EXPECT_EQ(sizeof(nd::array), array_tp.get_data_size());
  EXPECT_EQ(sizeof(nd::array), array_tp.get_data_alignment());
  EXPECT_FALSE(array_tp.is_expression());
//  EXPECT_EQ(array_tp, ndt::type(array_tp.str())); // Round trip through a string
}
