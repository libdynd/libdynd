//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/type.hpp>
#include <dynd/types/iteration_type.hpp>

using namespace std;
using namespace dynd;

TEST(IterationType, Constructor) {
  ndt::type iteration_tp = ndt::make_type<ndt::iteration_type>();
  EXPECT_EQ(iteration_id, iteration_tp.get_id());
  EXPECT_EQ(any_kind_id, iteration_tp.get_base_id());
  EXPECT_EQ(0u, iteration_tp.get_data_size());
  EXPECT_EQ(1u, iteration_tp.get_data_alignment());
  EXPECT_FALSE(iteration_tp.is_expression());
  EXPECT_TRUE(iteration_tp.is_symbolic());
  EXPECT_EQ(iteration_tp, ndt::type(iteration_tp.str())); // Round trip through a string
}
