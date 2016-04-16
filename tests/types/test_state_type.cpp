//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/type.hpp>
#include <dynd/types/state_type.hpp>

using namespace std;
using namespace dynd;

TEST(StateType, Constructor) {
  ndt::type state_tp = ndt::make_type<ndt::state_type>();
  EXPECT_EQ(state_id, state_tp.get_id());
  EXPECT_EQ(any_kind_id, state_tp.get_base_id());
  EXPECT_EQ(0u, state_tp.get_data_size());
  EXPECT_EQ(1u, state_tp.get_data_alignment());
  EXPECT_FALSE(state_tp.is_expression());
  EXPECT_TRUE(state_tp.is_symbolic());
  EXPECT_EQ(state_tp, ndt::type(state_tp.str())); // Round trip through a string
}
