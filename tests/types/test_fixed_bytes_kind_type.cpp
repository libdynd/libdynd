//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/types/fixed_bytes_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(FixedBytesKindType, Construction) {
  ndt::type tp = ndt::make_type<ndt::fixed_bytes_kind_type>();
  EXPECT_EQ(fixed_bytes_id, tp.get_id());
  EXPECT_EQ(bytes_kind_id, tp.get_base_id());
  EXPECT_EQ(0u, tp.get_data_alignment());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_FALSE(tp.is_expression());
  EXPECT_TRUE(tp.is_scalar());
  EXPECT_TRUE(tp.is_symbolic());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
}
