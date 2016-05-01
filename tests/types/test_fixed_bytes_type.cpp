//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "../dynd_assertions.hpp"
#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/types/fixed_bytes_type.hpp>

using namespace std;
using namespace dynd;

TEST(FixedBytesDType, Create) {
  ndt::type d;

  // Strings with various encodings and sizes
  d = ndt::make_type<ndt::fixed_bytes_type>(7, 1);
  EXPECT_EQ(fixed_bytes_id, d.get_id());
  EXPECT_EQ(bytes_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(7u, d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // Strings with various encodings and sizes
  d = ndt::make_type<ndt::fixed_bytes_type>(12, 4);
  EXPECT_EQ(fixed_bytes_id, d.get_id());
  EXPECT_EQ(bytes_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(12u, d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // Larger element than data size
  EXPECT_THROW(ndt::make_type<ndt::fixed_bytes_type>(1, 2), runtime_error);
  // Invalid alignment
  EXPECT_THROW(ndt::make_type<ndt::fixed_bytes_type>(6, 3), runtime_error);
  EXPECT_THROW(ndt::make_type<ndt::fixed_bytes_type>(10, 5), runtime_error);
  // Alignment must divide size
  EXPECT_THROW(ndt::make_type<ndt::fixed_bytes_type>(9, 4), runtime_error);
}

TEST(FixedBytesDType, Repr) {
  std::vector<const char *> roundtrip{"fixed_bytes[1600, align=8]", "fixed_bytes[10]"};

  for (auto s : roundtrip) {
    EXPECT_TYPE_REPR_EQ(s, ndt::type(s));
  }
}
