//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

TEST(BytesDType, Create)
{
  ndt::type d;

  // Strings with various alignments
  d = ndt::bytes_type::make(1);
  EXPECT_EQ(bytes_type_id, d.get_type_id());
  EXPECT_EQ(bytes_kind, d.get_kind());
  EXPECT_EQ(alignof(bytes), d.get_data_alignment());
  EXPECT_EQ(sizeof(bytes), d.get_data_size());
  EXPECT_EQ(1u, d.extended<ndt::bytes_type>()->get_target_alignment());
  EXPECT_EQ(1u, d.p("target_alignment").as<size_t>());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::bytes_type::make(2);
  EXPECT_EQ(bytes_type_id, d.get_type_id());
  EXPECT_EQ(bytes_kind, d.get_kind());
  EXPECT_EQ(alignof(bytes), d.get_data_alignment());
  EXPECT_EQ(sizeof(bytes), d.get_data_size());
  EXPECT_EQ(2u, d.extended<ndt::bytes_type>()->get_target_alignment());
  EXPECT_EQ(2u, d.p("target_alignment").as<size_t>());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::bytes_type::make(4);
  EXPECT_EQ(bytes_type_id, d.get_type_id());
  EXPECT_EQ(bytes_kind, d.get_kind());
  EXPECT_EQ(alignof(bytes), d.get_data_alignment());
  EXPECT_EQ(sizeof(bytes), d.get_data_size());
  EXPECT_EQ(4u, d.extended<ndt::bytes_type>()->get_target_alignment());
  EXPECT_EQ(4u, d.p("target_alignment").as<size_t>());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::bytes_type::make(8);
  EXPECT_EQ(bytes_type_id, d.get_type_id());
  EXPECT_EQ(bytes_kind, d.get_kind());
  EXPECT_EQ(alignof(bytes), d.get_data_alignment());
  EXPECT_EQ(sizeof(bytes), d.get_data_size());
  EXPECT_EQ(8u, d.extended<ndt::bytes_type>()->get_target_alignment());
  EXPECT_EQ(8u, d.p("target_alignment").as<size_t>());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::bytes_type::make(16);
  EXPECT_EQ(bytes_type_id, d.get_type_id());
  EXPECT_EQ(bytes_kind, d.get_kind());
  EXPECT_EQ(alignof(bytes), d.get_data_alignment());
  EXPECT_EQ(16u, d.extended<ndt::bytes_type>()->get_target_alignment());
  EXPECT_EQ(16u, d.p("target_alignment").as<size_t>());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(BytesDType, Assign)
{
  nd::array a, b, c;

  // Round-trip a string through a bytes assignment
  a = nd::array("testing").view_scalars(ndt::bytes_type::make(1));
  EXPECT_EQ(a.get_type(), ndt::bytes_type::make(1));
  b = nd::empty(ndt::bytes_type::make(1));
  b.vals() = a;
  c = b.view_scalars(ndt::string_type::make());
  EXPECT_EQ(c.get_type(), ndt::string_type::make());
  EXPECT_EQ("testing", c.as<std::string>());
}

TEST(BytesDType, Alignment)
{
  nd::array a;
  const bytes_type_data *btd;

  int64_t data[2] = {1, 2};
  a = nd::make_bytes_array(reinterpret_cast<const char *>(&data[0]), sizeof(data), 16);
  EXPECT_EQ(ndt::type("bytes[align=16]"), a.get_type());
  btd = reinterpret_cast<const bytes_type_data *>(a.get_readonly_originptr());
  EXPECT_TRUE(offset_is_aligned(reinterpret_cast<size_t>(btd->begin()), 16));
}

TEST(Bytes, Summary)
{
  stringstream ss;
  char x[100];
  for (size_t i = 0; i < sizeof(x); ++i) {
    x[i] = (char)i;
  }
  hexadecimal_print_summarized(ss, x, 10, 20);
  EXPECT_EQ("00010203040506070809", ss.str());
  ss.str("");
  ss.clear();
  hexadecimal_print_summarized(ss, x, 10, 12);
  EXPECT_EQ("0001 ... 09", ss.str());
}
