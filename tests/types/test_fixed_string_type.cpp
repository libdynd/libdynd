//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

TEST(FixedstringDType, Create)
{
  ndt::type d;

  // Strings with various encodings and sizes
  d = ndt::fixed_string_type::make(3, string_encoding_utf_8);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(3u, d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::fixed_string_type::make(129, string_encoding_utf_8);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::fixed_string_type::make(129, string_encoding_ascii);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::fixed_string_type::make(129, string_encoding_utf_16);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_EQ(2u, d.get_data_alignment());
  EXPECT_EQ(2u * 129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::fixed_string_type::make(129, string_encoding_utf_32);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(string_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(4u * 129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(FixedStringDType, Encoding)
{
  ndt::type t;

  t = ndt::fixed_string_type::make(10, string_encoding_ascii);
  EXPECT_EQ("ascii", t.p<std::string>("encoding"));

  t = ndt::fixed_string_type::make(10, string_encoding_ucs_2);
  EXPECT_EQ("ucs2", t.p<std::string>("encoding"));

  t = ndt::fixed_string_type::make(10, string_encoding_utf_8);
  EXPECT_EQ("utf8", t.p<std::string>("encoding"));

  t = ndt::fixed_string_type::make(10, string_encoding_utf_16);
  EXPECT_EQ("utf16", t.p<std::string>("encoding"));

  EXPECT_THROW(ndt::fixed_string_type::make(10, string_encoding_invalid), std::runtime_error);
}

TEST(FixedStringDType, Basic)
{
  nd::array a;

  // Trivial string going in and out of the system
  a = "abcdefg";
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), a.get_type());
  // Convert to a fixed_string type for testing
  nd::array b = nd::empty(ndt::fixed_string_type::make(7, string_encoding_utf_8));
  b.assign(a);
  EXPECT_EQ("abcdefg", b.as<std::string>());
}

TEST(FixedstringDType, Casting)
{
  nd::array a;

  a = nd::empty(ndt::fixed_string_type::make(16, string_encoding_utf_16));
  // Fill up the string with values
  a.vals() = "0123456789012345";
  EXPECT_EQ("0123456789012345", a.as<std::string>());
  // Confirm that now assigning a smaller string works
  a.vals() = "abc";
  EXPECT_EQ("abc", a.as<std::string>());
}

TEST(FixedstringDType, CanonicalDType)
{
  EXPECT_EQ((ndt::fixed_string_type::make(12, string_encoding_ascii)),
            (ndt::fixed_string_type::make(12, string_encoding_ascii).get_canonical_type()));
  EXPECT_EQ((ndt::fixed_string_type::make(14, string_encoding_utf_8)),
            (ndt::fixed_string_type::make(14, string_encoding_utf_8).get_canonical_type()));
  EXPECT_EQ((ndt::fixed_string_type::make(17, string_encoding_utf_16)),
            (ndt::fixed_string_type::make(17, string_encoding_utf_16).get_canonical_type()));
  EXPECT_EQ((ndt::fixed_string_type::make(21, string_encoding_utf_32)),
            (ndt::fixed_string_type::make(21, string_encoding_utf_32).get_canonical_type()));
}
