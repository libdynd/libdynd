//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(FixedstringDType, Create) {
  ndt::type d;

  // Strings with various encodings and sizes
  d = ndt::make_type<ndt::fixed_string_type>(3, string_encoding_utf_8);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(fixed_string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(3u, d.get_data_size());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::make_type<ndt::fixed_string_type>(129, string_encoding_utf_8);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(fixed_string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::make_type<ndt::fixed_string_type>(129, string_encoding_ascii);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(fixed_string_kind_id, d.get_base_id());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::make_type<ndt::fixed_string_type>(129, string_encoding_utf_16);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(fixed_string_kind_id, d.get_base_id());
  EXPECT_EQ(2u, d.get_data_alignment());
  EXPECT_EQ(2u * 129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::make_type<ndt::fixed_string_type>(129, string_encoding_utf_32);
  EXPECT_EQ(fixed_string_id, d.get_id());
  EXPECT_EQ(fixed_string_kind_id, d.get_base_id());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(4u * 129u, d.get_data_size());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(FixedStringDType, Encoding) {
  ndt::type t;

  t = ndt::make_type<ndt::fixed_string_type>(10, string_encoding_ascii);
  EXPECT_EQ("ascii", t.p<std::string>("encoding"));

  t = ndt::make_type<ndt::fixed_string_type>(10, string_encoding_ucs_2);
  EXPECT_EQ("ucs2", t.p<std::string>("encoding"));

  t = ndt::make_type<ndt::fixed_string_type>(10, string_encoding_utf_8);
  EXPECT_EQ("utf8", t.p<std::string>("encoding"));

  t = ndt::make_type<ndt::fixed_string_type>(10, string_encoding_utf_16);
  EXPECT_EQ("utf16", t.p<std::string>("encoding"));

  EXPECT_THROW(ndt::make_type<ndt::fixed_string_type>(10, string_encoding_invalid), std::runtime_error);
}

TEST(FixedStringDType, Basic) {
  nd::array a;

  // Trivial string going in and out of the system
  a = "abcdefg";
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), a.get_type());
  // Convert to a fixed_string type for testing
  nd::array b = nd::empty(ndt::make_type<ndt::fixed_string_type>(7, string_encoding_utf_8));
  b.assign(a);
  EXPECT_EQ("abcdefg", b.as<std::string>());
}

TEST(FixedstringDType, Casting) {
  nd::array a;

  a = nd::empty(ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_16));
  // Fill up the string with values
  a.vals() = "0123456789012345";
  EXPECT_EQ("0123456789012345", a.as<std::string>());
  // Confirm that now assigning a smaller string works
  a.vals() = "abc";
  EXPECT_EQ("abc", a.as<std::string>());
}

TEST(FixedstringDType, CanonicalDType) {
  EXPECT_EQ((ndt::make_type<ndt::fixed_string_type>(12, string_encoding_ascii)),
            (ndt::make_type<ndt::fixed_string_type>(12, string_encoding_ascii).get_canonical_type()));
  EXPECT_EQ((ndt::make_type<ndt::fixed_string_type>(14, string_encoding_utf_8)),
            (ndt::make_type<ndt::fixed_string_type>(14, string_encoding_utf_8).get_canonical_type()));
  EXPECT_EQ((ndt::make_type<ndt::fixed_string_type>(17, string_encoding_utf_16)),
            (ndt::make_type<ndt::fixed_string_type>(17, string_encoding_utf_16).get_canonical_type()));
  EXPECT_EQ((ndt::make_type<ndt::fixed_string_type>(21, string_encoding_utf_32)),
            (ndt::make_type<ndt::fixed_string_type>(21, string_encoding_utf_32).get_canonical_type()));
}

TEST(FixedstringDType, Repr) {
  std::vector<const char *> roundtrip{"fixed_string[10, 'utf32']", "fixed_string[10]"};

  for (auto s : roundtrip) {
    EXPECT_TYPE_REPR_EQ(s, ndt::type(s));
  }
}
