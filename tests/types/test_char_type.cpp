//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>

using namespace std;
using namespace dynd;

TEST(CharDType, Create) {
    ndt::type d;

    // Chars of various encodings
    d = ndt::char_type::make();
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("char", d.str());
    EXPECT_EQ(ndt::type("char"), d);
    EXPECT_NE(ndt::type("char['ascii']"), d);
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::char_type::make(string_encoding_ascii);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_data_size());
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("char['ascii']", d.str());
    EXPECT_NE(ndt::type("char"), d);
    EXPECT_EQ(ndt::type("char['ascii']"), d);
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::char_type::make(string_encoding_ucs_2);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_data_size());
    EXPECT_EQ(2u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("char['ucs2']", d.str());
    EXPECT_NE(ndt::type("char"), d);
    EXPECT_EQ(ndt::type("char['ucs2']"), d);
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::char_type::make(string_encoding_utf_32);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("char", d.str());
    EXPECT_EQ(ndt::type("char"), d);
    EXPECT_EQ(ndt::type("char['utf32']"), d);
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(CharDType, Assign) {
    nd::array a, b, c;

    // Round-trip a string through a char assignment
    a = nd::array("t");
    EXPECT_EQ(a.get_type(), ndt::string_type::make());
    b = nd::empty(ndt::char_type::make());
    b.vals() = a;
    c = b.cast(ndt::string_type::make()).eval();
    EXPECT_EQ(c.get_type(), ndt::string_type::make());
    EXPECT_EQ("t", c.as<std::string>());
}
