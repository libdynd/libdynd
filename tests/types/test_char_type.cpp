//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>

using namespace std;
using namespace dynd;

TEST(CharDType, Create) {
    ndt::type d;

    // Chars of various encodings
    d = ndt::make_char();
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());

    d = ndt::make_char(string_encoding_ascii);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_data_size());
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());

    d = ndt::make_char(string_encoding_ucs_2);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_data_size());
    EXPECT_EQ(2u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());

    d = ndt::make_char(string_encoding_utf_32);
    EXPECT_EQ(char_type_id, d.get_type_id());
    EXPECT_EQ(char_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
}

TEST(CharDType, Assign) {
    nd::array a, b, c;

    // Round-trip a string through a char assignment
    a = nd::array("t");
    EXPECT_EQ(a.get_type(), ndt::make_string());
    b = nd::empty(ndt::make_char());
    b.vals() = a;
    c = b.cast(ndt::make_string()).eval();
    EXPECT_EQ(c.get_type(), ndt::make_string());
    EXPECT_EQ("t", c.as<string>());
}

