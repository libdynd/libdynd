//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DateDType, Create) {
    dtype d;

    // Strings with various encodings and sizes
    d = make_fixedstring_dtype(string_encoding_utf_8, 3);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(3u, d.get_element_size());

    d = make_fixedstring_dtype(string_encoding_utf_8, 129);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(129u, d.get_element_size());

    d = make_fixedstring_dtype(string_encoding_ascii, 129);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(129u, d.get_element_size());

    d = make_fixedstring_dtype(string_encoding_utf_16, 129);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_alignment());
    EXPECT_EQ(2u*129u, d.get_element_size());

    d = make_fixedstring_dtype(string_encoding_utf_32, 129);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u*129u, d.get_element_size());
}
