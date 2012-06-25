//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

TEST(FixedstringDType, Create) {
    dtype d;

    // Strings with various encodings and sizes
    d = make_fixedstring_dtype(string_encoding_utf8, 3);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(3, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf8, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_ascii, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf16, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(2, d.alignment());
    EXPECT_EQ(2*129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf32, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(4, d.alignment());
    EXPECT_EQ(4*129, d.element_size());
}

TEST(FixedstringDType, Basic) {
    ndarray a;

    // Trivial string going in and out of the system
    a = std::string("abcdefg");
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf8, 7), a.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());

    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf16, 7));
    EXPECT_EQ(make_conversion_dtype(make_fixedstring_dtype(string_encoding_utf16, 7), make_fixedstring_dtype(string_encoding_utf8, 7)),
                a.get_dtype());
    a = a.vals();
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf16, 7),
                    a.get_dtype());
    //cout << a << endl;
}
