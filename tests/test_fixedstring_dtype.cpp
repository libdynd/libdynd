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
    d = make_fixedstring_dtype(string_encoding_utf_8, 3);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(3, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf_8, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_ascii, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf_16, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(2, d.alignment());
    EXPECT_EQ(2*129, d.element_size());

    d = make_fixedstring_dtype(string_encoding_utf_32, 129);
    EXPECT_EQ(fixedstring_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(4, d.alignment());
    EXPECT_EQ(4*129, d.element_size());
}

TEST(FixedstringDType, Basic) {
    ndarray a;

    // Trivial string going in and out of the system
    a = std::string("abcdefg");
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 7), a.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());

    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_16, 7));
    EXPECT_EQ(make_conversion_dtype(make_fixedstring_dtype(string_encoding_utf_16, 7), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                a.get_dtype());
    a = a.vals();
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_16, 7),
                    a.get_dtype());
    //cout << a << endl;
}

TEST(FixedstringDType, SingleCompare) {
    ndarray a(2, make_fixedstring_dtype(string_encoding_utf_8, 7));
    single_compare_kernel_instance k;

    a(0).vals() = std::string("abc");
    a(1).vals() = std::string("abd");

    // test ascii kernel
    a = a.vals();
    a.get_dtype().get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() < a(1).as<std::string>());
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() <= a(1).as<std::string>());
    EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() == a(1).as<std::string>());
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() != a(1).as<std::string>());
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() >= a(1).as<std::string>());
    EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), a(0).as<std::string>() > a(1).as<std::string>());

    // TODO: means for not hardcoding expected results in utf string comparison tests

    // test utf8 kernel
    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_8, 7));
    a = a.vals();
    a.get_dtype().get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);

    // test utf16 kernel
    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_16, 7));
    a = a.vals();
    a.get_dtype().get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);

    // test utf32 kernel
    a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_32, 7));
    a = a.vals();
    a.get_dtype().get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_originptr(), (char *)a(1).get_originptr(), k.auxdata), false);

}
