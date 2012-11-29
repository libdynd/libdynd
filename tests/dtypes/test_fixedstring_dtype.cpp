//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(FixedstringDType, Create) {
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

TEST(FixedstringDType, Basic) {
    ndobject a;

    // Trivial string going in and out of the system
    a = std::string("abcdefg");
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), a.get_dtype());
    // Convert to a fixedstring dtype for testing
    a = a.cast_scalars(make_fixedstring_dtype(string_encoding_utf_8, 7)).vals();
    EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());

    a = a.cast_scalars(make_fixedstring_dtype(string_encoding_utf_16, 7));
    EXPECT_EQ(make_convert_dtype(make_fixedstring_dtype(string_encoding_utf_16, 7), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                a.get_dtype());
    a = a.vals();
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_16, 7),
                    a.get_dtype());
    //cout << a << endl;
}

TEST(FixedstringDType, Casting) {
    ndobject a;

    a = ndobject(make_fixedstring_dtype(string_encoding_utf_16, 16));
    // Fill up the string with values
    a.vals() = std::string("0123456789012345");
    EXPECT_EQ("0123456789012345", a.as<std::string>());
    // Confirm that now assigning a smaller string works
    a.vals() = std::string("abc");
    EXPECT_EQ("abc", a.as<std::string>());
}

TEST(FixedstringDType, SingleCompare) {
    ndobject a = make_strided_ndobject(2, make_fixedstring_dtype(string_encoding_utf_8, 7));
    single_compare_kernel_instance k;

    a.at(0).vals() = std::string("abc");
    a.at(1).vals() = std::string("abd");

    // test ascii kernel
    a = a.vals();
    a.get_dtype().at(0).get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() < a.at(1).as<std::string>());
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() <= a.at(1).as<std::string>());
    EXPECT_EQ(k.comparisons[equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() == a.at(1).as<std::string>());
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() != a.at(1).as<std::string>());
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() >= a.at(1).as<std::string>());
    EXPECT_EQ(k.comparisons[greater_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), a.at(0).as<std::string>() > a.at(1).as<std::string>());

    // TODO: means for not hardcoding expected results in utf string comparison tests

    // test utf8 kernel
    a = a.cast_scalars(make_fixedstring_dtype(string_encoding_utf_8, 7));
    a = a.vals();
    a.get_dtype().at(0).get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);

    // test utf16 kernel
    a = a.cast_scalars(make_fixedstring_dtype(string_encoding_utf_16, 7));
    a = a.vals();
    a.get_dtype().at(0).get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);

    // test utf32 kernel
    a = a.cast_scalars(make_fixedstring_dtype(string_encoding_utf_32, 7));
    a = a.vals();
    a.get_dtype().at(0).get_single_compare_kernel(k);
    EXPECT_EQ(k.comparisons[less_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[less_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[not_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), true);
    EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
    EXPECT_EQ(k.comparisons[greater_id]((char *)a.at(0).get_readonly_originptr(), (char *)a.at(1).get_readonly_originptr(), k.auxdata), false);
}

TEST(FixedstringDType, CanonicalDType) {
    // The canonical dtype of a fixedstring dtype is always a UTF8 string dtype
    EXPECT_EQ((make_fixedstring_dtype(string_encoding_ascii, 12)),
                (make_fixedstring_dtype(string_encoding_ascii, 12).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(string_encoding_utf_8, 14)),
                (make_fixedstring_dtype(string_encoding_utf_8, 14).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(string_encoding_utf_16, 17)),
                (make_fixedstring_dtype(string_encoding_utf_16, 17).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(string_encoding_utf_32, 21)),
                (make_fixedstring_dtype(string_encoding_utf_32, 21).get_canonical_dtype()));
}
