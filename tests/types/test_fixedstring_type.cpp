//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>

using namespace std;
using namespace dynd;

TEST(FixedstringDType, Create) {
    ndt::type d;

    // Strings with various encodings and sizes
    d = ndt::make_fixedstring(3, string_encoding_utf_8);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_EQ(3u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_fixedstring(129, string_encoding_utf_8);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_EQ(129u, d.get_data_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_fixedstring(129, string_encoding_ascii);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_EQ(129u, d.get_data_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_fixedstring(129, string_encoding_utf_16);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_data_alignment());
    EXPECT_EQ(2u*129u, d.get_data_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_fixedstring(129, string_encoding_utf_32);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_EQ(4u*129u, d.get_data_size());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(FixedstringDType, Basic) {
    nd::array a;

    // Trivial string going in and out of the system
    a = "abcdefg";
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8), a.get_type());
    // Convert to a fixedstring type for testing
    a = a.ucast(ndt::make_fixedstring(7, string_encoding_utf_8)).eval();
    EXPECT_EQ("abcdefg", a.as<string>());

    a = a.ucast(ndt::make_fixedstring(7, string_encoding_utf_16));
    EXPECT_EQ(ndt::make_convert(ndt::make_fixedstring(7, string_encoding_utf_16),
                    ndt::make_fixedstring(7, string_encoding_utf_8)),
                a.get_type());
    a = a.eval();
    EXPECT_EQ(ndt::make_fixedstring(7, string_encoding_utf_16),
                    a.get_type());
    //cout << a << endl;
}

TEST(FixedstringDType, Casting) {
    nd::array a;

    a = nd::empty(ndt::make_fixedstring(16, string_encoding_utf_16));
    // Fill up the string with values
    a.vals() = "0123456789012345";
    EXPECT_EQ("0123456789012345", a.as<std::string>());
    // Confirm that now assigning a smaller string works
    a.vals() = "abc";
    EXPECT_EQ("abc", a.as<std::string>());
}

TEST(FixedstringDType, SingleCompare) {
    nd::array a = nd::empty(2, ndt::make_fixedstring(7, string_encoding_utf_8));

    a(0).vals() = "abc";
    a(1).vals() = "abd";

    // test ascii kernel
    a = a.eval();
    EXPECT_TRUE(a(0).op_sorting_less(a(1)));
    EXPECT_TRUE(a(0) < a(1));
    EXPECT_TRUE(a(0) <= a(1));
    EXPECT_FALSE(a(0) == a(1));
    EXPECT_TRUE(a(0) != a(1));
    EXPECT_FALSE(a(0) >= a(1));
    EXPECT_FALSE(a(0) > a(1));
    EXPECT_FALSE(a(0).equals_exact(a(1)));
    EXPECT_TRUE(a(0).equals_exact(a(0)));

    // TODO: means for not hardcoding expected results in utf string comparison tests

    // test utf8 kernel
    a = a.ucast(ndt::make_fixedstring(7, string_encoding_utf_8));
    a = a.eval();
    EXPECT_TRUE(a(0).op_sorting_less(a(1)));
    EXPECT_TRUE(a(0) < a(1));
    EXPECT_TRUE(a(0) <= a(1));
    EXPECT_FALSE(a(0) == a(1));
    EXPECT_TRUE(a(0) != a(1));
    EXPECT_FALSE(a(0) >= a(1));
    EXPECT_FALSE(a(0) > a(1));
    EXPECT_FALSE(a(0).equals_exact(a(1)));
    EXPECT_TRUE(a(0).equals_exact(a(0)));

    // test utf16 kernel
    a = a.ucast(ndt::make_fixedstring(7, string_encoding_utf_16));
    a = a.eval();
    EXPECT_TRUE(a(0).op_sorting_less(a(1)));
    EXPECT_TRUE(a(0) < a(1));
    EXPECT_TRUE(a(0) <= a(1));
    EXPECT_FALSE(a(0) == a(1));
    EXPECT_TRUE(a(0) != a(1));
    EXPECT_FALSE(a(0) >= a(1));
    EXPECT_FALSE(a(0) > a(1));
    EXPECT_FALSE(a(0).equals_exact(a(1)));
    EXPECT_TRUE(a(0).equals_exact(a(0)));

    // test utf32 kernel
    a = a.ucast(ndt::make_fixedstring(7, string_encoding_utf_32));
    a = a.eval();
    EXPECT_TRUE(a(0).op_sorting_less(a(1)));
    EXPECT_TRUE(a(0) < a(1));
    EXPECT_TRUE(a(0) <= a(1));
    EXPECT_FALSE(a(0) == a(1));
    EXPECT_TRUE(a(0) != a(1));
    EXPECT_FALSE(a(0) >= a(1));
    EXPECT_FALSE(a(0) > a(1));
    EXPECT_FALSE(a(0).equals_exact(a(1)));
    EXPECT_TRUE(a(0).equals_exact(a(0)));
}

TEST(FixedstringDType, CanonicalDType) {
    EXPECT_EQ((ndt::make_fixedstring(12, string_encoding_ascii)),
                (ndt::make_fixedstring(12, string_encoding_ascii).get_canonical_type()));
    EXPECT_EQ((ndt::make_fixedstring(14, string_encoding_utf_8)),
                (ndt::make_fixedstring(14, string_encoding_utf_8).get_canonical_type()));
    EXPECT_EQ((ndt::make_fixedstring(17, string_encoding_utf_16)),
                (ndt::make_fixedstring(17, string_encoding_utf_16).get_canonical_type()));
    EXPECT_EQ((ndt::make_fixedstring(21, string_encoding_utf_32)),
                (ndt::make_fixedstring(21, string_encoding_utf_32).get_canonical_type()));
}
