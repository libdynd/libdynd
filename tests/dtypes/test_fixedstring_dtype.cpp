//
// Copyright (C) 2011-13, DyND Developers
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
    d = make_fixedstring_dtype(3, string_encoding_utf_8);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(3u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());

    d = make_fixedstring_dtype(129, string_encoding_utf_8);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(129u, d.get_data_size());

    d = make_fixedstring_dtype(129, string_encoding_ascii);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(129u, d.get_data_size());

    d = make_fixedstring_dtype(129, string_encoding_utf_16);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(2u, d.get_alignment());
    EXPECT_EQ(2u*129u, d.get_data_size());

    d = make_fixedstring_dtype(129, string_encoding_utf_32);
    EXPECT_EQ(fixedstring_type_id, d.get_type_id());
    EXPECT_EQ(string_kind, d.get_kind());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u*129u, d.get_data_size());
}

TEST(FixedstringDType, Basic) {
    ndobject a;

    // Trivial string going in and out of the system
    a = "abcdefg";
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), a.get_dtype());
    // Convert to a fixedstring dtype for testing
    a = a.cast_scalars(make_fixedstring_dtype(7, string_encoding_utf_8)).eval();
    EXPECT_EQ("abcdefg", a.as<string>());

    a = a.cast_scalars(make_fixedstring_dtype(7, string_encoding_utf_16));
    EXPECT_EQ(make_convert_dtype(make_fixedstring_dtype(7, string_encoding_utf_16),
                    make_fixedstring_dtype(7, string_encoding_utf_8)),
                a.get_dtype());
    a = a.eval();
    EXPECT_EQ(make_fixedstring_dtype(7, string_encoding_utf_16),
                    a.get_dtype());
    //cout << a << endl;
}

TEST(FixedstringDType, Casting) {
    ndobject a;

    a = empty(make_fixedstring_dtype(16, string_encoding_utf_16));
    // Fill up the string with values
    a.vals() = "0123456789012345";
    EXPECT_EQ("0123456789012345", a.as<std::string>());
    // Confirm that now assigning a smaller string works
    a.vals() = "abc";
    EXPECT_EQ("abc", a.as<std::string>());
}

TEST(FixedstringDType, SingleCompare) {
    ndobject a = make_strided_ndobject(2,
                    make_fixedstring_dtype(7, string_encoding_utf_8));

    a.at(0).vals() = "abc";
    a.at(1).vals() = "abd";

    // test ascii kernel
    a = a.eval();
    EXPECT_TRUE(a.at(0).op_sorting_less(a.at(1)));
    EXPECT_TRUE(a.at(0) < a.at(1));
    EXPECT_TRUE(a.at(0) <= a.at(1));
    EXPECT_FALSE(a.at(0) == a.at(1));
    EXPECT_TRUE(a.at(0) != a.at(1));
    EXPECT_FALSE(a.at(0) >= a.at(1));
    EXPECT_FALSE(a.at(0) > a.at(1));
    EXPECT_FALSE(a.at(0).equals_exact(a.at(1)));
    EXPECT_TRUE(a.at(0).equals_exact(a.at(0)));

    // TODO: means for not hardcoding expected results in utf string comparison tests

    // test utf8 kernel
    a = a.cast_scalars(make_fixedstring_dtype(7, string_encoding_utf_8));
    a = a.eval();
    EXPECT_TRUE(a.at(0).op_sorting_less(a.at(1)));
    EXPECT_TRUE(a.at(0) < a.at(1));
    EXPECT_TRUE(a.at(0) <= a.at(1));
    EXPECT_FALSE(a.at(0) == a.at(1));
    EXPECT_TRUE(a.at(0) != a.at(1));
    EXPECT_FALSE(a.at(0) >= a.at(1));
    EXPECT_FALSE(a.at(0) > a.at(1));
    EXPECT_FALSE(a.at(0).equals_exact(a.at(1)));
    EXPECT_TRUE(a.at(0).equals_exact(a.at(0)));

    // test utf16 kernel
    a = a.cast_scalars(make_fixedstring_dtype(7, string_encoding_utf_16));
    a = a.eval();
    EXPECT_TRUE(a.at(0).op_sorting_less(a.at(1)));
    EXPECT_TRUE(a.at(0) < a.at(1));
    EXPECT_TRUE(a.at(0) <= a.at(1));
    EXPECT_FALSE(a.at(0) == a.at(1));
    EXPECT_TRUE(a.at(0) != a.at(1));
    EXPECT_FALSE(a.at(0) >= a.at(1));
    EXPECT_FALSE(a.at(0) > a.at(1));
    EXPECT_FALSE(a.at(0).equals_exact(a.at(1)));
    EXPECT_TRUE(a.at(0).equals_exact(a.at(0)));

    // test utf32 kernel
    a = a.cast_scalars(make_fixedstring_dtype(7, string_encoding_utf_32));
    a = a.eval();
    EXPECT_TRUE(a.at(0).op_sorting_less(a.at(1)));
    EXPECT_TRUE(a.at(0) < a.at(1));
    EXPECT_TRUE(a.at(0) <= a.at(1));
    EXPECT_FALSE(a.at(0) == a.at(1));
    EXPECT_TRUE(a.at(0) != a.at(1));
    EXPECT_FALSE(a.at(0) >= a.at(1));
    EXPECT_FALSE(a.at(0) > a.at(1));
    EXPECT_FALSE(a.at(0).equals_exact(a.at(1)));
    EXPECT_TRUE(a.at(0).equals_exact(a.at(0)));
}

TEST(FixedstringDType, CanonicalDType) {
    EXPECT_EQ((make_fixedstring_dtype(12, string_encoding_ascii)),
                (make_fixedstring_dtype(12, string_encoding_ascii).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(14, string_encoding_utf_8)),
                (make_fixedstring_dtype(14, string_encoding_utf_8).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(17, string_encoding_utf_16)),
                (make_fixedstring_dtype(17, string_encoding_utf_16).get_canonical_dtype()));
    EXPECT_EQ((make_fixedstring_dtype(21, string_encoding_utf_32)),
                (make_fixedstring_dtype(21, string_encoding_utf_32).get_canonical_dtype()));
}
