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
#include <dnd/dtypes/categorical_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dnd;

TEST(CategoricalDType, Create) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("baz");


    dtype d;

    // Strings with various encodings and sizes
    d = make_categorical_dtype(a);
    EXPECT_EQ(categorical_type_id, d.type_id());
    EXPECT_EQ(custom_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(4, d.element_size());

    d.extended()->print_dtype(cout);

}

TEST(CategoricalDType, Compare) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("baz");

    ndarray b(2, make_fixedstring_dtype(string_encoding_ascii, 3));
    b(0).vals() = std::string("foo");
    b(1).vals() = std::string("bar");

    dtype da = make_categorical_dtype(a);
    dtype da2 = make_categorical_dtype(a);
    dtype db = make_categorical_dtype(b);

    EXPECT_EQ(true, da == da);
    EXPECT_EQ(true, da == da2);
    EXPECT_EQ(false, da == db);

    ndarray i(3, make_dtype<int32_t>());
    i(0).vals() = 0;
    i(1).vals() = 10;
    i(2).vals() = 100;

    dtype di = make_categorical_dtype(i);
    EXPECT_EQ(false, da == di);
    di.extended()->print_dtype(cout);

}

TEST(CategoricalDType, Values) {

    ndarray a(2, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(1).vals() = std::string("baz");

    dtype dt = make_categorical_dtype(a);

    EXPECT_EQ(0, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a(0).get_readonly_originptr()));
    EXPECT_EQ(1, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a(1).get_readonly_originptr()));
    EXPECT_EQ(-1, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("aaa"));
    EXPECT_EQ(-1, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("ddd"));
    EXPECT_EQ(-1, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("zzz"));

}


