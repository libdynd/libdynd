//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/ndobject_arange.hpp>

using namespace std;
using namespace dynd;

TEST(CategoricalDType, Create) {
    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "baz";

    dtype d;
    d = make_categorical_dtype(a);
    EXPECT_EQ(categorical_type_id, d.get_type_id());
    EXPECT_EQ(custom_kind, d.get_kind());
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(1u, d.get_data_size());
    EXPECT_FALSE(d.is_expression());

    // With < 256 categories, storage is a uint8
    a = arange(256);
    d = make_categorical_dtype(a);
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_EQ(1u, d.get_data_size());

    // With < 32768 categories, storage is a uint16
    a = arange(257);
    d = make_categorical_dtype(a);
    EXPECT_EQ(2u, d.get_alignment());
    EXPECT_EQ(2u, d.get_data_size());
    a = arange(32768);
    d = make_categorical_dtype(a);
    EXPECT_EQ(2u, d.get_alignment());
    EXPECT_EQ(2u, d.get_data_size());

    // Otherwise, storage is a uint32
    a = arange(32769);
    d = make_categorical_dtype(a);
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u, d.get_data_size());
}

TEST(CategoricalDType, Convert) {
    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "baz";

    dtype cd = make_categorical_dtype(a);
    dtype sd = make_string_dtype(string_encoding_utf_8);

    EXPECT_TRUE(is_lossless_assignment(sd, cd));
    EXPECT_FALSE(is_lossless_assignment(cd, sd));

    // This operation was crashing, hence the test
    dtype cvt = make_convert_dtype(sd, cd);
    EXPECT_EQ(cd, cvt.operand_dtype());
    EXPECT_EQ(sd, cvt.value_dtype());
}

TEST(CategoricalDType, Compare) {

    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "baz";

    ndobject b = make_strided_ndobject(2, make_fixedstring_dtype(string_encoding_ascii, 3));
    b.at(0).vals() = "foo";
    b.at(1).vals() = "bar";

    dtype da = make_categorical_dtype(a);
    dtype da2 = make_categorical_dtype(a);
    dtype db = make_categorical_dtype(b);

    EXPECT_TRUE(da == da);
    EXPECT_TRUE( da == da2);
    EXPECT_FALSE(da == db);

    ndobject i = make_strided_ndobject(3, make_dtype<int32_t>());
    i.at(0).vals() = 0;
    i.at(1).vals() = 10;
    i.at(2).vals() = 100;

    dtype di = make_categorical_dtype(i);
    EXPECT_FALSE(da == di);
}

TEST(CategoricalDType, Unique) {

    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "foo";

    EXPECT_THROW(make_categorical_dtype(a), std::runtime_error);

    ndobject i = make_strided_ndobject(3, make_dtype<int32_t>());
    i.at(0).vals() = 0;
    i.at(1).vals() = 10;
    i.at(2).vals() = 10;

    EXPECT_THROW(make_categorical_dtype(i), std::runtime_error);

}

TEST(CategoricalDType, FactorFixedString) {
    ndobject string_cats = make_strided_ndobject(2, make_fixedstring_dtype(string_encoding_ascii, 3));
    string_cats.at(0).vals() = "bar";
    string_cats.at(1).vals() = "foo";

    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "foo";

    dtype da = factor_categorical_dtype(a);
    EXPECT_EQ(make_categorical_dtype(string_cats), da);
}

TEST(CategoricalDType, FactorString) {
    const char *cats_vals[] = {"bar", "foo", "foot"};
    const char *a_vals[] = {"foo", "bar", "foot", "foo", "bar"};
    ndobject cats = cats_vals, a = a_vals;

    dtype da = factor_categorical_dtype(a);
    EXPECT_EQ(make_categorical_dtype(cats), da);
}

TEST(CategoricalDType, FactorStringLonger) {
    const char *cats_vals[] = {"a", "abcdefghijklmnopqrstuvwxyz", "bar", "foo", "foot", "z"};
    const char *a_vals[] = {"foo", "bar", "foot", "foo", "bar", "abcdefghijklmnopqrstuvwxyz",
                    "foot", "foo", "z", "a", "abcdefghijklmnopqrstuvwxyz"};
    dtype da = factor_categorical_dtype(a_vals);
    EXPECT_EQ(make_categorical_dtype(cats_vals), da);
}

TEST(CategoricalDType, FactorInt) {
    ndobject int_cats = make_strided_ndobject(2, make_dtype<int32_t>());
    int_cats.at(0).vals() = 0;
    int_cats.at(1).vals() = 10;

    ndobject i = make_strided_ndobject(3, make_dtype<int32_t>());
    i.at(0).vals() = 10;
    i.at(1).vals() = 10;
    i.at(2).vals() = 0;

    dtype di = factor_categorical_dtype(i);
    EXPECT_EQ(make_categorical_dtype(int_cats), di);
}

TEST(CategoricalDType, Values) {
    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "baz";

    dtype dt = make_categorical_dtype(a);

    EXPECT_EQ(0u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(0)));
    EXPECT_EQ(1u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(1)));
    EXPECT_EQ(2u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(2)));
    EXPECT_EQ(0u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("foo"));
    EXPECT_EQ(1u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("bar"));
    EXPECT_EQ(2u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("baz"));
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("aaa"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("ddd"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("zzz"), std::runtime_error);
}

TEST(CategoricalDType, ValuesLonger) {
    const char *cats_vals[] = {"foo", "abcdefghijklmnopqrstuvwxyz", "z", "bar", "a", "foot"};
    const char *a_vals[] = {"foo", "z", "abcdefghijklmnopqrstuvwxyz",
                    "z", "bar", "a", "foot", "a", "abcdefghijklmnopqrstuvwxyz", "foo", "bar", "foo", "foot"};
    uint32_t a_uints[] = {0, 2, 1, 2, 3, 4, 5, 4, 1, 0, 3, 0, 5};
    int cats_count = sizeof(cats_vals) / sizeof(cats_vals[0]);
    int a_count = sizeof(a_uints) / sizeof(a_uints[0]);

    dtype dt = make_categorical_dtype(cats_vals);
    ndobject a = ndobject(a_vals).cast_udtype(dt).vals();
    ndobject a_view = a.p("category_ints");

    // Check that the categories got the right values
    for (int i = 0; i < cats_count; ++i) {
        EXPECT_EQ((uint32_t)i, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(cats_vals[i]));
    }
    // Check that everything in 'a' is right
    for (int i = 0; i < a_count; ++i) {
        EXPECT_EQ(a_vals[i], a.at(i).as<string>());
        EXPECT_EQ(a_uints[i], a_view.at(i).as<uint32_t>());
    }
}

TEST(CategoricalDType, AssignFixedString) {
    ndobject cat = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    cat.at(0).vals() = "foo";
    cat.at(1).vals() = "bar";
    cat.at(2).vals() = "baz";

    dtype dt = make_categorical_dtype(cat);

    ndobject a = make_strided_ndobject(3, dt);
    a.val_assign(cat);
    EXPECT_EQ("foo", a.at(0).as<string>());
    EXPECT_EQ("bar", a.at(1).as<string>());
    EXPECT_EQ("baz", a.at(2).as<string>());
    a.at(0).vals() = cat.at(2);
    EXPECT_EQ("baz", a.at(0).as<string>());

    cat.at(0).vals() = string("zzz");
    EXPECT_THROW(a.at(0).vals() = cat.at(0), std::runtime_error);

    // TODO implicit conversion?
    //a(0).vals() = string("bar");
    //cout << a << endl;

    ndobject tmp = make_strided_ndobject(3, cat.get_dtype().at(0));
    tmp.val_assign(a);
    EXPECT_EQ("baz", tmp.at(0).as<string>());
    EXPECT_EQ("bar", tmp.at(1).as<string>());
    EXPECT_EQ("baz", tmp.at(2).as<string>());
    tmp.at(0).vals() = a.at(1);
    EXPECT_EQ("bar", tmp.at(0).as<string>());
}

TEST(CategoricalDType, AssignInt) {

    ndobject cat = make_strided_ndobject(3, make_dtype<int32_t>());
    cat.at(0).vals() = 10;
    cat.at(1).vals() = 100;
    cat.at(2).vals() = 1000;

    dtype dt = make_categorical_dtype(cat);

    ndobject a = make_strided_ndobject(3, dt);
    a.val_assign(cat);
    EXPECT_EQ(10, a.at(0).as<int32_t>());
    EXPECT_EQ(100, a.at(1).as<int32_t>());
    EXPECT_EQ(1000, a.at(2).as<int32_t>());
    a.at(0).vals() = cat.at(2);
    EXPECT_EQ(1000, a.at(0).as<int32_t>());

    // TODO implicit conversion?
    //a(0).vals() = string("bar");
    //cout << a << endl;

    ndobject tmp = make_strided_ndobject(3, cat.get_dtype().at(0));
    tmp.val_assign(a);
    EXPECT_EQ(1000, tmp.at(0).as<int32_t>());
    EXPECT_EQ(100, tmp.at(1).as<int32_t>());
    EXPECT_EQ(1000, tmp.at(2).as<int32_t>());
    tmp.at(0).vals() = a.at(1);
    EXPECT_EQ(100, tmp.at(0).as<int32_t>());

}

TEST(CategoricalDType, AssignRange) {

    ndobject cat = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    cat.at(0).vals() = "foo";
    cat.at(1).vals() = "bar";
    cat.at(2).vals() = "baz";

    dtype dt = make_categorical_dtype(cat);

    ndobject a = make_strided_ndobject(9, dt);
    ndobject b = a.at(0 <= irange() < 3);
    b.val_assign(cat);
    ndobject c = a.at(3 <= irange() < 6 );
    c.val_assign(cat.at(0));
    ndobject d = a.at(6 <= irange() / 2 < 9 );
    d.val_assign(cat.at(1));
    a.at(7).vals() = cat.at(2);

    EXPECT_EQ("foo", a.at(0).as<string>());
    EXPECT_EQ("bar", a.at(1).as<string>());
    EXPECT_EQ("baz", a.at(2).as<string>());
    EXPECT_EQ("foo", a.at(3).as<string>());
    EXPECT_EQ("foo", a.at(4).as<string>());
    EXPECT_EQ("foo", a.at(5).as<string>());
    EXPECT_EQ("bar", a.at(6).as<string>());
    EXPECT_EQ("baz", a.at(7).as<string>());
    EXPECT_EQ("bar", a.at(8).as<string>());
}

TEST(CategoricalDType, CategoriesProperty) {
    const char *cats_vals[] = {"this", "is", "a", "test"};
    ndobject cats = cats_vals;
    dtype cd = make_categorical_dtype(cats_vals);
    EXPECT_TRUE(cats.equals_exact(cd.p("categories")));
}
