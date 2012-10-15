//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndarray.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(CategoricalDType, Create) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("baz");

    dtype d;

    d = make_categorical_dtype(a);
    EXPECT_EQ(categorical_type_id, d.type_id());
    EXPECT_EQ(custom_kind, d.kind());
    EXPECT_EQ(4u, d.alignment());
    EXPECT_EQ(4u, d.element_size());

    cout << d << endl;

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

    EXPECT_TRUE(da == da);
    EXPECT_TRUE( da == da2);
    EXPECT_FALSE(da == db);

    ndarray i(3, make_dtype<int32_t>());
    i(0).vals() = 0;
    i(1).vals() = 10;
    i(2).vals() = 100;

    dtype di = make_categorical_dtype(i);
    EXPECT_FALSE(da == di);

    // cout << di << endl;

}

TEST(CategoricalDType, Unique) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("foo");

    EXPECT_THROW(make_categorical_dtype(a), std::runtime_error);

    ndarray i(3, make_dtype<int32_t>());
    i(0).vals() = 0;
    i(1).vals() = 10;
    i(2).vals() = 10;

    EXPECT_THROW(make_categorical_dtype(i), std::runtime_error);

}

TEST(CategoricalDType, Factor) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("foo");

    dtype da = factor_categorical_dtype(a);

    cout << da << endl;

    ndarray i(3, make_dtype<int32_t>());
    i(0).vals() = 10;
    i(1).vals() = 10;
    i(2).vals() = 0;

    dtype di = factor_categorical_dtype(i);

    cout << di << endl;

}

TEST(CategoricalDType, Values) {

    ndarray a(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a(0).vals() = std::string("foo");
    a(1).vals() = std::string("bar");
    a(2).vals() = std::string("baz");

    dtype dt = make_categorical_dtype(a);

    EXPECT_EQ(0u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a(0).get_readonly_originptr()));
    EXPECT_EQ(1u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a(1).get_readonly_originptr()));
    EXPECT_EQ(2u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a(2).get_readonly_originptr()));
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("aaa"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("ddd"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("zzz"), std::runtime_error);

}

TEST(CategoricalDType, AssignFixedString) {

    ndarray cat(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    cat(0).vals() = std::string("foo");
    cat(1).vals() = std::string("bar");
    cat(2).vals() = std::string("baz");

    dtype dt = make_categorical_dtype(cat);

    ndarray a(3, dt);
    a.val_assign(cat);
    EXPECT_EQ("foo", a(0).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("bar", a(1).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("baz", a(2).as_dtype(cat.get_dtype()).as<std::string>());
    cout << a << endl;
    a(0).vals() = cat(2);
    EXPECT_EQ("baz", a(0).as_dtype(cat.get_dtype()).as<std::string>());
    cout << a << endl;

    cat(0).vals() = std::string("zzz");
    EXPECT_THROW(a(0).vals() = cat(0), std::runtime_error);

    // TODO implicit conversion?
    //a(0).vals() = std::string("bar");
    //cout << a << endl;

    ndarray tmp(3, cat.get_dtype());
    tmp.val_assign(a);
    EXPECT_EQ("baz", tmp(0).as<std::string>());
    EXPECT_EQ("bar", tmp(1).as<std::string>());
    EXPECT_EQ("baz", tmp(2).as<std::string>());
    cout << tmp << endl;
    tmp(0).vals() = a(1);
    EXPECT_EQ("bar", tmp(0).as<std::string>());
    cout << tmp << endl;

}

TEST(CategoricalDType, AssignInt) {

    ndarray cat(3, make_dtype<int32_t>());
    cat(0).vals() = 10;
    cat(1).vals() = 100;
    cat(2).vals() = 1000;

    dtype dt = make_categorical_dtype(cat);

    ndarray a(3, dt);
    a.val_assign(cat);
    EXPECT_EQ(10, a(0).as_dtype(cat.get_dtype()).as<int32_t>());
    EXPECT_EQ(100, a(1).as_dtype(cat.get_dtype()).as<int32_t>());
    EXPECT_EQ(1000, a(2).as_dtype(cat.get_dtype()).as<int32_t>());
    cout << a << endl;
    a(0).vals() = cat(2);
    EXPECT_EQ(1000, a(0).as_dtype(cat.get_dtype()).as<int32_t>());
    cout << a << endl;

    // TODO implicit conversion?
    //a(0).vals() = std::string("bar");
    //cout << a << endl;

    ndarray tmp(3, cat.get_dtype());
    tmp.val_assign(a);
    EXPECT_EQ(1000, tmp(0).as_dtype(cat.get_dtype()).as<int32_t>());
    EXPECT_EQ(100, tmp(1).as_dtype(cat.get_dtype()).as<int32_t>());
    EXPECT_EQ(1000, tmp(2).as_dtype(cat.get_dtype()).as<int32_t>());
    cout << tmp << endl;
    tmp(0).vals() = a(1);
    EXPECT_EQ(100, tmp(0).as_dtype(cat.get_dtype()).as<int32_t>());
    cout << tmp << endl;

}

TEST(CategoricalDType, AssignRange) {

    ndarray cat(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    cat(0).vals() = std::string("foo");
    cat(1).vals() = std::string("bar");
    cat(2).vals() = std::string("baz");

    dtype dt = make_categorical_dtype(cat);

    ndarray a(9, dt);
    ndarray b = a(0 <= irange() < 3);
    b.val_assign(cat);
    ndarray c = a(3 <= irange() < 6 );
    c.val_assign(cat(0));
    ndarray d = a(6 <= irange() / 2 < 9 );
    d.val_assign(cat(1));
    a(7).vals() = cat(2);

    EXPECT_EQ("foo", a(0).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("bar", a(1).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("baz", a(2).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("foo", a(3).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("foo", a(4).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("foo", a(5).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("bar", a(6).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("baz", a(7).as_dtype(cat.get_dtype()).as<std::string>());
    EXPECT_EQ("bar", a(8).as_dtype(cat.get_dtype()).as<std::string>());

}

