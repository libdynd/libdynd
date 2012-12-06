//
// Copyright (C) 2011-12, Dynamic NDArray Developers
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
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(4u, d.get_element_size());
    EXPECT_FALSE(d.is_expression());

    cout << d << endl;
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

    // cout << di << endl;

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

TEST(CategoricalDType, Factor) {

    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "foo";

    dtype da = factor_categorical_dtype(a);

    cout << da << endl;

    ndobject i = make_strided_ndobject(3, make_dtype<int32_t>());
    i.at(0).vals() = 10;
    i.at(1).vals() = 10;
    i.at(2).vals() = 0;

    //dtype di = factor_categorical_dtype(i);

    //cout << di << endl;

}

TEST(CategoricalDType, Values) {

    ndobject a = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    a.at(0).vals() = "foo";
    a.at(1).vals() = "bar";
    a.at(2).vals() = "baz";

    dtype dt = make_categorical_dtype(a);

    EXPECT_EQ(0u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(0).get_readonly_originptr()));
    EXPECT_EQ(1u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(1).get_readonly_originptr()));
    EXPECT_EQ(2u, static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category(a.at(2).get_readonly_originptr()));
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("aaa"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("ddd"), std::runtime_error);
    EXPECT_THROW(static_cast<const categorical_dtype*>(dt.extended())->get_value_from_category("zzz"), std::runtime_error);

}

TEST(CategoricalDType, AssignFixedString) {
    ndobject cat = make_strided_ndobject(3, make_fixedstring_dtype(string_encoding_ascii, 3));
    cat.at(0).vals() = "foo";
    cat.at(1).vals() = "bar";
    cat.at(2).vals() = "baz";

    dtype dt = make_categorical_dtype(cat);

    ndobject a = make_strided_ndobject(3, dt);
    a.val_assign(cat);
    EXPECT_EQ("foo", a.at(0).as<std::string>());
    EXPECT_EQ("bar", a.at(1).as<std::string>());
    EXPECT_EQ("baz", a.at(2).as<std::string>());
    cout << a << endl;
    a.at(0).vals() = cat.at(2);
    EXPECT_EQ("baz", a.at(0).as<std::string>());
    cout << a << endl;

    cat.at(0).vals() = std::string("zzz");
    EXPECT_THROW(a.at(0).vals() = cat.at(0), std::runtime_error);

    // TODO implicit conversion?
    //a(0).vals() = std::string("bar");
    //cout << a << endl;

    ndobject tmp = make_strided_ndobject(3, cat.get_dtype().at(0));
    tmp.val_assign(a);
    EXPECT_EQ("baz", tmp.at(0).as<std::string>());
    EXPECT_EQ("bar", tmp.at(1).as<std::string>());
    EXPECT_EQ("baz", tmp.at(2).as<std::string>());
    cout << tmp << endl;
    tmp.at(0).vals() = a.at(1);
    EXPECT_EQ("bar", tmp.at(0).as<std::string>());
    cout << tmp << endl;

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
    cout << a << endl;
    a.at(0).vals() = cat.at(2);
    EXPECT_EQ(1000, a.at(0).as<int32_t>());
    cout << a << endl;

    // TODO implicit conversion?
    //a(0).vals() = std::string("bar");
    //cout << a << endl;

    ndobject tmp = make_strided_ndobject(3, cat.get_dtype().at(0));
    tmp.val_assign(a);
    EXPECT_EQ(1000, tmp.at(0).as<int32_t>());
    EXPECT_EQ(100, tmp.at(1).as<int32_t>());
    EXPECT_EQ(1000, tmp.at(2).as<int32_t>());
    cout << tmp << endl;
    tmp.at(0).vals() = a.at(1);
    EXPECT_EQ(100, tmp.at(0).as<int32_t>());
    cout << tmp << endl;

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

    EXPECT_EQ("foo", a.at(0).as<std::string>());
    EXPECT_EQ("bar", a.at(1).as<std::string>());
    EXPECT_EQ("baz", a.at(2).as<std::string>());
    EXPECT_EQ("foo", a.at(3).as<std::string>());
    EXPECT_EQ("foo", a.at(4).as<std::string>());
    EXPECT_EQ("foo", a.at(5).as<std::string>());
    EXPECT_EQ("bar", a.at(6).as<std::string>());
    EXPECT_EQ("baz", a.at(7).as<std::string>());
    EXPECT_EQ("bar", a.at(8).as<std::string>());

}

