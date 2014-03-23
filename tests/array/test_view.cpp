//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../test_memory.hpp"

#include <dynd/json_parser.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;

TEST(View, SameType) {
    // When the type is the same, nd::view should return the
    // exact same array
    nd::array a = nd::empty("5 * 3 * int32");
    nd::array b;

    b = nd::view(a, a.get_type());
    EXPECT_EQ(a.get_ndo(), b.get_ndo());

    // collapse the array into strided instead of fixed dims
    a = a(irange(), irange());
    EXPECT_EQ(ndt::type("strided * strided * int32"), a.get_type());
    b = nd::view(a, a.get_type());
    EXPECT_EQ(a.get_ndo(), b.get_ndo());

    // test also with a var dim and string type
    a = parse_json("3 * var * string",
                   "[[\"this\", \"is\", \"for\"], [\"testing\"], []]");
    b = nd::view(a, a.get_type());
    EXPECT_EQ(a.get_ndo(), b.get_ndo());
}

TEST(View, FixedToStrided) {
    nd::array a = nd::empty("5 * 3 * int32");
    nd::array b;

    b = nd::view(a, ndt::type("strided * 3 * int32"));
    EXPECT_EQ(ndt::type("strided * 3 * int32"), b.get_type());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());

    b = nd::view(a, ndt::type("strided * strided * int32"));
    EXPECT_EQ(ndt::type("strided * strided * int32"), b.get_type());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());
}

TEST(View, StridedToFixed) {
    nd::array a = nd::empty("5 * 3 * int32");
    nd::array b;

    // Make them strided dimensions
    a = a(irange(), irange());
    EXPECT_EQ(ndt::type("strided * strided * int32"), a.get_type());

    b = nd::view(a, ndt::type("strided * 3 * int32"));
    EXPECT_EQ(ndt::type("strided * 3 * int32"), b.get_type());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());

    b = nd::view(a, ndt::type("5 * 3 * int32"));
    EXPECT_EQ(ndt::type("5 * 3 * int32"), b.get_type());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());
}

TEST(View, Errors) {
    nd::array a = nd::empty("5 * 3 * int32");

    // Shape mismatches
    EXPECT_THROW(nd::view(a, ndt::type("strided * 2 * int32")), type_error);
    EXPECT_THROW(nd::view(a, ndt::type("5 * 2 * int32")), type_error);
    EXPECT_THROW(nd::view(a, ndt::type("6 * 3 * int32")), type_error);
    // DType mismatches
    EXPECT_THROW(nd::view(a, ndt::type("5 * 3 * uint32")), type_error);

    // Also starting from strided dimensions
    a = a(irange(), irange());
    EXPECT_EQ(ndt::type("strided * strided * int32"), a.get_type());

    // Shape mismatches
    EXPECT_THROW(nd::view(a, ndt::type("strided * 2 * int32")), type_error);
    EXPECT_THROW(nd::view(a, ndt::type("5 * 2 * int32")), type_error);
    EXPECT_THROW(nd::view(a, ndt::type("6 * 3 * int32")), type_error);
    // DType mismatches
    EXPECT_THROW(nd::view(a, ndt::type("5 * 3 * uint32")), type_error);
}
