//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../test_memory_new.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/func/option.hpp>

using namespace std;
using namespace dynd;


TEST(Option, IsAvail) {
    nd::array x = nd::empty("?int64");
    nd::assign_na(x);
    EXPECT_FALSE(nd::is_avail(x).as<bool>());

    x = nd::empty("?datetime");
    nd::assign_na(x);
    EXPECT_FALSE(nd::is_avail(x).as<bool>());

    x = parse_json("?time", "\"11:00:13\"");
    EXPECT_TRUE(nd::is_avail(x).as<bool>());

    x = parse_json("?date", "\"2014-01-10\"");
    EXPECT_TRUE(nd::is_avail(x).as<bool>());
}


TEST(Option, IsAvailArray) {
    nd::array data = parse_json("3 * ?int", "[0, null, 2]");
    nd::array expected{true, false, true};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("3 * ?int", "[null, null, null]");
    expected = {false, false, false};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("3 * ?datetime", "[null, null, null]");
    expected = {false, false, false};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("2 * ?date", "[\"2014-01-31\", null]");
    expected = {true, false};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("2 * ?date", "[null, \"2014-01-31\"]");
    expected = {false, true};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("2 * ?datetime", "[\"2014-01-31T12:13:14Z\", null]");
    expected = {true, false};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("2 * ?datetime", "[null, \"2014-01-31T12:13:14Z\"]");
    expected = {false, true};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("3 * ?time", "[\"11:12:11\", \"11:12:12\", \"11:12:13\"]");
    expected = {true, true, true};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("3 * ?void", "[null, null, null]");
    expected = {false, false, false};
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("2 * 3 * ?float64", "[[1.0, null, 3.0], [null, \"NaN\", 3.0]]");
    expected = parse_json("2 * 3 * bool", "[[true, false, true], [false, false, true]]");
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);

    data = parse_json("0 * ?int64", "[]");
    expected = parse_json("0 * bool", "[]");
    EXPECT_ARRAY_EQ(nd::is_avail(data), expected);
}


TEST(Option, AssignNA) {
    nd::array x = nd::assign_na_decl(kwds("dst_tp", ndt::type("?int64")));
    EXPECT_FALSE(nd::is_avail(x).as<bool>());
}


TEST(Option, AssignNAArray) {
    nd::array a = nd::empty("3 * ?int64");
    a(0).vals() = nd::assign_na_decl(kwds("dst_tp", ndt::type("?int64")));
    a(1).vals() = 1.0;
    a(2).vals() = 3.0;
    nd::array expected = {false, true, true};
    EXPECT_ARRAY_EQ(nd::is_avail(a), expected);
}
