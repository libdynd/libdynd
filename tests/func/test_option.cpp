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

#include <dynd/option.hpp>

using namespace std;
using namespace dynd;

TEST(Option, IsAvail)
{
  nd::array x = nd::empty("?int64");
  x.assign_na();
  EXPECT_TRUE(nd::is_na(x).as<bool>());
}

TEST(Option, IsAvailArray)
{
  nd::array data = parse_json("3 * ?int", "[0, null, 2]");
  nd::array expected{false, true, false};
  EXPECT_ARRAY_EQ(nd::is_na(data), expected);

  data = parse_json("3 * ?int", "[null, null, null]");
  expected = {true, true, true};
  EXPECT_ARRAY_EQ(nd::is_na(data), expected);

  data = parse_json("3 * ?void", "[null, null, null]");
  expected = {true, true, true};
  EXPECT_ARRAY_EQ(nd::is_na(data), expected);

  data = parse_json("2 * 3 * ?float64", "[[1.0, null, 3.0], [null, \"NaN\", 3.0]]");
  expected = parse_json("2 * 3 * bool", "[[false, true, false], [true, true, false]]");
  EXPECT_ARRAY_EQ(nd::is_na(data), expected);

  data = parse_json("0 * ?int64", "[]");
  expected = parse_json("0 * bool", "[]");
  EXPECT_ARRAY_EQ(nd::is_na(data), expected);
}

TEST(Option, AssignNA)
{
  nd::array x = nd::assign_na({}, {{"dst_tp", ndt::type("?int64")}});
  EXPECT_TRUE(nd::is_na(x).as<bool>());
}

TEST(Option, AssignNAArray)
{
  nd::array a = nd::empty("3 * ?int64");
  a(0).vals() = nd::assign_na({}, {{"dst_tp", ndt::type("?int64")}});
  a(1).vals() = 1.0;
  a(2).vals() = 3.0;
  nd::array expected = {true, false, false};
  EXPECT_ARRAY_EQ(nd::is_na(a), expected);
}
