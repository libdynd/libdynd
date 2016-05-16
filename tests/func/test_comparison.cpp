//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>
#include <inc_gtest.hpp>
#include <iostream>
#include <stdexcept>

#include "../dynd_assertions.hpp"
#include "../test_memory_new.hpp"

#include <dynd/array.hpp>
#include <dynd/comparison.hpp>
#include <dynd/json_parser.hpp>

using namespace dynd;

TEST(Comparison, OptionScalar) {
  nd::array NA = nd::empty(ndt::type("?int32"));
  NA.assign_na();
  EXPECT_ALL_TRUE(nd::is_na(NA < 1));
  EXPECT_ALL_TRUE(nd::is_na(NA > 1));
  EXPECT_ALL_TRUE(nd::is_na(NA >= 1));
  EXPECT_ALL_TRUE(nd::is_na(NA <= 1));
  EXPECT_ALL_TRUE(nd::is_na(NA == 1));
  EXPECT_ALL_TRUE(nd::is_na(NA != 1));

  EXPECT_ALL_TRUE(nd::is_na(1 < NA));
  EXPECT_ALL_TRUE(nd::is_na(1 > NA));
  EXPECT_ALL_TRUE(nd::is_na(1 >= NA));
  EXPECT_ALL_TRUE(nd::is_na(1 <= NA));
  EXPECT_ALL_TRUE(nd::is_na(1 == NA));
  EXPECT_ALL_TRUE(nd::is_na(1 != NA));

  EXPECT_ALL_TRUE(nd::is_na(NA < NA));
  EXPECT_ALL_TRUE(nd::is_na(NA > NA));
  EXPECT_ALL_TRUE(nd::is_na(NA >= NA));
  EXPECT_ALL_TRUE(nd::is_na(NA <= NA));
  EXPECT_ALL_TRUE(nd::is_na(NA == NA));
  EXPECT_ALL_TRUE(nd::is_na(NA != NA));
}

TEST(Comparison, OptionArray) {
  nd::array data = parse_json("5 * ?int32", "[null, 0, 40, null, 1]");
  nd::array expected = nd::array{true, false, false, true, false};
  EXPECT_ARRAY_EQ(nd::is_na(data < 1), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data > 1), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data >= 1), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data <= 1), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data == 1), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data != 1), expected);

  EXPECT_ARRAY_EQ(nd::is_na(1 < data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(1 > data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(1 >= data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(1 <= data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(1 == data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(1 != data), expected);

  EXPECT_ARRAY_EQ(nd::is_na(data < data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data > data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data >= data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data <= data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data == data), expected);
  EXPECT_ARRAY_EQ(nd::is_na(data != data), expected);
}
