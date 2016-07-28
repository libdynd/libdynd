//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../test_memory_new.hpp"

#include <dynd/array.hpp>
#include <dynd/comparison.hpp>
#include <dynd/json_parser.hpp>
#include <dynd_assertions.hpp>

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

/*
TEST(Equals, Tuple) {
  nd::array a = nd::tuple({{0, 1, 2}, 6, 7});
  nd::array b = nd::tuple({{0, 1, 2}, 8, 9});

  std::cout << (a == b) << std::endl;
  std::exit(-1);
}
*/

TEST(AllEqual, Int) {
  EXPECT_ARRAY_EQ(true, nd::all_equal(4, 4));
  EXPECT_ARRAY_EQ(true, nd::all_equal(1, 1));
}

TEST(AllEqual, Fixed) {
  EXPECT_ARRAY_EQ(true, nd::all_equal(nd::array{0, 1, 2}, nd::array{0, 1, 2}));
  EXPECT_ARRAY_EQ(false, nd::all_equal(nd::array{0, 1, 2}, nd::array{0, 1, 3}));
}
