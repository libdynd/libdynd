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

#include <dynd/func/arithmetic.hpp>
#include <dynd/func/comparison.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>

using namespace dynd;

/*
TEST(Comparison, Simple)
{
    nd::array a = {1, -1, 3};
    nd::array b = {0, 1, 2};

    std::cout << a << std::endl;
    std::cout << (a > b) << std::endl;
}
*/

TEST(Comparison, OptionScalar)
{
  nd::array NA = nd::empty(ndt::type("?int32"));
  nd::assign_na(NA);
  EXPECT_ALL_FALSE(nd::is_avail(NA < 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA > 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA >= 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA <= 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA == 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA != 1));

  EXPECT_ALL_FALSE(nd::is_avail(1 < NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 > NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 >= NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 <= NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 == NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 != NA));

  EXPECT_ALL_FALSE(nd::is_avail(NA < NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA > NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA >= NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA <= NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA == NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA != NA));
}


TEST(Comparison, OptionArray)
{
  nd::array data = parse_json("5 * ?int32", "[null, 0, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  EXPECT_ARRAY_EQ(nd::is_avail(data < 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data > 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data >= 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data <= 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data == 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data != 1), expected);

  EXPECT_ARRAY_EQ(nd::is_avail(1 < data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 > data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 >= data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 <= data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 == data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 != data), expected);

  EXPECT_ARRAY_EQ(nd::is_avail(data < data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data > data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data >= data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data <= data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data == data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data != data), expected);
}
