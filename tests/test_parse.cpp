//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/parse.hpp>

using namespace std;
using namespace dynd;

TEST(Parse, UInt16)
{
  EXPECT_EQ(0, parse<uint16_t>("0"));
  EXPECT_EQ(0, parse<uint16_t>("0", nocheck));

  EXPECT_EQ(1, parse<uint16_t>("1"));
  EXPECT_EQ(1, parse<uint16_t>("1", nocheck));

  EXPECT_EQ(123, parse<uint16_t>("123"));
  EXPECT_EQ(123, parse<uint16_t>("123", nocheck));

  EXPECT_THROW(parse<uint16_t>("65536"), overflow_error);
  EXPECT_EQ(0, parse<uint16_t>("65536", nocheck));

  EXPECT_THROW(parse<uint16_t>("-1"), invalid_argument);
  EXPECT_EQ(0, parse<uint16_t>("-1", nocheck));
}

TEST(Parse, UInt32)
{
  EXPECT_EQ(0U, parse<uint32_t>("0"));
  EXPECT_EQ(0U, parse<uint32_t>("0", nocheck));

  EXPECT_EQ(1U, parse<uint32_t>("1"));
  EXPECT_EQ(1U, parse<uint32_t>("1", nocheck));

  EXPECT_EQ(123U, parse<uint32_t>("123"));
  EXPECT_EQ(123U, parse<uint32_t>("123", nocheck));

  EXPECT_THROW(parse<uint32_t>("4294967296"), overflow_error);
  EXPECT_EQ(0U, parse<uint32_t>("4294967296", nocheck));

  EXPECT_THROW(parse<uint32_t>("-1"), invalid_argument);
  EXPECT_EQ(0U, parse<uint32_t>("-1", nocheck));
}

TEST(JSONParse, Number)
{
  //  EXPECT_ARRAY_EQ(32, nd::json::parse("32", ndt::make_type<int>()));
  EXPECT_ARRAY_EQ(32u, nd::json::parse("32", ndt::make_type<unsigned int>()));
}

TEST(JSONParse, Option)
{
  EXPECT_TRUE(nd::json::parse("null", ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>())).is_missing());
//  std::cout << nd::json::parse("23", ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>())) << std::endl;
  //std::exit(-1);
//  EXPECT_TRUE(.is_missing());
}

TEST(JSONParse, List)
{
  EXPECT_ARRAY_EQ((nd::array{0u}),
                  nd::json::parse("[0]", ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<unsigned int>())));
  EXPECT_ARRAY_EQ((nd::array{0u, 1u}),
                  nd::json::parse("[0, 1]", ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>())));
}

TEST(JSONParse, ListOfLists)
{
  EXPECT_ARRAY_EQ(
      (nd::array{{0u, 1u}}),
      nd::json::parse("[[0, 1]]", ndt::make_type<ndt::fixed_dim_type>(
                                      1, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>()))));
}
