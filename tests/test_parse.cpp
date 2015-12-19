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

TEST(Parse, Int)
{
  EXPECT_EQ(0, parse<int>("0"));
  EXPECT_EQ(1, parse<int>("1"));
  EXPECT_EQ(2, parse<int>("2"));
  EXPECT_EQ(3, parse<int>("3"));
  EXPECT_EQ(4, parse<int>("4"));
  EXPECT_EQ(5, parse<int>("5"));
  EXPECT_EQ(6, parse<int>("6"));
  EXPECT_EQ(7, parse<int>("7"));
}

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

  EXPECT_ARRAY_EQ(-1, nd::json::parse("-1", ndt::make_type<int>()));
  EXPECT_ARRAY_EQ(7, nd::json::parse("7", ndt::make_type<int>()));

  EXPECT_ARRAY_EQ(0u, nd::json::parse("0", ndt::make_type<unsigned int>())); // Minimum value
  EXPECT_ARRAY_EQ(32u, nd::json::parse("32", ndt::make_type<unsigned int>()));
  //  EXPECT_ARRAY_EQ(numeric_limits<unsigned int>::max(), nd::json::parse("0", ndt::make_type<unsigned int>())); //
  //  Minimum value
}

TEST(JSONParse, Option)
{
  ndt::type tp;
  nd::array a;

  tp = ndt::make_type<ndt::option_type>(ndt::make_type<int>());
  a = nd::json::parse("null", tp);
  EXPECT_EQ(tp, a.get_type());
  EXPECT_TRUE(a.is_missing());

  EXPECT_TRUE(nd::json::parse("null", ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>())).is_missing());
  EXPECT_FALSE(nd::json::parse("23", ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>())).is_missing());
}

TEST(JSONParse, List)
{
  EXPECT_ARRAY_EQ((nd::array{0u}),
                  nd::json::parse("[0]", ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<unsigned int>())));
  EXPECT_ARRAY_EQ((nd::array{0u, 1u}),
                  nd::json::parse("[0, 1]", ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>())));
}

TEST(JSONParse, ListOfOption)
{
  ndt::type tp;
  nd::array a;

  tp = ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse("[0, 1]", tp);
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_EQ(1, a(1).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse("[null, 1, 2]", tp);
  EXPECT_TRUE(a(0).is_missing());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse("[0, null, 2]", tp);
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_TRUE(a(1).is_missing());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse("[0, 1, null]", tp);
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_TRUE(a(2).is_missing());
}

TEST(JSONParse, ListOfLists)
{
  EXPECT_ARRAY_EQ(
      (nd::array{{0u, 1u}}),
      nd::json::parse("[[0, 1]]", ndt::make_type<ndt::fixed_dim_type>(
                                      1, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>()))));
}
