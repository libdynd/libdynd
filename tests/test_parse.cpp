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
  EXPECT_EQ(-2, parse<int>("-2"));
  EXPECT_EQ(-1, parse<int>("-1"));
  EXPECT_EQ(0, parse<int>("0"));
  EXPECT_EQ(1, parse<int>("1"));
  EXPECT_EQ(2, parse<int>("2"));
  EXPECT_EQ(3, parse<int>("3"));
  EXPECT_EQ(4, parse<int>("4"));
  EXPECT_EQ(5, parse<int>("5"));
  EXPECT_EQ(6, parse<int>("6"));
  EXPECT_EQ(7, parse<int>("7"));
}

TEST(Parse, IntLimits)
{
  EXPECT_EQ(numeric_limits<int>::min(), parse<int>(to_string(numeric_limits<int>::min())));
  EXPECT_EQ(numeric_limits<int>::max(), parse<int>(to_string(numeric_limits<int>::max())));
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

TEST(Parse, FloatInf)
{
  // Inf
  EXPECT_TRUE(isinf(parse<float>("Inf")));
  EXPECT_FALSE(signbit(parse<float>("Inf")));

  // +Inf
  EXPECT_TRUE(isinf(parse<float>("+Inf")));
  EXPECT_FALSE(signbit(parse<float>("+Inf")));

  // -Inf
  EXPECT_TRUE(isinf(parse<float>("-Inf")));
  EXPECT_TRUE(signbit(parse<float>("-Inf")));

  // inf
  EXPECT_TRUE(isinf(parse<float>("inf")));
  EXPECT_FALSE(signbit(parse<float>("inf")));

  // +inf
  EXPECT_TRUE(isinf(parse<float>("+inf")));
  EXPECT_FALSE(signbit(parse<float>("+inf")));

  // -inf
  EXPECT_TRUE(isinf(parse<float>("-inf")));
  EXPECT_TRUE(signbit(parse<float>("-inf")));

  // Infinity
  EXPECT_TRUE(isinf(parse<float>("Infinity")));
  EXPECT_FALSE(signbit(parse<float>("Infinity")));

  // +Infinity
  EXPECT_TRUE(isinf(parse<float>("+Infinity")));
  EXPECT_FALSE(signbit(parse<float>("+Infinity")));

  // -Infinity
  EXPECT_TRUE(isinf(parse<float>("-Infinity")));
  EXPECT_TRUE(signbit(parse<float>("-Infinity")));

  // 1.#INF
  EXPECT_TRUE(isinf(parse<float>("1.#INF")));
  EXPECT_FALSE(signbit(parse<float>("1.#INF")));

  // -1.#INF
  EXPECT_TRUE(isinf(parse<float>("-1.#INF")));
  EXPECT_TRUE(signbit(parse<float>("-1.#INF")));
}

TEST(Parse, FloatNaN)
{
  // NaN
  EXPECT_TRUE(std::isnan(parse<float>("NaN")));
  EXPECT_FALSE(signbit(parse<float>("NaN")));

  // +NaN
  EXPECT_TRUE(std::isnan(parse<float>("+NaN")));
  EXPECT_FALSE(signbit(parse<float>("+NaN")));

  // -NaN
  EXPECT_TRUE(std::isnan(parse<float>("-NaN")));
  EXPECT_TRUE(signbit(parse<float>("-NaN")));

  // nan
  EXPECT_TRUE(std::isnan(parse<float>("nan")));
  EXPECT_FALSE(signbit(parse<float>("nan")));

  // +nan
  EXPECT_TRUE(std::isnan(parse<float>("+nan")));
  EXPECT_FALSE(signbit(parse<float>("+nan")));

  // -nan
  EXPECT_TRUE(std::isnan(parse<float>("-nan")));
  EXPECT_TRUE(signbit(parse<float>("-nan")));

  // 1.#QNAN
  EXPECT_TRUE(std::isnan(parse<float>("1.#QNAN")));
  EXPECT_FALSE(signbit(parse<float>("1.#QNAN")));

  // -1.#IND
  EXPECT_TRUE(std::isnan(parse<float>("-1.#IND")));
  EXPECT_TRUE(signbit(parse<float>("-1.#IND")));
}

TEST(Parse, DoubleInf)
{
  // Inf
  EXPECT_TRUE(std::isinf(parse<double>("Inf")));
  EXPECT_FALSE(signbit(parse<double>("Inf")));

  // +Inf
  EXPECT_TRUE(std::isinf(parse<double>("+Inf")));
  EXPECT_FALSE(signbit(parse<double>("+Inf")));

  // -Inf
  EXPECT_TRUE(std::isinf(parse<double>("-Inf")));
  EXPECT_TRUE(signbit(parse<double>("-Inf")));

  // inf
  EXPECT_TRUE(std::isinf(parse<double>("inf")));
  EXPECT_FALSE(signbit(parse<double>("inf")));

  // +inf
  EXPECT_TRUE(std::isinf(parse<double>("+inf")));
  EXPECT_FALSE(signbit(parse<double>("+inf")));

  // -inf
  EXPECT_TRUE(std::isinf(parse<double>("-inf")));
  EXPECT_TRUE(signbit(parse<double>("-inf")));

  // Infinity
  EXPECT_TRUE(std::isinf(parse<double>("Infinity")));
  EXPECT_FALSE(signbit(parse<double>("Infinity")));

  // +Infinity
  EXPECT_TRUE(std::isinf(parse<double>("+Infinity")));
  EXPECT_FALSE(signbit(parse<double>("+Infinity")));

  // -Infinity
  EXPECT_TRUE(std::isinf(parse<double>("-Infinity")));
  EXPECT_TRUE(signbit(parse<double>("-Infinity")));

  // 1.#INF
  EXPECT_TRUE(std::isinf(parse<double>("1.#INF")));
  EXPECT_FALSE(signbit(parse<double>("1.#INF")));

  // -1.#INF
  EXPECT_TRUE(std::isinf(parse<double>("-1.#INF")));
  EXPECT_TRUE(signbit(parse<double>("-1.#INF")));
}

TEST(Parse, DoubleNaN)
{
  // +NaN
  EXPECT_TRUE(std::isnan(parse<double>("NaN")));
  EXPECT_FALSE(signbit(parse<double>("NaN")));

  // -NaN
  EXPECT_TRUE(std::isnan(parse<double>("-NaN")));
  EXPECT_TRUE(signbit(parse<double>("-NaN")));

  // +nan
  EXPECT_TRUE(std::isnan(parse<double>("nan")));
  EXPECT_FALSE(signbit(parse<double>("nan")));

  // -nan
  EXPECT_TRUE(std::isnan(parse<double>("-nan")));
  EXPECT_TRUE(signbit(parse<double>("-nan")));

  // 1.#QNAN
  EXPECT_TRUE(std::isnan(parse<double>("1.#QNAN")));
  EXPECT_FALSE(signbit(parse<double>("1.#QNAN")));

  // -1.#IND
  EXPECT_TRUE(std::isnan(parse<double>("-1.#IND")));
  EXPECT_TRUE(signbit(parse<double>("-1.#IND")));
}

TEST(JSONParse, Number)
{
  //  EXPECT_ARRAY_EQ(32, nd::json::parse("32", ndt::make_type<int>()));

  EXPECT_ARRAY_EQ(-1, nd::json::parse(ndt::make_type<int>(), "-1"));
  EXPECT_ARRAY_EQ(7, nd::json::parse(ndt::make_type<int>(), "7"));

  EXPECT_ARRAY_EQ(0u, nd::json::parse(ndt::make_type<unsigned int>(), "0")); // Minimum value
  EXPECT_ARRAY_EQ(32u, nd::json::parse(ndt::make_type<unsigned int>(), "32"));
  //  EXPECT_ARRAY_EQ(numeric_limits<unsigned int>::max(), nd::json::parse("0", ndt::make_type<unsigned int>())); //
  //  Minimum value
}

TEST(JSONParse, Option)
{
  ndt::type tp;
  nd::array a;

  tp = ndt::make_type<ndt::option_type>(ndt::make_type<int>());
  a = nd::json::parse(tp, "null");
  EXPECT_EQ(tp, a.get_type());
  EXPECT_TRUE(a.is_missing());

  EXPECT_TRUE(nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()), "null").is_missing());
  EXPECT_FALSE(nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()), "23").is_missing());
}

TEST(JSONParse, List)
{
  EXPECT_ARRAY_EQ((nd::array{0u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<unsigned int>()), "[0]"));
  EXPECT_ARRAY_EQ((nd::array{0u, 1u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>()), "[0, 1]"));
}

TEST(JSONParse, ListOfOption)
{
  ndt::type tp;
  nd::array a;

  tp = ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[0, 1]");
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_EQ(1, a(1).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[null, 1, 2]");
  EXPECT_TRUE(a(0).is_missing());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[0, null, 2]");
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_TRUE(a(1).is_missing());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[0, 1, null]");
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_TRUE(a(2).is_missing());
}

TEST(JSONParse, ListOfLists)
{
  EXPECT_ARRAY_EQ((nd::array{{0u, 1u}}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(
                                      1, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>())),
                                  "[[0, 1]]"));
}
