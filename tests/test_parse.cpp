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

/*
TEST(StringType, StringToBool)
{
  EXPECT_TRUE(nd::array("true").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array(" True").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("TRUE ").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("T").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("yes  ").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("Yes").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("Y").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array(" on").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("On").ucast<bool1>().as<bool>());
  EXPECT_TRUE(nd::array("1").ucast<bool1>().as<bool>());

  EXPECT_FALSE(nd::array("false").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("False").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("FALSE ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("F ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array(" no").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("No").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("N").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("off ").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("Off").ucast<bool1>().as<bool>());
  EXPECT_FALSE(nd::array("0 ").ucast<bool1>().as<bool>());

  // By default, conversion to bool is not permissive
  EXPECT_THROW(nd::array(nd::array("").ucast<bool1>().eval()), invalid_argument);
  EXPECT_THROW(nd::array(nd::array("2").ucast<bool1>().eval()), invalid_argument);
  EXPECT_THROW(nd::array(nd::array("flase").ucast<bool1>().eval()), invalid_argument);

  // In "nocheck" mode, it's a bit more permissive
  eval::eval_context tmp_ectx;
  tmp_ectx.errmode = assign_error_nocheck;
  EXPECT_FALSE(nd::array(nd::array("").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
  EXPECT_TRUE(nd::array(nd::array("2").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
  EXPECT_TRUE(nd::array(nd::array("flase").ucast<bool1>().eval(&tmp_ectx)).as<bool>());
}
*/

TEST(Parse, Bool)
{
  EXPECT_TRUE(parse<bool>("true", nocheck));
  EXPECT_TRUE(parse<bool>("True", nocheck));
  EXPECT_TRUE(parse<bool>("TRUE", nocheck));
  EXPECT_TRUE(parse<bool>("T", nocheck));
  EXPECT_TRUE(parse<bool>("yes", nocheck));
  EXPECT_TRUE(parse<bool>("Yes", nocheck));
  EXPECT_TRUE(parse<bool>("Y", nocheck));
  EXPECT_TRUE(parse<bool>("on", nocheck));
  EXPECT_TRUE(parse<bool>("On", nocheck));
  EXPECT_TRUE(parse<bool>("1", nocheck));

  EXPECT_FALSE(parse<bool>("false", nocheck));
}

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

TEST(Parse, LongLimits)
{
  EXPECT_EQ(numeric_limits<long>::min(), parse<long>(to_string(numeric_limits<long>::min())));
  EXPECT_EQ(numeric_limits<long>::max(), parse<long>(to_string(numeric_limits<long>::max())));
}

TEST(Parse, UIntLimits)
{
  EXPECT_EQ(numeric_limits<unsigned int>::min(), parse<unsigned int>(to_string(numeric_limits<unsigned int>::min())));
  EXPECT_EQ(numeric_limits<unsigned int>::max(), parse<unsigned int>(to_string(numeric_limits<unsigned int>::max())));
}

TEST(Parse, UInt8Limits)
{
  EXPECT_EQ(0, parse<uint8_t>("0"));
  EXPECT_EQ(255, parse<uint8_t>("255"));
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
  EXPECT_TRUE(std::isinf(parse<float>("Inf")));
  EXPECT_FALSE(signbit(parse<float>("Inf")));

  // +Inf
  EXPECT_TRUE(std::isinf(parse<float>("+Inf")));
  EXPECT_FALSE(signbit(parse<float>("+Inf")));

  // -Inf
  EXPECT_TRUE(std::isinf(parse<float>("-Inf")));
  EXPECT_TRUE(signbit(parse<float>("-Inf")));

  // inf
  EXPECT_TRUE(std::isinf(parse<float>("inf")));
  EXPECT_FALSE(signbit(parse<float>("inf")));

  // +inf
  EXPECT_TRUE(std::isinf(parse<float>("+inf")));
  EXPECT_FALSE(signbit(parse<float>("+inf")));

  // -inf
  EXPECT_TRUE(std::isinf(parse<float>("-inf")));
  EXPECT_TRUE(signbit(parse<float>("-inf")));

  // Infinity
  EXPECT_TRUE(std::isinf(parse<float>("Infinity")));
  EXPECT_FALSE(signbit(parse<float>("Infinity")));

  // +Infinity
  EXPECT_TRUE(std::isinf(parse<float>("+Infinity")));
  EXPECT_FALSE(signbit(parse<float>("+Infinity")));

  // -Infinity
  EXPECT_TRUE(std::isinf(parse<float>("-Infinity")));
  EXPECT_TRUE(signbit(parse<float>("-Infinity")));

  // 1.#INF
  EXPECT_TRUE(std::isinf(parse<float>("1.#INF")));
  EXPECT_FALSE(signbit(parse<float>("1.#INF")));

  // -1.#INF
  EXPECT_TRUE(std::isinf(parse<float>("-1.#INF")));
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

TEST(JSONParse, Bool)
{
  EXPECT_ARRAY_EQ(true, nd::json::parse(ndt::make_type<bool1>(), "true"));
  EXPECT_ARRAY_EQ(false, nd::json::parse(ndt::make_type<bool1>(), "false"));

  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "null"), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "flase"), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "\"flase\""), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "\"\""), invalid_argument);
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

#define EXPECT_MISSING(TP, ACTUAL)                                                                                     \
  EXPECT_EQ(TP, ACTUAL.get_type());                                                                                    \
  EXPECT_TRUE(ACTUAL.is_missing());

#define EXPECT_AVAILABLE(TP, ACTUAL)                                                                                   \
  EXPECT_EQ(TP, ACTUAL.get_type());                                                                                    \
  EXPECT_FALSE(ACTUAL.is_missing());

TEST(JSONParse, Option)
{
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()), "null"));
  //  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()),
  //               nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), "\"null\""));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()), "NULL"));
  //  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()),
  //               nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), "\"NULL\""));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool>()), "NA"));
  //  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()),
  //               nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<bool1>()), "\"NA\""));

  EXPECT_AVAILABLE(ndt::make_type<ndt::option_type>(ndt::make_type<short>()),
                   nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<short>()), "123"));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<short>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<short>()), "null"));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<int>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<int>()), "null"));

  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()), "null"));
}

template <typename T>
T na();

template <>
inline int na()
{
  return DYND_INT32_NA;
}

TEST(JSONParse, FixedDim)
{
  nd::array actual;

  EXPECT_ARRAY_EQ((nd::array{0u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<unsigned int>()), "[0]"));
  EXPECT_ARRAY_EQ((nd::array{0u, 1u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>()), "[0, 1]"));

  actual =
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(9, ndt::make_type<ndt::option_type>(ndt::make_type<int>())),
                      "[null, 3, null, -1000, 1, 3, null, null, null]");
  EXPECT_TRUE(actual(0).is_missing());
  EXPECT_EQ(3, actual(1).as<int>());
  EXPECT_TRUE(actual(2).is_missing());
  EXPECT_EQ(-1000, actual(3).as<int>());
  EXPECT_EQ(1, actual(4).as<int>());
  EXPECT_EQ(3, actual(5).as<int>());
  EXPECT_TRUE(actual(6).is_missing());
  EXPECT_TRUE(actual(7).is_missing());
  EXPECT_TRUE(actual(8).is_missing());

/*
  actual =
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(9, ndt::make_type<ndt::option_type>(ndt::make_type<long>())),
                      "[null, 3, null, -1000, 1, 3, null, null, null]");
  EXPECT_TRUE(actual(0).is_missing());
  EXPECT_EQ(3, actual(1).as<long>());
  EXPECT_TRUE(actual(2).is_missing());
  EXPECT_EQ(-1000, actual(3).as<long>());
  EXPECT_EQ(1, actual(4).as<long>());
  EXPECT_EQ(3, actual(5).as<long>());
  EXPECT_TRUE(actual(6).is_missing());
  EXPECT_TRUE(actual(7).is_missing());
  EXPECT_TRUE(actual(8).is_missing());
*/
}

TEST(JSONParse, VarDim)
{
  nd::array expected;

  expected = nd::empty(ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>())).assign({0, 1});
  EXPECT_ARRAY_EQ(expected, nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>()), "[0, 1]"));

  expected = nd::empty(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()))
                 .assign({true, true, true, true, true, true, true});
  EXPECT_ARRAY_EQ(expected, nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()),
                                            "[true, \"true\", 1, \"T\", \"y\", \"On\", \"yes\"]"));
  expected = nd::empty(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()))
                 .assign({false, false, false, false, false, false, false});
  EXPECT_ARRAY_EQ(expected, nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()),
                                            "[false, \"false\", 0, \"F\", \"n\", \"Off\", \"no\"]"));
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
