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
#include <dynd/types/date_type.hpp>

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

TEST(Parse, SignedChar)
{
  EXPECT_EQ(static_cast<signed char>(-1), parse<signed char>("-1"));
  EXPECT_EQ(static_cast<signed char>(6), parse<signed char>("6"));

  EXPECT_EQ(numeric_limits<signed char>::min(), parse<signed char>(to_string(numeric_limits<signed char>::min())));
  EXPECT_EQ(numeric_limits<signed char>::max(), parse<signed char>(to_string(numeric_limits<signed char>::max())));
}

TEST(Parse, Short)
{
  EXPECT_EQ(numeric_limits<short>::min(), parse<short>(to_string(numeric_limits<short>::min())));
  EXPECT_EQ(numeric_limits<short>::max(), parse<short>(to_string(numeric_limits<short>::max())));
}

TEST(Parse, Int)
{
  EXPECT_EQ(-2, parse<int>("-2"));
  EXPECT_EQ(5, parse<int>("5"));
  EXPECT_EQ(17, parse<int>("17"));

  EXPECT_EQ(numeric_limits<int>::min(), parse<int>(to_string(numeric_limits<int>::min())));
  EXPECT_EQ(numeric_limits<int>::max(), parse<int>(to_string(numeric_limits<int>::max())));
}

TEST(Parse, LongLimits)
{
  EXPECT_EQ(numeric_limits<long>::min(), parse<long>(to_string(numeric_limits<long>::min())));
  EXPECT_EQ(numeric_limits<long>::max(), parse<long>(to_string(numeric_limits<long>::max())));
}

TEST(Parse, LongLongLimits)
{
  EXPECT_EQ(numeric_limits<long long>::min(), parse<long long>(to_string(numeric_limits<long long>::min())));
  EXPECT_EQ(numeric_limits<long long>::max(), parse<long long>(to_string(numeric_limits<long long>::max())));
}

TEST(Parse, UInt)
{
  EXPECT_EQ(numeric_limits<unsigned int>::min(), parse<unsigned int>(to_string(numeric_limits<unsigned int>::min())));
  EXPECT_EQ(numeric_limits<unsigned int>::max(), parse<unsigned int>(to_string(numeric_limits<unsigned int>::max())));

  EXPECT_THROW(parse<unsigned int>("-1"), invalid_argument);
  EXPECT_THROW(parse<unsigned int>("-7"), invalid_argument);
}

TEST(Parse, UInt8)
{
  EXPECT_EQ(static_cast<uint8_t>(0), parse<uint8_t>("0"));
  EXPECT_EQ(static_cast<uint8_t>(255), parse<uint8_t>("255"));

  EXPECT_THROW(parse<uint8_t>("256"), out_of_range);
  EXPECT_THROW(parse<uint8_t>("391602"), out_of_range);
  EXPECT_THROW(parse<uint8_t>("-1"), invalid_argument);
  EXPECT_THROW(parse<uint8_t>("-7"), invalid_argument);
}

TEST(Parse, UInt16)
{
  EXPECT_EQ(0, parse<uint16_t>("0"));
  EXPECT_EQ(0, parse<uint16_t>("0", nocheck));

  EXPECT_EQ(1, parse<uint16_t>("1"));
  EXPECT_EQ(1, parse<uint16_t>("1", nocheck));

  EXPECT_EQ(123, parse<uint16_t>("123"));
  EXPECT_EQ(123, parse<uint16_t>("123", nocheck));

  EXPECT_THROW(parse<uint16_t>("65536"), out_of_range);
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

  EXPECT_THROW(parse<uint32_t>("4294967296"), out_of_range);
  EXPECT_EQ(0U, parse<uint32_t>("4294967296", nocheck));

  EXPECT_THROW(parse<uint32_t>("-1"), invalid_argument);
  EXPECT_EQ(0U, parse<uint32_t>("-1", nocheck));
}

/*
ToDo: Valgrind does not like this on some Travis CI setups, need to fix that.

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
*/

/*
ToDo: Valgrind does not like this on some Travis CI setups, need to fix that.

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
*/

TEST(JSONParse, Bool)
{
  EXPECT_ARRAY_EQ(true, nd::json::parse(ndt::make_type<bool1>(), "true"));
  EXPECT_ARRAY_EQ(false, nd::json::parse(ndt::make_type<bool1>(), "false"));

  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "null"), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "flase"), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "\"flase\""), invalid_argument);
  EXPECT_THROW(nd::json::parse(ndt::make_type<bool1>(), "\"\""), invalid_argument);
}

TEST(JSONParse, Int)
{
  EXPECT_ARRAY_EQ(static_cast<char>(123), nd::json::parse(ndt::make_type<signed char>(), "123"));
  EXPECT_ARRAY_EQ(static_cast<short>(-30000), nd::json::parse(ndt::make_type<short>(), "-30000"));
  EXPECT_ARRAY_EQ(-1, nd::json::parse(ndt::make_type<int>(), "-1"));
  EXPECT_ARRAY_EQ(7, nd::json::parse(ndt::make_type<int>(), "7"));
  EXPECT_ARRAY_EQ(500000, nd::json::parse(ndt::make_type<int>(), "500000"));
  EXPECT_ARRAY_EQ(-3000000000LL, nd::json::parse(ndt::make_type<long long>(), "-3000000000"));

  //  std::cout << nd::json::parse(ndt::make_type<int>(), "\"123456\"") << std::endl;
  // std::exit(-1);
}

TEST(JSONParse, UInt)
{
  EXPECT_ARRAY_EQ(static_cast<unsigned char>(123), nd::json::parse(ndt::make_type<unsigned char>(), "123"));
  EXPECT_ARRAY_EQ(static_cast<unsigned short>(50000), nd::json::parse(ndt::make_type<unsigned short>(), "50000"));
  EXPECT_ARRAY_EQ(500000U, nd::json::parse(ndt::make_type<unsigned int>(), "500000"));
  EXPECT_ARRAY_EQ(3000000000ULL, nd::json::parse(ndt::make_type<unsigned long long>(), "3000000000"));
}

TEST(JSONParse, String)
{
  EXPECT_ARRAY_EQ("testing one two three",
                  nd::json::parse(ndt::make_type<ndt::string_type>(), "\"testing one two three\""));
  EXPECT_ARRAY_EQ(
      " \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 ",
      nd::json::parse(ndt::make_type<ndt::string_type>(), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \""));

  EXPECT_THROW(nd::json::parse(ndt::make_type<ndt::string_type>(), "false"), invalid_argument);
}

#define EXPECT_MISSING(TP, ACTUAL)                                                                                     \
  EXPECT_EQ(TP, ACTUAL.get_type());                                                                                    \
  EXPECT_TRUE(ACTUAL.is_na());

#define EXPECT_AVAILABLE(EXPECTED, ACTUAL)                                                                             \
  EXPECT_ARRAY_VALS_EQ(EXPECTED, ACTUAL);                                                                              \
//  EXPECT_EQ(EXPECTEDACTUAL.get_type());

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

  //    EXPECT_AVAILABLE(ndt::make_type<ndt::option_type>(ndt::make_type<short>()),
  //                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<short>()), "123"));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<short>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<short>()), "null"));
  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<int>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<int>()), "null"));

  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<unsigned int>()), "null"));

  EXPECT_AVAILABLE(
      "testing 1 2 3",
      nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<ndt::string_type>()), "\"testing 1 2 3\""));

  EXPECT_MISSING(ndt::make_type<ndt::option_type>(ndt::make_type<ndt::string_type>()),
                 nd::json::parse(ndt::make_type<ndt::option_type>(ndt::make_type<ndt::string_type>()), "null"));

  //  std::cout << nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(
  //                                 9, ndt::make_type<ndt::option_type>(ndt::make_type<dynd::string>())),
  //                           "[null, \"123\", null, \"456\", \"0\", \"789\", null, null, null]")
  //      << std::endl;
  //  std::exit(-1);
}

/*
TEST(JSONParser, OptionString)
{
  nd::array a, b, c;

  a = parse_json(ndt::type("?string"), "\"testing 1 2 3\"");
  EXPECT_EQ(ndt::type("?string"), a.get_type());
  EXPECT_EQ("testing 1 2 3", a.as<std::string>());
  a = parse_json(ndt::type("?string"), "\"null\"");
  EXPECT_EQ("null", a.as<std::string>());
  a = parse_json(ndt::type("?string"), "\"NA\"");
  EXPECT_EQ("NA", a.as<std::string>());
  a = parse_json(ndt::type("?string"), "\"\"");
  EXPECT_EQ("", a.as<std::string>());

  a = parse_json(ndt::type("?string"), "null");
  EXPECT_EQ(ndt::type("?string"), a.get_type());
  EXPECT_EQ(NULL, reinterpret_cast<const dynd::string *>(a.cdata())->begin());
  EXPECT_THROW(a.as<std::string>(), overflow_error);

  a = parse_json("9 * ?string", "[null, \"123\", null, \"456\", \"0\", \"789\", null, null, null]");
  EXPECT_EQ(ndt::type("9 * option[string]"), a.get_type());
  b = nd::empty("9 * string");
  EXPECT_THROW(b.vals() = a, overflow_error);
  // Assign this to another option type
  b = nd::empty("9 * ?int");
  b.vals() = a;
  c = parse_json("9 * ?int", "[null, 123, null, 456, 0, 789, null, null, null]");
  EXPECT_TRUE(nd::view(b, "9 * int").equals_exact(nd::view(c, "9 * int")));
}
*/

template <typename T>
T na();

template <>
inline int na()
{
  return DYND_INT32_NA;
}

TEST(ParseJSON, Struct)
{
  EXPECT_ARRAY_EQ(nd::as_struct({{"x", 2}, {"y", 3}}),
                  nd::json::parse(ndt::struct_type::make({"x", "y"}, {ndt::make_type<int>(), ndt::make_type<int>()}),
                                  "{\"x\":2,\"y\":3}"));

  /*
    nd::array n;
    ndt::type sdt = ndt::struct_type::make(
        {"id", "amount", "name", "when"},
        {ndt::make_type<int>(), ndt::make_type<double>(), ndt::make_type<ndt::string_type>(), ndt::date_type::make()});

    // A straightforward struct
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,"
                        " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_type());
    EXPECT_EQ(24601, n(0).as<int>());
    EXPECT_EQ(3.75, n(1).as<double>());
    EXPECT_EQ("Jean", n(2).as<std::string>());
    EXPECT_EQ("2012-09-19", n(3).as<std::string>());

    // Default parsing policy discards extra JSON fields
    n = parse_json(sdt, "{\"amount\":3.75,\"id\":24601,\"discarded\":[1,2,3],"
                        " \"when\":\"2012-09-19\",\"name\":\"Jean\"}");
    EXPECT_EQ(sdt, n.get_type());
    EXPECT_EQ(24601, n(0).as<int>());
    EXPECT_EQ(3.75, n(1).as<double>());
    EXPECT_EQ("Jean", n(2).as<std::string>());
    EXPECT_EQ("2012-09-19", n(3).as<std::string>());

    // Every field must be populated, though
    EXPECT_THROW(parse_json(sdt, "{\"amount\":3.75,\"discarded\":[1,2,3],"
                                 " \"when\":\"2012-09-19\",\"name\":\"Jean\"}"),
                 invalid_argument);
  */
}

TEST(JSONParse, FixedDim)
{
  EXPECT_ARRAY_EQ((nd::array{true, true, false, false}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(4, ndt::make_type<bool>()),
                                  "  [true, true, false, false]  "));

  EXPECT_THROW(
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(4, ndt::make_type<bool>()), "[true, true, false, false] 3.5"),
      invalid_argument);
  EXPECT_THROW(
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<bool>()), "[true, true, false, false]"),
      invalid_argument);
  EXPECT_THROW(
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(5, ndt::make_type<bool>()), "[true, true, false, false]"),
      invalid_argument);

  EXPECT_ARRAY_EQ((nd::array{0u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<unsigned int>()), "[0]"));
  EXPECT_ARRAY_EQ((nd::array{0u, 1u}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>()), "[0, 1]"));

  nd::array actual =
      nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(9, ndt::make_type<ndt::option_type>(ndt::make_type<int>())),
                      "[null, 3, null, -1000, 1, 3, null, null, null]");
  EXPECT_TRUE(actual(0).is_na());
  EXPECT_EQ(3, actual(1).as<int>());
  EXPECT_TRUE(actual(2).is_na());
  EXPECT_EQ(-1000, actual(3).as<int>());
  EXPECT_EQ(1, actual(4).as<int>());
  EXPECT_EQ(3, actual(5).as<int>());
  EXPECT_TRUE(actual(6).is_na());
  EXPECT_TRUE(actual(7).is_na());
  EXPECT_TRUE(actual(8).is_na());

  /*
    actual =
        nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(9,
    ndt::make_type<ndt::option_type>(ndt::make_type<long>())),
                        "[null, 3, null, -1000, 1, 3, null, null, null]");
    EXPECT_TRUE(actual(0).is_na());
    EXPECT_EQ(3, actual(1).as<long>());
    EXPECT_TRUE(actual(2).is_na());
    EXPECT_EQ(-1000, actual(3).as<long>());
    EXPECT_EQ(1, actual(4).as<long>());
    EXPECT_EQ(3, actual(5).as<long>());
    EXPECT_TRUE(actual(6).is_na());
    EXPECT_TRUE(actual(7).is_na());
    EXPECT_TRUE(actual(8).is_na());
  */

  actual = nd::json::parse(ndt::make_fixed_dim(3, ndt::var_dim_type::make(ndt::make_type<int>())),
                           "  [[1,2,3], [4,5], [6,7,-10,1000] ]  ");
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::var_dim_type::make(ndt::make_type<int>())), actual.get_type());
  EXPECT_EQ(1, actual(0, 0).as<int>());
  EXPECT_EQ(2, actual(0, 1).as<int>());
  EXPECT_EQ(3, actual(0, 2).as<int>());
  EXPECT_EQ(4, actual(1, 0).as<int>());
  EXPECT_EQ(5, actual(1, 1).as<int>());
  EXPECT_EQ(6, actual(2, 0).as<int>());
  EXPECT_EQ(7, actual(2, 1).as<int>());
  EXPECT_EQ(-10, actual(2, 2).as<int>());
  EXPECT_EQ(1000, actual(2, 3).as<int>());
}

TEST(JSONParse, VarDim)
{
  EXPECT_ARRAY_EQ(
      nd::empty(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>())).assign({true, true, false, false}),
      nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()), "  [true, true, false, false]  "));

  EXPECT_THROW(
      nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<bool>()), "[true, true, false, false] 3.5"),
      invalid_argument);

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

  nd::array actual =
      nd::json::parse(ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<int>())),
                      "  [[1,2,3], [4,5,2] ]  ");
  EXPECT_EQ(ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<int>())),
            actual.get_type());
  EXPECT_EQ(1, actual(0, 0).as<int>());
  EXPECT_EQ(2, actual(0, 1).as<int>());
  EXPECT_EQ(3, actual(0, 2).as<int>());
  EXPECT_EQ(4, actual(1, 0).as<int>());
  EXPECT_EQ(5, actual(1, 1).as<int>());
  EXPECT_EQ(2, actual(1, 2).as<int>());
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
  EXPECT_TRUE(a(0).is_na());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[0, null, 2]");
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_TRUE(a(1).is_na());
  EXPECT_EQ(2, a(2).as<int>());

  tp = ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<ndt::option_type>(ndt::make_type<int>()));
  a = nd::json::parse(tp, "[0, 1, null]");
  EXPECT_EQ(0, a(0).as<int>());
  EXPECT_EQ(1, a(1).as<int>());
  EXPECT_TRUE(a(2).is_na());
}

TEST(JSONParse, ListOfLists)
{
  EXPECT_ARRAY_EQ((nd::array{{0u, 1u}}),
                  nd::json::parse(ndt::make_type<ndt::fixed_dim_type>(
                                      1, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<unsigned int>())),
                                  "[[0, 1]]"));
}
