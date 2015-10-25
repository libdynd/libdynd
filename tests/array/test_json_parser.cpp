//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/view.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/option_type.hpp>

using namespace std;
using namespace dynd;

TEST(JSON, DiscoverBool)
{
  EXPECT_EQ(ndt::type::make<bool1>(), ndt::json::discover("true"));
  EXPECT_EQ(ndt::type::make<bool1>(), ndt::json::discover("false"));
}

TEST(JSON, DiscoverInt64)
{
  EXPECT_EQ(ndt::type::make<int64>(), ndt::json::discover("0"));
  EXPECT_EQ(ndt::type::make<int64>(), ndt::json::discover("3"));
  EXPECT_EQ(ndt::type::make<int64>(), ndt::json::discover("11"));

  EXPECT_EQ(ndt::type::make<int64>(), ndt::json::discover("-1"));
  EXPECT_EQ(ndt::type::make<int64>(), ndt::json::discover("-5"));
}

TEST(JSON, DiscoverFloat64)
{
  EXPECT_EQ(ndt::type::make<float64>(), ndt::json::discover("0.5"));
  EXPECT_EQ(ndt::type::make<float64>(), ndt::json::discover("3.14"));
}

TEST(JSON, DiscoverString)
{
  EXPECT_EQ(ndt::type(string_type_id), ndt::json::discover("\"Hello, world!\""));
}

TEST(JSON, DiscoverOption)
{
  EXPECT_EQ(ndt::type("?Any"), ndt::json::discover("null"));
}

TEST(JSON, DiscoverArray)
{
  EXPECT_EQ(ndt::type("()"), ndt::json::discover("[]"));

  EXPECT_EQ(ndt::type("1 * int64"), ndt::json::discover("[0]"));
  EXPECT_EQ(ndt::type("1 * string"), ndt::json::discover("[\"JSON\"]"));
  EXPECT_EQ(ndt::type("1 * {x: int64, y: float64}"), ndt::json::discover("[{\"x\": 3, \"y\": 0.75}]"));
  EXPECT_EQ(ndt::type("2 * int64"), ndt::json::discover("[11, -3]"));
  EXPECT_EQ(ndt::type("5 * int64"), ndt::json::discover("[0, 1, 2, 3, 4]"));
  EXPECT_EQ(ndt::type("5 * float64"), ndt::json::discover("[0, 1, 2.5, 3, 4]"));
  EXPECT_EQ(ndt::type("5 * float64"), ndt::json::discover("[0.5, 1, 2, 3, 4]"));

  EXPECT_EQ(ndt::type("5 * ?int64"), ndt::json::discover("[null, 1, 2, 3, 4]"));
  EXPECT_EQ(ndt::type("5 * ?int64"), ndt::json::discover("[0, 1, null, 3, 4]"));
  EXPECT_EQ(ndt::type("5 * ?int64"), ndt::json::discover("[0, 1, 2, 3, null]"));
  EXPECT_EQ(ndt::type("3 * ?float64"), ndt::json::discover("[null, -7, 3.3]"));

  EXPECT_EQ(ndt::type("(int64, string)"), ndt::json::discover("[2, \"Hello, world!\"]"));

  EXPECT_EQ(ndt::type("2 * 1 * int64"), ndt::json::discover("[[10], [7]]"));
  EXPECT_EQ(ndt::type("1 * 2 * int64"), ndt::json::discover("[[10, 7]]"));
  EXPECT_EQ(ndt::type("2 * 2 * int64"), ndt::json::discover("[[0, 1], [2, 3]]"));
  EXPECT_EQ(ndt::type("2 * 3 * int64"), ndt::json::discover("[[0, 1, 11], [2, 3, -4]]"));

  EXPECT_EQ(ndt::type("2 * ?2 * int64"), ndt::json::discover("[null, [2, 3]]"));
  EXPECT_EQ(ndt::type("2 * ?3 * int64"), ndt::json::discover("[[0, -1, 1], null]"));
  EXPECT_EQ(ndt::type("2 * ?2 * float64"), ndt::json::discover("[[0, 1.2], null]"));

  EXPECT_EQ(ndt::type("2 * var * int64"), ndt::json::discover("[[0, 1], [2]]"));
  EXPECT_EQ(ndt::type("2 * var * var * int64"), ndt::json::discover("[[[0, 1], [2]], [[3]]]"));

  EXPECT_EQ(ndt::type("2 * var * 3 * int64"), ndt::json::discover("[[[0, 1, 4], [2, 5, 3]], [[3, 1, 2]]]"));

  EXPECT_EQ(ndt::type("2 * var * float64"), ndt::json::discover("[[0.5, 1], [2]]"));
  EXPECT_EQ(ndt::type("2 * var * ?float64"), ndt::json::discover("[[0.5, 1], [null]]"));

  EXPECT_EQ(ndt::type("2 * var * ?int64"), ndt::json::discover("[[0, null], [2]]"));
}

TEST(JSON, DiscoverObject)
{
  EXPECT_EQ(ndt::type("{}"), ndt::json::discover("{}"));

  EXPECT_EQ(ndt::type("{a: int64}"), ndt::json::discover("{\"a\": 3}"));
  EXPECT_EQ(ndt::type("{a: float64}"), ndt::json::discover("{\"a\": 3.14}"));

  EXPECT_EQ(ndt::type("{x: float64, y: 3 * int64}"), ndt::json::discover("{\"x\": 3.14, \"y\": [1, 2, 3]}"));
}

TEST(JSON, ParserWithMissingValue)
{
  nd::array a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{\"x\": 7}");
  EXPECT_ARRAY_VALS_EQ(a.p("x"), 7);
  EXPECT_TRUE(a.p("y").is_missing());

  a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{\"x\": 7, \"y\": 11.5}");
  EXPECT_ARRAY_VALS_EQ(a.p("x"), 7);
  EXPECT_ARRAY_VALS_EQ(a.p("y"), 11.5);

  a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{}");
  EXPECT_TRUE(a.p("x").is_missing());
  EXPECT_TRUE(a.p("y").is_missing());
}

TEST(JSONParser, BuiltinsFromBool)
{
  nd::array a;

  EXPECT_ARRAY_EQ(true, parse_json(ndt::type::make<bool1>(), "true"));
  EXPECT_ARRAY_EQ(false, parse_json(ndt::type::make<bool1>(), "false"));

  a = parse_json("var * bool", "[true, \"true\", 1, \"T\", \"y\", \"On\", \"yes\"]");
  EXPECT_EQ(7, a.get_dim_size());
  for (intptr_t i = 0, i_end = a.get_dim_size(); i < i_end; ++i) {
    EXPECT_TRUE(a(i).as<bool>());
  }
  a = parse_json("var * bool", "[false, \"false\", 0, \"F\", \"n\", \"Off\", \"no\"]");
  EXPECT_EQ(7, a.get_dim_size());
  for (intptr_t i = 0, i_end = a.get_dim_size(); i < i_end; ++i) {
    EXPECT_FALSE(a(i).as<bool>());
  }

  // Handling of NULL with option[bool]
  a = parse_json(ndt::option_type::make(ndt::type::make<bool1>()), "null");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<bool1>()), a.get_type());
  EXPECT_EQ(DYND_BOOL_NA, *a.get_readonly_originptr());
  a = parse_json(ndt::option_type::make(ndt::type::make<bool1>()), "\"NULL\"");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<bool1>()), a.get_type());
  EXPECT_EQ(DYND_BOOL_NA, *a.get_readonly_originptr());
  a = parse_json(ndt::option_type::make(ndt::type::make<bool1>()), "\"NA\"");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<bool1>()), a.get_type());
  EXPECT_EQ(DYND_BOOL_NA, *a.get_readonly_originptr());
  a = parse_json(ndt::option_type::make(ndt::type::make<bool1>()), "\"\"");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<bool1>()), a.get_type());
  EXPECT_EQ(DYND_BOOL_NA, *a.get_readonly_originptr());

  // Handling of an NULL, invalid token, string with junk in it, empty string
  EXPECT_THROW(parse_json(ndt::type::make<bool1>(), "null"), invalid_argument);
  EXPECT_THROW(parse_json(ndt::type::make<bool1>(), "flase"), invalid_argument);
  EXPECT_THROW(parse_json(ndt::type::make<bool1>(), "\"flase\""), invalid_argument);
  EXPECT_THROW(parse_json(ndt::type::make<bool1>(), "\"\""), invalid_argument);
  eval::eval_context ectx;
  ectx.errmode = assign_error_nocheck;
  a = parse_json(ndt::type::make<bool1>(), "null", &ectx);
  EXPECT_FALSE(a.as<bool>());
  a = parse_json(ndt::type::make<bool1>(), "\"flase\"", &ectx);
  EXPECT_TRUE(a.as<bool>());
  a = parse_json(ndt::type::make<bool1>(), "\"\"", &ectx);
  EXPECT_FALSE(a.as<bool>());
}

TEST(JSONParser, BuiltinsFromInteger)
{
  nd::array n;

  n = parse_json(ndt::type::make<int8_t>(), "123");
  EXPECT_EQ(ndt::type::make<int8_t>(), n.get_type());
  EXPECT_EQ(123, n.as<int8_t>());
  n = parse_json(ndt::type::make<int16_t>(), "-30000");
  EXPECT_EQ(ndt::type::make<int16_t>(), n.get_type());
  EXPECT_EQ(-30000, n.as<int16_t>());
  n = parse_json(ndt::type::make<int32_t>(), "500000");
  EXPECT_EQ(ndt::type::make<int32_t>(), n.get_type());
  EXPECT_EQ(500000, n.as<int32_t>());
  n = parse_json(ndt::type::make<int64_t>(), "-3000000000");
  EXPECT_EQ(ndt::type::make<int64_t>(), n.get_type());
  EXPECT_EQ(-3000000000LL, n.as<int64_t>());
  n = parse_json(ndt::type::make<int128>(), "-12345678901234567890123");
  EXPECT_EQ(ndt::type::make<int128>(), n.get_type());
  EXPECT_EQ(0xfffffffffffffd62ULL, n.as<int128>().m_hi);
  EXPECT_EQ(0xbd49b1898ebdbb35ULL, n.as<int128>().m_lo);

  n = parse_json(ndt::type::make<uint8_t>(), "123");
  EXPECT_EQ(ndt::type::make<uint8_t>(), n.get_type());
  EXPECT_EQ(123u, n.as<uint8_t>());
  n = parse_json(ndt::type::make<uint16_t>(), "50000");
  EXPECT_EQ(ndt::type::make<uint16_t>(), n.get_type());
  EXPECT_EQ(50000u, n.as<uint16_t>());
  n = parse_json(ndt::type::make<uint32_t>(), "500000");
  EXPECT_EQ(ndt::type::make<uint32_t>(), n.get_type());
  EXPECT_EQ(500000u, n.as<uint32_t>());
  n = parse_json(ndt::type::make<uint64_t>(), "3000000000");
  EXPECT_EQ(ndt::type::make<uint64_t>(), n.get_type());
  EXPECT_EQ(3000000000ULL, n.as<uint64_t>());
  n = parse_json(ndt::type::make<uint128>(), "1234567890123456789012345678");
  EXPECT_EQ(ndt::type::make<uint128>(), n.get_type());
  EXPECT_EQ(0x3fd35ebULL, n.as<uint128>().m_hi);
  EXPECT_EQ(0x6d797a91be38f34eULL, n.as<uint128>().m_lo);
}

TEST(JSONParser, OptionInt)
{
  nd::array a, b, c;

  a = parse_json(ndt::option_type::make(ndt::type::make<int8_t>()), "123");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<int8_t>()), a.get_type());
  EXPECT_EQ(123, a.as<int8_t>());
  a = parse_json(ndt::option_type::make(ndt::type::make<int8_t>()), "null");
  EXPECT_EQ(ndt::option_type::make(ndt::type::make<int8_t>()), a.get_type());
  EXPECT_EQ(DYND_INT8_NA, *reinterpret_cast<const int8_t *>(a.get_readonly_originptr()));
  EXPECT_THROW(a.as<int8_t>(), overflow_error);

  a = parse_json("9 * ?int32", "[null, 3, null, -1000, 1, 3, null, null, null]");
  EXPECT_EQ(ndt::type("9 * option[int32]"), a.get_type());
  b = nd::empty("9 * int32");
  EXPECT_THROW(b.vals() = a, overflow_error);
  // Assigning from ?int32 to ?int64 should match exactly with parsing to ?int64
  b = nd::empty("9 * ?int64");
  b.vals() = a;
  c = parse_json("9 * ?int64", "[null, 3, null, -1000, 1, 3, null, null, null]");
  EXPECT_TRUE(nd::view(b, "9 * int64").equals_exact(nd::view(c, "9 * int64")));
}

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
  EXPECT_EQ(NULL, reinterpret_cast<const dynd::string *>(a.get_readonly_originptr())->begin());
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

TEST(JSONParser, SignedIntegerLimits)
{
  nd::array n;

  n = parse_json(ndt::type::make<int8_t>(), "-128");
  EXPECT_EQ(-128, n.as<int8_t>());
  n = parse_json(ndt::type::make<int8_t>(), "127");
  EXPECT_EQ(127, n.as<int8_t>());
  n = parse_json(ndt::type::make<int16_t>(), "-32768");
  EXPECT_EQ(-32768, n.as<int16_t>());
  n = parse_json(ndt::type::make<int16_t>(), "32767");
  EXPECT_EQ(32767, n.as<int16_t>());
  n = parse_json(ndt::type::make<int32_t>(), "-2147483648");
  EXPECT_EQ(-2147483648LL, n.as<int32_t>());
  n = parse_json(ndt::type::make<int32_t>(), "2147483647");
  EXPECT_EQ(2147483647, n.as<int32_t>());
  n = parse_json(ndt::type::make<int64_t>(), "-9223372036854775808");
  EXPECT_EQ(-9223372036854775807LL - 1, n.as<int64_t>());
  n = parse_json(ndt::type::make<int64_t>(), "9223372036854775807");
  EXPECT_EQ(9223372036854775807LL, n.as<int64_t>());
  n = parse_json(ndt::type::make<int128>(), "-170141183460469231731687303715884105728");
  EXPECT_EQ(0x8000000000000000ULL, n.as<int128>().m_hi);
  EXPECT_EQ(0ULL, n.as<int128>().m_lo);
  n = parse_json(ndt::type::make<int128>(), "170141183460469231731687303715884105727");
  EXPECT_EQ(0x7fffffffffffffffULL, n.as<int128>().m_hi);
  EXPECT_EQ(0xffffffffffffffffULL, n.as<int128>().m_lo);
  EXPECT_THROW(parse_json(ndt::type::make<int8_t>(), "-129"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int8_t>(), "128"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int16_t>(), "-32769"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int16_t>(), "32768"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int32_t>(), "-2147483649"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int32_t>(), "2147483648"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int64_t>(), "-9223372036854775809"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int64_t>(), "9223372036854775808"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int128>(), "-170141183460469231731687303715884105729"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<int128>(), "170141183460469231731687303715884105728"), exception);
}

TEST(JSONParser, UnsignedIntegerLimits)
{
  nd::array n;

  n = parse_json(ndt::type::make<uint8_t>(), "0");
  EXPECT_EQ(0u, n.as<uint8_t>());
  n = parse_json(ndt::type::make<uint8_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint8_t>());
  n = parse_json(ndt::type::make<uint8_t>(), "255");
  EXPECT_EQ(255, n.as<uint8_t>());
  n = parse_json(ndt::type::make<uint16_t>(), "0");
  EXPECT_EQ(0u, n.as<uint16_t>());
  n = parse_json(ndt::type::make<uint16_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint16_t>());
  n = parse_json(ndt::type::make<uint16_t>(), "65535");
  EXPECT_EQ(65535, n.as<uint16_t>());
  n = parse_json(ndt::type::make<uint32_t>(), "0");
  EXPECT_EQ(0u, n.as<uint32_t>());
  n = parse_json(ndt::type::make<uint32_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint32_t>());
  n = parse_json(ndt::type::make<uint32_t>(), "4294967295");
  EXPECT_EQ(4294967295U, n.as<uint32_t>());
  n = parse_json(ndt::type::make<uint64_t>(), "0");
  EXPECT_EQ(0u, n.as<uint64_t>());
  n = parse_json(ndt::type::make<uint64_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint64_t>());
  n = parse_json(ndt::type::make<uint64_t>(), "18446744073709551615");
  EXPECT_EQ(18446744073709551615ULL, n.as<uint64_t>());
  n = parse_json(ndt::type::make<uint128>(), "0");
  EXPECT_EQ(0u, n.as<uint128>());
  n = parse_json(ndt::type::make<uint128>(), "-0");
  EXPECT_EQ(0u, n.as<uint128>());
  n = parse_json(ndt::type::make<uint128>(), "340282366920938463463374607431768211455");
  EXPECT_EQ(0xffffffffffffffffULL, n.as<uint128>().m_lo);
  EXPECT_EQ(0xffffffffffffffffULL, n.as<uint128>().m_hi);
  EXPECT_THROW(parse_json(ndt::type::make<uint8_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint8_t>(), "256"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint16_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint16_t>(), "65536"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint32_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint32_t>(), "4294967296"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint64_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint64_t>(), "18446744073709551616"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint128>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::type::make<uint128>(), "340282366920938463463374607431768211456"), exception);
}

TEST(JSONParser, IntFromString)
{
  nd::array a;

  a = parse_json(ndt::type::make<int>(), "\"123456\"");
  EXPECT_EQ(123456, a.as<int>());
  a = parse_json(ndt::type::make<int>(), "\"-123456\"");
  EXPECT_EQ(-123456, a.as<int>());

  EXPECT_THROW(parse_json(ndt::type::make<int>(), "\"-12356blarg\""), exception);
  eval::eval_context ectx_nocheck;
  ectx_nocheck.errmode = assign_error_nocheck;
  a = parse_json(ndt::type::make<int>(), "\"-12356blarg\"", &ectx_nocheck);
  EXPECT_EQ(-12356, a.as<int>());
}

TEST(JSONParser, BuiltinsFromFloat)
{
  nd::array n;

  n = parse_json(ndt::type::make<float>(), "123");
  EXPECT_EQ(ndt::type::make<float>(), n.get_type());
  EXPECT_EQ(123.f, n.as<float>());
  n = parse_json(ndt::type::make<float>(), "1.5");
  EXPECT_EQ(ndt::type::make<float>(), n.get_type());
  EXPECT_EQ(1.5f, n.as<float>());
  n = parse_json(ndt::type::make<float>(), "1.5e2");
  EXPECT_EQ(ndt::type::make<float>(), n.get_type());
  EXPECT_EQ(1.5e2f, n.as<float>());

  n = parse_json(ndt::type::make<double>(), "123");
  EXPECT_EQ(ndt::type::make<double>(), n.get_type());
  EXPECT_EQ(123., n.as<double>());
  n = parse_json(ndt::type::make<double>(), "1.5");
  EXPECT_EQ(ndt::type::make<double>(), n.get_type());
  EXPECT_EQ(1.5, n.as<double>());
  n = parse_json(ndt::type::make<double>(), "1.5e2");
  EXPECT_EQ(ndt::type::make<double>(), n.get_type());
  EXPECT_EQ(1.5e2, n.as<double>());
}

TEST(JSONParser, String)
{
  nd::array n;

  n = parse_json(ndt::string_type::make(), "\"testing one two three\"");
  EXPECT_EQ(ndt::string_type::make(), n.get_type());
  EXPECT_EQ("testing one two three", n.as<std::string>());
  n = parse_json(ndt::string_type::make(), "\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u0020 \"");
  EXPECT_EQ(ndt::string_type::make(), n.get_type());
  EXPECT_EQ(" \" \\ / \b \f \n \r \t   ", n.as<std::string>());

  EXPECT_THROW(parse_json(ndt::string_type::make(), "false"), invalid_argument);
}

TEST(JSONParser, ListBools)
{
  nd::array n;

  n = parse_json(ndt::var_dim_type::make(ndt::type::make<bool1>()), "  [true, true, false, false]  ");
  EXPECT_EQ(ndt::var_dim_type::make(ndt::type::make<bool1>()), n.get_type());
  EXPECT_TRUE(n(0).as<bool>());
  EXPECT_TRUE(n(1).as<bool>());
  EXPECT_FALSE(n(2).as<bool>());
  EXPECT_FALSE(n(3).as<bool>());

  n = parse_json(ndt::make_fixed_dim(4, ndt::type::make<bool1>()), "  [true, true, false, false]  ");
  EXPECT_EQ(ndt::make_fixed_dim(4, ndt::type::make<bool1>()), n.get_type());
  EXPECT_TRUE(n(0).as<bool>());
  EXPECT_TRUE(n(1).as<bool>());
  EXPECT_FALSE(n(2).as<bool>());
  EXPECT_FALSE(n(3).as<bool>());

  EXPECT_THROW(parse_json(ndt::var_dim_type::make(ndt::type::make<bool1>()), "[true, true, false, false] 3.5"),
               invalid_argument);
  EXPECT_THROW(parse_json(ndt::make_fixed_dim(4, ndt::type::make<bool1>()), "[true, true, false, false] 3.5"),
               invalid_argument);
  EXPECT_THROW(parse_json(ndt::make_fixed_dim(3, ndt::type::make<bool1>()), "[true, true, false, false]"),
               invalid_argument);
  EXPECT_THROW(parse_json(ndt::make_fixed_dim(5, ndt::type::make<bool1>()), "[true, true, false, false]"),
               invalid_argument);
}

TEST(JSONParser, NestedListInts)
{
  nd::array n;

  n = parse_json(ndt::make_fixed_dim(3, ndt::var_dim_type::make(ndt::type::make<int>())),
                 "  [[1,2,3], [4,5], [6,7,-10,1000] ]  ");
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::var_dim_type::make(ndt::type::make<int>())), n.get_type());
  EXPECT_EQ(1, n(0, 0).as<int>());
  EXPECT_EQ(2, n(0, 1).as<int>());
  EXPECT_EQ(3, n(0, 2).as<int>());
  EXPECT_EQ(4, n(1, 0).as<int>());
  EXPECT_EQ(5, n(1, 1).as<int>());
  EXPECT_EQ(6, n(2, 0).as<int>());
  EXPECT_EQ(7, n(2, 1).as<int>());
  EXPECT_EQ(-10, n(2, 2).as<int>());
  EXPECT_EQ(1000, n(2, 3).as<int>());

  n = parse_json(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::type::make<int>())), "  [[1,2,3], [4,5,2] ]  ");
  EXPECT_EQ(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::type::make<int>())), n.get_type());
  EXPECT_EQ(1, n(0, 0).as<int>());
  EXPECT_EQ(2, n(0, 1).as<int>());
  EXPECT_EQ(3, n(0, 2).as<int>());
  EXPECT_EQ(4, n(1, 0).as<int>());
  EXPECT_EQ(5, n(1, 1).as<int>());
  EXPECT_EQ(2, n(1, 2).as<int>());
}

TEST(JSONParser, Struct)
{
  nd::array n;
  ndt::type sdt =
      ndt::struct_type::make({"id", "amount", "name", "when"}, {ndt::type::make<int>(),   ndt::type::make<double>(),
                                                                ndt::string_type::make(), ndt::date_type::make()});

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
}

TEST(JSONParser, NestedStruct)
{
  nd::array n;
  ndt::type sdt = ndt::struct_type::make(
      {"position", "amount", "data"},
      {ndt::make_fixed_dim(3, ndt::type::make<float>()), ndt::type::make<double>(),
       ndt::struct_type::make({"name", "when"}, {ndt::string_type::make(), ndt::date_type::make()})});

  n = parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                      "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }");
  EXPECT_EQ(sdt, n.get_type());
  EXPECT_EQ(3.5, n(0, 0).as<float>());
  EXPECT_EQ(1.0, n(0, 1).as<float>());
  EXPECT_EQ(1e10, n(0, 2).as<float>());
  EXPECT_EQ(10.5, n(1).as<double>());
  EXPECT_EQ("Harvey", n(2, 0).as<std::string>());
  EXPECT_EQ("1970-02-13", n(2, 1).as<std::string>());

  // Too many entries in "position"
  EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                               "\"amount\": 10.5, \"position\": [1.5,3.5,1.0,1e10] }"),
               invalid_argument);

  // Too few entries in "position"
  EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, "
                               "\"amount\": 10.5, \"position\": [1.0,1e10] }"),
               invalid_argument);

  // Missing field "when"
  EXPECT_THROW(parse_json(sdt, "{\"data\":{\"name\":\"Harvey\", \"when2\":\"1970-02-13\"}, "
                               "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] }"),
               invalid_argument);
}

TEST(JSONParser, ListOfStruct)
{
  nd::array n;
  ndt::type sdt = ndt::var_dim_type::make(ndt::struct_type::make(
      {"position", "amount", "data"},
      {ndt::make_fixed_dim(3, ndt::type::make<float>()), ndt::type::make<double>(),
       ndt::struct_type::make({"name", "when"}, {ndt::string_type::make(), ndt::date_type::make()})}));

  n = parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                      "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                      "{\"position\":[1,2,3], \"amount\": 3.125,\n"
                      "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]");
  EXPECT_EQ(3.5, n(0, 0, 0).as<float>());
  EXPECT_EQ(1.0, n(0, 0, 1).as<float>());
  EXPECT_EQ(1e10, n(0, 0, 2).as<float>());
  EXPECT_EQ(10.5, n(0, 1).as<double>());
  EXPECT_EQ("Harvey", n(0, 2, 0).as<std::string>());
  EXPECT_EQ("1970-02-13", n(0, 2, 1).as<std::string>());
  EXPECT_EQ(1, n(1, 0, 0).as<float>());
  EXPECT_EQ(2, n(1, 0, 1).as<float>());
  EXPECT_EQ(3, n(1, 0, 2).as<float>());
  EXPECT_EQ(3.125, n(1, 1).as<double>());
  EXPECT_EQ("Frank", n(1, 2, 0).as<std::string>());
  EXPECT_EQ("2013-12-25", n(1, 2, 1).as<std::string>());

  // Spurious '#' inserted
  EXPECT_THROW(parse_json(sdt, "[{\"data\":{\"name\":\"Harvey\", \"when\":\"1970-02-13\"}, \n"
                               "\"amount\": 10.5, \"position\": [3.5,1.0,1e10] },\n"
                               "{\"position\":[1,2,3], \"amount\": 3.125#,\n"
                               "\"data\":{ \"when\":\"2013-12-25\", \"name\":\"Frank\"}}]"),
               invalid_argument);
}
