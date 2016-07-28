//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/callable.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/parse.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/view.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(JSONParser, UnsignedIntegerLimits) {
  nd::array n;

  n = parse_json(ndt::make_type<uint8_t>(), "0");
  EXPECT_EQ(0u, n.as<uint8_t>());
  n = parse_json(ndt::make_type<uint8_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint8_t>());
  n = parse_json(ndt::make_type<uint8_t>(), "255");
  EXPECT_EQ(255, n.as<uint8_t>());
  n = parse_json(ndt::make_type<uint16_t>(), "0");
  EXPECT_EQ(0u, n.as<uint16_t>());
  n = parse_json(ndt::make_type<uint16_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint16_t>());
  n = parse_json(ndt::make_type<uint16_t>(), "65535");
  EXPECT_EQ(65535, n.as<uint16_t>());
  n = parse_json(ndt::make_type<uint32_t>(), "0");
  EXPECT_EQ(0u, n.as<uint32_t>());
  n = parse_json(ndt::make_type<uint32_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint32_t>());
  n = parse_json(ndt::make_type<uint32_t>(), "4294967295");
  EXPECT_EQ(4294967295U, n.as<uint32_t>());
  n = parse_json(ndt::make_type<uint64_t>(), "0");
  EXPECT_EQ(0u, n.as<uint64_t>());
  n = parse_json(ndt::make_type<uint64_t>(), "-0");
  EXPECT_EQ(0u, n.as<uint64_t>());

  n = parse_json(ndt::make_type<uint64_t>(), "18446744073709551615");
  EXPECT_EQ(18446744073709551615ULL, n.as<uint64_t>());
  n = parse_json(ndt::make_type<uint128>(), "0");
  EXPECT_EQ(0u, n.as<uint128>());
  n = parse_json(ndt::make_type<uint128>(), "-0");
  EXPECT_EQ(0u, n.as<uint128>());
  n = parse_json(ndt::make_type<uint128>(), "340282366920938463463374607431768211455");
  EXPECT_EQ(0xffffffffffffffffULL, n.as<uint128>().m_lo);
  EXPECT_EQ(0xffffffffffffffffULL, n.as<uint128>().m_hi);
  EXPECT_THROW(parse_json(ndt::make_type<uint8_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint8_t>(), "256"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint16_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint16_t>(), "65536"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint32_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint32_t>(), "4294967296"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint64_t>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint64_t>(), "18446744073709551616"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint128>(), "-1"), exception);
  EXPECT_THROW(parse_json(ndt::make_type<uint128>(), "340282366920938463463374607431768211456"), exception);
}

TEST(JSONParser, IntFromString) {
  nd::array a;

  a = parse_json(ndt::make_type<int>(), "\"123456\"");
  EXPECT_EQ(123456, a.as<int>());
  a = parse_json(ndt::make_type<int>(), "\"-123456\"");
  EXPECT_EQ(-123456, a.as<int>());

  EXPECT_THROW(parse_json(ndt::make_type<int>(), "\"-12356blarg\""), exception);
  eval::eval_context ectx_nocheck;
  ectx_nocheck.errmode = assign_error_nocheck;
  a = parse_json(ndt::make_type<int>(), "\"-12356blarg\"", &ectx_nocheck);
  EXPECT_EQ(-12356, a.as<int>());
}

TEST(JSONParser, BuiltinsFromFloat) {
  nd::array n;

  n = parse_json(ndt::make_type<float>(), "123");
  EXPECT_EQ(ndt::make_type<float>(), n.get_type());
  EXPECT_EQ(123.f, n.as<float>());
  n = parse_json(ndt::make_type<float>(), "1.5");
  EXPECT_EQ(ndt::make_type<float>(), n.get_type());
  EXPECT_EQ(1.5f, n.as<float>());
  n = parse_json(ndt::make_type<float>(), "1.5e2");
  EXPECT_EQ(ndt::make_type<float>(), n.get_type());
  EXPECT_EQ(1.5e2f, n.as<float>());

  n = parse_json(ndt::make_type<double>(), "123");
  EXPECT_EQ(ndt::make_type<double>(), n.get_type());
  EXPECT_EQ(123., n.as<double>());
  n = parse_json(ndt::make_type<double>(), "1.5");
  EXPECT_EQ(ndt::make_type<double>(), n.get_type());
  EXPECT_EQ(1.5, n.as<double>());
  n = parse_json(ndt::make_type<double>(), "1.5e2");
  EXPECT_EQ(ndt::make_type<double>(), n.get_type());
  EXPECT_EQ(1.5e2, n.as<double>());
}

TEST(JSONParser, Struct) {
  nd::array n;
  ndt::type sdt = ndt::make_type<ndt::struct_type>({{ndt::make_type<int>(), "id"},
                                                    {ndt::make_type<double>(), "amount"},
                                                    {ndt::make_type<ndt::string_type>(), "name"},
                                                    {ndt::make_type<ndt::string_type>(), "when"}});

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

TEST(JSONParser, NestedStruct) {
  nd::array n;
  ndt::type sdt = ndt::make_type<ndt::struct_type>(
      {{ndt::make_fixed_dim(3, ndt::make_type<float>()), "position"},
       {ndt::make_type<double>(), "amount"},
       {ndt::make_type<ndt::struct_type>(
            {{ndt::make_type<ndt::string_type>(), "name"}, {ndt::make_type<ndt::string_type>(), "when"}}),
        "data"}});

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

TEST(JSONParser, ListOfStruct) {
  nd::array n;
  ndt::type sdt = ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::struct_type>(
      {{ndt::make_fixed_dim(3, ndt::make_type<float>()), "position"},
       {ndt::make_type<double>(), "amount"},
       {ndt::make_type<ndt::struct_type>(
            {{ndt::make_type<ndt::string_type>(), "name"}, {ndt::make_type<ndt::string_type>(), "when"}}),
        "data"}}));

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

TEST(JSON, ParserWithMissingValue) {
  nd::array a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{\"x\": 7}");
  EXPECT_ARRAY_VALS_EQ(a.p("x"), 7);
  EXPECT_TRUE(a.p("y").is_na());

  a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{\"x\": 7, \"y\": 11.5}");
  EXPECT_ARRAY_VALS_EQ(a.p("x"), 7);
  EXPECT_ARRAY_VALS_EQ(a.p("y"), 11.5);

  a = parse_json(ndt::type("{x: ?int32, y: ?float64}"), "{}");
  EXPECT_TRUE(a.p("x").is_na());
  EXPECT_TRUE(a.p("y").is_na());
}

/*
TEST(JSON, DiscoverBool)
{
  EXPECT_EQ(ndt::make_type<bool1>(), ndt::json::discover("true"));
  EXPECT_EQ(ndt::make_type<bool1>(), ndt::json::discover("false"));
}

TEST(JSON, DiscoverInt64)
{
  EXPECT_EQ(ndt::make_type<int64>(), ndt::json::discover("0"));
  EXPECT_EQ(ndt::make_type<int64>(), ndt::json::discover("3"));
  EXPECT_EQ(ndt::make_type<int64>(), ndt::json::discover("11"));

  EXPECT_EQ(ndt::make_type<int64>(), ndt::json::discover("-1"));
  EXPECT_EQ(ndt::make_type<int64>(), ndt::json::discover("-5"));
}

TEST(JSON, DiscoverFloat64)
{
  EXPECT_EQ(ndt::make_type<float64>(), ndt::json::discover("0.5"));
  EXPECT_EQ(ndt::make_type<float64>(), ndt::json::discover("3.14"));
}

TEST(JSON, DiscoverString) { EXPECT_EQ(ndt::type(string_id), ndt::json::discover("\"Hello, world!\"")); }

TEST(JSON, DiscoverOption) { EXPECT_EQ(ndt::type("?Any"), ndt::json::discover("null")); }

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
*/
