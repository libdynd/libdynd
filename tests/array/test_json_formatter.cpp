//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/json_formatter.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

TEST(JSONFormatter, Builtins)
{
  nd::array a;
  a = true;
  EXPECT_EQ("true", format_json(a).as<std::string>());
  a = false;
  EXPECT_EQ("false", format_json(a).as<std::string>());
  a = (int8_t) - 27;
  EXPECT_EQ("-27", format_json(a).as<std::string>());
  a = (int16_t) - 30000;
  EXPECT_EQ("-30000", format_json(a).as<std::string>());
  a = (int32_t) - 2000000000;
  EXPECT_EQ("-2000000000", format_json(a).as<std::string>());
  a = (int64_t) - 10000000000LL;
  EXPECT_EQ("-10000000000", format_json(a).as<std::string>());
  a = (uint8_t)200;
  EXPECT_EQ("200", format_json(a).as<std::string>());
  a = (uint16_t)40000;
  EXPECT_EQ("40000", format_json(a).as<std::string>());
  a = (uint32_t)3000000000u;
  EXPECT_EQ("3000000000", format_json(a).as<std::string>());
  a = (uint64_t)10000000000LL;
  EXPECT_EQ("10000000000", format_json(a).as<std::string>());
  a = 3.125f;
  EXPECT_EQ("3.125", format_json(a).as<std::string>());
  a = 3.125;
  EXPECT_EQ("3.125", format_json(a).as<std::string>());
  a = parse_json("?bool", "null");
  EXPECT_EQ("null", format_json(a).as<std::string>());
}

TEST(JSONFormatter, String)
{
  nd::array a;
  a = "testing string";
  EXPECT_EQ("\"testing string\"", format_json(a).as<std::string>());
  a = " \" \\ / \b \f \n \r \t ";
  EXPECT_EQ("\" \\\" \\\\ \\/ \\b \\f \\n \\r \\t \"", format_json(a).as<std::string>());
}

TEST(JSONFormatter, DateTime)
{
  nd::array a;
  a = nd::array("2013-12-15").ucast(ndt::date_type::make());
  EXPECT_EQ("\"2013-12-15\"", format_json(a).as<std::string>());
  a = nd::array("2013-12-15T13:14:22.19").ucast(ndt::datetime_type::make());
  EXPECT_EQ("\"2013-12-15T13:14:22.19\"", format_json(a).as<std::string>());
  a = nd::array("2013-12-15T13:14:22.19").ucast(ndt::datetime_type::make(tz_utc));
  EXPECT_EQ("\"2013-12-15T13:14:22.19Z\"", format_json(a).as<std::string>());
}

TEST(JSONFormatter, UniformDim)
{
  nd::array a;
  float vals[] = {3.5f, -1.25f, 4.75f};
  // Strided dimension
  a = vals;
  EXPECT_EQ("[3.5,-1.25,4.75]", format_json(a).as<std::string>());
  // Variable-sized dimension
  a = parse_json("var * string", "[\"testing\", \"one\", \"two\"] ");
  EXPECT_EQ("[\"testing\",\"one\",\"two\"]", format_json(a).as<std::string>());
  // Fixed dimension
  a = parse_json("3 * string", "[\"testing\", \"one\", \"two\"] ");
  EXPECT_EQ("[\"testing\",\"one\",\"two\"]", format_json(a).as<std::string>());
  // Some option types
  a = parse_json("var * ?int", "[null, 32, 21, null, null, 222]");
  EXPECT_EQ("[null,32,21,null,null,222]", format_json(a).as<std::string>());
  a = parse_json("var * ?bool", "[true, false, null, 1, 0, \"NA\"]");
  EXPECT_EQ("[true,false,null,true,false,null]", format_json(a).as<std::string>());
  a = parse_json("var * ?real", "[1.5, null, 3.125, 9.25, null, null]");
  EXPECT_EQ("[1.5,null,3.125,9.25,null,null]", format_json(a).as<std::string>());
}
