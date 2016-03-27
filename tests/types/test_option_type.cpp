//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/option.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(OptionType, Create)
{
  ndt::type d;

  d = ndt::make_type<ndt::option_type>(ndt::make_type<int16_t>());
  EXPECT_EQ(option_id, d.get_id());
  EXPECT_EQ(any_kind_id, d.get_base_id());
  EXPECT_EQ(2u, d.get_data_alignment());
  EXPECT_EQ(2u, d.get_data_size());
  EXPECT_EQ(ndt::make_type<int16_t>(), d.extended<ndt::option_type>()->get_value_type());
  EXPECT_TRUE(d.is_scalar());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
  EXPECT_EQ("?int16", d.str());
  EXPECT_EQ(d, ndt::type("?int16"));
  EXPECT_EQ(d, ndt::type("option[int16]"));

  d = ndt::make_type<ndt::option_type>(ndt::make_type<ndt::string_type>());
  EXPECT_EQ(option_id, d.get_id());
  EXPECT_EQ(any_kind_id, d.get_base_id());
  EXPECT_EQ(ndt::make_type<ndt::string_type>().get_data_alignment(), d.get_data_alignment());
  EXPECT_EQ(ndt::make_type<ndt::string_type>().get_data_size(), d.get_data_size());
  EXPECT_EQ(ndt::make_type<ndt::string_type>(), d.extended<ndt::option_type>()->get_value_type());
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
  EXPECT_EQ("?string", d.str());

  // No option of option allowed
  EXPECT_THROW(ndt::make_type<ndt::option_type>(ndt::make_type<ndt::option_type>(ndt::make_type<int>())), type_error);
  EXPECT_THROW(ndt::type("option[option[bool]]"), type_error);
}

TEST(OptionType, OptionIntAssign)
{
  nd::array a, b, c;
  eval::eval_context tmp_ectx;

  // Assignment from option[S] to option[T]
  a = parse_json("2 * ?int8", "[-10, null]");
  b = nd::empty("2 * ?int16");
  b.vals() = a;
  EXPECT_JSON_EQ_ARR("[-10, -32768]", nd::old_view(b, "2 * int16"));
  tmp_ectx.errmode = assign_error_nocheck;
  b.assign(a, tmp_ectx.errmode);
  EXPECT_JSON_EQ_ARR("[-10, -32768]", nd::old_view(b, "2 * int16"));

  // Assignment from option[T] to T without any NAs
  a = parse_json("3 * ?int32", "[1, 2, 3]");
  b = nd::empty("3 * int32");
  b.vals() = a;
  EXPECT_ARRAY_EQ(nd::old_view(a, "3 * int32"), b);

  // Assignment from T to option[T]
  a = parse_json("3 * int32", "[1, 3, 5]");
  b = nd::empty("3 * ?int32");
  b.vals() = a;
  EXPECT_ARRAY_EQ(a, nd::old_view(b, "3 * int32"));

  // Assignment from string to option[int]
  a = parse_json("5 * string", "[\"null\", \"12\", \"NA\", \"34\", \"\"]");
  b = nd::empty("5 * ?int32");
  b.vals() = a;
  c = parse_json("5 * ?int32", "[null, 12, null, 34, null]");
  EXPECT_ARRAY_EQ(nd::old_view(c, "Fixed * int32"), nd::old_view(b, "Fixed * int32"));
}

TEST(OptionType, Cast)
{
  nd::array a, b;

  a = parse_json("3 * string", "[\"null\", \"NA\", \"25\"]");
  b = nd::empty(ndt::type("3 * ?int"));
  b.assign(a);
  EXPECT_ARRAY_EQ(nd::old_view(parse_json("3 * ?int", "[null, null, 25]"), "3 * int"), nd::old_view(b, "3 * int"));
}

TEST(OptionType, FloatNAvsNaN)
{
  nd::array a = nd::empty("3 * ?float64");

  parse_json(a, "[0, null, \"nan\"]");
  // This is matching R's behavior with floating point NaN
  EXPECT_FALSE(nd::is_na(a(0)).as<bool>());
  EXPECT_TRUE(nd::is_na(a(1)).as<bool>());
  EXPECT_TRUE(nd::is_na(a(2)).as<bool>());
  // TODO: An isnan arrfunc should return false, NA, true
}

TEST(OptionType, Float)
{
  nd::array a = nd::empty("5 * float64");

  parse_json(a, "[12, 0, \"nan\", -99, \"nan\"]");

  // Assigning from a float NaN value to an ?int type converts NaN
  // to NA values
  nd::array b = nd::empty(5, "?int");
  b.vals() = a;
  EXPECT_EQ(12, b(0).as<int>());
  EXPECT_EQ(0, b(1).as<int>());
  EXPECT_TRUE(nd::is_na(b(2)).as<bool>());
  EXPECT_EQ(-99, b(3).as<int>());
  EXPECT_TRUE(nd::is_na(b(4)).as<bool>());
}

/*
TEST(OptionType, String)
{
  nd::array a = nd::empty("5 * ?string");

  parse_json(a, "[null, \"testing\", \"NA\", \"\","
                " \"valid\"]");
  EXPECT_TRUE(nd::is_na(a(0)).as<bool>());
  EXPECT_EQ("testing", a(1).as<std::string>());
  EXPECT_EQ("NA", a(2).as<std::string>());
  EXPECT_EQ("", a(3).as<std::string>());
  EXPECT_EQ("valid", a(4).as<std::string>());

  a = nd::empty("5 * ?string");
  a.vals_at(0) = "";
  std::cout << a(0).as<std::string>() << std::endl;
  EXPECT_EQ("", a(0).as<std::string>());
  a.vals_at(1) = "NA";
  EXPECT_EQ("NA", a(1).as<std::string>());
}
*/
