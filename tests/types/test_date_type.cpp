//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/date_util.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(DateType, Create)
{
  ndt::type d;

  d = ndt::date_type::make();
  EXPECT_EQ(4u, d.get_data_size());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(ndt::date_type::make(), ndt::date_type::make());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DateType, ValueCreation)
{
  ndt::type d = ndt::date_type::make(), di = ndt::type::make<int32_t>();

  EXPECT_EQ((1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 - 365,
            nd::array("1599-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((1600 - 1970) * 365 - (1972 - 1600) / 4 + 3,
            nd::array("1600-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 + 366,
            nd::array("1601-01-01").ucast(d).view_scalars(di).as<int32_t>());

  EXPECT_EQ((1900 - 1970) * 365 - (1970 - 1900) / 4,
            nd::array("1900-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((1900 - 1970) * 365 - (1970 - 1900) / 4 + 365,
            nd::array("1901-01-01").ucast(d).view_scalars(di).as<int32_t>());

  EXPECT_EQ(-3 * 365 - 1,
            nd::array("1967-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(-2 * 365 - 1,
            nd::array("1968-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(-1 * 365,
            nd::array("1969-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(0 * 365,
            nd::array("1970-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(1 * 365,
            nd::array("1971-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(2 * 365,
            nd::array("1972-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(3 * 365 + 1,
            nd::array("1973-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ(4 * 365 + 1,
            nd::array("1974-01-01").ucast(d).view_scalars(di).as<int32_t>());

  EXPECT_EQ((2000 - 1970) * 365 + (2000 - 1972) / 4,
            nd::array("2000-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((2000 - 1970) * 365 + (2000 - 1972) / 4 + 366,
            nd::array("2001-01-01").ucast(d).view_scalars(di).as<int32_t>());

  EXPECT_EQ((2400 - 1970) * 365 + (2400 - 1972) / 4 - 3,
            nd::array("2400-01-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((2400 - 1970) * 365 + (2400 - 1972) / 4 - 3 + 366,
            nd::array("2401-01-01").ucast(d).view_scalars(di).as<int32_t>());

  EXPECT_EQ((1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 + 31 + 28,
            nd::array("1600-02-29").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 + 31 + 29,
            nd::array("1600-03-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((2000 - 1970) * 365 + (2000 - 1972) / 4 + 31 + 28,
            nd::array("2000-02-29").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((2000 - 1970) * 365 + (2000 - 1972) / 4 + 31 + 29,
            nd::array("2000-03-01").ucast(d).view_scalars(di).as<int32_t>());
  EXPECT_EQ((2000 - 1970) * 365 + (2000 - 1972) / 4 + 366 + 31 + 28 + 21,
            nd::array("2001-03-22").ucast(d).view_scalars(di).as<int32_t>());

  // Unambiguous date format which has extra empty time attached
  EXPECT_EQ(
      (2000 - 1970) * 365 + (2000 - 1972) / 4 + 366 + 31 + 28 + 21,
      nd::array("2001-03-22 00:00:00").ucast(d).view_scalars(di).as<int32_t>());
}

TEST(DateType, BadInputStrings)
{
  ndt::type d = ndt::date_type::make();

  // Arbitrary bad string
  EXPECT_THROW(nd::array("badvalue").ucast(d).eval(), invalid_argument);
  // Character after year must be '-'
  EXPECT_THROW(nd::array("1980X").ucast(d).eval(), invalid_argument);
  // Cannot have trailing '-'
  EXPECT_THROW(nd::array("1980-").ucast(d).eval(), invalid_argument);
  // Month must be in range [1,12]
  EXPECT_THROW(nd::array("1980-00").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-13").ucast(d).eval(), invalid_argument);
  // Month must have one or two digits
  EXPECT_THROW(nd::array("1980-1").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-1-023").ucast(d).eval(), invalid_argument);
  // 'Mor' is not a valid month
  EXPECT_THROW(nd::array("1980-Mor").ucast(d).eval(), invalid_argument);
  // Cannot have trailing '-'
  EXPECT_THROW(nd::array("1980-01-").ucast(d).eval(), invalid_argument);
  // Day must be in range [1,len(month)]
  EXPECT_THROW(nd::array("1980-01-0").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-01-00").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-01-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1979-02-29").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-02-30").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-03-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-04-31").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-05-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-06-31").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-07-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-08-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-09-31").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-10-32").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-11-31").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-12-32").ucast(d).eval(), invalid_argument);
  // Cannot have trailing characters
  EXPECT_THROW(nd::array("1980-02-03%").ucast(d).eval(), invalid_argument);
  EXPECT_THROW(nd::array("1980-02-03 q").ucast(d).eval(), invalid_argument);
}

TEST(DateType, DateProperties)
{
  ndt::type d = ndt::date_type::make();
  nd::array a;

  a = nd::array("1955-03-13").ucast(d).eval();
  EXPECT_EQ(ndt::property_type::make(d, "year"), a.p("year").get_type());
  EXPECT_EQ(ndt::property_type::make(d, "month"), a.p("month").get_type());
  EXPECT_EQ(ndt::property_type::make(d, "day"), a.p("day").get_type());
  EXPECT_EQ(1955, a.p("year").as<int32_t>());
  EXPECT_EQ(3, a.p("month").as<int32_t>());
  EXPECT_EQ(13, a.p("day").as<int32_t>());

  const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
  a = nd::array(strs).ucast(d).eval();
  EXPECT_EQ(1931, a.p("year")(0).as<int32_t>());
  EXPECT_EQ(12, a.p("month")(0).as<int32_t>());
  EXPECT_EQ(12, a.p("day")(0).as<int32_t>());
  EXPECT_EQ(2013, a.p("year")(1).as<int32_t>());
  EXPECT_EQ(5, a.p("month")(1).as<int32_t>());
  EXPECT_EQ(14, a.p("day")(1).as<int32_t>());
  EXPECT_EQ(2012, a.p("year")(2).as<int32_t>());
  EXPECT_EQ(12, a.p("month")(2).as<int32_t>());
  EXPECT_EQ(25, a.p("day")(2).as<int32_t>());
}

TEST(DateType, DatePropertyConvertOfString)
{
  nd::array a, b, c;
  const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
  a = nd::array(strs)
          .ucast(ndt::fixed_string_type::make(10, string_encoding_ascii))
          .eval();
  b = a.ucast(ndt::date_type::make());
  EXPECT_EQ(ndt::make_fixed_dim(
                3, ndt::fixed_string_type::make(10, string_encoding_ascii)),
            a.get_type());
  EXPECT_EQ(ndt::make_fixed_dim(
                3, ndt::convert_type::make(ndt::date_type::make(),
                                           ndt::fixed_string_type::make(
                                               10, string_encoding_ascii))),
            b.get_type());

  // year property
  c = b.p("year");
  EXPECT_EQ(property_type_id, c.get_dtype().get_type_id());
  c = c.eval();
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::type::make<int>()), c.get_type());
  EXPECT_EQ(1931, c(0).as<int>());
  EXPECT_EQ(2013, c(1).as<int>());
  EXPECT_EQ(2012, c(2).as<int>());

  // weekday function
  c = b.f("weekday");
  EXPECT_EQ(property_type_id, c.get_dtype().get_type_id());
  c = c.eval();
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::type::make<int>()), c.get_type());
  EXPECT_EQ(5, c(0).as<int>());
  EXPECT_EQ(1, c(1).as<int>());
  EXPECT_EQ(1, c(2).as<int>());
}

TEST(DateType, ToStructFunction)
{
  ndt::type d = ndt::date_type::make();
  nd::array a, b;

  a = nd::array("1955-03-13").ucast(d).eval();
  b = a.f("to_struct");
  EXPECT_EQ(ndt::property_type::make(d, "struct"), b.get_type());
  b = b.eval();
  EXPECT_EQ(ndt::struct_type::make({"year", "month", "day"},
                                   {ndt::type::make<int16_t>(),
                                    ndt::type::make<int8_t>(),
                                    ndt::type::make<int8_t>()}),
            b.get_type());
  EXPECT_EQ(1955, b.p("year").as<int32_t>());
  EXPECT_EQ(3, b.p("month").as<int32_t>());
  EXPECT_EQ(13, b.p("day").as<int32_t>());

  // Do it again, but now with a chain of expressions
  a = nd::array("1955-03-13").ucast(d).f("to_struct");
  EXPECT_EQ(1955, a.p("year").as<int32_t>());
  EXPECT_EQ(3, a.p("month").as<int32_t>());
  EXPECT_EQ(13, a.p("day").as<int32_t>());
}

TEST(DateType, ToStruct)
{
  ndt::type d = ndt::date_type::make(), ds;
  nd::array a, b;

  a = nd::array("1955-03-13").ucast(d).eval();

  // This is the default struct produced
  ds = ndt::struct_type::make({"year", "month", "day"},
                              {ndt::type::make<int32_t>(),
                               ndt::type::make<int8_t>(),
                               ndt::type::make<int8_t>()});
  b = nd::empty(ds);
  b.vals() = a;
  EXPECT_EQ(1955, b(0).as<int32_t>());
  EXPECT_EQ(3, b(1).as<int8_t>());
  EXPECT_EQ(13, b(2).as<int8_t>());

  // This should work too
  ds = ndt::struct_type::make({"month", "year", "day"},
                              {ndt::type::make<int16_t>(),
                               ndt::type::make<int16_t>(),
                               ndt::type::make<float>()});
  b = nd::empty(ds);
  b.vals() = a;
  EXPECT_EQ(1955, b(1).as<int16_t>());
  EXPECT_EQ(3, b(0).as<int16_t>());
  EXPECT_EQ(13, b(2).as<float>());
}

TEST(DateType, FromStruct)
{
  ndt::type d = ndt::date_type::make(), ds;
  nd::array a, b;

  // This is the default struct accepted
  ds = ndt::struct_type::make({"year", "month", "day"},
                              {ndt::type::make<int32_t>(),
                               ndt::type::make<int8_t>(),
                               ndt::type::make<int8_t>()});
  a = nd::empty(ds);
  a(0).vals() = 1955;
  a(1).vals() = 3;
  a(2).vals() = 13;
  b = nd::empty(d);
  b.vals() = a;
  EXPECT_EQ(1955, b.p("year").as<int32_t>());
  EXPECT_EQ(3, b.p("month").as<int32_t>());
  EXPECT_EQ(13, b.p("day").as<int32_t>());

  // This should work too
  ds = ndt::struct_type::make({"month", "year", "day"},
                              {ndt::type::make<int16_t>(),
                               ndt::type::make<int16_t>(),
                               ndt::type::make<float>()});
  a = nd::empty(ds);
  a.p("year").vals() = 1955;
  a.p("month").vals() = 3;
  a.p("day").vals() = 13;
  b = nd::empty(d);
  b.vals() = a;
  EXPECT_EQ(1955, b.p("year").as<int32_t>());
  EXPECT_EQ(3, b.p("month").as<int32_t>());
  EXPECT_EQ(13, b.p("day").as<int32_t>());
}

TEST(DateType, StrFTime)
{
  ndt::type d = ndt::date_type::make(), ds;
  nd::array a, b;

  a = nd::array("1955-03-13").ucast(d).eval();

  b = a.f("strftime", "%Y");
  EXPECT_EQ("1955", b.as<std::string>());
  b = a.f("strftime", "%m/%d/%y");
  EXPECT_EQ("03/13/55", b.as<std::string>());
  b = a.f("strftime", "%Y and %j");
  EXPECT_EQ("1955 and 072", b.as<std::string>());

  const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
  a = nd::array(strs).ucast(d).eval();

  b = a.f("strftime", "%Y-%m-%d %j %U %w %W");
  EXPECT_EQ("1931-12-12 346 49 6 49", b(0).as<std::string>());
  EXPECT_EQ("2013-05-14 134 19 2 19", b(1).as<std::string>());
  EXPECT_EQ("2012-12-25 360 52 2 52", b(2).as<std::string>());
}

TEST(DateType, StrFTimeOfConvert)
{
  // First create a date array which is still a convert expression type
  const char *vals[] = {"1920-03-12", "2013-01-01", "2000-12-25"};
  nd::array a = nd::array(vals).ucast(ndt::date_type::make());
  EXPECT_EQ(
      ndt::make_fixed_dim(3, ndt::convert_type::make(ndt::date_type::make(),
                                                     ndt::string_type::make())),
      a.get_type());

  nd::array b = a.f("strftime", "%Y %m %d");
  EXPECT_EQ("1920 03 12", b(0).as<std::string>());
  EXPECT_EQ("2013 01 01", b(1).as<std::string>());
  EXPECT_EQ("2000 12 25", b(2).as<std::string>());
}

TEST(DateType, StrFTimeOfMultiDim)
{
  const char *vals_0[] = {"1920-03-12", "2013-01-01"};
  const char *vals_1[] = {"2000-12-25"};
  nd::array a = nd::empty(2, -1, ndt::date_type::make());
  a.vals_at(0) = vals_0;
  a.vals_at(1) = vals_1;

  a = a.f("strftime", "%d/%m/%Y");
  EXPECT_EQ("12/03/1920", a(0, 0).as<std::string>());
  EXPECT_EQ("01/01/2013", a(0, 1).as<std::string>());
  EXPECT_EQ("25/12/2000", a(1, 0).as<std::string>());
}

#if defined(_MSC_VER)
// Only the Windows strftime seems to support this behavior without
// writing our own strftime format parser.
TEST(DateType, StrFTimeBadFormat)
{
  ndt::type d = ndt::date_type::make();
  nd::array a;

  a = nd::array("1955-03-13").ucast(d).eval();
  // Invalid format string should raise an error.
  EXPECT_THROW(a.f("strftime", "%Y %x %s").eval(), runtime_error);
}
#endif

TEST(DateType, WeekDay)
{
  ndt::type d = ndt::date_type::make();
  nd::array a;

  a = nd::array("1955-03-13").ucast(d).eval();
  EXPECT_EQ(6, a.f("weekday").as<int32_t>());
  a = nd::array("2002-12-04").ucast(d).eval();
  EXPECT_EQ(2, a.f("weekday").as<int32_t>());
}

TEST(DateType, Replace)
{
  ndt::type d = ndt::date_type::make();
  nd::array a;

  a = nd::array("1955-03-13").ucast(d).eval();
  EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<std::string>());
  EXPECT_EQ("2012-12-13", a.f("replace", 2012, 12).as<std::string>());
  EXPECT_EQ("2012-12-15", a.f("replace", 2012, 12, 15).as<std::string>());
  // Custom extension, allow -1 indexing from the end for months and days
  EXPECT_EQ("2012-12-30", a.f("replace", 2012, -1, 30).as<std::string>());
  EXPECT_EQ("2012-05-31", a.f("replace", 2012, -8, -1).as<std::string>());
  // The C++ call interface doesn't let you skip arguments (yet, there is no
  // keyword argument mechanism),
  // so test this manually
  nd::array param =
      a.find_dynamic_function("replace").get_default_parameters().eval_copy(
          nd::readwrite_access_flags);
  *reinterpret_cast<void **>(param(0).get_readwrite_originptr()) =
      (void *)a.get_ndo();
  param(2).vals() = 7;
  EXPECT_EQ(
      "1955-07-13",
      a.find_dynamic_function("replace").call_generic(param).as<std::string>());
  param(3).vals() = -1;
  EXPECT_EQ(
      "1955-07-31",
      a.find_dynamic_function("replace").call_generic(param).as<std::string>());
  param(2).vals() = 2;
  EXPECT_EQ(
      "1955-02-28",
      a.find_dynamic_function("replace").call_generic(param).as<std::string>());
  param(1).vals() = 2012;
  EXPECT_EQ(
      "2012-02-29",
      a.find_dynamic_function("replace").call_generic(param).as<std::string>());
  // Should throw an exception when no arguments or out of bounds arguments are
  // provided
  EXPECT_THROW(a.f("replace").eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, -13).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, 0).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, 13).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 1900, 2, -29).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 1900, 2, 0).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 1900, 2, 29).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, 2, -30).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, 2, 0).eval(), runtime_error);
  EXPECT_THROW(a.f("replace", 2000, 2, 30).eval(), runtime_error);
}

TEST(DateType, ReplaceOfConvert)
{
  nd::array a;

  // Make an expression type with value type 'date'
  a = nd::array("1955-03-13").ucast(ndt::date_type::make());
  EXPECT_EQ(
      ndt::convert_type::make(ndt::date_type::make(), ndt::string_type::make()),
      a.get_type());
  // Call replace on it
  EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<std::string>());
}

TEST(DateType, NumPyCompatibleProperty)
{
  int64_t vals64[] = {-16730, 0, 11001, numeric_limits<int64_t>::min()};

  nd::array a = nd::array_rw(vals64);
  nd::array a_date = a.adapt(ndt::date_type::make(), "days since 1970-01-01");
  // Reading from the 'int64 as date' view
  EXPECT_EQ("1924-03-13", a_date(0).as<std::string>());
  EXPECT_EQ("1970-01-01", a_date(1).as<std::string>());
  EXPECT_EQ("2000-02-14", a_date(2).as<std::string>());
  EXPECT_EQ("NA", a_date(3).as<std::string>());

  // Writing to the 'int64 as date' view
  a_date(0).vals() = "1975-01-30";
  EXPECT_EQ(1855, a(0).as<int64_t>());
  a_date(0).vals() = "NA";
  EXPECT_EQ(numeric_limits<int64_t>::min(), a(0).as<int64_t>());
}

TEST(DateType, AdaptFromInt)
{
  nd::array a, b;

  // int32
  a = nd::array_rw(25);
  b = a.adapt(ndt::date_type::make(), "days since 2012-03-02");
  EXPECT_EQ("2012-03-27", b.as<std::string>());
  // This adapter works both ways
  b.vals() = "2012-03-01";
  EXPECT_EQ(-1, a.as<int>());

  // int64
  a = nd::array_rw(365LL);
  b = a.adapt(ndt::date_type::make(), "days since 1925-03-02");
  EXPECT_EQ("1926-03-02", b.as<std::string>());
  b.vals() = "1925-04-02";
  EXPECT_EQ(31, a.as<int>());

  // Array of int32
  const char *s_vals[] = {"2000-01-01", "2100-02-01", "2099-12-31"};
  int32_t i32_vals[] = {-5, 10, 0};
  a = nd::array_rw(i32_vals);
  b = a.adapt(ndt::date_type::make(), "days since 2100-1-1");
  EXPECT_EQ("2099-12-27", b(0).as<std::string>());
  EXPECT_EQ("2100-01-11", b(1).as<std::string>());
  EXPECT_EQ("2100-01-01", b(2).as<std::string>());
  b.vals() = s_vals;
  EXPECT_EQ(-365 * 100 - 25, a(0).as<int>());
  EXPECT_EQ(31, a(1).as<int>());
  EXPECT_EQ(-1, a(2).as<int>());
}

TEST(DateType, AdaptAsInt)
{
  nd::array a, b;

  a = parse_json("3 * date",
                 "[\"2001-01-05\", \"1999-12-20\", \"2000-01-01\"]");
  b = a.adapt(ndt::type::make<int>(), "days since 2000-01-01");
  EXPECT_EQ(370, b(0).as<int>());
  EXPECT_EQ(-12, b(1).as<int>());
  EXPECT_EQ(0, b(2).as<int>());
}

TEST(DateYMD, LeapYear)
{
  EXPECT_TRUE(date_ymd::is_leap_year(1600));
  EXPECT_FALSE(date_ymd::is_leap_year(1700));
  EXPECT_FALSE(date_ymd::is_leap_year(1800));
  EXPECT_FALSE(date_ymd::is_leap_year(1900));
  EXPECT_TRUE(date_ymd::is_leap_year(2000));
  EXPECT_FALSE(date_ymd::is_leap_year(2001));
  EXPECT_FALSE(date_ymd::is_leap_year(2002));
  EXPECT_FALSE(date_ymd::is_leap_year(2003));
  EXPECT_TRUE(date_ymd::is_leap_year(2004));
}

TEST(DateYMD, ToDays)
{
  date_ymd d;

  // Some extreme edge cases
  d.year = -29200;
  d.month = 2;
  d.day = 29;
  EXPECT_EQ(-11384550, d.to_days());
  d.year = 32000;
  d.month = 1;
  d.day = 31;
  EXPECT_EQ(10968262, d.to_days());
  d.year = -1234;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ(-1170237, d.to_days());
  d.year = -1;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ(-719529, d.to_days());
  d.year = 0;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ(-719528, d.to_days());
  d.year = 0;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ(-719163, d.to_days());
  d.year = 1;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ(-719162, d.to_days());

  // Values around the 1970 epoch
  d.year = 1969;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ(-1, d.to_days());
  d.year = 1970;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ(0, d.to_days());
  d.year = 1970;
  d.month = 1;
  d.day = 2;
  EXPECT_EQ(1, d.to_days());

  // Values around Feb 29 1968 (leap year prior to 1970)
  d.year = 1968;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(-673, d.to_days());
  d.year = 1968;
  d.month = 2;
  d.day = 29;
  EXPECT_EQ(-672, d.to_days());
  d.year = 1968;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(-671, d.to_days());

  // Values areound Feb 29, 1972 (leap year after 1970)
  d.year = 1972;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(788, d.to_days());
  d.year = 1972;
  d.month = 2;
  d.day = 29;
  EXPECT_EQ(789, d.to_days());
  d.year = 1972;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(790, d.to_days());

  // Values around Feb 28 1900 (special non-leap year prior to 1970)
  d.year = 1900;
  d.month = 2;
  d.day = 27;
  EXPECT_EQ(-25510, d.to_days());
  d.year = 1900;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(-25509, d.to_days());
  d.year = 1900;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(-25508, d.to_days());

  // Values around Feb 29 1600 (special leap year prior to 1970)
  d.year = 1600;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(-135082, d.to_days());
  d.year = 1600;
  d.month = 2;
  d.day = 29;
  EXPECT_EQ(-135081, d.to_days());
  d.year = 1600;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(-135080, d.to_days());

  // Values around Feb 29 2000 (special leap year after 1970)
  d.year = 2000;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(11015, d.to_days());
  d.year = 2000;
  d.month = 2;
  d.day = 29;
  EXPECT_EQ(11016, d.to_days());
  d.year = 2000;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(11017, d.to_days());

  // Values around Feb 28 2100 (special non-leap year after 1970)
  d.year = 2100;
  d.month = 2;
  d.day = 27;
  EXPECT_EQ(47539, d.to_days());
  d.year = 2100;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(47540, d.to_days());
  d.year = 2100;
  d.month = 3;
  d.day = 1;
  EXPECT_EQ(47541, d.to_days());

  // The last of every month of a non-leap year
  d.year = 1979;
  d.month = 1;
  d.day = 31;
  EXPECT_EQ(3317, d.to_days());
  d.year = 1979;
  d.month = 2;
  d.day = 28;
  EXPECT_EQ(3345, d.to_days());
  d.year = 1979;
  d.month = 3;
  d.day = 31;
  EXPECT_EQ(3376, d.to_days());
  d.year = 1979;
  d.month = 4;
  d.day = 30;
  EXPECT_EQ(3406, d.to_days());
  d.year = 1979;
  d.month = 5;
  d.day = 31;
  EXPECT_EQ(3437, d.to_days());
  d.year = 1979;
  d.month = 6;
  d.day = 30;
  EXPECT_EQ(3467, d.to_days());
  d.year = 1979;
  d.month = 7;
  d.day = 31;
  EXPECT_EQ(3498, d.to_days());
  d.year = 1979;
  d.month = 8;
  d.day = 31;
  EXPECT_EQ(3529, d.to_days());
  d.year = 1979;
  d.month = 9;
  d.day = 30;
  EXPECT_EQ(3559, d.to_days());
  d.year = 1979;
  d.month = 10;
  d.day = 31;
  EXPECT_EQ(3590, d.to_days());
  d.year = 1979;
  d.month = 11;
  d.day = 30;
  EXPECT_EQ(3620, d.to_days());
  d.year = 1979;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ(3651, d.to_days());
}

TEST(DateYMD, SetFromDays)
{
  date_ymd d;

  // Some extreme edge cases
  d.set_from_days(-11384550);
  EXPECT_EQ(-29200, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(29, d.day);
  d.set_from_days(10968262);
  EXPECT_EQ(32000, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(31, d.day);
  d.set_from_days(-1170237);
  EXPECT_EQ(-1234, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(1, d.day);
  d.set_from_days(-719529);
  EXPECT_EQ(-1, d.year);
  EXPECT_EQ(12, d.month);
  EXPECT_EQ(31, d.day);
  d.set_from_days(-719528);
  EXPECT_EQ(0, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(1, d.day);
  d.set_from_days(-719163);
  EXPECT_EQ(0, d.year);
  EXPECT_EQ(12, d.month);
  EXPECT_EQ(31, d.day);
  d.set_from_days(-719162);
  EXPECT_EQ(1, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(1, d.day);

  // Values around the 1970 epoch
  d.set_from_days(-1);
  EXPECT_EQ(1969, d.year);
  EXPECT_EQ(12, d.month);
  EXPECT_EQ(31, d.day);
  d.set_from_days(0);
  EXPECT_EQ(1970, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(1, d.day);
  d.set_from_days(1);
  EXPECT_EQ(1970, d.year);
  EXPECT_EQ(1, d.month);
  EXPECT_EQ(2, d.day);

  // Values around Feb 29 1968 (leap year prior to 1970)
  d.set_from_days(-673);
  EXPECT_EQ(1968, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(-672);
  EXPECT_EQ(1968, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(29, d.day);
  d.set_from_days(-671);
  EXPECT_EQ(1968, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);

  // Values areound Feb 29, 1972 (leap year after 1970)
  d.set_from_days(788);
  EXPECT_EQ(1972, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(789);
  EXPECT_EQ(1972, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(29, d.day);
  d.set_from_days(790);
  EXPECT_EQ(1972, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);

  // Values around Feb 28 1900 (special non-leap year prior to 1970)
  d.set_from_days(-25510);
  EXPECT_EQ(1900, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(27, d.day);
  d.set_from_days(-25509);
  EXPECT_EQ(1900, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(-25508);
  EXPECT_EQ(1900, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);

  // Values around Feb 29 1600 (special leap year prior to 1970)
  d.set_from_days(-135082);
  EXPECT_EQ(1600, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(-135081);
  EXPECT_EQ(1600, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(29, d.day);
  d.set_from_days(-135080);
  EXPECT_EQ(1600, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);

  // Values around Feb 29 2000 (special leap year after 1970)
  d.set_from_days(11015);
  EXPECT_EQ(2000, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(11016);
  EXPECT_EQ(2000, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(29, d.day);
  d.set_from_days(11017);
  EXPECT_EQ(2000, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);

  // Values around Feb 28 2100 (special non-leap year after 1970)
  d.set_from_days(47539);
  EXPECT_EQ(2100, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(27, d.day);
  d.set_from_days(47540);
  EXPECT_EQ(2100, d.year);
  EXPECT_EQ(2, d.month);
  EXPECT_EQ(28, d.day);
  d.set_from_days(47541);
  EXPECT_EQ(2100, d.year);
  EXPECT_EQ(3, d.month);
  EXPECT_EQ(1, d.day);
}

TEST(DateYMD, ToStr)
{
  date_ymd d;

  d.year = 2000;
  d.month = 10;
  d.day = 5;
  EXPECT_EQ("2000-10-05", d.to_str());
  d.year = 1973;
  d.month = 12;
  d.day = 26;
  EXPECT_EQ("1973-12-26", d.to_str());
  d.year = 1;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ("0001-01-01", d.to_str());
  d.year = 0;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ("+000000-12-31", d.to_str());
  d.year = 9999;
  d.month = 12;
  d.day = 31;
  EXPECT_EQ("9999-12-31", d.to_str());
  d.year = 10000;
  d.month = 1;
  d.day = 1;
  EXPECT_EQ("+010000-01-01", d.to_str());
  d.year = 25386;
  d.month = 3;
  d.day = 19;
  EXPECT_EQ("+025386-03-19", d.to_str());
  d.year = -25386;
  d.month = 3;
  d.day = 19;
  EXPECT_EQ("-025386-03-19", d.to_str());
}

TEST(DateYMD, SetFromStr)
{
  date_ymd d;

  // ISO 8601 date strings
  d.set_from_str("1925-12-30");
  EXPECT_EQ(d.year, 1925);
  EXPECT_EQ(d.month, 12);
  EXPECT_EQ(d.day, 30);
  d.set_from_str("9999-12-31");
  EXPECT_EQ(d.year, 9999);
  EXPECT_EQ(d.month, 12);
  EXPECT_EQ(d.day, 31);
  d.set_from_str(" \t  0001-01-01    ");
  EXPECT_EQ(d.year, 1);
  EXPECT_EQ(d.month, 1);
  EXPECT_EQ(d.day, 1);
  d.set_from_str("+000000-06-02");
  EXPECT_EQ(d.year, 0);
  EXPECT_EQ(d.month, 6);
  EXPECT_EQ(d.day, 2);
  d.set_from_str("-025000-03-12");
  EXPECT_EQ(d.year, -25000);
  EXPECT_EQ(d.month, 3);
  EXPECT_EQ(d.day, 12);
  d.set_from_str("+025000-03-12");
  EXPECT_EQ(d.year, 25000);
  EXPECT_EQ(d.month, 3);
  EXPECT_EQ(d.day, 12);
  d.set_from_str("20081231");
  EXPECT_EQ("2008-12-31", d.to_str());

  // year-month-day with a string month
  d.set_from_str("1993-OCT-12");
  EXPECT_EQ(d.year, 1993);
  EXPECT_EQ(d.month, 10);
  EXPECT_EQ(d.day, 12);
  d.set_from_str("1422-Jan-22");
  EXPECT_EQ(d.year, 1422);
  EXPECT_EQ(d.month, 1);
  EXPECT_EQ(d.day, 22);
  d.set_from_str("2000-feb-29");
  EXPECT_EQ(d.year, 2000);
  EXPECT_EQ(d.month, 2);
  EXPECT_EQ(d.day, 29);
  d.set_from_str("2014-december-25");
  EXPECT_EQ(d.year, 2014);
  EXPECT_EQ(d.month, 12);
  EXPECT_EQ(d.day, 25);

  // day-month-year with a string month
  d.set_from_str("12-October-1993");
  EXPECT_EQ(d.year, 1993);
  EXPECT_EQ(d.month, 10);
  EXPECT_EQ(d.day, 12);
  d.set_from_str("22-january-1422");
  EXPECT_EQ(d.year, 1422);
  EXPECT_EQ(d.month, 1);
  EXPECT_EQ(d.day, 22);
  d.set_from_str("29-Feb-2000");
  EXPECT_EQ(d.year, 2000);
  EXPECT_EQ(d.month, 2);
  EXPECT_EQ(d.day, 29);
  d.set_from_str("25-DEC-2014");
  EXPECT_EQ(d.year, 2014);
  EXPECT_EQ(d.month, 12);
  EXPECT_EQ(d.day, 25);

  // Some variations
  d.set_from_str("2012/11/13");
  EXPECT_EQ(d.year, 2012);
  EXPECT_EQ(d.month, 11);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("2015.08.13");
  EXPECT_EQ(d.year, 2015);
  EXPECT_EQ(d.month, 8);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("1990/MAY/13");
  EXPECT_EQ(d.year, 1990);
  EXPECT_EQ(d.month, 5);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("2015.Nov.13");
  EXPECT_EQ(d.year, 2015);
  EXPECT_EQ(d.month, 11);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("13/apr/1490");
  EXPECT_EQ(d.year, 1490);
  EXPECT_EQ(d.month, 4);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("13.April.2020");
  EXPECT_EQ(d.year, 2020);
  EXPECT_EQ(d.month, 4);
  EXPECT_EQ(d.day, 13);
  d.set_from_str("Jun 5, 1919");
  EXPECT_EQ(d.year, 1919);
  EXPECT_EQ(d.month, 6);
  EXPECT_EQ(d.day, 5);
  d.set_from_str("Fri, Jun 6, 1919");
  EXPECT_EQ(d.year, 1919);
  EXPECT_EQ(d.month, 6);
  EXPECT_EQ(d.day, 6);
  d.set_from_str("Saturday, Jun 7, 1919");
  EXPECT_EQ(d.year, 1919);
  EXPECT_EQ(d.month, 6);
  EXPECT_EQ(d.day, 7);
  d.set_from_str("  June  05,\t1919 ");
  EXPECT_EQ(d.year, 1919);
  EXPECT_EQ(d.month, 6);
  EXPECT_EQ(d.day, 5);
  d.set_from_str("  jul  16,\t3022 ");
  EXPECT_EQ("3022-07-16", d.to_str());
  d.set_from_str("Monday, 01-Aug-2011");
  EXPECT_EQ("2011-08-01", d.to_str());
  d.set_from_str("1982-2-20");
  EXPECT_EQ("1982-02-20", d.to_str());
  d.set_from_str("1982-2-3");
  EXPECT_EQ("1982-02-03", d.to_str());
  d.set_from_str("29Apr2002");
  EXPECT_EQ("2002-04-29", d.to_str());
  d.set_from_str("01SEP1990");
  EXPECT_EQ("1990-09-01", d.to_str());
  d.set_from_str("Jan. 2, 1960");
  EXPECT_EQ("1960-01-02", d.to_str());
  d.set_from_str("Jun 1,2008");
  EXPECT_EQ("2008-06-01", d.to_str());

  // Parsing a datetime with the time == midnight is ok too, and
  // ignores the time zone
  d.set_from_str("1918-05-23T00");
  EXPECT_EQ("1918-05-23", d.to_str());
  d.set_from_str("1919-06-24T00Z");
  EXPECT_EQ("1919-06-24", d.to_str());
  d.set_from_str("1920-07-25T00:00");
  EXPECT_EQ("1920-07-25", d.to_str());
  d.set_from_str("1921-08-26T00:00+05");
  EXPECT_EQ("1921-08-26", d.to_str());
  d.set_from_str("1922-09-27T00:00:00");
  EXPECT_EQ("1922-09-27", d.to_str());
  d.set_from_str("1923-10-28 00:00:00-0600");
  EXPECT_EQ("1923-10-28", d.to_str());
  d.set_from_str("1924-11-29T00:00:00.0");
  EXPECT_EQ("1924-11-29", d.to_str());
  d.set_from_str("1925-12-30T00:00:00.00+10:30");
  EXPECT_EQ("1925-12-30", d.to_str());

  // RFC2822 date syntax
  d.set_from_str("Mon, 25 Dec 1995 00:00:00 GMT");
  EXPECT_EQ("1995-12-25", d.to_str());
  d.set_from_str("Tue, 26 Dec 1995 00:00:00 +0430");
  EXPECT_EQ("1995-12-26", d.to_str());

  // Ambiguous formats with MDY order
  d.set_from_str("01-02-2003", date_parse_mdy);
  EXPECT_EQ("2003-01-02", d.to_str());
  d.set_from_str("2/3/2004", date_parse_mdy);
  EXPECT_EQ("2004-02-03", d.to_str());

  // Ambiguous formats with DMY order
  d.set_from_str("01-02-2004", date_parse_dmy);
  EXPECT_EQ("2004-02-01", d.to_str());
  d.set_from_str("2-3-2005", date_parse_dmy);
  EXPECT_EQ("2005-03-02", d.to_str());

  // Two digit years, resolved with a sliding window starting 70 years ago
  d.set_from_str("01-02-03", date_parse_mdy);
  EXPECT_EQ("2003-01-02", d.to_str());
  d.set_from_str("01-02-03", date_parse_dmy);
  EXPECT_EQ("2003-02-01", d.to_str());
  d.set_from_str("01-02-03", date_parse_ymd);
  EXPECT_EQ("2001-02-03", d.to_str());
  d.set_from_str("01-02-99", date_parse_mdy);
  EXPECT_EQ("1999-01-02", d.to_str());
  d.set_from_str("01/dec/99", date_parse_mdy);
  EXPECT_EQ("1999-12-01", d.to_str());
  d.set_from_str("2-MAR-03", date_parse_mdy);
  EXPECT_EQ("2003-03-02", d.to_str());
  d.set_from_str("02-MAR-4", date_parse_ymd);
  EXPECT_EQ("2002-03-04", d.to_str());
  d.set_from_str("January 10, 98");
  EXPECT_EQ("1998-01-10", d.to_str());
  d.set_from_str("12.4.03", date_parse_dmy);
  EXPECT_EQ("2003-04-12", d.to_str());
}

TEST(DateYMD, SetFromStr_Errors)
{
  date_ymd d;

  EXPECT_THROW(d.set_from_str("123-01-01"), invalid_argument);
  EXPECT_THROW(d.set_from_str("2000-01-01X"), invalid_argument);
  EXPECT_THROW(d.set_from_str("2000-02-30"), invalid_argument);
  EXPECT_THROW(d.set_from_str("2001-02-29"), invalid_argument);
  EXPECT_THROW(d.set_from_str("2012-01/01"), invalid_argument);
  EXPECT_THROW(d.set_from_str("2012-rec-01"), invalid_argument);
  EXPECT_THROW(d.set_from_str("Banuary 5, 1992"), invalid_argument);
  // Ambiguous formats
  EXPECT_THROW(d.set_from_str("01-02-03"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01/02/03"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01.02.03"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01-02-2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01/02/2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01.02.2003"), invalid_argument);
  // Rejecting the ambiguous format, even if the value can resolve the date
  EXPECT_THROW(d.set_from_str("01-14-2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01/14/2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("01.14.2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("14-02-2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("14/02/2003"), invalid_argument);
  EXPECT_THROW(d.set_from_str("14.02.2003"), invalid_argument);
  // Rejecting two digit years if set to disallow
  EXPECT_THROW(d.set_from_str("01-02-99", date_parse_mdy, 0), invalid_argument);
  EXPECT_THROW(d.set_from_str("01-02-99", date_parse_dmy, 0), invalid_argument);
  EXPECT_THROW(d.set_from_str("99-02-01", date_parse_ymd, 0), invalid_argument);
  EXPECT_THROW(d.set_from_str("01-Feb-99", date_parse_dmy, 0),
               invalid_argument);
}

TEST(DateYMD, TwoDigitYear_FixedWindow)
{
  EXPECT_EQ(1944, date_ymd::resolve_2digit_year_fixed_window(44, 1929));
  EXPECT_EQ(1944, date_ymd::resolve_2digit_year(44, 1929));
  EXPECT_EQ(2000, date_ymd::resolve_2digit_year_fixed_window(0, 1929));
  EXPECT_EQ(2000, date_ymd::resolve_2digit_year(0, 1929));

  // Just before/after the window boundary
  EXPECT_EQ(1929, date_ymd::resolve_2digit_year_fixed_window(29, 1929));
  EXPECT_EQ(1929, date_ymd::resolve_2digit_year(29, 1929));
  EXPECT_EQ(2028, date_ymd::resolve_2digit_year_fixed_window(28, 1929));
  EXPECT_EQ(2028, date_ymd::resolve_2digit_year(28, 1929));

  // End/beginning of the century
  EXPECT_EQ(1999, date_ymd::resolve_2digit_year_fixed_window(99, 1929));
  EXPECT_EQ(1999, date_ymd::resolve_2digit_year(99, 1929));
  EXPECT_EQ(2000, date_ymd::resolve_2digit_year_fixed_window(0, 1929));
  EXPECT_EQ(2000, date_ymd::resolve_2digit_year(0, 1929));

  // Window starting on a century boundary
  EXPECT_EQ(1200, date_ymd::resolve_2digit_year_fixed_window(0, 1200));
  EXPECT_EQ(1200, date_ymd::resolve_2digit_year(0, 1200));
  EXPECT_EQ(1299, date_ymd::resolve_2digit_year_fixed_window(99, 1200));
  EXPECT_EQ(1299, date_ymd::resolve_2digit_year(99, 1200));
}

TEST(DateYMD, TwoDigitYear_SlidingWindow)
{
  // NOTE: These tests should start to fail in the year 2030,
  //       and will need to be updated then.

  // Sliding window starting 70 years ago
  EXPECT_EQ(1960, date_ymd::resolve_2digit_year_sliding_window(60, 70));
  EXPECT_EQ(1960, date_ymd::resolve_2digit_year(60, 70));
  EXPECT_EQ(1999, date_ymd::resolve_2digit_year_sliding_window(99, 70));
  EXPECT_EQ(1999, date_ymd::resolve_2digit_year(99, 70));
  EXPECT_EQ(2043, date_ymd::resolve_2digit_year_sliding_window(43, 70));
  EXPECT_EQ(2043, date_ymd::resolve_2digit_year(43, 70));

  // Sliding window starting 20 years ago
  EXPECT_EQ(2010, date_ymd::resolve_2digit_year_sliding_window(10, 20));
  EXPECT_EQ(2010, date_ymd::resolve_2digit_year(10, 20));
  EXPECT_EQ(2050, date_ymd::resolve_2digit_year_sliding_window(50, 20));
  EXPECT_EQ(2050, date_ymd::resolve_2digit_year(50, 20));
  EXPECT_EQ(2093, date_ymd::resolve_2digit_year_sliding_window(93, 20));
  EXPECT_EQ(2093, date_ymd::resolve_2digit_year(93, 20));
}
