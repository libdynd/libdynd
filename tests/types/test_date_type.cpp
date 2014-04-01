//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/call_callable.hpp>

using namespace std;
using namespace dynd;

TEST(DateDType, Create) {
    ndt::type d;

    d = ndt::make_date();
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_data_alignment());
    EXPECT_EQ(ndt::make_date(), ndt::make_date());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DateDType, ValueCreation) {
    ndt::type d = ndt::make_date(), di = ndt::make_type<int32_t>();

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 - 365,
                    nd::array("1599-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3,
                    nd::array("1600-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 366,
                    nd::array("1601-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4,
                    nd::array("1900-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4 + 365,
                    nd::array("1901-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ(-3*365 - 1,
                    nd::array("1967-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-2*365 - 1,
                    nd::array("1968-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-1*365,
                    nd::array("1969-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(0*365,
                    nd::array("1970-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(1*365,
                    nd::array("1971-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(2*365,
                    nd::array("1972-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(3*365 + 1,
                    nd::array("1973-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(4*365 + 1,
                    nd::array("1974-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4,
                    nd::array("2000-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366,
                    nd::array("2001-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3,
                    nd::array("2400-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3 + 366,
                    nd::array("2401-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 28,
                    nd::array("1600-02-29").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 29,
                    nd::array("1600-03-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 28,
                    nd::array("2000-02-29").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 29,
                    nd::array("2000-03-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366 + 31 + 28 + 21,
                    nd::array("2001-03-22").ucast(d).view_scalars(di).as<int32_t>());

    // Unambiguous date format which has extra empty time attached
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366 + 31 + 28 + 21,
                    nd::array("2001-03-22 00:00:00").ucast(d).view_scalars(di).as<int32_t>());
}

TEST(DateDType, BadInputStrings) {
    ndt::type d = ndt::make_date();

    // Arbitrary bad string
    EXPECT_THROW(nd::array("badvalue").ucast(d).eval(), invalid_argument);
    // Character after year must be '-'
    EXPECT_THROW(nd::array("1980X").ucast(d).eval(), invalid_argument);
    // Cannot have trailing '-'
    EXPECT_THROW(nd::array("1980-").ucast(d).eval(), invalid_argument);
    // Month must be in range [1,12]
    EXPECT_THROW(nd::array("1980-00").ucast(d).eval(), invalid_argument);
    EXPECT_THROW(nd::array("1980-13").ucast(d).eval(), invalid_argument);
    // Month must have two digits
    EXPECT_THROW(nd::array("1980-1").ucast(d).eval(), invalid_argument);
    EXPECT_THROW(nd::array("1980-1-02").ucast(d).eval(), invalid_argument);
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

TEST(DateDType, DateProperties) {
    ndt::type d = ndt::make_date();
    nd::array a;

    a = nd::array("1955-03-13").ucast(d).eval();
    EXPECT_EQ(ndt::make_property(d, "year"), a.p("year").get_type());
    EXPECT_EQ(ndt::make_property(d, "month"), a.p("month").get_type());
    EXPECT_EQ(ndt::make_property(d, "day"), a.p("day").get_type());
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

TEST(DateDType, DatePropertyConvertOfString) {
    nd::array a, b, c;
    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = nd::array(strs).ucast(ndt::make_fixedstring(10, string_encoding_ascii)).eval();
    b = a.ucast(ndt::make_date());
    EXPECT_EQ(ndt::make_strided_dim(
                    ndt::make_fixedstring(10, string_encoding_ascii)),
                    a.get_type());
    EXPECT_EQ(ndt::make_strided_dim(
                    ndt::make_convert(ndt::make_date(),
                        ndt::make_fixedstring(10, string_encoding_ascii))),
                    b.get_type());

    // year property
    c = b.p("year");
    EXPECT_EQ(property_type_id, c.get_dtype().get_type_id());
    c = c.eval();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int>()), c.get_type());
    EXPECT_EQ(1931, c(0).as<int>());
    EXPECT_EQ(2013, c(1).as<int>());
    EXPECT_EQ(2012, c(2).as<int>());

    // weekday function
    c = b.f("weekday");
    EXPECT_EQ(property_type_id, c.get_dtype().get_type_id());
    c = c.eval();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int>()), c.get_type());
    EXPECT_EQ(5, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(1, c(2).as<int>());
}

TEST(DateDType, ToStructFunction) {
    ndt::type d = ndt::make_date();
    nd::array a, b;

    a = nd::array("1955-03-13").ucast(d).eval();
    b = a.f("to_struct");
    EXPECT_EQ(ndt::make_property(d, "struct"),
                    b.get_type());
    b = b.eval();
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<int16_t>(), "year",
                        ndt::make_type<int8_t>(), "month",
                        ndt::make_type<int8_t>(), "day"),
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

TEST(DateDType, ToStruct) {
    ndt::type d = ndt::make_date(), ds;
    nd::array a, b;

    a = nd::array("1955-03-13").ucast(d).eval();

    // This is the default struct produced
    ds = ndt::make_cstruct(ndt::make_type<int32_t>(), "year", ndt::make_type<int8_t>(), "month", ndt::make_type<int8_t>(), "day");
    b = nd::empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b(0).as<int32_t>());
    EXPECT_EQ(3, b(1).as<int8_t>());
    EXPECT_EQ(13, b(2).as<int8_t>());

    // This should work too
    ds = ndt::make_cstruct(ndt::make_type<int16_t>(), "month", ndt::make_type<int16_t>(), "year", ndt::make_type<float>(), "day");
    b = nd::empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b(1).as<int16_t>());
    EXPECT_EQ(3, b(0).as<int16_t>());
    EXPECT_EQ(13, b(2).as<float>());

    // This should work too
    ds = ndt::make_struct(ndt::make_type<int16_t>(), "month", ndt::make_type<int16_t>(), "year", ndt::make_type<float>(), "day");
    b = nd::empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b(1).as<int16_t>());
    EXPECT_EQ(3, b(0).as<int16_t>());
    EXPECT_EQ(13, b(2).as<float>());
}

TEST(DateDType, FromStruct) {
    ndt::type d = ndt::make_date(), ds;
    nd::array a, b;

    // This is the default struct accepted
    ds = ndt::make_cstruct(ndt::make_type<int32_t>(), "year", ndt::make_type<int8_t>(), "month", ndt::make_type<int8_t>(), "day");
    a = nd::empty(ds);
    a(0).vals() = 1955;
    a(1).vals() = 3;
    a(2).vals() = 13;
    b = nd::empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = ndt::make_cstruct(ndt::make_type<int16_t>(), "month", ndt::make_type<int16_t>(), "year", ndt::make_type<float>(), "day");
    a = nd::empty(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = nd::empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = ndt::make_struct(ndt::make_type<int16_t>(), "month", ndt::make_type<int16_t>(), "year", ndt::make_type<float>(), "day");
    a = nd::empty(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = nd::empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());
}

TEST(DateDType, StrFTime) {
    ndt::type d = ndt::make_date(), ds;
    nd::array a, b;

    a = nd::array("1955-03-13").ucast(d).eval();

    b = a.f("strftime", "%Y");
    EXPECT_EQ("1955", b.as<string>());
    b = a.f("strftime", "%m/%d/%y");
    EXPECT_EQ("03/13/55", b.as<string>());
    b = a.f("strftime", "%Y and %j");
    EXPECT_EQ("1955 and 072", b.as<string>());

    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = nd::array(strs).ucast(d).eval();

    b = a.f("strftime", "%Y-%m-%d %j %U %w %W");
    EXPECT_EQ("1931-12-12 346 49 6 49", b(0).as<string>());
    EXPECT_EQ("2013-05-14 134 19 2 19", b(1).as<string>());
    EXPECT_EQ("2012-12-25 360 52 2 52", b(2).as<string>());
}

TEST(DateDType, StrFTimeOfConvert) {
    // First create a date array which is still a convert expression type
    const char *vals[] = {"1920-03-12", "2013-01-01", "2000-12-25"};
    nd::array a = nd::array(vals).ucast(ndt::make_date());
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_convert(ndt::make_date(), ndt::make_string())),
                    a.get_type());

    nd::array b = a.f("strftime", "%Y %m %d");
    EXPECT_EQ("1920 03 12", b(0).as<string>());
    EXPECT_EQ("2013 01 01", b(1).as<string>());
    EXPECT_EQ("2000 12 25", b(2).as<string>());
}

TEST(DateDType, StrFTimeOfMultiDim) {
    const char *vals_0[] = {"1920-03-12", "2013-01-01"};
    const char *vals_1[] = {"2000-12-25"};
    nd::array a = nd::make_strided_array(2, -1, ndt::make_date());
    a.vals_at(0) = vals_0;
    a.vals_at(1) = vals_1;

    a = a.f("strftime", "%d/%m/%Y");
    EXPECT_EQ("12/03/1920", a(0, 0).as<string>());
    EXPECT_EQ("01/01/2013", a(0, 1).as<string>());
    EXPECT_EQ("25/12/2000", a(1, 0).as<string>());
}

#if defined(_MSC_VER)
// Only the Windows strftime seems to support this behavior without
// writing our own strftime format parser.
TEST(DateDType, StrFTimeBadFormat) {
    ndt::type d = ndt::make_date();
    nd::array a;

    a = nd::array("1955-03-13").ucast(d).eval();
    // Invalid format string should raise an error.
    EXPECT_THROW(a.f("strftime", "%Y %x %s").eval(), runtime_error);
}
#endif

TEST(DateDType, WeekDay) {
    ndt::type d = ndt::make_date();
    nd::array a;

    a = nd::array("1955-03-13").ucast(d).eval();
    EXPECT_EQ(6, a.f("weekday").as<int32_t>());
    a = nd::array("2002-12-04").ucast(d).eval();
    EXPECT_EQ(2, a.f("weekday").as<int32_t>());
}

TEST(DateDType, Replace) {
    ndt::type d = ndt::make_date();
    nd::array a;

    a = nd::array("1955-03-13").ucast(d).eval();
    EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<string>());
    EXPECT_EQ("2012-12-13", a.f("replace", 2012, 12).as<string>());
    EXPECT_EQ("2012-12-15", a.f("replace", 2012, 12, 15).as<string>());
    // Custom extension, allow -1 indexing from the end for months and days
    EXPECT_EQ("2012-12-30", a.f("replace", 2012, -1, 30).as<string>());
    EXPECT_EQ("2012-05-31", a.f("replace", 2012, -8, -1).as<string>());
    // The C++ call interface doesn't let you skip arguments (yet, there is no keyword argument mechanism),
    // so test this manually
    nd::array param = a.find_dynamic_function("replace").get_default_parameters().eval_copy(nd::readwrite_access_flags);
    *reinterpret_cast<void **>(param(0).get_readwrite_originptr()) = (void*)a.get_ndo();
    param(2).vals() = 7;
    EXPECT_EQ("1955-07-13", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param(3).vals() = -1;
    EXPECT_EQ("1955-07-31", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param(2).vals() = 2;
    EXPECT_EQ("1955-02-28", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param(1).vals() = 2012;
    EXPECT_EQ("2012-02-29", a.find_dynamic_function("replace").call_generic(param).as<string>());
    // Should throw an exception when no arguments or out of bounds arguments are provided
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

TEST(DateDType, ReplaceOfConvert) {
    nd::array a;

    // Make an expression type with value type 'date'
    a = nd::array("1955-03-13").ucast(ndt::make_date());
    EXPECT_EQ(ndt::make_convert(ndt::make_date(), ndt::make_string()),
                    a.get_type());
    // Call replace on it
    EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<string>());
}

TEST(DateDType, NumPyCompatibleProperty) {
    int64_t vals64[] = {-16730, 0, 11001, numeric_limits<int64_t>::min()};

    nd::array a = vals64;
    nd::array a_date = a.view_scalars(ndt::make_reversed_property(ndt::make_date(),
                    ndt::make_type<int64_t>(), "days_after_1970_int64"));
    // Reading from the 'int64 as date' view
    EXPECT_EQ("1924-03-13", a_date(0).as<string>());
    EXPECT_EQ("1970-01-01", a_date(1).as<string>());
    EXPECT_EQ("2000-02-14", a_date(2).as<string>());
    EXPECT_EQ("NA",         a_date(3).as<string>());

    // Writing to the 'int64 as date' view
    a_date(0).vals() = "1975-01-30";
    EXPECT_EQ(1855, a(0).as<int64_t>());
    a_date(0).vals() = "NA";
    EXPECT_EQ(numeric_limits<int64_t>::min(), a(0).as<int64_t>());
}

TEST(DateYMD, LeapYear) {
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

TEST(DateYMD, ToDays) {
    date_ymd d;

    // Some extreme edge cases
    d.year = -29200; d.month = 2; d.day = 29;
    EXPECT_EQ(-11384550, d.to_days());
    d.year = 32000; d.month = 1; d.day = 31;
    EXPECT_EQ(10968262, d.to_days());
    d.year = -1234; d.month = 1; d.day = 1;
    EXPECT_EQ(-1170237, d.to_days());
    d.year = -1; d.month = 12; d.day = 31;
    EXPECT_EQ(-719529, d.to_days());
    d.year = 0; d.month = 1; d.day = 1;
    EXPECT_EQ(-719528, d.to_days());
    d.year = 0; d.month = 12; d.day = 31;
    EXPECT_EQ(-719163, d.to_days());
    d.year = 1; d.month = 1; d.day = 1;
    EXPECT_EQ(-719162, d.to_days());

    // Values around the 1970 epoch
    d.year = 1969; d.month = 12; d.day = 31;
    EXPECT_EQ(-1, d.to_days());
    d.year = 1970; d.month = 1; d.day = 1;
    EXPECT_EQ(0, d.to_days());
    d.year = 1970; d.month = 1; d.day = 2;
    EXPECT_EQ(1, d.to_days());

    // Values around Feb 29 1968 (leap year prior to 1970)
    d.year = 1968; d.month = 2; d.day = 28;
    EXPECT_EQ(-673, d.to_days());
    d.year = 1968; d.month = 2; d.day = 29;
    EXPECT_EQ(-672, d.to_days());
    d.year = 1968; d.month = 3; d.day = 1;
    EXPECT_EQ(-671, d.to_days());

    // Values areound Feb 29, 1972 (leap year after 1970)
    d.year = 1972; d.month = 2; d.day = 28;
    EXPECT_EQ(788, d.to_days());
    d.year = 1972; d.month = 2; d.day = 29;
    EXPECT_EQ(789, d.to_days());
    d.year = 1972; d.month = 3; d.day = 1;
    EXPECT_EQ(790, d.to_days());

    // Values around Feb 28 1900 (special non-leap year prior to 1970)
    d.year = 1900; d.month = 2; d.day = 27;
    EXPECT_EQ(-25510, d.to_days());
    d.year = 1900; d.month = 2; d.day = 28;
    EXPECT_EQ(-25509, d.to_days());
    d.year = 1900; d.month = 3; d.day = 1;
    EXPECT_EQ(-25508, d.to_days());

    // Values around Feb 29 1600 (special leap year prior to 1970)
    d.year = 1600; d.month = 2; d.day = 28;
    EXPECT_EQ(-135082, d.to_days());
    d.year = 1600; d.month = 2; d.day = 29;
    EXPECT_EQ(-135081, d.to_days());
    d.year = 1600; d.month = 3; d.day = 1;
    EXPECT_EQ(-135080, d.to_days());

    // Values around Feb 29 2000 (special leap year after 1970)
    d.year = 2000; d.month = 2; d.day = 28;
    EXPECT_EQ(11015, d.to_days());
    d.year = 2000; d.month = 2; d.day = 29;
    EXPECT_EQ(11016, d.to_days());
    d.year = 2000; d.month = 3; d.day = 1;
    EXPECT_EQ(11017, d.to_days());

    // Values around Feb 28 2100 (special non-leap year after 1970)
    d.year = 2100; d.month = 2; d.day = 27;
    EXPECT_EQ(47539, d.to_days());
    d.year = 2100; d.month = 2; d.day = 28;
    EXPECT_EQ(47540, d.to_days());
    d.year = 2100; d.month = 3; d.day = 1;
    EXPECT_EQ(47541, d.to_days());

    // The last of every month of a non-leap year
    d.year = 1979; d.month = 1; d.day = 31;
    EXPECT_EQ(3317, d.to_days());
    d.year = 1979; d.month = 2; d.day = 28;
    EXPECT_EQ(3345, d.to_days());
    d.year = 1979; d.month = 3; d.day = 31;
    EXPECT_EQ(3376, d.to_days());
    d.year = 1979; d.month = 4; d.day = 30;
    EXPECT_EQ(3406, d.to_days());
    d.year = 1979; d.month = 5; d.day = 31;
    EXPECT_EQ(3437, d.to_days());
    d.year = 1979; d.month = 6; d.day = 30;
    EXPECT_EQ(3467, d.to_days());
    d.year = 1979; d.month = 7; d.day = 31;
    EXPECT_EQ(3498, d.to_days());
    d.year = 1979; d.month = 8; d.day = 31;
    EXPECT_EQ(3529, d.to_days());
    d.year = 1979; d.month = 9; d.day = 30;
    EXPECT_EQ(3559, d.to_days());
    d.year = 1979; d.month = 10; d.day = 31;
    EXPECT_EQ(3590, d.to_days());
    d.year = 1979; d.month = 11; d.day = 30;
    EXPECT_EQ(3620, d.to_days());
    d.year = 1979; d.month = 12; d.day = 31;
    EXPECT_EQ(3651, d.to_days());
}

TEST(DateYMD, SetFromDays) {
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

TEST(DateYMD, ToStr) {
    date_ymd d;

    d.year = 2000; d.month = 10; d.day = 5;
    EXPECT_EQ("2000-10-05", d.to_str());
    d.year = 1973; d.month = 12; d.day = 26;
    EXPECT_EQ("1973-12-26", d.to_str());
    d.year = 1; d.month = 1; d.day = 1;
    EXPECT_EQ("0001-01-01", d.to_str());
    d.year = 0; d.month = 12; d.day = 31;
    EXPECT_EQ("+000000-12-31", d.to_str());
    d.year = 9999; d.month = 12; d.day = 31;
    EXPECT_EQ("9999-12-31", d.to_str());
    d.year = 10000; d.month = 1; d.day = 1;
    EXPECT_EQ("+010000-01-01", d.to_str());
    d.year = 25386; d.month = 3; d.day = 19;
    EXPECT_EQ("+025386-03-19", d.to_str());
    d.year = -25386; d.month = 3; d.day = 19;
    EXPECT_EQ("-025386-03-19", d.to_str());
}

TEST(DateYMD, SetFromStr) {
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
    EXPECT_EQ(d.year, 3022);
    EXPECT_EQ(d.month, 7);
    EXPECT_EQ(d.day, 16);
    d.set_from_str("Monday, 01-Aug-2011");
    EXPECT_EQ(d.year, 2011);
    EXPECT_EQ(d.month, 8);
    EXPECT_EQ(d.day, 1);

    // Parsing a datetime with the time == midnight is ok too, and
    // ignores the time zone
    d.set_from_str("1925-12-30T00");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00Z");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00:00");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00:00+05");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00:00:00");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30 00:00:00-0600");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00:00:00.0");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);
    d.set_from_str("1925-12-30T00:00:00.00+10:30");
    EXPECT_EQ(d.year, 1925);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 30);

    // RFC2822 date syntax
    d.set_from_str( "Mon, 25 Dec 1995 00:00:00 GMT");
    EXPECT_EQ(d.year, 1995);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 25);
    d.set_from_str( "Tue, 26 Dec 1995 00:00:00 +0430");
    EXPECT_EQ(d.year, 1995);
    EXPECT_EQ(d.month, 12);
    EXPECT_EQ(d.day, 26);
}

TEST(DateYMD, SetFromStr_Errors) {
    date_ymd d;
    
    EXPECT_THROW(d.set_from_str("123-01-01"), invalid_argument);
    EXPECT_THROW(d.set_from_str("2000-01-01X"), invalid_argument);
    EXPECT_THROW(d.set_from_str("2000-02-30"), invalid_argument);
    EXPECT_THROW(d.set_from_str("2001-02-29"), invalid_argument);
    EXPECT_THROW(d.set_from_str("2012-01/01"), invalid_argument);
    EXPECT_THROW(d.set_from_str("2012-rec-01"), invalid_argument);
    EXPECT_THROW(d.set_from_str("Banuary 5, 1992"), invalid_argument);
}
