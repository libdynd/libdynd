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
    EXPECT_THROW(nd::array("badvalue").ucast(d).eval(), runtime_error);
    // Character after year must be '-'
    EXPECT_THROW(nd::array("1980X").ucast(d).eval(), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(nd::array("1980-").ucast(d).eval(), runtime_error);
    // Month must be in range [1,12]
    EXPECT_THROW(nd::array("1980-00").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-13").ucast(d).eval(), runtime_error);
    // Month must have two digits
    EXPECT_THROW(nd::array("1980-1").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-1-02").ucast(d).eval(), runtime_error);
    // 'Mor' is not a valid month
    EXPECT_THROW(nd::array("1980-Mor").ucast(d).eval(), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(nd::array("1980-01-").ucast(d).eval(), runtime_error);
    // Day must be in range [1,len(month)]
    EXPECT_THROW(nd::array("1980-01-0").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-01-00").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-01-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1979-02-29").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-02-30").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-03-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-04-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-05-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-06-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-07-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-08-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-09-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-10-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-11-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-12-32").ucast(d).eval(), runtime_error);
    // Cannot have trailing characters
    EXPECT_THROW(nd::array("1980-02-03%").ucast(d).eval(), runtime_error);
    EXPECT_THROW(nd::array("1980-02-03 q").ucast(d).eval(), runtime_error);
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
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<int32_t>(), "year",
                        ndt::make_type<int16_t>(), "month",
                        ndt::make_type<int16_t>(), "day"),
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
