//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/cstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/call_callable.hpp>

using namespace std;
using namespace dynd;

TEST(DateDType, Create) {
    dtype d;

    d = make_date_dtype();
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_alignment());
    EXPECT_EQ(make_date_dtype(), make_date_dtype());
}

TEST(DateDType, ValueCreation) {
    dtype d = make_date_dtype(), di = make_dtype<int32_t>();

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 - 365,
                    ndobject("1599-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3,
                    ndobject("1600-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 366,
                    ndobject("1601-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4,
                    ndobject("1900-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4 + 365,
                    ndobject("1901-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ(-3*365 - 1,
                    ndobject("1967-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-2*365 - 1,
                    ndobject("1968-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-1*365,
                    ndobject("1969-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(0*365,
                    ndobject("1970-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(1*365,
                    ndobject("1971-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(2*365,
                    ndobject("1972-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(3*365 + 1,
                    ndobject("1973-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(4*365 + 1,
                    ndobject("1974-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4,
                    ndobject("2000-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366,
                    ndobject("2001-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3,
                    ndobject("2400-01-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3 + 366,
                    ndobject("2401-01-01").ucast(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 28,
                    ndobject("1600-02-29").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 29,
                    ndobject("1600-03-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 28,
                    ndobject("2000-02-29").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 29,
                    ndobject("2000-03-01").ucast(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366 + 31 + 28 + 21,
                    ndobject("2001-03-22").ucast(d).view_scalars(di).as<int32_t>());
}

TEST(DateDType, BadInputStrings) {
    dtype d = make_date_dtype();

    // Arbitrary bad string
    EXPECT_THROW(ndobject("badvalue").ucast(d).eval(), runtime_error);
    // Character after year must be '-'
    EXPECT_THROW(ndobject("1980X").ucast(d).eval(), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(ndobject("1980-").ucast(d).eval(), runtime_error);
    // Month must be in range [1,12]
    EXPECT_THROW(ndobject("1980-00").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-13").ucast(d).eval(), runtime_error);
    // Month must have two digits
    EXPECT_THROW(ndobject("1980-1").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-1-02").ucast(d).eval(), runtime_error);
    // 'Mor' is not a valid month
    EXPECT_THROW(ndobject("1980-Mor").ucast(d).eval(), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(ndobject("1980-01-").ucast(d).eval(), runtime_error);
    // Day must be in range [1,len(month)]
    EXPECT_THROW(ndobject("1980-01-0").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-01-00").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-01-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1979-02-29").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-02-30").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-03-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-04-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-05-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-06-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-07-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-08-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-09-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-10-32").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-11-31").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-12-32").ucast(d).eval(), runtime_error);
    // Cannot have trailing characters
    EXPECT_THROW(ndobject("1980-02-03%").ucast(d).eval(), runtime_error);
    EXPECT_THROW(ndobject("1980-02-03 q").ucast(d).eval(), runtime_error);
}

TEST(DateDType, DateProperties) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").ucast(d).eval();
    EXPECT_EQ(make_property_dtype(d, "year"), a.p("year").get_dtype());
    EXPECT_EQ(make_property_dtype(d, "month"), a.p("month").get_dtype());
    EXPECT_EQ(make_property_dtype(d, "day"), a.p("day").get_dtype());
    EXPECT_EQ(1955, a.p("year").as<int32_t>());
    EXPECT_EQ(3, a.p("month").as<int32_t>());
    EXPECT_EQ(13, a.p("day").as<int32_t>());

    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = ndobject(strs).ucast(d).eval();
    EXPECT_EQ(1931, a.p("year").at(0).as<int32_t>());
    EXPECT_EQ(12, a.p("month").at(0).as<int32_t>());
    EXPECT_EQ(12, a.p("day").at(0).as<int32_t>());
    EXPECT_EQ(2013, a.p("year").at(1).as<int32_t>());
    EXPECT_EQ(5, a.p("month").at(1).as<int32_t>());
    EXPECT_EQ(14, a.p("day").at(1).as<int32_t>());
    EXPECT_EQ(2012, a.p("year").at(2).as<int32_t>());
    EXPECT_EQ(12, a.p("month").at(2).as<int32_t>());
    EXPECT_EQ(25, a.p("day").at(2).as<int32_t>());
}

TEST(DateDType, DatePropertyConvertOfString) {
    ndobject a, b, c;
    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = ndobject(strs).ucast(make_fixedstring_dtype(10, string_encoding_ascii)).eval();
    b = a.ucast(make_date_dtype());
    EXPECT_EQ(make_strided_dim_dtype(
                    make_fixedstring_dtype(10, string_encoding_ascii)),
                    a.get_dtype());
    EXPECT_EQ(make_strided_dim_dtype(
                    make_convert_dtype(make_date_dtype(),
                        make_fixedstring_dtype(10, string_encoding_ascii))),
                    b.get_dtype());

    // year property
    c = b.p("year");
    EXPECT_EQ(property_type_id, c.get_udtype().get_type_id());
    c = c.eval();
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), c.get_dtype());
    EXPECT_EQ(1931, c.at(0).as<int>());
    EXPECT_EQ(2013, c.at(1).as<int>());
    EXPECT_EQ(2012, c.at(2).as<int>());

    // weekday function
    c = b.f("weekday");
    EXPECT_EQ(property_type_id, c.get_udtype().get_type_id());
    c = c.eval();
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), c.get_dtype());
    EXPECT_EQ(5, c.at(0).as<int>());
    EXPECT_EQ(1, c.at(1).as<int>());
    EXPECT_EQ(1, c.at(2).as<int>());
}

TEST(DateDType, ToStructFunction) {
    dtype d = make_date_dtype();
    ndobject a, b;

    a = ndobject("1955-03-13").ucast(d).eval();
    b = a.f("to_struct");
    EXPECT_EQ(make_property_dtype(d, "struct"),
                    b.get_dtype());
    b = b.eval();
    EXPECT_EQ(make_cstruct_dtype(make_dtype<int32_t>(), "year",
                        make_dtype<int16_t>(), "month",
                        make_dtype<int16_t>(), "day"),
                    b.get_dtype());
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3, b.p("month").as<int32_t>());
    EXPECT_EQ(13, b.p("day").as<int32_t>());

    // Do it again, but now with a chain of expressions
    a = ndobject("1955-03-13").ucast(d).f("to_struct");
    EXPECT_EQ(1955, a.p("year").as<int32_t>());
    EXPECT_EQ(3, a.p("month").as<int32_t>());
    EXPECT_EQ(13, a.p("day").as<int32_t>());
}

TEST(DateDType, ToStruct) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    a = ndobject("1955-03-13").ucast(d).eval();

    // This is the default struct produced
    ds = make_cstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day");
    b = empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b.at(0).as<int32_t>());
    EXPECT_EQ(3, b.at(1).as<int8_t>());
    EXPECT_EQ(13, b.at(2).as<int8_t>());

    // This should work too
    ds = make_cstruct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    b = empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());
    EXPECT_EQ(13, b.at(2).as<float>());

    // This should work too
    ds = make_struct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    b = empty(ds);
    b.vals() = a;
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());
    EXPECT_EQ(13, b.at(2).as<float>());
}

TEST(DateDType, FromStruct) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    // This is the default struct accepted
    ds = make_cstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day");
    a = empty(ds);
    a.at(0).vals() = 1955;
    a.at(1).vals() = 3;
    a.at(2).vals() = 13;
    b = empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = make_cstruct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    a = empty(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = make_struct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    a = empty(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = empty(d);
    b.vals() = a;
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());
}

TEST(DateDType, StrFTime) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    a = ndobject("1955-03-13").ucast(d).eval();

    b = a.f("strftime", "%Y");
    EXPECT_EQ("1955", b.as<string>());
    b = a.f("strftime", "%m/%d/%y");
    EXPECT_EQ("03/13/55", b.as<string>());
    b = a.f("strftime", "%Y and %j");
    EXPECT_EQ("1955 and 072", b.as<string>());

    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = ndobject(strs).ucast(d).eval();

    b = a.f("strftime", "%Y-%m-%d %j %U %w %W");
    EXPECT_EQ("1931-12-12 346 49 6 49", b.at(0).as<string>());
    EXPECT_EQ("2013-05-14 134 19 2 19", b.at(1).as<string>());
    EXPECT_EQ("2012-12-25 360 52 2 52", b.at(2).as<string>());
}

TEST(DateDType, StrFTimeOfConvert) {
    // First create a date array which is still a convert expression dtype
    const char *vals[] = {"1920-03-12", "2013-01-01", "2000-12-25"};
    ndobject a = ndobject(vals).ucast(make_date_dtype());
    EXPECT_EQ(make_strided_dim_dtype(make_convert_dtype(make_date_dtype(), make_string_dtype())),
                    a.get_dtype());

    ndobject b = a.f("strftime", "%Y %m %d");
    EXPECT_EQ("1920 03 12", b.at(0).as<string>());
    EXPECT_EQ("2013 01 01", b.at(1).as<string>());
    EXPECT_EQ("2000 12 25", b.at(2).as<string>());
}

TEST(DateDType, StrFTimeOfMultiDim) {
    const char *vals_0[] = {"1920-03-12", "2013-01-01"};
    const char *vals_1[] = {"2000-12-25"};
    ndobject a = make_strided_ndobject(2, -1, make_date_dtype());
    a.vals_at(0) = vals_0;
    a.vals_at(1) = vals_1;

    a = a.f("strftime", "%d/%m/%Y");
    EXPECT_EQ("12/03/1920", a.at(0, 0).as<string>());
    EXPECT_EQ("01/01/2013", a.at(0, 1).as<string>());
    EXPECT_EQ("25/12/2000", a.at(1, 0).as<string>());
}

#if defined(_MSC_VER)
// Only the Windows strftime seems to support this behavior without
// writing our own strftime format parser.
TEST(DateDType, StrFTimeBadFormat) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").ucast(d).eval();
    // Invalid format string should raise an error.
    EXPECT_THROW(a.f("strftime", "%Y %x %s").eval(), runtime_error);
}
#endif

TEST(DateDType, WeekDay) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").ucast(d).eval();
    EXPECT_EQ(6, a.f("weekday").as<int32_t>());
    a = ndobject("2002-12-04").ucast(d).eval();
    EXPECT_EQ(2, a.f("weekday").as<int32_t>());
}

TEST(DateDType, Replace) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").ucast(d).eval();
    EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<string>());
    EXPECT_EQ("2012-12-13", a.f("replace", 2012, 12).as<string>());
    EXPECT_EQ("2012-12-15", a.f("replace", 2012, 12, 15).as<string>());
    // Custom extension, allow -1 indexing from the end for months and days
    EXPECT_EQ("2012-12-30", a.f("replace", 2012, -1, 30).as<string>());
    EXPECT_EQ("2012-05-31", a.f("replace", 2012, -8, -1).as<string>());
    // The C++ call interface doesn't let you skip arguments (yet, there is no keyword argument mechanism),
    // so test this manually
    ndobject param = a.find_dynamic_function("replace").get_default_parameters().eval_copy();
    *reinterpret_cast<void **>(param.at(0).get_readwrite_originptr()) = (void*)a.get_ndo();
    param.at(2).vals() = 7;
    EXPECT_EQ("1955-07-13", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param.at(3).vals() = -1;
    EXPECT_EQ("1955-07-31", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param.at(2).vals() = 2;
    EXPECT_EQ("1955-02-28", a.find_dynamic_function("replace").call_generic(param).as<string>());
    param.at(1).vals() = 2012;
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
    ndobject a;

    // Make an expression dtype with value type 'date'
    a = ndobject("1955-03-13").ucast(make_date_dtype());
    EXPECT_EQ(make_convert_dtype(make_date_dtype(), make_string_dtype()),
                    a.get_dtype());
    // Call replace on it
    EXPECT_EQ("2013-03-13", a.f("replace", 2013).as<string>());
}

TEST(DateDType, NumPyCompatibleProperty) {
    int64_t vals64[] = {-16730, 0, 11001, numeric_limits<int64_t>::min()};

    ndobject a = vals64;
    ndobject a_date = a.view_scalars(make_reversed_property_dtype(make_date_dtype(),
                    make_dtype<int64_t>(), "days_after_1970_int64"));
    // Reading from the 'int64 as date' view
    EXPECT_EQ("1924-03-13", a_date.at(0).as<string>());
    EXPECT_EQ("1970-01-01", a_date.at(1).as<string>());
    EXPECT_EQ("2000-02-14", a_date.at(2).as<string>());
    EXPECT_EQ("NA",         a_date.at(3).as<string>());

    // Writing to the 'int64 as date' view
    a_date.at(0).vals() = "1975-01-30";
    EXPECT_EQ(1855, a.at(0).as<int64_t>());
    a_date.at(0).vals() = "NA";
    EXPECT_EQ(numeric_limits<int64_t>::min(), a.at(0).as<int64_t>());
}
