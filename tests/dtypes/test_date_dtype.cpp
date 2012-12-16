//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/date_property_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/gfunc/callable.hpp>

using namespace std;
using namespace dynd;

TEST(DateDType, Create) {
    dtype d;
    const date_dtype *dd;

    d = make_date_dtype();
    EXPECT_EQ(4u, d.get_data_size());
    EXPECT_EQ(4u, d.get_alignment());
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_day);

    d = make_date_dtype(date_unit_month);
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_month);

    d = make_date_dtype(date_unit_year);
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_year);
}

TEST(DateDType, Equality) {
    EXPECT_EQ(make_date_dtype(), make_date_dtype(date_unit_day));
    EXPECT_EQ(make_date_dtype(date_unit_month), make_date_dtype(date_unit_month));
    EXPECT_EQ(make_date_dtype(date_unit_year), make_date_dtype(date_unit_year));
    EXPECT_FALSE(make_date_dtype() == make_date_dtype(date_unit_month));
    EXPECT_FALSE(make_date_dtype() == make_date_dtype(date_unit_year));
}

TEST(DateDType, ValueCreation) {
    dtype d = make_date_dtype(), di = make_dtype<int32_t>();

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 - 365,
                    ndobject("1599-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3,
                    ndobject("1600-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 366,
                    ndobject("1601-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4,
                    ndobject("1900-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1900-1970)*365 - (1970-1900)/4 + 365,
                    ndobject("1901-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ(-3*365 - 1,
                    ndobject("1967-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-2*365 - 1,
                    ndobject("1968-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(-1*365,
                    ndobject("1969-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(0*365,
                    ndobject("1970-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(1*365,
                    ndobject("1971-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(2*365,
                    ndobject("1972-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(3*365 + 1,
                    ndobject("1973-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ(4*365 + 1,
                    ndobject("1974-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4,
                    ndobject("2000-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366,
                    ndobject("2001-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3,
                    ndobject("2400-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2400 - 1970)*365 + (2400 - 1972)/4 - 3 + 366,
                    ndobject("2401-01-01").cast_scalars(d).view_scalars(di).as<int32_t>());

    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 28,
                    ndobject("1600-02-29").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((1600-1970)*365 - (1972-1600)/4 + 3 + 31 + 29,
                    ndobject("1600-03-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 28,
                    ndobject("2000-02-29").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 31 + 29,
                    ndobject("2000-03-01").cast_scalars(d).view_scalars(di).as<int32_t>());
    EXPECT_EQ((2000 - 1970)*365 + (2000 - 1972)/4 + 366 + 31 + 28 + 21,
                    ndobject("2001-03-22").cast_scalars(d).view_scalars(di).as<int32_t>());
}

TEST(DateDType, BadInputStrings) {
    dtype d = make_date_dtype();

    // Arbitrary bad string
    EXPECT_THROW(ndobject(ndobject("badvalue").cast_scalars(d).vals()), runtime_error);
    // Character after year must be '-'
    EXPECT_THROW(ndobject(ndobject("1980X").cast_scalars(d).vals()), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(ndobject(ndobject("1980-").cast_scalars(d).vals()), runtime_error);
    // Month must be in range [1,12]
    EXPECT_THROW(ndobject(ndobject("1980-00").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-13").cast_scalars(d).vals()), runtime_error);
    // Month must have two digits
    EXPECT_THROW(ndobject(ndobject("1980-1").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-1-02").cast_scalars(d).vals()), runtime_error);
    // 'Mor' is not a valid month
    EXPECT_THROW(ndobject(ndobject("1980-Mor").cast_scalars(d).vals()), runtime_error);
    // Cannot have trailing '-'
    EXPECT_THROW(ndobject(ndobject("1980-01-").cast_scalars(d).vals()), runtime_error);
    // Day must be in range [1,len(month)]
    EXPECT_THROW(ndobject(ndobject("1980-01-0").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-01-00").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-01-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1979-02-29").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-02-30").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-03-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-04-31").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-05-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-06-31").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-07-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-08-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-09-31").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-10-32").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-11-31").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-12-32").cast_scalars(d).vals()), runtime_error);
    // Cannot have trailing characters
    EXPECT_THROW(ndobject(ndobject("1980-02-03%").cast_scalars(d).vals()), runtime_error);
    EXPECT_THROW(ndobject(ndobject("1980-02-03 q").cast_scalars(d).vals()), runtime_error);
}

TEST(DateDType, DateDaysUnitProperties) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").cast_scalars(d).vals();
    EXPECT_EQ(make_date_property_dtype(d, "year"), a.p("year").get_dtype());
    EXPECT_EQ(make_date_property_dtype(d, "month"), a.p("month").get_dtype());
    EXPECT_EQ(make_date_property_dtype(d, "day"), a.p("day").get_dtype());
    EXPECT_EQ(1955, a.p("year").as<int32_t>());
    EXPECT_EQ(3, a.p("month").as<int32_t>());
    EXPECT_EQ(13, a.p("day").as<int32_t>());

    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = ndobject(strs).cast_scalars(d).vals();
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

TEST(DateDType, DateDaysUnitStructFunction) {
    dtype d = make_date_dtype();
    ndobject a, b;

    a = ndobject("1955-03-13").cast_scalars(d).vals();
    b = a.f("to_struct").call(a);
    EXPECT_EQ(make_convert_dtype(make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day"), d),
                    b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day"),
                    b.get_dtype());
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3, b.p("month").as<int32_t>());
    EXPECT_EQ(13, b.p("day").as<int32_t>());
}

TEST(DateDType, DateMonthsUnitProperties) {
    dtype d = make_date_dtype(date_unit_month);
    ndobject a = ndobject("1955-03").cast_scalars(d).vals();
    EXPECT_EQ(make_date_property_dtype(d, "year"), a.p("year").get_dtype());
    EXPECT_EQ(make_date_property_dtype(d, "month"), a.p("month").get_dtype());
    EXPECT_THROW(make_date_property_dtype(d, "day"), runtime_error);
    EXPECT_EQ(1955, a.p("year").as<int32_t>());
    EXPECT_EQ(3, a.p("month").as<int32_t>());
    EXPECT_THROW(a.p("day"), runtime_error);

    const char *strs[] = {"1931-12", "2013-05", "2012-12"};
    a = ndobject(strs).cast_scalars(d).vals();
    EXPECT_EQ(1931, a.p("year").at(0).as<int32_t>());
    EXPECT_EQ(12, a.p("month").at(0).as<int32_t>());
    EXPECT_EQ(2013, a.p("year").at(1).as<int32_t>());
    EXPECT_EQ(5, a.p("month").at(1).as<int32_t>());
    EXPECT_EQ(2012, a.p("year").at(2).as<int32_t>());
    EXPECT_EQ(12, a.p("month").at(2).as<int32_t>());
    EXPECT_THROW(a.p("day"), runtime_error);
}

TEST(DateDType, DateYearsUnitProperties) {
    dtype d = make_date_dtype(date_unit_year);
    ndobject a = ndobject("1955").cast_scalars(d).vals();
    EXPECT_EQ(make_date_property_dtype(d, "year"), a.p("year").get_dtype());
    EXPECT_THROW(make_date_property_dtype(d, "month"), runtime_error);
    EXPECT_THROW(make_date_property_dtype(d, "day"), runtime_error);
    EXPECT_EQ(1955, a.p("year").as<int32_t>());
    EXPECT_THROW(a.p("month"), runtime_error);
    EXPECT_THROW(a.p("day"), runtime_error);

    const char *strs[] = {"1931", "2013", "2012"};
    a = ndobject(strs).cast_scalars(d).vals();
    EXPECT_EQ(1931, a.p("year").at(0).as<int32_t>());
    EXPECT_EQ(2013, a.p("year").at(1).as<int32_t>());
    EXPECT_EQ(2012, a.p("year").at(2).as<int32_t>());
    EXPECT_THROW(a.p("month"), runtime_error);
    EXPECT_THROW(a.p("day"), runtime_error);
}

TEST(DateDType, DaysUnitToStruct) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    a = ndobject("1955-03-13").cast_scalars(d).vals();

    // This is the default struct produced
    ds = make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(0).as<int32_t>());
    EXPECT_EQ(3, b.at(1).as<int8_t>());
    EXPECT_EQ(13, b.at(2).as<int8_t>());

    // This should work too
    ds = make_fixedstruct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());
    EXPECT_EQ(13, b.at(2).as<float>());

    // This should work too
    ds = make_struct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());
    EXPECT_EQ(13, b.at(2).as<float>());
}

TEST(DateDType, MonthsUnitToStruct) {
    dtype d = make_date_dtype(date_unit_month), ds;
    ndobject a, b;

    a = ndobject("1955-03").cast_scalars(d).vals();

    // This is the default struct produced
    ds = make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(0).as<int32_t>());
    EXPECT_EQ(3, b.at(1).as<int8_t>());

    // This should work too
    ds = make_fixedstruct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());

    // This should work too
    ds = make_struct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(1).as<int16_t>());
    EXPECT_EQ(3, b.at(0).as<int16_t>());
}

TEST(DateDType, YearsUnitToStruct) {
    dtype d = make_date_dtype(date_unit_year), ds;
    ndobject a, b;

    a = ndobject("1955").cast_scalars(d).vals();

    // This is the default struct produced
    ds = make_fixedstruct_dtype(make_dtype<int32_t>(), "year");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(0).as<int32_t>());

    // This should work too
    ds = make_fixedstruct_dtype(make_dtype<float>(), "year");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(0).as<float>());

    // This should work too
    ds = make_struct_dtype(make_dtype<float>(), "year");
    b = ndobject(ds);
    b.val_assign(a);
    EXPECT_EQ(1955, b.at(0).as<float>());
}

TEST(DateDType, StructToDaysUnit) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    // This is the default struct accepted
    ds = make_fixedstruct_dtype(make_dtype<int32_t>(), "year", make_dtype<int8_t>(), "month", make_dtype<int8_t>(), "day");
    a = ndobject(ds);
    a.at(0).vals() = 1955;
    a.at(1).vals() = 3;
    a.at(2).vals() = 13;
    b = ndobject(d);
    b.val_assign(a);
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = make_fixedstruct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    a = ndobject(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = ndobject(d);
    b.val_assign(a);
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());

    // This should work too
    ds = make_struct_dtype(make_dtype<int16_t>(), "month", make_dtype<int16_t>(), "year", make_dtype<float>(), "day");
    a = ndobject(ds);
    a.p("year").vals() = 1955;
    a.p("month").vals() = 3;
    a.p("day").vals() = 13;
    b = ndobject(d);
    b.val_assign(a);
    EXPECT_EQ(1955, b.p("year").as<int32_t>());
    EXPECT_EQ(3,    b.p("month").as<int32_t>());
    EXPECT_EQ(13,   b.p("day").as<int32_t>());
}

TEST(DateDType, StrFTime) {
    dtype d = make_date_dtype(), ds;
    ndobject a, b;

    a = ndobject("1955-03-13").cast_scalars(d).vals();

    b = a.f("strftime").call(a, "%Y");
    EXPECT_EQ("1955", b.as<string>());
    b = a.f("strftime").call(a, "%m/%d/%y");
    EXPECT_EQ("03/13/55", b.as<string>());
    b = a.f("strftime").call(a, "%Y and %j");
    EXPECT_EQ("1955 and 072", b.as<string>());

    const char *strs[] = {"1931-12-12", "2013-05-14", "2012-12-25"};
    a = ndobject(strs).cast_scalars(d).vals();

    b = a.f("strftime").call(a, "%Y-%m-%d %j %U %w %W");
    EXPECT_EQ("1931-12-12 346 49 6 49", b.at(0).as<string>());
    EXPECT_EQ("2013-05-14 134 19 2 19", b.at(1).as<string>());
    EXPECT_EQ("2012-12-25 360 52 2 52", b.at(2).as<string>());
}

#if defined(_MSC_VER)
// Only the Windows strftime seems to support this behavior without
// writing our own strftime format parser.
TEST(DateDType, StrFTimeBadFormat) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").cast_scalars(d).vals();
    // Invalid format string should raise an error.
    EXPECT_THROW(a.f("strftime").call(a, "%Y %x %s"), runtime_error);
}
#endif

TEST(DateDType, WeekDay) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").cast_scalars(d).vals();
    EXPECT_EQ(6, a.f("weekday").call(a).as<int32_t>());
    a = ndobject("2002-12-04").cast_scalars(d).vals();
    EXPECT_EQ(2, a.f("weekday").call(a).as<int32_t>());
}

TEST(DateDType, Replace) {
    dtype d = make_date_dtype();
    ndobject a;

    a = ndobject("1955-03-13").cast_scalars(d).vals();
    EXPECT_EQ("2013-03-13", a.f("replace").call(a, 2013).as<string>());
    EXPECT_EQ("2012-12-13", a.f("replace").call(a, 2012, 12).as<string>());
    EXPECT_EQ("2012-12-15", a.f("replace").call(a, 2012, 12, 15).as<string>());
    // Custom extension, allow -1 indexing from the end for months and days
    EXPECT_EQ("2012-12-30", a.f("replace").call(a, 2012, -1, 30).as<string>());
    EXPECT_EQ("2012-05-31", a.f("replace").call(a, 2012, -8, -1).as<string>());
    // The C++ call interface doesn't let you skip arguments (yet, there is no keyword argument mechanism),
    // so test this manually
    ndobject param = a.f("replace").get_default_parameters().eval_copy();
    *reinterpret_cast<void **>(param.at(0).get_readwrite_originptr()) = (void*)a.get_ndo();
    param.at(2).vals() = 7;
    EXPECT_EQ("1955-07-13", a.f("replace").call_generic(param).as<string>());
    param.at(3).vals() = -1;
    EXPECT_EQ("1955-07-31", a.f("replace").call_generic(param).as<string>());
    param.at(2).vals() = 2;
    EXPECT_EQ("1955-02-28", a.f("replace").call_generic(param).as<string>());
    param.at(1).vals() = 2012;
    EXPECT_EQ("2012-02-29", a.f("replace").call_generic(param).as<string>());
    // Should throw an exception when no arguments or out of bounds arguments are provided
    EXPECT_THROW(a.f("replace").call(a), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, -13), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, 0), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, 13), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 1900, 2, -29), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 1900, 2, 0), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 1900, 2, 29), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, 2, -30), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, 2, 0), runtime_error);
    EXPECT_THROW(a.f("replace").call(a, 2000, 2, 30), runtime_error);
}
