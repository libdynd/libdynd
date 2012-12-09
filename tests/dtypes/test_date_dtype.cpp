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
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DateDType, Create) {
    dtype d;
    const date_dtype *dd;

    d = make_date_dtype();
    EXPECT_EQ(4u, d.get_element_size());
    EXPECT_EQ(4u, d.get_alignment());
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_day);

    d = make_date_dtype(date_unit_week);
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_week);

    d = make_date_dtype(date_unit_month);
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_month);

    d = make_date_dtype(date_unit_year);
    dd = static_cast<const date_dtype *>(d.extended());
    EXPECT_EQ(dd->get_unit(), date_unit_year);
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