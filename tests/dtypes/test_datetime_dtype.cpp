//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/datetime_dtype.hpp>
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

TEST(DateTimeDType, Create) {
    dtype d;
    const datetime_dtype *dd;

    d = make_datetime_dtype(datetime_unit_minute, tz_abstract);
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(d, make_datetime_dtype(datetime_unit_minute, tz_abstract));
    EXPECT_EQ(datetime_unit_minute, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = make_datetime_dtype(datetime_unit_msecond, tz_utc);
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(d, make_datetime_dtype(datetime_unit_msecond, tz_utc));
    EXPECT_EQ(datetime_unit_msecond, dd->get_unit());
    EXPECT_EQ(tz_utc, dd->get_timezone());
}

TEST(DateTimeDType, CreateFromString) {
    dtype d;
    const datetime_dtype *dd;

    d = dtype("datetime('hour')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_hour, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = dtype("datetime('min')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_minute, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = dtype("datetime('sec')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_second, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = dtype("datetime('msec')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_msecond, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = dtype("datetime('usec')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_usecond, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    d = dtype("datetime('nsec')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_nsecond, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    // Explicit abstract timezone
    d = dtype("datetime('hour', 'abstract')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_hour, dd->get_unit());
    EXPECT_EQ(tz_abstract, dd->get_timezone());

    // UTC timezone
    d = dtype("datetime('hour', 'UTC')");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_dtype *>(d.extended());
    EXPECT_EQ(datetime_unit_hour, dd->get_unit());
    EXPECT_EQ(tz_utc, dd->get_timezone());
}

TEST(DateDType, ValueCreationAbstractMinutes) {
    dtype d = make_datetime_dtype(datetime_unit_minute, tz_abstract), di = make_dtype<int64_t>();

    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16,
                    ndobject("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45,
                    ndobject("1600-01-01 15:45").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL,
                    ndobject("1601-01-01T00").ucast(d).view_scalars(di).as<int64_t>());

    // Parsing Zulu timezone as abstract raises an error
    EXPECT_THROW(ndobject("2000-01-01T03:00Z").ucast(d).eval(), runtime_error);
    // Parsing specified timezone as abstract raises an error
    EXPECT_THROW(ndobject("2000-01-01T03:00+0300").ucast(d).eval(), runtime_error);

    // Parsing Zulu timezone as abstract with no error checking works though
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45,
                    ndobject("1600-01-01 15:45Z").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
    // Parsing specified timezone as abstract with no error checking throws away the time zone
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45,
                    ndobject("1600-01-01 15:45+0600").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
}


TEST(DateDType, ValueCreationUTCMinutes) {
    dtype d = make_datetime_dtype(datetime_unit_minute, tz_utc), di = make_dtype<int64_t>();

    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16,
                    ndobject("1599-01-01T04:16Z").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45,
                    ndobject("1600-01-01 14:45-0100").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL,
                    ndobject("1601-01-01T00Z").ucast(d).view_scalars(di).as<int64_t>());
}

TEST(DateDType, ConvertToString) {
    EXPECT_EQ("2013-02-16T12",
                    ndobject("2013-02-16T12").cast(dtype("datetime('hour')")).as<string>());
    EXPECT_EQ("2013-02-16T12Z",
                    ndobject("2013-02-16T12Z").cast(dtype("datetime('hour','UTC')")).as<string>());

    EXPECT_EQ("2013-02-16T12:13",
                    ndobject("2013-02-16T12:13").cast(dtype("datetime('min')")).as<string>());
    EXPECT_EQ("2013-02-16T12:13Z",
                    ndobject("2013-02-16T12:13Z").cast(dtype("datetime('min','UTC')")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19",
                    ndobject("2013-02-16T12:13:19").cast(dtype("datetime('sec')")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19Z",
                    ndobject("2013-02-16T12:13:19Z").cast(dtype("datetime('sec','UTC')")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012",
                    ndobject("2013-02-16T12:13:19.012").cast(dtype("datetime('msec')")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.012Z",
                    ndobject("2013-02-16T12:13:19.012Z").cast(dtype("datetime('msec','UTC')")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012345",
                    ndobject("2013-02-16T12:13:19.012345").cast(dtype("datetime('usec')")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.012345Z",
                    ndobject("2013-02-16T12:13:19.012345Z").cast(dtype("datetime('usec','UTC')")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012345678",
                    ndobject("2013-02-16T12:13:19.012345678").cast(dtype("datetime('nsec')")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.012345678Z",
                    ndobject("2013-02-16T12:13:19.012345678Z").cast(dtype("datetime('nsec','UTC')")).as<string>());
}
