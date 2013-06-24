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
