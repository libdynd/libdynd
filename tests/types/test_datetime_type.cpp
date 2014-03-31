//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/datetime_type.hpp>
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

TEST(DateTimeDType, Create) {
    ndt::type d;
    const datetime_type *dd;

    d = ndt::make_datetime(tz_abstract);
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_type *>(d.extended());
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(ndt::make_datetime(tz_abstract), d);
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_datetime(tz_utc);
    dd = static_cast<const datetime_type *>(d.extended());
    EXPECT_EQ(ndt::make_datetime(tz_utc), d);
    EXPECT_EQ(tz_utc, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DateTimeDType, CreateFromString) {
    ndt::type d;
    const datetime_type *dd;

    d = ndt::type("datetime");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_type *>(d.extended());
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::type("datetime[tz='UTC']");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = static_cast<const datetime_type *>(d.extended());
    EXPECT_EQ(tz_utc, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DateTimeDType, ValueCreationAbstract) {
    ndt::type d = ndt::make_datetime(tz_abstract), di = ndt::make_type<int64_t>();

    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 15:45").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
                    nd::array("1601-01-01T00").ucast(d).view_scalars(di).as<int64_t>());

    // Parsing Zulu timezone as abstract raises an error
    EXPECT_THROW(nd::array("2000-01-01T03:00Z").ucast(d).eval(), runtime_error);
    // Parsing specified timezone as abstract raises an error
    EXPECT_THROW(nd::array("2000-01-01T03:00+0300").ucast(d).eval(), runtime_error);

    // Parsing Zulu timezone as abstract with no error checking works though
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 15:45Z").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
    // Parsing specified timezone as abstract with no error checking throws away the time zone
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 15:45+0600").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
}


TEST(DateTimeDType, ValueCreationUTC) {
    ndt::type d = ndt::make_datetime(tz_utc), di = ndt::make_type<int64_t>();

    // TODO: enable this test
    //EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
    //                nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16Z").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 14:45-0100").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
                    nd::array("1601-01-01T00Z").ucast(d).view_scalars(di).as<int64_t>());
}

TEST(DateTimeDType, ConvertToString) {
    EXPECT_EQ("2013-02-16T12:00",
                    nd::array("2013-02-16T12").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:00Z",
                    nd::array("2013-02-16T12Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13",
                    nd::array("2013-02-16T12:13").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:13Z",
                    nd::array("2013-02-16T12:13Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19",
                    nd::array("2013-02-16T12:13:19").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19Z",
                    nd::array("2013-02-16T12:13:19Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012",
                    nd::array("2013-02-16T12:13:19.012").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.012Z",
                    nd::array("2013-02-16T12:13:19.012Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012345",
                    nd::array("2013-02-16T12:13:19.012345").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.012345Z",
                    nd::array("2013-02-16T12:13:19.012345Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    // Ticks resolution (100*nanoseconds)
    EXPECT_EQ("2013-02-16T12:13:19.0123456",
                    nd::array("2013-02-16T12:13:19.0123456").cast(ndt::type("datetime")).as<string>());
    EXPECT_EQ("2013-02-16T12:13:19.0123456Z",
                    nd::array("2013-02-16T12:13:19.0123456Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());
}

TEST(DateTimeDType, Properties) {
    nd::array n;

    n = nd::array("1963-02-28T16:12:14.123654").cast(ndt::type("datetime")).eval();
    EXPECT_EQ(1963, n.p("year").as<int32_t>());
    EXPECT_EQ(2, n.p("month").as<int32_t>());
    EXPECT_EQ(28, n.p("day").as<int32_t>());
    EXPECT_EQ(16, n.p("hour").as<int32_t>());
    EXPECT_EQ(12, n.p("minute").as<int32_t>());
    EXPECT_EQ(14, n.p("second").as<int32_t>());
    EXPECT_EQ(1236540, n.p("tick").as<int32_t>());
}
