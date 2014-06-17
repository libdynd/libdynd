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
#include <dynd/func/callable.hpp>
#include <dynd/func/call_callable.hpp>

using namespace std;
using namespace dynd;

TEST(DateTimeDType, Create) {
    ndt::type d;
    const datetime_type *dd;

    d = ndt::make_datetime();
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = d.tcast<datetime_type>();
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(ndt::make_datetime(tz_abstract), d);
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_datetime(tz_utc);
    dd = d.tcast<datetime_type>();
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
    dd = d.tcast<datetime_type>();
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::type("datetime[tz='UTC']");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = d.tcast<datetime_type>();
    EXPECT_EQ(tz_utc, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DateTimeDType, ValueCreationAbstract) {
    ndt::type d = ndt::make_datetime(), di = ndt::make_type<int64_t>();

    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 15:45").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
                    nd::array("1601-01-01T00").ucast(d).view_scalars(di).as<int64_t>());

    // Parsing Zulu timezone as abstract works ok (throws away time zone)
    EXPECT_EQ("2000-01-01T03:00",
              nd::array("2000-01-01T03:00Z").ucast(d).eval().as<string>());
    // Parsing specified timezone as abstract works ok (throws away time zone)
    EXPECT_EQ("2000-01-01T03:00",
              nd::array("2000-01-01T03:00+0300").ucast(d).eval().as<string>());

    // Parsing Zulu timezone as abstract with no error checking works though
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
//                    nd::array("1600-01-01 15:45Z").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
    // Parsing specified timezone as abstract with no error checking throws away the time zone
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
//                    nd::array("1600-01-01 15:45+0600").ucast(d, 0, assign_error_none).view_scalars(di).as<int64_t>());
}


TEST(DateTimeDType, ValueCreationUTC) {
    ndt::type d = ndt::make_datetime(tz_utc), di = ndt::make_type<int64_t>();

    // TODO: enable this test
    //EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
    //                nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
//                    nd::array("1599-01-01T04:16Z").ucast(d).view_scalars(di).as<int64_t>());
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
//                    nd::array("1600-01-01 14:45-0100").ucast(d).view_scalars(di).as<int64_t>());
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
//                    nd::array("1601-01-01T00Z").ucast(d).view_scalars(di).as<int64_t>());
}

TEST(DateTimeDType, ConvertToString) {
    EXPECT_EQ("2013-02-16T12:00",
                    nd::array("2013-02-16T12").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:00Z",
//                    nd::array("2013-02-16T12Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13",
                    nd::array("2013-02-16T12:13").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:13Z",
//                    nd::array("2013-02-16T12:13Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19",
                    nd::array("2013-02-16T12:13:19").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:13:19Z",
//                    nd::array("2013-02-16T12:13:19Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012",
                    nd::array("2013-02-16T12:13:19.012").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:13:19.012Z",
//                    nd::array("2013-02-16T12:13:19.012Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    EXPECT_EQ("2013-02-16T12:13:19.012345",
                    nd::array("2013-02-16T12:13:19.012345").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:13:19.012345Z",
//                    nd::array("2013-02-16T12:13:19.012345Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());

    // Ticks resolution (100*nanoseconds)
    EXPECT_EQ("2013-02-16T12:13:19.0123456",
                    nd::array("2013-02-16T12:13:19.0123456").cast(ndt::type("datetime")).as<string>());
//    EXPECT_EQ("2013-02-16T12:13:19.0123456Z",
//                    nd::array("2013-02-16T12:13:19.0123456Z").cast(ndt::type("datetime[tz='UTC']")).as<string>());
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

TEST(DateTimeStruct, FromToString) {
    datetime_struct dts;

    dts.set_from_str("1991-02-03 04:05:06");
    EXPECT_EQ("1991-02-03T04:05:06", dts.to_str());
    dts.set_from_str("11/12/1822 06:47:26.00", date_parse_mdy);
    EXPECT_EQ("1822-11-12T06:47:26", dts.to_str());
    dts.set_from_str("Fri Dec 19 15:10:11 1997");
    EXPECT_EQ("1997-12-19T15:10:11", dts.to_str());
    dts.set_from_str("Friday, November 11, 2005 17:56:21");
    EXPECT_EQ("2005-11-11T17:56:21", dts.to_str());
    dts.set_from_str("1982-2-20 5:02:00");
    EXPECT_EQ("1982-02-20T05:02", dts.to_str());
    dts.set_from_str("15MAR1985:14:15:22");
    EXPECT_EQ("1985-03-15T14:15:22", dts.to_str());
    dts.set_from_str("20030331 05:59:59.9");
    EXPECT_EQ("2003-03-31T05:59:59.9", dts.to_str());
    dts.set_from_str("Jul  6 2030  5:55PM");
    EXPECT_EQ("2030-07-06T17:55", dts.to_str());
    dts.set_from_str("1994-10-20 T 11:15");
    EXPECT_EQ("1994-10-20T11:15", dts.to_str());
    dts.set_from_str("201303041438");
    EXPECT_EQ("2013-03-04T14:38", dts.to_str());
    dts.set_from_str("20130304143805");
    EXPECT_EQ("2013-03-04T14:38:05", dts.to_str());
    dts.set_from_str("20130304143805.");
    EXPECT_EQ("2013-03-04T14:38:05", dts.to_str());
    dts.set_from_str("20130304143805.123");
    EXPECT_EQ("2013-03-04T14:38:05.123", dts.to_str());
}
