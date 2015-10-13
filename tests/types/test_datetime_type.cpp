//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/string.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(DatetimeType, Create) {
    ndt::type d;
    const ndt::datetime_type *dd;

    d = ndt::datetime_type::make();
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = d.extended<ndt::datetime_type>();
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(ndt::datetime_type::make(tz_abstract), d);
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::datetime_type::make(tz_utc);
    dd = d.extended<ndt::datetime_type>();
    EXPECT_EQ(ndt::datetime_type::make(tz_utc), d);
    EXPECT_EQ(tz_utc, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DatetimeType, CreateFromString) {
    ndt::type d;
    const ndt::datetime_type *dd;

    d = ndt::type("datetime");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = d.extended<ndt::datetime_type>();
    EXPECT_EQ(tz_abstract, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::type("datetime[tz='UTC']");
    ASSERT_EQ(datetime_type_id, d.get_type_id());
    dd = d.extended<ndt::datetime_type>();
    EXPECT_EQ(tz_utc, dd->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(DatetimeType, ValueCreationAbstract) {
    ndt::type d = ndt::datetime_type::make(), di = ndt::type::make<int64_t>();

    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
                    nd::array("1600-01-01 15:45").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
                    nd::array("1601-01-01T00").ucast(d).view_scalars(di).as<int64_t>());

    // Parsing Zulu timezone as abstract works ok (throws away time zone)
    EXPECT_EQ("2000-01-01T03:00",
              nd::array("2000-01-01T03:00Z").ucast(d).eval().as<std::string>());
    // Parsing specified timezone as abstract works ok (throws away time zone)
    EXPECT_EQ("2000-01-01T03:00",
              nd::array("2000-01-01T03:00+0300").ucast(d).eval().as<std::string>());

    // Parsing Zulu timezone as abstract with no error checking works though
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
//                    nd::array("1600-01-01 15:45Z").ucast(d, 0, assign_error_nocheck).view_scalars(di).as<int64_t>());
    // Parsing specified timezone as abstract with no error checking throws away the time zone
//    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
//                    nd::array("1600-01-01 15:45+0600").ucast(d, 0, assign_error_nocheck).view_scalars(di).as<int64_t>());
}


TEST(DatetimeType, ValueCreationUTC) {
    ndt::type d = ndt::datetime_type::make(tz_utc), di = ndt::type::make<int64_t>();

    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 - 365) * 1440LL + 4 * 60 + 16) * 60 * 10000000LL,
                    nd::array("1599-01-01T04:16Z").ucast(d).view_scalars(di).as<int64_t>());
    // TODO: enable this test
    //EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3) * 1440LL + 15 * 60 + 45) * 60 * 10000000LL,
    //                nd::array("1600-01-01 14:45-0100").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ((((1600-1970)*365 - (1972-1600)/4 + 3 + 366) * 1440LL) * 60 * 10000000LL,
                    nd::array("1601-01-01T00:00Z").ucast(d).view_scalars(di).as<int64_t>());
}

TEST(DatetimeType, ConvertToString) {
    EXPECT_EQ("2013-02-16T12:00",
                    nd::array("2013-02-16T12").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:00Z",
                    nd::array("2013-02-16T12").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());

    EXPECT_EQ("2013-02-16T12:13",
                    nd::array("2013-02-16T12:13").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:13Z",
                    nd::array("2013-02-16T12:13Z").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());

    EXPECT_EQ("2013-02-16T12:13:19",
                    nd::array("2013-02-16T12:13:19").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:13:19Z",
                    nd::array("2013-02-16T12:13:19Z").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());

    EXPECT_EQ("2013-02-16T12:13:19.012",
                    nd::array("2013-02-16T12:13:19.012").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:13:19.012Z",
                    nd::array("2013-02-16T12:13:19.012Z").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());

    EXPECT_EQ("2013-02-16T12:13:19.012345",
                    nd::array("2013-02-16T12:13:19.012345").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:13:19.012345Z",
                    nd::array("2013-02-16T12:13:19.012345Z").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());

    // Ticks resolution (100*nanoseconds)
    EXPECT_EQ("2013-02-16T12:13:19.0123456",
                    nd::array("2013-02-16T12:13:19.0123456").cast(ndt::type("datetime")).as<std::string>());
    EXPECT_EQ("2013-02-16T12:13:19.0123456Z",
                    nd::array("2013-02-16T12:13:19.0123456Z").cast(ndt::type("datetime[tz='UTC']")).as<std::string>());
}

TEST(DatetimeType, AbstractTZToUTC) {
    // Assigning from an abstract/naive timezone to UTC is allowed, the datetime
    // value adopts the destination time zone.
    nd::array a = parse_json("2 * datetime",
                             "[\"2013-01-15T12:30\", \"2010-03-12T11:15:59\"]");
    nd::array b = nd::empty("2 * datetime[tz='UTC']");
    b.vals() = a;
    EXPECT_EQ("2013-01-15T12:30Z", b(0).as<std::string>());
    EXPECT_EQ("2010-03-12T11:15:59Z", b(1).as<std::string>());

    // Assigning the other way is not ok by default, except in nocheck error
    // mode
    a = nd::empty("datetime");
    EXPECT_THROW(a.vals() = b(0), type_error);
    eval::eval_context ectx;
    ectx.errmode = assign_error_nocheck;
    a.val_assign(b(1), &ectx);
    EXPECT_EQ("2010-03-12T11:15:59", a.as<std::string>());
}

TEST(DatetimeType, Properties) {
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


TEST(DatetimeType, AdaptFromInt) {
    nd::array a, b;

    a = parse_json("3 * int64", "[31968000000, -999480000, 45296789]");
    b = a.adapt(ndt::datetime_type::make(), "milliseconds since 2000-01-01T00:00");
    EXPECT_EQ("2001-01-05T00:00", b(0).as<std::string>());
    EXPECT_EQ("1999-12-20T10:22", b(1).as<std::string>());
    EXPECT_EQ("2000-01-01T12:34:56.789", b(2).as<std::string>());

    a = parse_json("3 * int64", "[31968000000000, -999480000000, 45296789123]");
    b = a.adapt(ndt::datetime_type::make(), "microseconds since 2000-01-01");
    EXPECT_EQ("2001-01-05T00:00", b(0).as<std::string>());
    EXPECT_EQ("1999-12-20T10:22", b(1).as<std::string>());
    EXPECT_EQ("2000-01-01T12:34:56.789123", b(2).as<std::string>());

    a = parse_json("3 * int64", "[31968000000000000, -999480000000000, 45296789123456]");
    b = a.adapt(ndt::datetime_type::make(), "nanoseconds since 2000");
    EXPECT_EQ("2001-01-05T00:00", b(0).as<std::string>());
    EXPECT_EQ("1999-12-20T10:22", b(1).as<std::string>());
    EXPECT_EQ("2000-01-01T12:34:56.7891234", b(2).as<std::string>());
}

TEST(DatetimeType, AdaptAsInt) {
    nd::array a, b;

    a = parse_json("3 * datetime", "[\"2001-01-05T00:00\", \"1999-12-20T10:22\", \"2000-01-01T12:34:56\"]");
    b = a.adapt(ndt::type::make<int64_t>(), "seconds since 2000-01-01T00:00");
    EXPECT_EQ(370*24*60*60, b(0).as<int64_t>());
    EXPECT_EQ(-12*24*60*60 + 10*60*60 + 22*60, b(1).as<int64_t>());
    EXPECT_EQ(12*60*60 + 34*60 + 56, b(2).as<int64_t>());
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
    dts.set_from_str("Jan 2 2050");
    EXPECT_EQ("2050-01-02T00:00", dts.to_str());
}
