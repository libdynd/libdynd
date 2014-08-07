//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/time_type.hpp>
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

TEST(TimeDType, Create) {
    ndt::type d;
    const time_type *tt;

    d = ndt::make_time(tz_abstract);
    ASSERT_EQ(time_type_id, d.get_type_id());
    tt = d.tcast<time_type>();
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(ndt::make_time(tz_abstract), d);
    EXPECT_EQ(tz_abstract, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_time(tz_utc);
    tt = d.tcast<time_type>();
    EXPECT_EQ(ndt::make_time(tz_utc), d);
    EXPECT_EQ(tz_utc, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(TimeDType, CreateFromString) {
    ndt::type d;
    const time_type *tt;

    d = ndt::type("time");
    ASSERT_EQ(time_type_id, d.get_type_id());
    tt = d.tcast<time_type>();
    EXPECT_EQ(tz_abstract, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::type("time[tz='UTC']");
    ASSERT_EQ(time_type_id, d.get_type_id());
    tt = d.tcast<time_type>();
    EXPECT_EQ(tz_utc, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(TimeDType, ValueCreationAbstract) {
    ndt::type d = ndt::make_time(tz_abstract), di = ndt::make_type<int64_t>();

    EXPECT_EQ(0, nd::array("00:00").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(0, nd::array("12:00 am").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(12 * DYND_TICKS_PER_HOUR + 30 * DYND_TICKS_PER_MINUTE,
              nd::array("12:30").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(12 * DYND_TICKS_PER_HOUR + 34 * DYND_TICKS_PER_MINUTE +
                  56 * DYND_TICKS_PER_SECOND + 700 * DYND_TICKS_PER_MILLISECOND,
              nd::array("12:34:56.7").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(12 * DYND_TICKS_PER_HOUR + 34 * DYND_TICKS_PER_MINUTE +
                  56 * DYND_TICKS_PER_SECOND + 7890123,
              nd::array("12:34:56.7890123").ucast(d).view_scalars(di).as<int64_t>());
    EXPECT_EQ(DYND_TICKS_PER_DAY - 1,
              nd::array("23:59:59.9999999").ucast(d).view_scalars(di).as<int64_t>());
}

TEST(TimeDType, ConvertToString) {
    EXPECT_EQ("00:00",
              nd::array("00:00:00.000").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("00:00",
              nd::array("12:00:00.000 AM").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("12:00",
              nd::array("12:00:00.000 PM").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("23:59:59.9999999",
              nd::array("23:59:59.9999999").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("00:00:01",
              nd::array("00:00:01.000").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("00:00:00.0000001",
              nd::array("00:00:00.0000001").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("12:34:56.7",
              nd::array("12:34:56.700").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("23:59:56.78",
              nd::array("23:59:56.7800").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("23:59:59.789",
              nd::array("23:59:59.78900").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("23:59:59.78901",
              nd::array("23:59:59.789010").cast(ndt::type("time")).as<string>());
    EXPECT_EQ("12:34:56.7890123",
              nd::array("12:34:56.7890123").cast(ndt::type("time")).as<string>());
}

TEST(TimeDType, Properties) {
    nd::array a, b;

    a = nd::array("12:34:56.7890123").cast(ndt::type("time")).eval();
    EXPECT_EQ(12, a.p("hour").as<int32_t>());
    EXPECT_EQ(34, a.p("minute").as<int32_t>());
    EXPECT_EQ(56, a.p("second").as<int32_t>());
    EXPECT_EQ(789012, a.p("microsecond").as<int32_t>());
    EXPECT_EQ(7890123, a.p("tick").as<int32_t>());

    b = a.f("to_struct").eval();
    EXPECT_EQ(time_hmst::type(), b.get_type());
    EXPECT_EQ(12, b.p("hour").as<int32_t>());
    EXPECT_EQ(34, b.p("minute").as<int32_t>());
    EXPECT_EQ(56, b.p("second").as<int32_t>());
    EXPECT_EQ(7890123, b.p("tick").as<int32_t>());
}

TEST(TimeHMST, SetFromStr) {
    time_hmst hmst;

    hmst.set_from_str("00:00");
    EXPECT_EQ(0, hmst.hour);
    EXPECT_EQ(0, hmst.minute);
    EXPECT_EQ(0, hmst.second);
    EXPECT_EQ(0, hmst.tick);
    hmst.set_from_str("12:29p");
    EXPECT_EQ(12, hmst.hour);
    EXPECT_EQ(29, hmst.minute);
    EXPECT_EQ(0, hmst.second);
    EXPECT_EQ(0, hmst.tick);
    hmst.set_from_str("12:14:22a.m.");
    EXPECT_EQ(0, hmst.hour);
    EXPECT_EQ(14, hmst.minute);
    EXPECT_EQ(22, hmst.second);
    EXPECT_EQ(0, hmst.tick);
    hmst.set_from_str("3:30 pm");
    EXPECT_EQ(15, hmst.hour);
    EXPECT_EQ(30, hmst.minute);
    EXPECT_EQ(0, hmst.second);
    EXPECT_EQ(0, hmst.tick);
    hmst.set_from_str("12:34:56.7");
    EXPECT_EQ(12, hmst.hour);
    EXPECT_EQ(34, hmst.minute);
    EXPECT_EQ(56, hmst.second);
    EXPECT_EQ(700 * DYND_TICKS_PER_MILLISECOND, hmst.tick);
    hmst.set_from_str("12:34:56.789012345678901234 AM");
    EXPECT_EQ(0, hmst.hour);
    EXPECT_EQ(34, hmst.minute);
    EXPECT_EQ(56, hmst.second);
    EXPECT_EQ(7890123, hmst.tick);
    hmst.set_from_str("09:30:00:003");
    EXPECT_EQ(9, hmst.hour);
    EXPECT_EQ(30, hmst.minute);
    EXPECT_EQ(0, hmst.second);
    EXPECT_EQ(30000, hmst.tick);
}

TEST(TimeHMST, SetFromStr_Errors) {
    time_hmst hmst;

    EXPECT_THROW(hmst.set_from_str("00"), invalid_argument);
    EXPECT_THROW(hmst.set_from_str("00:00 AM"), invalid_argument);
    EXPECT_THROW(hmst.set_from_str("13:00 PM"), invalid_argument);
    EXPECT_THROW(hmst.set_from_str("13:"), invalid_argument);
    EXPECT_THROW(hmst.set_from_str("08:00:"), invalid_argument);
    EXPECT_THROW(hmst.set_from_str("08:00:00."), invalid_argument);
}
