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
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/call_callable.hpp>

using namespace std;
using namespace dynd;

TEST(TimeDType, Create) {
    ndt::type d;
    const time_type *tt;

    d = ndt::make_time(tz_abstract);
    ASSERT_EQ(time_type_id, d.get_type_id());
    tt = static_cast<const time_type *>(d.extended());
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<int64_t>::value, d.get_data_alignment());
    EXPECT_EQ(ndt::make_time(tz_abstract), d);
    EXPECT_EQ(tz_abstract, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::make_time(tz_utc);
    tt = static_cast<const time_type *>(d.extended());
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
    tt = static_cast<const time_type *>(d.extended());
    EXPECT_EQ(tz_abstract, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    d = ndt::type("time[tz='UTC']");
    ASSERT_EQ(time_type_id, d.get_type_id());
    tt = static_cast<const time_type *>(d.extended());
    EXPECT_EQ(tz_utc, tt->get_timezone());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(TimeDType, ValueCreationAbstract) {
    ndt::type d = ndt::make_time(tz_abstract), di = ndt::make_type<int64_t>();

    EXPECT_EQ(0, nd::array("00:00").ucast(d).view_scalars(di).as<int64_t>());
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
    nd::array n;

    n = nd::array("12:34:56.7890123").cast(ndt::type("time")).eval();
    EXPECT_EQ(12, n.p("hour").as<int32_t>());
    EXPECT_EQ(34, n.p("minute").as<int32_t>());
    EXPECT_EQ(56, n.p("second").as<int32_t>());
    EXPECT_EQ(789012, n.p("microsecond").as<int32_t>());
    EXPECT_EQ(7890123, n.p("tick").as<int32_t>());
}
