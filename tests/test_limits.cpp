//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/limits.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(Limits, Min) {
  EXPECT_ARRAY_EQ(numeric_limits<int8_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<int8_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<int16_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<int16_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<int32_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<int32_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<int64_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<int64_t>()}}));

  EXPECT_ARRAY_EQ(numeric_limits<uint8_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<uint8_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<uint16_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<uint16_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<uint32_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<uint32_t>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<uint64_t>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<uint64_t>()}}));

  EXPECT_ARRAY_EQ(numeric_limits<float>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<float>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<double>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<double>()}}));
}
