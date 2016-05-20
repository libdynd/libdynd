//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include <dynd/types/float_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(FloatKindType, Constructor) {
  ndt::type float_kind_tp = ndt::make_type<ndt::float_kind_type>();
  EXPECT_EQ(float_kind_id, float_kind_tp.get_id());
  EXPECT_EQ(scalar_kind_id, float_kind_tp.get_base_id());
  EXPECT_EQ(0u, float_kind_tp.get_data_size());
  EXPECT_EQ(1u, float_kind_tp.get_data_alignment());
  EXPECT_FALSE(float_kind_tp.is_expression());
  EXPECT_TRUE(float_kind_tp.is_symbolic());
  EXPECT_EQ(float_kind_tp, ndt::type(float_kind_tp.str())); // Round trip through a string
}

TEST(FloatKindType, Match) {
  ndt::type float_kind_tp = ndt::make_type<ndt::float_kind_type>();

  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<bool>()));

  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<int8_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<int16_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<int32_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<int64_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<int128>()));

  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<uint8_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<uint16_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<uint32_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<uint64_t>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<uint128>()));

  EXPECT_TRUE(float_kind_tp.match(ndt::make_type<float16>()));
  EXPECT_TRUE(float_kind_tp.match(ndt::make_type<float>()));
  EXPECT_TRUE(float_kind_tp.match(ndt::make_type<double>()));
  EXPECT_TRUE(float_kind_tp.match(ndt::make_type<float128>()));

  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<dynd::complex<float>>()));
  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<dynd::complex<double>>()));

  EXPECT_FALSE(float_kind_tp.match(ndt::make_type<void>()));
}

TEST(FloatKindType, IDOf) { EXPECT_EQ(float_kind_id, ndt::id_of<ndt::float_kind_type>::value); }
