//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include <dynd/types/uint_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(UIntKindType, Constructor) {
  ndt::type uint_kind_tp = ndt::make_type<ndt::uint_kind_type>();
  EXPECT_EQ(uint_kind_id, uint_kind_tp.get_id());
  EXPECT_EQ(scalar_kind_id, uint_kind_tp.get_base_id());
  EXPECT_EQ(0u, uint_kind_tp.get_data_size());
  EXPECT_EQ(1u, uint_kind_tp.get_data_alignment());
  EXPECT_FALSE(uint_kind_tp.is_expression());
  EXPECT_TRUE(uint_kind_tp.is_symbolic());
  EXPECT_EQ(uint_kind_tp, ndt::type(uint_kind_tp.str())); // Round trip through a string

  vector<ndt::type> bases{ndt::make_type<ndt::scalar_kind_type>(), ndt::make_type<ndt::any_kind_type>()};
  EXPECT_EQ(bases, uint_kind_tp.bases());
}

TEST(UIntKindType, Match) {
  ndt::type uint_kind_tp = ndt::make_type<ndt::uint_kind_type>();

  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<bool>()));

  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<int8_t>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<int16_t>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<int32_t>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<int64_t>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<int128>()));

  EXPECT_TRUE(uint_kind_tp.match(ndt::make_type<uint8_t>()));
  EXPECT_TRUE(uint_kind_tp.match(ndt::make_type<uint16_t>()));
  EXPECT_TRUE(uint_kind_tp.match(ndt::make_type<uint32_t>()));
  EXPECT_TRUE(uint_kind_tp.match(ndt::make_type<uint64_t>()));
  EXPECT_TRUE(uint_kind_tp.match(ndt::make_type<uint128>()));

  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<float16>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<float>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<double>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<float128>()));

  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<dynd::complex<float>>()));
  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<dynd::complex<double>>()));

  EXPECT_FALSE(uint_kind_tp.match(ndt::make_type<void>()));
}

TEST(UIntKindType, IDOf) { EXPECT_EQ(uint_kind_id, ndt::id_of<ndt::uint_kind_type>::value); }
