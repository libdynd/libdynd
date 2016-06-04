//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include <dynd/types/complex_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(ComplexKindType, Constructor) {
  ndt::type complex_kind_tp = ndt::make_type<ndt::complex_kind_type>();
  EXPECT_EQ(complex_kind_id, complex_kind_tp.get_id());
  EXPECT_EQ(scalar_kind_id, complex_kind_tp.get_base_id());
  EXPECT_EQ(0u, complex_kind_tp.get_data_size());
  EXPECT_EQ(1u, complex_kind_tp.get_data_alignment());
  EXPECT_FALSE(complex_kind_tp.is_expression());
  EXPECT_TRUE(complex_kind_tp.is_symbolic());
  EXPECT_EQ(complex_kind_tp, ndt::type(complex_kind_tp.str())); // Round trip through a string
}

TEST(ComplexKindType, Match) {
  ndt::type complex_kind_tp = ndt::make_type<ndt::complex_kind_type>();

  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<bool>()));

  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<int8_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<int16_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<int32_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<int64_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<int128>()));

  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<uint8_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<uint16_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<uint32_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<uint64_t>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<uint128>()));

  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<float16>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<float>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<double>()));
  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<float128>()));

  EXPECT_TRUE(complex_kind_tp.match(ndt::make_type<dynd::complex<float>>()));
  EXPECT_TRUE(complex_kind_tp.match(ndt::make_type<dynd::complex<double>>()));

  EXPECT_FALSE(complex_kind_tp.match(ndt::make_type<void>()));
}

TEST(ComplexKindType, IDOf) { EXPECT_EQ(complex_kind_id, ndt::id_of<ndt::complex_kind_type>::value); }
