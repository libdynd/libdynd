//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/adapt_type.hpp>

using namespace std;
using namespace dynd;

TEST(AdaptType, Byteswap)
{
  ndt::type tp = ndt::make_type<ndt::adapt_type>(
      ndt::make_type<float>(), ndt::make_fixed_bytes(sizeof(float), alignof(float)), nd::byteswap, nd::byteswap);
  // The value has the native byte-order type
  EXPECT_EQ(ndt::make_type<float>(), tp.value_type());
  // The storage is the a bytes type with matching storage and alignment
  EXPECT_EQ(ndt::make_fixed_bytes(sizeof(float), alignof(float)), tp.storage_type());
  EXPECT_TRUE(tp.is_expression());
  // The canonical type of a byteswap type is always the non-swapped version
  EXPECT_EQ(ndt::make_type<float>(), tp.get_canonical_type());

  tp = ndt::make_type<ndt::adapt_type>(
      ndt::make_type<dynd::complex<double>>(),
      ndt::make_fixed_bytes(sizeof(dynd::complex<double>), alignof(dynd::complex<double>)), nd::pairwise_byteswap,
      nd::pairwise_byteswap);
  // The value has the native byte-order type
  EXPECT_EQ(ndt::make_type<dynd::complex<double>>(), tp.value_type());
  // The storage is the a bytes type with matching storage and alignment
  EXPECT_EQ(ndt::make_fixed_bytes(sizeof(dynd::complex<double>), alignof(dynd::complex<double>)), tp.storage_type());
  EXPECT_TRUE(tp.is_expression());
}

TEST(AdaptType, ByteswapEval)
{
  nd::array a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<int16_t>(), ndt::make_fixed_bytes(sizeof(int16_t), alignof(int16_t)), nd::byteswap, nd::byteswap));
  a.assign(0x1362);
  EXPECT_EQ(0x6213, a.view<int16_t>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<int32_t>(), ndt::make_fixed_bytes(sizeof(int32_t), alignof(int32_t)), nd::byteswap, nd::byteswap));
  a.assign(0x12345678);
  EXPECT_EQ(0x78563412, a.view<int32_t>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<int64_t>(), ndt::make_fixed_bytes(sizeof(int64_t), alignof(int64_t)), nd::byteswap, nd::byteswap));
  a.assign(0x12345678abcdef01LL);
  EXPECT_EQ(0x01efcdab78563412LL, a.view<int64_t>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<float>(), ndt::make_fixed_bytes(sizeof(float), alignof(float)), nd::byteswap, nd::byteswap));
  a.assign(alias_cast<float>(0xDA0F4940));
  EXPECT_EQ(3.1415926f, a.view<float>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<double>(), ndt::make_fixed_bytes(sizeof(double), alignof(double)), nd::byteswap, nd::byteswap));
  a.assign(alias_cast<double>(0x112D4454FB210940LL));
  EXPECT_EQ(3.14159265358979, a.view<double>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<dynd::complex<float>>(),
      ndt::make_fixed_bytes(sizeof(dynd::complex<float>), alignof(dynd::complex<float>)), nd::pairwise_byteswap,
      nd::pairwise_byteswap));
  a.assign(dynd::complex<float>(alias_cast<float>(0xDA0F4940), alias_cast<float>(0xC1B88FD3)));
  EXPECT_EQ(dynd::complex<float>(3.1415926f, -1.23456e12f), a.view<dynd::complex<float>>());

  a = nd::empty(ndt::make_type<ndt::adapt_type>(
      ndt::make_type<dynd::complex<double>>(),
      ndt::make_fixed_bytes(sizeof(dynd::complex<double>), alignof(dynd::complex<double>)), nd::pairwise_byteswap,
      nd::pairwise_byteswap));
  a.assign(dynd::complex<double>(alias_cast<double>(0x112D4454FB210940LL), alias_cast<double>(0x002892B01FF771C2LL)));
  EXPECT_EQ(dynd::complex<double>(3.14159265358979, -1.2345678912345e12), a.view<dynd::complex<double>>());
}
