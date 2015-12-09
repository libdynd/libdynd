//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/new_adapt_type.hpp>

using namespace std;
using namespace dynd;

TEST(AdaptType, Construction)
{
  nd::callable forward = nd::functional::apply([](int x) { return ++x; });
  nd::callable inverse = nd::functional::apply([](int y) { return --y; });

  ndt::type tp = ndt::make_type<ndt::new_adapt_type>(forward, inverse);

  nd::array a = nd::empty(tp);
  a.val_assign(4);
}

TEST(AdaptType, Byteswap)
{
  nd::callable forward = nd::functional::apply([](int x) { return ++x; });
  nd::callable inverse = nd::functional::apply([](int y) { return --y; });

  ndt::type tp = ndt::make_type<ndt::new_adapt_type>(forward, inverse);

  EXPECT_EQ(tp.value_type(), ndt::type::make<int>());
  EXPECT_EQ(tp.storage_type(), ndt::type::make<int>());
}

TEST(ByteswapDType, Create)
{
  ndt::type d;

  d = ndt::byteswap_type::make(ndt::type::make<float>());
  // The value has the native byte-order type
  EXPECT_EQ(d.value_type(), ndt::type::make<float>());
  // The storage is the a bytes type with matching storage and alignment
  EXPECT_EQ(d.storage_type(), ndt::make_fixed_bytes(4, 4));
  EXPECT_TRUE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  d = ndt::byteswap_type::make(ndt::type::make<dynd::complex<double>>());
  // The value has the native byte-order type
  EXPECT_EQ(d.value_type(), ndt::type::make<dynd::complex<double>>());
  // The storage is the a bytes type with matching storage and alignment
  EXPECT_EQ(d.storage_type(), ndt::make_fixed_bytes(16, scalar_align_of<dynd::complex<double>>::value));
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // Only basic built-in types can be used to make a byteswap type
  EXPECT_THROW(d = ndt::byteswap_type::make(ndt::convert_type::make(ndt::type::make<int>(), ndt::type::make<float>())),
               dynd::type_error);
}

TEST(ByteswapDType, Basic)
{
  nd::array a;

  int16_t value16 = 0x1362;
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<int16_t>()), (char *)&value16);
  EXPECT_EQ(0x6213, a.as<int16_t>());

  int32_t value32 = 0x12345678;
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<int32_t>()), (char *)&value32);
  EXPECT_EQ(0x78563412, a.as<int32_t>());

  int64_t value64 = 0x12345678abcdef01LL;
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<int64_t>()), (char *)&value64);
  EXPECT_EQ(0x01efcdab78563412LL, a.as<int64_t>());

  value32 = 0xDA0F4940;
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<float>()), (char *)&value32);
  EXPECT_EQ(3.1415926f, a.as<float>());

  value64 = 0x112D4454FB210940LL;
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<double>()), (char *)&value64);
  EXPECT_EQ(3.14159265358979, a.as<double>());
  a = a.eval();
  EXPECT_EQ(3.14159265358979, a.as<double>());

  uint32_t value32_pair[2] = {0xDA0F4940, 0xC1B88FD3};
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<dynd::complex<float>>()), (char *)&value32_pair);
  EXPECT_EQ(dynd::complex<float>(3.1415926f, -1.23456e12f), a.as<dynd::complex<float>>());

  int64_t value64_pair[2] = {0x112D4454FB210940LL, 0x002892B01FF771C2LL};
  a = nd::make_pod_array(ndt::byteswap_type::make(ndt::type::make<dynd::complex<double>>()), (char *)&value64_pair);
  EXPECT_EQ(dynd::complex<double>(3.14159265358979, -1.2345678912345e12), a.as<dynd::complex<double>>());
}

TEST(ByteswapDType, CanonicalDType)
{
  // The canonical type of a byteswap type is always the non-swapped version
  EXPECT_EQ((ndt::type::make<float>()), (ndt::byteswap_type::make(ndt::type::make<float>()).get_canonical_type()));
}
