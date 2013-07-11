//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>

using namespace std;
using namespace dynd;

TEST(ByteswapDType, Create) {
    ndt::type d;

    d = ndt::make_byteswap<float>();
    // The value has the native byte-order type
    EXPECT_EQ(d.value_type(), ndt::make_type<float>());
    // The storage is the a bytes type with matching storage and alignment
    EXPECT_EQ(d.storage_type(), ndt::make_fixedbytes(4, 4));
    EXPECT_TRUE(d.is_expression());

    d = ndt::make_byteswap<complex<double> >();
    // The value has the native byte-order type
    EXPECT_EQ(d.value_type(), ndt::make_type<complex<double> >());
    // The storage is the a bytes type with matching storage and alignment
    EXPECT_EQ(d.storage_type(), ndt::make_fixedbytes(16, scalar_align_of<complex<double> >::value));

    // Only basic built-in types can be used to make a byteswap type
    EXPECT_THROW(d = ndt::make_byteswap(ndt::make_convert<int, float>()), runtime_error);
}

TEST(ByteswapDType, Basic) {
    nd::array a;

    int16_t value16 = 0x1362;
    a = nd::make_pod_array(ndt::make_byteswap<int16_t>(), (char *)&value16);
    EXPECT_EQ(0x6213, a.as<int16_t>());

    int32_t value32 = 0x12345678;
    a = nd::make_pod_array(ndt::make_byteswap<int32_t>(), (char *)&value32);
    EXPECT_EQ(0x78563412, a.as<int32_t>());

    int64_t value64 = 0x12345678abcdef01LL;
    a = nd::make_pod_array(ndt::make_byteswap<int64_t>(), (char *)&value64);
    EXPECT_EQ(0x01efcdab78563412LL, a.as<int64_t>());

    value32 = 0xDA0F4940;
    a = nd::make_pod_array(ndt::make_byteswap<float>(), (char *)&value32);
    EXPECT_EQ(3.1415926f, a.as<float>());

    value64 = 0x112D4454FB210940LL;
    a = nd::make_pod_array(ndt::make_byteswap<double>(), (char *)&value64);
    EXPECT_EQ(3.14159265358979, a.as<double>());
    a = a.eval();
    EXPECT_EQ(3.14159265358979, a.as<double>());

    uint32_t value32_pair[2] = {0xDA0F4940, 0xC1B88FD3};
    a = nd::make_pod_array(ndt::make_byteswap<complex<float> >(), (char *)&value32_pair);
    EXPECT_EQ(complex<float>(3.1415926f, -1.23456e12f), a.as<complex<float> >());

    int64_t value64_pair[2] = {0x112D4454FB210940LL, 0x002892B01FF771C2LL};
    a = nd::make_pod_array(ndt::make_byteswap<complex<double> >(), (char *)&value64_pair);
    EXPECT_EQ(complex<double>(3.14159265358979, -1.2345678912345e12), a.as<complex<double> >());
}

TEST(ByteswapDType, CanonicalDType) {
    // The canonical type of a byteswap type is always the non-swapped version
    EXPECT_EQ((ndt::make_type<float>()), (ndt::make_byteswap<float>().get_canonical_type()));
}

