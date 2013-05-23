//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(ByteswapDType, Create) {
    dtype d;

    d = make_byteswap_dtype<float>();
    // The value has the native byte-order dtype
    EXPECT_EQ(d.value_dtype(), make_dtype<float>());
    // The storage is the a bytes dtype with matching storage and alignment
    EXPECT_EQ(d.storage_dtype(), make_fixedbytes_dtype(4, 4));
    EXPECT_TRUE(d.is_expression());

    d = make_byteswap_dtype<complex<double> >();
    // The value has the native byte-order dtype
    EXPECT_EQ(d.value_dtype(), make_dtype<complex<double> >());
    // The storage is the a bytes dtype with matching storage and alignment
    EXPECT_EQ(d.storage_dtype(), make_fixedbytes_dtype(16, 8));

    // Only basic built-in dtypes can be used to make a byteswap dtype
    EXPECT_THROW(d = make_byteswap_dtype(make_convert_dtype<int, float>()), runtime_error);
}

TEST(ByteswapDType, Basic) {
    ndobject a;

    int16_t value16 = 0x1362;
    a = make_pod_ndobject(make_byteswap_dtype<int16_t>(), (char *)&value16);
    EXPECT_EQ(0x6213, a.as<int16_t>());

    int32_t value32 = 0x12345678;
    a = make_pod_ndobject(make_byteswap_dtype<int32_t>(), (char *)&value32);
    EXPECT_EQ(0x78563412, a.as<int32_t>());

    int64_t value64 = 0x12345678abcdef01LL;
    a = make_pod_ndobject(make_byteswap_dtype<int64_t>(), (char *)&value64);
    EXPECT_EQ(0x01efcdab78563412LL, a.as<int64_t>());

    value32 = 0xDA0F4940;
    a = make_pod_ndobject(make_byteswap_dtype<float>(), (char *)&value32);
    EXPECT_EQ(3.1415926f, a.as<float>());

    value64 = 0x112D4454FB210940LL;
    a = make_pod_ndobject(make_byteswap_dtype<double>(), (char *)&value64);
    EXPECT_EQ(3.14159265358979, a.as<double>());
    a = a.eval();
    EXPECT_EQ(3.14159265358979, a.as<double>());

    uint32_t value32_pair[2] = {0xDA0F4940, 0xC1B88FD3};
    a = make_pod_ndobject(make_byteswap_dtype<complex<float> >(), (char *)&value32_pair);
    EXPECT_EQ(complex<float>(3.1415926f, -1.23456e12f), a.as<complex<float> >());

    int64_t value64_pair[2] = {0x112D4454FB210940LL, 0x002892B01FF771C2LL};
    a = make_pod_ndobject(make_byteswap_dtype<complex<double> >(), (char *)&value64_pair);
    EXPECT_EQ(complex<double>(3.14159265358979, -1.2345678912345e12), a.as<complex<double> >());
}

TEST(ByteswapDType, CanonicalDType) {
    // The canonical dtype of a byteswap dtype is always the non-swapped version
    EXPECT_EQ((make_dtype<float>()), (make_byteswap_dtype<float>().get_canonical_dtype()));
}

