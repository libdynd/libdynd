//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/dtypes/type_alignment.hpp>
#include <dynd/dtypes/byteswap_type.hpp>
#include <dynd/dtypes/fixedbytes_type.hpp>

using namespace std;
using namespace dynd;

TEST(AlignDType, Create) {
    ndt::type d;

    d = ndt::make_unaligned<float>();
    // The value has the native byte-order dtype
    EXPECT_EQ(d.value_type(), ndt::make_type<float>());
    // The storage is bytes with alignment 1
    EXPECT_EQ(d.storage_type(), ndt::make_fixedbytes(4, 1));
    // The alignment of the dtype is 1
    EXPECT_EQ(1u, d.get_data_alignment());
    EXPECT_TRUE(d.is_expression());

    // TODO: Make sure it raises if an object dtype is attempted
    //EXPECT_THROW(d = make_unaligned([[some object dtype]]), runtime_error);
}

TEST(AlignDType, Basic) {
    nd::array a;

    union {
        char data[16];
        uint64_t value[2];
    } storage;

    int32_t value16 = 0x1234;
    memcpy(storage.data + 1, &value16, sizeof(value16));
    a = nd::make_pod_array(ndt::make_unaligned<int16_t>(), storage.data + 1);
    EXPECT_EQ(0x1234, a.as<int16_t>());

    int32_t value32 = 0x12345678;
    memcpy(storage.data + 1, &value32, sizeof(value32));
    a = nd::make_pod_array(ndt::make_unaligned<int32_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678, a.as<int32_t>());

    int64_t value64 = 0x12345678abcdef01LL;
    memcpy(storage.data + 1, &value64, sizeof(value64));
    a = nd::make_pod_array(ndt::make_unaligned<int64_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678abcdef01LL, a.as<int64_t>());
}

TEST(AlignDType, Chained) {
    // The unaligned dtype can give back an expression type as the value dtype,
    // make sure that is handled properly at the dtype object level.
    ndt::type dt = make_unaligned(ndt::make_byteswap<int>());
    EXPECT_EQ(ndt::make_byteswap(ndt::make_type<int>(), ndt::make_view(ndt::make_fixedbytes(4, 4), ndt::make_fixedbytes(4, 1))), dt);
    EXPECT_EQ(ndt::make_fixedbytes(4, 1), dt.storage_type());
    EXPECT_EQ(ndt::make_type<int>(), dt.value_type());
}

TEST(AlignDType, CanonicalDType) {
    // The canonical type of an alignment result is always the aligned dtype
    EXPECT_EQ((ndt::make_type<float>()), (ndt::make_unaligned<float>().get_canonical_type()));
}

