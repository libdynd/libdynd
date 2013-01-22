//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(AlignDType, Create) {
    dtype d;

    d = make_unaligned_dtype<float>();
    // The value has the native byte-order dtype
    EXPECT_EQ(d.value_dtype(), make_dtype<float>());
    // The storage is bytes with alignment 1
    EXPECT_EQ(d.storage_dtype(), make_fixedbytes_dtype(4, 1));
    // The alignment of the dtype is 1
    EXPECT_EQ(1u, d.get_alignment());
    EXPECT_TRUE(d.is_expression());

    // TODO: Make sure it raises if an object dtype is attempted
    //EXPECT_THROW(d = make_unaligned_dtype([[some object dtype]]), runtime_error);
}

TEST(AlignDType, Basic) {
    ndobject a;

    union {
        char data[16];
        uint64_t value[2];
    } storage;

    int32_t value16 = 0x1234;
    memcpy(storage.data + 1, &value16, sizeof(value16));
    a = make_scalar_ndobject(make_unaligned_dtype<int16_t>(), storage.data + 1);
    EXPECT_EQ(0x1234, a.as<int16_t>());

    int32_t value32 = 0x12345678;
    memcpy(storage.data + 1, &value32, sizeof(value32));
    a = make_scalar_ndobject(make_unaligned_dtype<int32_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678, a.as<int32_t>());

    int64_t value64 = 0x12345678abcdef01LL;
    memcpy(storage.data + 1, &value64, sizeof(value64));
    a = make_scalar_ndobject(make_unaligned_dtype<int64_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678abcdef01LL, a.as<int64_t>());
}

TEST(AlignDType, Chained) {
    // The unaligned dtype can give back an expression dtype as the value dtype,
    // make sure that is handled properly at the dtype object level.
    dtype dt = make_unaligned_dtype(make_byteswap_dtype<int>());
    EXPECT_EQ(make_byteswap_dtype(make_dtype<int>(), make_view_dtype(make_fixedbytes_dtype(4, 4), make_fixedbytes_dtype(4, 1))), dt);
    EXPECT_EQ(make_fixedbytes_dtype(4, 1), dt.storage_dtype());
    EXPECT_EQ(make_dtype<int>(), dt.value_dtype());
}

TEST(AlignDType, CanonicalDType) {
    // The canonical dtype of an alignment result is always the aligned dtype
    EXPECT_EQ((make_dtype<float>()), (make_unaligned_dtype<float>().get_canonical_dtype()));
}