#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/unaligned_dtype.hpp>

using namespace std;
using namespace dnd;

TEST(UnalignedDType, Create) {
    dtype d;

    d = make_unaligned_dtype<float>();
    // The value has the native byte-order dtype
    EXPECT_EQ(d.value_dtype(), make_dtype<float>());
    // The storage is the byteswap dtype itself
    EXPECT_EQ(d.storage_dtype(), d);
    // The alignment of the dtype is 1
    EXPECT_EQ(1, d.alignment());

    // TODO: Make sure it raises if an object dtype is attempted
    //EXPECT_THROW(d = make_unaligned_dtype([[some object dtype]]), runtime_error);
}

TEST(UnalignedDType, Basic) {
    ndarray a;

    union {
        char data[16];
        uint64_t value[2];
    } storage;

    int32_t value16 = 0x1234;
    memcpy(storage.data + 1, &value16, sizeof(value16));
    a = ndarray(make_unaligned_dtype<int16_t>(), storage.data + 1);
    EXPECT_EQ(0x1234, a.as<int16_t>());

    int32_t value32 = 0x12345678;
    memcpy(storage.data + 1, &value32, sizeof(value32));
    a = ndarray(make_unaligned_dtype<int32_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678, a.as<int32_t>());

    int64_t value64 = 0x12345678abcdef01LL;
    memcpy(storage.data + 1, &value64, sizeof(value64));
    a = ndarray(make_unaligned_dtype<int64_t>(), storage.data + 1);
    EXPECT_EQ(0x12345678abcdef01LL, a.as<int64_t>());
    // This should raise an exception if the
    // preprocessor symbol DND_ALIGNMENT_ASSERTIONS is defined,
    // (or segfault on platforms that don't support unaligned access)
    //a = ndarray(make_dtype<int64_t>(), storage.data + 1);
}
