#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/dtype.hpp"

using namespace std;
using namespace dnd;

TEST(DType, BasicConstructor) {
    dtype d;

    // Default-constructed dtype properties
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(generic_type_id, d.type_id());
    EXPECT_EQ(generic_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(0u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // bool dtype
    d = dtype(bool_type_id);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(bool_type_id, d.type_id());
    EXPECT_EQ(bool_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int8 dtype
    d = dtype(int8_type_id);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int8_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int16 dtype
    d = dtype(int16_type_id);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int16_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(2, d.alignment());
    EXPECT_EQ(2u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int32 dtype
    d = dtype(int32_type_id, 4);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(4, d.alignment());
    EXPECT_EQ(4u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int
    d = make_dtype<int>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(int), d.alignment());
    EXPECT_EQ(sizeof(int), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // long
    d = make_dtype<long>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(long), d.alignment());
    EXPECT_EQ(sizeof(long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // long long
    d = make_dtype<long long>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int64_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ((int)sizeof(long long), d.alignment());
    EXPECT_EQ(sizeof(long long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned int
    d = make_dtype<unsigned int>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint32_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned int), d.alignment());
    EXPECT_EQ(sizeof(unsigned int), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long
    d = make_dtype<unsigned long>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long long
    d = make_dtype<unsigned long long>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint64_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ((int)sizeof(unsigned long long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // float
    d = make_dtype<float>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(float32_type_id, d.type_id());
    EXPECT_EQ(float_kind, d.kind());
    EXPECT_EQ((int)sizeof(float), d.alignment());
    EXPECT_EQ(sizeof(float), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // double
    d = make_dtype<double>();
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(float64_type_id, d.type_id());
    EXPECT_EQ(float_kind, d.kind());
    EXPECT_EQ((int)sizeof(double), d.alignment());
    EXPECT_EQ(sizeof(double), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // For fixed-size types, can't specify a bad size
    EXPECT_THROW(dtype(int32_type_id, 8), runtime_error);
}

TEST(DType, UTF8Constructor) {
    dtype d;

    // UTF8 with various string sizes
    d = dtype(utf8_type_id, 3);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(utf8_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(3u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    d = dtype(utf8_type_id, 129);
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(utf8_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(129u, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // If the size is big enough, needs to use an extended_dtype
    d = dtype(utf8_type_id, ((intptr_t)1 << (8*sizeof(intptr_t)-2)));
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(utf8_type_id, d.type_id());
    EXPECT_EQ(string_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(((uintptr_t)1 << (8*sizeof(intptr_t)-2)), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // Can't specify a negative size (dtype accepts a uintptr_t now)
    //EXPECT_THROW(dtype(utf8_type_id, -13), runtime_error);
}
