#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "dnd/dtype.hpp"

using namespace std;
using namespace dnd;

TEST(DType, BasicConstructor) {
    dtype d;

    // Default-constructed dtype properties
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(generic_type_id, d.type_id());
    EXPECT_EQ(generic_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(0, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // Boolean dtype
    d = dtype(bool_type_id);
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(bool_type_id, d.type_id());
    EXPECT_EQ(bool_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int8 dtype
    d = dtype(int8_type_id);
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int8_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(1, d.alignment());
    EXPECT_EQ(1, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int16 dtype
    d = dtype(int16_type_id);
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int16_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(2, d.alignment());
    EXPECT_EQ(2, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int32 dtype
    d = dtype(int32_type_id, 4);
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(4, d.alignment());
    EXPECT_EQ(4, d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // int
    d = mkdtype<int>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int32_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(sizeof(int), d.alignment());
    EXPECT_EQ(sizeof(int), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // long
    d = mkdtype<long>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(sizeof(long), d.alignment());
    EXPECT_EQ(sizeof(long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // long long
    d = mkdtype<long long>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(int64_type_id, d.type_id());
    EXPECT_EQ(int_kind, d.kind());
    EXPECT_EQ(sizeof(long long), d.alignment());
    EXPECT_EQ(sizeof(long long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned int
    d = mkdtype<unsigned int>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint32_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ(sizeof(unsigned int), d.alignment());
    EXPECT_EQ(sizeof(unsigned int), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long
    d = mkdtype<unsigned long>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ(sizeof(unsigned long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // unsigned long long
    d = mkdtype<unsigned long long>();
    EXPECT_TRUE(d.is_trivial());
    EXPECT_FALSE(d.is_byteswapped());
    EXPECT_EQ(uint64_type_id, d.type_id());
    EXPECT_EQ(uint_kind, d.kind());
    EXPECT_EQ(sizeof(unsigned long long), d.alignment());
    EXPECT_EQ(sizeof(unsigned long long), d.itemsize());
    EXPECT_EQ(NULL, d.extended());

    // For fixed-size types, can't specify a bad size
    EXPECT_THROW(dtype(int32_type_id, 8), runtime_error);
}
