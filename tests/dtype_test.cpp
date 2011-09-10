#include <iostream>
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
}
