//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/pointer_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(PointerDType, VoidPointer) {
    dtype d;

    d = make_pointer_dtype<void>();
    EXPECT_EQ(void_pointer_type_id, d.get_type_id());
    EXPECT_EQ(void_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(blockref_memory_management, d.get_memory_management());
    EXPECT_FALSE(d.is_expression());
}

TEST(PointerDType, PointerToBuiltIn) {
    dtype d;

    d = make_pointer_dtype<char>();
    EXPECT_EQ(pointer_type_id, d.get_type_id());
    EXPECT_EQ(expression_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_alignment());
    EXPECT_EQ(blockref_memory_management, d.get_memory_management());
    EXPECT_EQ(make_dtype<char>(), d.value_dtype());
    EXPECT_EQ(make_pointer_dtype<void>(), d.operand_dtype());
    EXPECT_EQ(make_pointer_dtype<void>(), d.storage_dtype());
    // As a special case, the pointer_dtype says it isn't an expression dtype,
    // even though it is derived from base_expression_dtype
    EXPECT_FALSE(d.is_expression());
}

