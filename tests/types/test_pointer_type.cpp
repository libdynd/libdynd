//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/pointer_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(PointerType, VoidPointer) {
    ndt::type d;

    d = ndt::make_pointer<void>();
    EXPECT_EQ(void_pointer_type_id, d.get_type_id());
    EXPECT_EQ(void_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_NE(0u, d.get_flags()&type_flag_blockref);
    EXPECT_FALSE(d.is_expression());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(PointerType, PointerToBuiltIn) {
    ndt::type d;

    d = ndt::make_pointer<char>();
    EXPECT_EQ(pointer_type_id, d.get_type_id());
    EXPECT_EQ(expr_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_NE(0u, d.get_flags()&type_flag_blockref);
    EXPECT_EQ(ndt::make_type<char>(), d.value_type());
    EXPECT_EQ(ndt::make_type<char>(), d.p("target_type").as<ndt::type>());
    EXPECT_EQ(ndt::make_pointer<void>(), d.operand_type());
    EXPECT_EQ(ndt::make_pointer<void>(), d.storage_type());
    // As a special case, the pointer_type says it isn't an expression type,
    // even though it is derived from base_expr_type
    EXPECT_FALSE(d.is_expression());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(PointerType, IsTypeSubarray) {
    EXPECT_TRUE(ndt::type("pointer[int32]").is_type_subarray(ndt::type("pointer[int32]")));
    EXPECT_TRUE(ndt::type("strided * 3 * pointer[int32]").is_type_subarray(ndt::type("3 * pointer[int32]")));
    EXPECT_TRUE(ndt::type("3 * 10 * pointer[int32]").is_type_subarray(ndt::type("pointer[int32]")));
    EXPECT_TRUE(ndt::type("pointer[int32]").is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::type("pointer[int32]")));
}
