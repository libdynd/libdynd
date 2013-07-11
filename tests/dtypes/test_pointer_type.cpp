//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/pointer_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(PointerDType, VoidPointer) {
    ndt::type d;

    d = ndt::make_pointer<void>();
    EXPECT_EQ(void_pointer_type_id, d.get_type_id());
    EXPECT_EQ(void_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_NE(0u, d.get_flags()&type_flag_blockref);
    EXPECT_FALSE(d.is_expression());
}

TEST(PointerDType, PointerToBuiltIn) {
    ndt::type d;

    d = ndt::make_pointer<char>();
    EXPECT_EQ(pointer_type_id, d.get_type_id());
    EXPECT_EQ(expression_kind, d.get_kind());
    EXPECT_EQ(sizeof(void *), d.get_data_size());
    EXPECT_EQ(sizeof(void *), d.get_data_alignment());
    EXPECT_NE(0u, d.get_flags()&type_flag_blockref);
    EXPECT_EQ(ndt::make_type<char>(), d.value_type());
    EXPECT_EQ(ndt::make_type<char>(), d.p("target_dtype").as<ndt::type>());
    EXPECT_EQ(ndt::make_pointer<void>(), d.operand_type());
    EXPECT_EQ(ndt::make_pointer<void>(), d.storage_type());
    // As a special case, the pointer_type says it isn't an expression type,
    // even though it is derived from base_expression_type
    EXPECT_FALSE(d.is_expression());
}

