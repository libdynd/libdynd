//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/pointer_type.hpp>
#include <dynd/array.hpp>
#include <dynd/callable_registry.hpp>

using namespace std;
using namespace dynd;

TEST(PointerType, PointerToBuiltIn)
{
  ndt::type d;

  d = ndt::pointer_type::make(ndt::make_type<char>());
  EXPECT_EQ(pointer_id, d.get_id());
  EXPECT_EQ(any_kind_id, d.get_base_id());
  EXPECT_EQ(sizeof(void *), d.get_data_size());
  EXPECT_EQ(sizeof(void *), d.get_data_alignment());
  EXPECT_NE(0u, d.get_flags() & type_flag_blockref);
  EXPECT_EQ(ndt::make_type<char>(), *d.p("target_type").type);
  // As a special case, the pointer_type says it isn't an expression type,
  // even though it is derived from base_expr_type
  EXPECT_FALSE(d.is_expression());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(PointerType, IsTypeSubarray)
{
  EXPECT_TRUE(ndt::type("pointer[int32]").is_type_subarray(ndt::type("pointer[int32]")));
  EXPECT_TRUE(ndt::type("Fixed * 3 * pointer[int32]").is_type_subarray(ndt::type("3 * pointer[int32]")));
  EXPECT_TRUE(ndt::type("3 * 10 * pointer[int32]").is_type_subarray(ndt::type("pointer[int32]")));
  EXPECT_TRUE(ndt::type("pointer[int32]").is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::type("pointer[int32]")));
}
