//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/types/any_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(AnyKindType, Constructor) {
  ndt::type any_kind_tp = ndt::make_type<ndt::any_kind_type>();
  EXPECT_EQ(any_kind_id, any_kind_tp.get_id());
  EXPECT_EQ(0, any_kind_tp.get_data_size());
  EXPECT_EQ(1, any_kind_tp.get_data_alignment());
  EXPECT_FALSE(any_kind_tp.is_expression());
  EXPECT_TRUE(any_kind_tp.is_symbolic());
  EXPECT_EQ(any_kind_tp, ndt::type(any_kind_tp.str())); // Round trip through a string
}

TEST(SymbolicTypes, AnySym) {
  ndt::type tp;

  tp = ndt::make_type<ndt::any_kind_type>();
  EXPECT_EQ(any_kind_id, tp.get_id());
  EXPECT_EQ("Any", tp.str());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  // The "Any" type's variadic-ness should propagate through dimension types
  tp = ndt::type("3 * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("var * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("Fixed * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("?3 * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("pointer[3 * Any]");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  // The "Any" type's variadic-ness should not propagate through struct/tuple
  // types
  tp = ndt::type("(Any, Any)");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("{x: Any, y: Any}");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(AnyKindType, IDOf) { EXPECT_EQ(any_kind_id, ndt::id_of<ndt::any_kind_type>::value); }
