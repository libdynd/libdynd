//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../dynd_assertions.hpp"

#include <dynd/struct.hpp>

using namespace std;
using namespace dynd;


TEST(Struct, FieldAccess)
{
  nd::array struct_arg = nd::as_struct({{"x", 7}, {"y", 0.5}});

  // XXX: with return type void: exception "no child found"
  // XXX: with return type Any: exception "The dynd type Any is not concrete as required"
  // EXPECT_EQ(7, nd::field_access(struct_arg, "x"));
  // EXPECT_EQ(0.5, nd::field_access(struct_arg, "y"));
}


