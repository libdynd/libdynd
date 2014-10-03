//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/take_arrfunc.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/func/chain_arrfunc.hpp>
#include <dynd/func/copy_arrfunc.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/func/arrfunc_registry.hpp>

using namespace std;
using namespace dynd;

TEST(ChainArrFunc, Simple) {
  const nd::arrfunc &copy = make_copy_arrfunc();
  const nd::arrfunc &chained = make_chain_arrfunc(
      copy, func::get_regfunction("sin"), ndt::make_type<double>());
  nd::array a = nd::empty<double>();
  chained.call_out("0", a);
  EXPECT_EQ(0., a.as<double>());
  chained.call_out("1.5", a);
  EXPECT_DOUBLE_EQ(sin(1.5), a.as<double>());
  chained.call_out(3.1, a);
  EXPECT_DOUBLE_EQ(sin(3.1), a.as<double>());
}
