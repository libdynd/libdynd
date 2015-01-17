//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/apply.hpp>
#include <dynd/func/permute.hpp>

static void func0(double &x, int y) { x = y; }

/*
TEST(Permute, Simple)
{
  nd::arrfunc af = nd::functional::apply(&func0);
  std::cout << af.get_array_type() << std::endl;

  nd::arrfunc paf = nd::functional::permute(af, {-1, 0});
  std::cout << paf.get_array_type() << std::endl;  

  std::cout << paf(2) << std::endl;

  std::exit(-1);
}
*/