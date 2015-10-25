//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/sort.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Sort, 1D)
{
  std::cout << nd::sort(nd::array{1, 2, 3}) << std::endl;
  std::exit(-1);
}
