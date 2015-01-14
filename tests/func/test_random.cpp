//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

TEST(Random, Uniform)
{

  ndt::type tp = ndt::make_type<int32_t>();
  std::cout << nd::uniform(kwds("a", 0, "b", 10, "tp", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("a", 0, "b", 10, "tp", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("a", 0, "b", 10, "tp", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("a", 0, "b", 10, "tp", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("a", 0, "b", 10, "tp", nd::array(tp))) << std::endl;

//  std::exit(-1);
}