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
  std::cout << ((nd::arrfunc) nd::uniform).get_array_type() << std::endl;


  ndt::type tp = ndt::make_type<int>();
  std::cout << nd::uniform(kwds("type", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("type", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("type", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("type", nd::array(tp))) << std::endl;
  std::cout << nd::uniform(kwds("type", nd::array(tp))) << std::endl;

  std::exit(-1);
}