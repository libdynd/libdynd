//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/io.hpp>

using namespace std;
using namespace dynd;

TEST(IO, Serialize)
{
  std::cout << nd::serialize << std::endl;

  std::cout << "--" << std::endl;
  nd::array a = nd::serialize({3}, {{"identity", nd::empty(ndt::type("bytes"))}});
  std::cout << "--" << std::endl;

  std::cout << a << std::endl;

//  std::exit(-1);
}
