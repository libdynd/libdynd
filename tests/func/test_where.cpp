//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include "../dynd_assertions.hpp"
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

/*
TEST(Where, Untitled) {
  nd::callable f = nd::functional::where([](int DYND_UNUSED(x)) { return false; });

  std::cout << f << std::endl;
  nd::array res = f(nd::array{0, 1, 2, 3, 4});
  std::cout << res << std::endl;

  std::exit(-1);
}
*/
