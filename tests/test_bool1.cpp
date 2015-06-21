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

#include <dynd/config.hpp>

#include <typeinfo>

using namespace std;
using namespace dynd;

/*
TEST(Builtin, Bool1)
{
  std::cout << typeid(typename std::common_type<bool, int8_t>::type).name() <<
std::endl;
  std::cout << typeid(int).name() << std::endl;

  std::cout << typeid(typename std::common_type<bool1, int8_t>::type).name() <<
std::endl;
  std::cout << typeid(int8_t).name() << std::endl;

  EXPECT_TRUE(
      (std::is_same<typename std::common_type<bool, int8_t>::type,
                    typename std::common_type<bool1, int8>::type>::value));

  std::exit(-1);
}
*/