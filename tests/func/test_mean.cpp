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

#include <dynd/func/mean.hpp>

#include <dynd/func/apply.hpp>
#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

TEST(Mean, 1D)
{
  EXPECT_ARRAY_EQ(0.0, nd::mean(nd::array{0.0}));
  EXPECT_ARRAY_EQ(1.0, nd::mean(nd::array{1.0}));
  EXPECT_ARRAY_EQ(2.0, nd::mean(nd::array{0.0, 2.0, 4.0}));
  EXPECT_ARRAY_EQ(3.0, nd::mean(nd::array{1.0, 3.0, 5.0}));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
}

#if _MSC_VER >= 1900

TEST(Mean, 2D)
{
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{0.0, 1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0, 9.0}})));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{9.0, 8.0, 7.0, 6.0, 5.0}, {4.0, 3.0, 2.0, 1.0, 0.0}})));
}

#endif

TEST(Plugin, Untitled)
{
  // nd::callable f = nd::functional::apply([](int x, int y) { return x + y; });
  nd::callable f = nd::add::children[int32_type_id][int32_type_id];

  std::cout << f(1, 2) << std::endl;
  std::cout << (f.get_single().func == NULL) << std::endl;
  std::cout << (f.get_single().ir == NULL) << std::endl;
  std::cout << std::string(const_cast<char *>(f.get_single().ir)) << std::endl;
  std::exit(-1);
}
