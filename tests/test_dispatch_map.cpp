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

#include <dynd/dispatch_map.hpp>

using namespace std;
using namespace dynd;

TEST(Sort, TopologicalSort)
{
  std::vector<std::vector<intptr_t>> edges{{}, {}, {3}, {1}, {0, 1}, {0, 2}};

  std::vector<intptr_t> res(6);
  topological_sort(std::vector<intptr_t>{0, 1, 2, 3, 4, 5}, edges, res.begin());

  EXPECT_EQ(5, res[0]);
  EXPECT_EQ(4, res[1]);
  EXPECT_EQ(2, res[2]);
  EXPECT_EQ(3, res[3]);
  EXPECT_EQ(1, res[4]);
  EXPECT_EQ(0, res[5]);
}

TEST(Dispatch, Map)
{
  // body

  dispatch_map<int, 2> m{{{{int32_id, int64_id}}, 3}, {{{int32_id, float64_id}}, 3}};

//  m.at({{int32_id, int64_id}});
}
