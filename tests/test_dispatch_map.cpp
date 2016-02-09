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

TEST(DispatchMap, Binary)
{
  dispatch_map<int, 2> map{{{int32_id, int64_id}, 0}, {{float32_id, int64_id}, 1}, {{scalar_kind_id, int64_id}, 2}};

  EXPECT_EQ(0, map.find({int32_id, int64_id})->second);
  EXPECT_EQ(1, map.find({float32_id, int64_id})->second);
  EXPECT_EQ(2, map.find({float64_id, int64_id})->second);
  EXPECT_EQ(2, map.find({int64_id, int64_id})->second);
  EXPECT_EQ(map.end(), map.find({int64_id, int32_id}));

  /*
    for (auto x : map) {
      const std::array<type_id_t, 2> &key = x.first;
      std::cout << "{" << key[0] << ", " << key[1] << "}" << std::endl;
    }
  */

  /*
    for (auto x : map.cache()) {
      const std::array<type_id_t, 2> &key = x.first;
      std::cout << "{" << key[0] << ", " << key[1] << "}" << std::endl;
    }
  */
}
