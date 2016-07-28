//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

struct push_back_size {
  template <typename T>
  void on_each(vector<size_t> &res) const {
    res.push_back(sizeof(T));
  }
};

TEST(TypeSequence, ForEach) {
  typedef type_sequence<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double> S;

  vector<size_t> res;
  for_each<S>(push_back_size(), res);
  EXPECT_EQ(sizeof(int8_t), res[0]);
  EXPECT_EQ(sizeof(int16_t), res[1]);
  EXPECT_EQ(sizeof(int32_t), res[2]);
  EXPECT_EQ(sizeof(int64_t), res[3]);
  EXPECT_EQ(sizeof(uint8_t), res[4]);
  EXPECT_EQ(sizeof(uint16_t), res[5]);
  EXPECT_EQ(sizeof(uint32_t), res[6]);
  EXPECT_EQ(sizeof(uint64_t), res[7]);
  EXPECT_EQ(sizeof(float), res[8]);
  EXPECT_EQ(sizeof(double), res[9]);
}
