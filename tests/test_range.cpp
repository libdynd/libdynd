//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/range.hpp>

using namespace std;
using namespace dynd;

TEST(Range, Range) {
  EXPECT_ARRAY_EQ(nd::array({0, 1, 2, 3, 4}), nd::range({{"stop", 5}}));
  EXPECT_ARRAY_EQ(nd::array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), nd::range({{"stop", 10}}));

  EXPECT_ARRAY_EQ(nd::array({1, 2, 3, 4, 5, 6, 7, 8, 9}), nd::range({{"start", 1}, {"stop", 10}}));
  EXPECT_ARRAY_EQ(nd::array({5, 6, 7, 8, 9}), nd::range({{"start", 5}, {"stop", 10}}));
  EXPECT_ARRAY_EQ(nd::array({3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
                  nd::range({{"start", 3}, {"stop", 20}}));
  EXPECT_ARRAY_NEAR(nd::array({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
                    nd::range({{"start", 1.0}, {"stop", 10.0}}));

  EXPECT_ARRAY_EQ(nd::array({0, 2, 4, 6, 8}), nd::range({{"stop", 10}, {"step", 2}}));

  EXPECT_ARRAY_EQ(nd::array({5, 8}), nd::range({{"start", 5}, {"stop", 10}, {"step", 3}}));
  EXPECT_ARRAY_EQ(nd::array({10, 9, 8, 7, 6}), nd::range({{"start", 10}, {"stop", 5}, {"step", -1}}));
  EXPECT_ARRAY_NEAR(
      nd::array({1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5}),
      nd::range({{"start", 1.0}, {"stop", 10.0}, {"step", 0.5}}));
  EXPECT_ARRAY_NEAR(nd::array({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}),
                    nd::range({{"start", 0.0}, {"stop", 1.0}, {"step", 0.1}}));
  EXPECT_ARRAY_NEAR(
      nd::array({0.00f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.1f, 0.11f, 0.12f, 0.13f, 0.14f,
                 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f, 0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f, 0.29f,
                 0.3f, 0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f, 0.4f, 0.41f, 0.42f, 0.43f, 0.44f,
                 0.45f, 0.46f, 0.47f, 0.48f, 0.49f, 0.5f, 0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f, 0.58f, 0.59f,
                 0.6f, 0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f, 0.67f, 0.68f, 0.69f, 0.7f, 0.71f, 0.72f, 0.73f, 0.74f,
                 0.75f, 0.76f, 0.77f, 0.78f, 0.79f, 0.8f, 0.81f, 0.82f, 0.83f, 0.84f, 0.85f, 0.86f, 0.87f, 0.88f, 0.89f,
                 .9f, 0.91f, 0.92f, 0.93f, 0.94f, 0.95f, 0.96f, 0.97f, 0.98f, 0.99f}),
      nd::range({{"start", 0.0f}, {"stop", 1.0f}, {"step", 0.01f}}));
}

TEST(Range, Linspace) {
  // ...
}
