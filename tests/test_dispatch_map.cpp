//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"

#include <dynd/dispatcher.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/types/bool_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypeRegistry, Bases) {
  EXPECT_EQ(vector<type_id_t>({any_kind_id}), base_ids(scalar_kind_id));

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), base_ids(bool_kind_id));
  EXPECT_EQ(vector<type_id_t>({bool_kind_id, scalar_kind_id, any_kind_id}), base_ids(bool_id));

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), base_ids(int_kind_id));
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), base_ids(int8_id));
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), base_ids(int16_id));
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), base_ids(int32_id));
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), base_ids(int64_id));
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), base_ids(int128_id));

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), base_ids(uint_kind_id));
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}), base_ids(uint8_id));
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}), base_ids(uint16_id));
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}), base_ids(uint32_id));
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}), base_ids(uint64_id));
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}), base_ids(uint128_id));

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), base_ids(float_kind_id));
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}), base_ids(float16_id));
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}), base_ids(float32_id));
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}), base_ids(float64_id));
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}), base_ids(float128_id));

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), base_ids(complex_kind_id));
  EXPECT_EQ(vector<type_id_t>({complex_kind_id, scalar_kind_id, any_kind_id}), base_ids(complex_float32_id));
  EXPECT_EQ(vector<type_id_t>({complex_kind_id, scalar_kind_id, any_kind_id}), base_ids(complex_float64_id));

  EXPECT_EQ(vector<type_id_t>({any_kind_id}), base_ids(void_id));
}

TEST(TypeRegistry, Ambiguous) {
  EXPECT_TRUE(
      ambiguous(std::array<type_id_t, 2>{int_kind_id, int32_id}, std::array<type_id_t, 2>{int32_id, int_kind_id}));
  EXPECT_FALSE(
      ambiguous(std::array<type_id_t, 2>{int32_id, int32_id}, std::array<type_id_t, 2>{int32_id, int_kind_id}));
}

TEST(TypeRegistry, IsBaseIDOf) {
  for (type_id_t id = min_id(); id < void_id; id = static_cast<type_id_t>(id + 1)) {
    for (type_id_t base_id : base_ids(id)) {
      EXPECT_TRUE(is_base_id_of(base_id, id));
    }
  }

  EXPECT_FALSE(is_base_id_of(scalar_kind_id, fixed_dim_id));
  EXPECT_FALSE(is_base_id_of(int8_id, int32_id));
  EXPECT_FALSE(is_base_id_of(uint_kind_id, float64_id));
}

TEST(Sort, TopologicalSort) {
  vector<int> res(6);
  topological_sort({0, 1, 2, 3, 4, 5}, {{}, {}, {3}, {1}, {0, 1}, {0, 2}}, res.begin());
  EXPECT_EQ((vector<int>{5, 4, 2, 3, 1, 0}), res);
}

/*
TEST(Dispatcher, Unary) {
  dispatcher<1, int> dispatcher{
      {{scalar_kind_id}, 1}, {{int_kind_id}, 2}, {{int32_id}, 3}, {{float32_id}, 4}, {{float64_id}, 5}};
  EXPECT_EQ(1, dispatcher(bool_id));
  EXPECT_EQ(2, dispatcher(int16_id));
  EXPECT_EQ(3, dispatcher(int32_id));
  EXPECT_EQ(2, dispatcher(int64_id));
  EXPECT_EQ(4, dispatcher(float32_id));
  EXPECT_EQ(5, dispatcher(float64_id));
  EXPECT_EQ(1, dispatcher(float128_id));
  EXPECT_THROW(dispatcher(option_id), out_of_range);
  EXPECT_THROW(dispatcher(fixed_dim_id), out_of_range);

  dispatcher.insert({{any_kind_id}, 0});
  EXPECT_EQ(1, dispatcher(bool_id));
  EXPECT_EQ(2, dispatcher(int16_id));
  EXPECT_EQ(3, dispatcher(int32_id));
  EXPECT_EQ(2, dispatcher(int64_id));
  EXPECT_EQ(4, dispatcher(float32_id));
  EXPECT_EQ(5, dispatcher(float64_id));
  EXPECT_EQ(1, dispatcher(float128_id));
  EXPECT_EQ(0, dispatcher(option_id));
  EXPECT_EQ(0, dispatcher(fixed_dim_id));
}

TEST(Dispatcher, Binary) {
  dispatcher<2, int> dispatcher{{{any_kind_id, int64_id}, 0},
                                {{scalar_kind_id, int64_id}, 1},
                                {{int32_id, int64_id}, 2},
                                {{float32_id, int64_id}, 3}};

  EXPECT_EQ(2, dispatcher(int32_id, int64_id));
  EXPECT_EQ(3, dispatcher(float32_id, int64_id));
  EXPECT_EQ(1, dispatcher(float64_id, int64_id));
  EXPECT_EQ(1, dispatcher(int64_id, int64_id));
  EXPECT_EQ(0, dispatcher(option_id, int64_id));
  EXPECT_THROW(dispatcher(int64_id, float32_id), out_of_range);
}

TEST(Dispatcher, Ternary) {
  dispatcher<2, int> dispatcher{{{any_kind_id, int64_id}, 0},
                                {{scalar_kind_id, int64_id}, 1},
                                {{int32_id, int64_id}, 2},
                                {{float32_id, int64_id}, 3}};

  EXPECT_EQ(2, dispatcher(int32_id, int64_id));
  EXPECT_EQ(3, dispatcher(float32_id, int64_id));
  EXPECT_EQ(1, dispatcher(float64_id, int64_id));
  EXPECT_EQ(1, dispatcher(int64_id, int64_id));
  EXPECT_EQ(0, dispatcher(option_id, int64_id));
}
*/
