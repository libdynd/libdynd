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
#include <dynd/type_registry.hpp>
#include <dynd/types/type_id.hpp>

using namespace std;
using namespace dynd;

TEST(TypeRegistry, Bases)
{
  EXPECT_EQ(vector<type_id_t>({any_kind_id}), ndt::type_registry[scalar_kind_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), ndt::type_registry[bool_kind_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({bool_kind_id, scalar_kind_id, any_kind_id}), ndt::type_registry[bool_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), ndt::type_registry[int_kind_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), ndt::type_registry[int8_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), ndt::type_registry[int16_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), ndt::type_registry[int32_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}), ndt::type_registry[int64_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({int_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[int128_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), ndt::type_registry[uint_kind_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[uint8_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[uint16_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[uint32_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[uint64_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({uint_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[uint128_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), ndt::type_registry[float_kind_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[float16_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[float32_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[float64_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({float_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[float128_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({scalar_kind_id, any_kind_id}), ndt::type_registry[complex_kind_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({complex_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[complex_float32_id].get_base_ids());
  EXPECT_EQ(vector<type_id_t>({complex_kind_id, scalar_kind_id, any_kind_id}),
            ndt::type_registry[complex_float64_id].get_base_ids());

  EXPECT_EQ(vector<type_id_t>({any_kind_id}), ndt::type_registry[void_id].get_base_ids());
}

TEST(TypeRegistry, IsBaseIDOf)
{
  for (type_id_t id = ndt::type_registry.min(); id < void_id; id = static_cast<type_id_t>(id + 1)) {
    for (type_id_t base_id : ndt::type_registry[id].get_base_ids()) {
      EXPECT_TRUE(is_base_id_of(base_id, id));
    }
  }

  EXPECT_FALSE(is_base_id_of(scalar_kind_id, fixed_dim_id));
  EXPECT_FALSE(is_base_id_of(int8_id, int32_id));
  EXPECT_FALSE(is_base_id_of(uint_kind_id, float64_id));
}

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

/*
TEST(DispatchMap, Unary)
{
  typedef dispatch_map<int, 1> map_type;

  map_type map{{any_kind_id, 0}, {scalar_kind_id, 1}, {int32_id, 2}, {float32_id, 3}};
  EXPECT_EQ(map_type::value_type(int32_id, 2), *map.find(int32_id));
  EXPECT_EQ(map_type::value_type(float32_id, 3), *map.find(float32_id));
  EXPECT_EQ(map_type::value_type(scalar_kind_id, 1), *map.find(float64_id));
  EXPECT_EQ(map_type::value_type(scalar_kind_id, 1), *map.find(int64_id));
  EXPECT_EQ(map_type::value_type(any_kind_id, 0), *map.find(option_id));
}

TEST(DispatchMap, Binary)
{
  typedef dispatch_map<int, 2> map_type;

  map_type map{{{any_kind_id, int64_id}, 0},
               {{scalar_kind_id, int64_id}, 1},
               {{int32_id, int64_id}, 2},
               {{float32_id, int64_id}, 3}};
  EXPECT_EQ(map_type::value_type({int32_id, int64_id}, 2), *map.find({int32_id, int64_id}));
  EXPECT_EQ(map_type::value_type({float32_id, int64_id}, 3), *map.find({float32_id, int64_id}));
  EXPECT_EQ(map_type::value_type({scalar_kind_id, int64_id}, 1), *map.find({float64_id, int64_id}));
  EXPECT_EQ(map_type::value_type({scalar_kind_id, int64_id}, 1), *map.find({int64_id, int64_id}));
  EXPECT_EQ(map_type::value_type({any_kind_id, int64_id}, 0), *map.find({option_id, int64_id}));
  EXPECT_EQ(map.end(), map.find({int64_id, int32_id}));
}
*/
