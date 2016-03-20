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

TEST(Dispatcher, Unary)
{
  dispatcher<int> dispatcher{{{any_kind_id}, 0}, {{scalar_kind_id}, 1}, {{int_kind_id}, 2},
                             {{int32_id}, 3},    {{float32_id}, 4},     {{float64_id}, 5}};

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

TEST(Dispatcher, Binary)
{
  dispatcher<int> dispatcher{{{any_kind_id, int64_id}, 0},
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

TEST(Dispatcher, Ternary)
{
  dispatcher<int> dispatcher{{{any_kind_id, int64_id}, 0},
                             {{scalar_kind_id, int64_id}, 1},
                             {{int32_id, int64_id}, 2},
                             {{float32_id, int64_id}, 3}};

  EXPECT_EQ(2, dispatcher(int32_id, int64_id));
  EXPECT_EQ(3, dispatcher(float32_id, int64_id));
  EXPECT_EQ(1, dispatcher(float64_id, int64_id));
  EXPECT_EQ(1, dispatcher(int64_id, int64_id));
  EXPECT_EQ(0, dispatcher(option_id, int64_id));
}
