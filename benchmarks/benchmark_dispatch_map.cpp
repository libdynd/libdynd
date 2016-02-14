//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <benchmark/benchmark.h>

#include <dynd/type.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/dispatch_map.hpp>

using namespace std;
using namespace dynd;

static void BM_Unary_Dispatch(benchmark::State &state)
{
  dispatch_map<int, 1> map{{any_kind_id, 0}, {scalar_kind_id, 1}, {bool_id, 2},     {int8_id, 3},     {int16_id, 4},
                           {int32_id, 5},    {int64_id, 6},       {int128_id, 7},   {uint8_id, 8},    {uint16_id, 9},
                           {uint32_id, 10},  {uint64_id, 11},     {uint128_id, 12}, {float32_id, 13}, {float64_id, 14}};

  while (state.KeepRunning()) {
    map[static_cast<type_id_t>(state.range_x())];
  }
}

BENCHMARK(BM_Unary_Dispatch)->Range(bool_id, ndt::type_registry.size() - 1);

static void BM_Binary_Dispatch(benchmark::State &state)
{
  typedef dispatch_map<int, 2> map_type;

  map_type map{{{any_kind_id, int64_id}, 0},
               {{scalar_kind_id, int64_id}, 1},
               {{int32_id, int64_id}, 2},
               {{float32_id, int64_id}, 3}};
  while (state.KeepRunning()) {
    map[{float64_id, int64_id}];
  }
}

BENCHMARK(BM_Binary_Dispatch);
