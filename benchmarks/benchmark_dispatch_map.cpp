//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>

#include <benchmark/benchmark.h>

#include <dispatcher.hpp>

#include <dynd/type.hpp>
#include <dynd/type_registry.hpp>
#include <dynd/dispatch_map.hpp>

using namespace std;
using namespace dynd;

template <size_t N>
class DispatchFixture : public ::benchmark::Fixture {
public:
  vector<pair<array<type_id_t, N>, array<type_id_t, N>>> pairs;

  void SetUp(const benchmark::State &state)
  {
    pairs.resize(state.range_x());

    default_random_engine generator;
    uniform_int_distribution<underlying_type_t<type_id_t>> d(bool_id, callable_id);

    for (auto &pair : pairs) {
      for (size_t i = 0; i < N; ++i) {
        pair.first[i] = static_cast<type_id_t>(d(generator));
        pair.second[i] = static_cast<type_id_t>(d(generator));
      }
    }
  }
};

template <>
class DispatchFixture<1> : public ::benchmark::Fixture {
public:
  vector<pair<type_id_t, type_id_t>> pairs;

  void SetUp(const benchmark::State &state)
  {
    pairs.resize(state.range_x());

    default_random_engine generator;
    uniform_int_distribution<underlying_type_t<type_id_t>> d(bool_id, callable_id);

    for (auto &pair : pairs) {
      pair.first = static_cast<type_id_t>(d(generator));
      pair.second = static_cast<type_id_t>(d(generator));
    }
  }
};

typedef DispatchFixture<1> X;

BENCHMARK_DEFINE_F(X, BM_IsBaseIDOf)(benchmark::State &state)
{
  while (state.KeepRunning()) {
    for (const auto &pair : pairs) {
      benchmark::DoNotOptimize(is_base_id_of(pair.first, pair.second));
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());
}

// BENCHMARK_REGISTER_F(X, BM_IsBaseIDOf)->Arg(100)->Arg(1000)->Arg(10000);

typedef DispatchFixture<1> UnaryDispatchFixture;
typedef DispatchFixture<2> BinaryDispatchFixture;

BENCHMARK_DEFINE_F(BinaryDispatchFixture, BM_Supercedes)(benchmark::State &state)
{
  while (state.KeepRunning()) {
    for (const auto &pair : pairs) {
      benchmark::DoNotOptimize(supercedes(pair.first, pair.second));
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());
}

// BENCHMARK_REGISTER_F(BinaryDispatchFixture, BM_Supercedes)->Arg(10)->Arg(100)->Arg(1000);

BENCHMARK_DEFINE_F(UnaryDispatchFixture, BM_UnaryDispatch)(benchmark::State &state)
{
  dispatch_map<int, 1> map{{any_kind_id, 0}, {scalar_kind_id, 1}, {bool_id, 2},     {int8_id, 3},     {int16_id, 4},
                           {int32_id, 5},    {int64_id, 6},       {int128_id, 7},   {uint8_id, 8},    {uint16_id, 9},
                           {uint32_id, 10},  {uint64_id, 11},     {uint128_id, 12}, {float32_id, 13}, {float64_id, 14}};
  while (state.KeepRunning()) {
    for (const auto &pair : pairs) {
      benchmark::DoNotOptimize(map[pair.first]);
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());
}

BENCHMARK_REGISTER_F(UnaryDispatchFixture, BM_UnaryDispatch)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_VirtualDispatch(benchmark::State &state)
{
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize((*items[0])());
  }
}

BENCHMARK(BM_VirtualDispatch);

BENCHMARK_DEFINE_F(BinaryDispatchFixture, BM_BinaryDispatch)(benchmark::State &state)
{
  typedef dispatch_map<int, 2> map_type;

  map_type map{{{any_kind_id, int64_id}, 0},
               {{scalar_kind_id, int64_id}, 1},
               {{int32_id, int64_id}, 2},
               {{float32_id, int64_id}, 3}};
  while (state.KeepRunning()) {
    for (const auto &pair : pairs) {
      benchmark::DoNotOptimize(map.find(pair.first));
    }
  }
  state.SetItemsProcessed(state.iterations() * state.range_x());
}

// BENCHMARK_REGISTER_F(BinaryDispatchFixture, BM_BinaryDispatch)->Arg(1000);
