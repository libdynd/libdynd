//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <benchmark/benchmark.h>

#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

template <typename T>
static void BM_Array_BuiltinEmpty(benchmark::State &state)
{
  ndt::type tp = ndt::type::make<T>();
  while (state.KeepRunning()) {
    nd::empty(tp);
  }
}
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, int32_t);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, int64_t);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, float);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, double);

template <typename T>
static void BM_Array_1DEmpty(benchmark::State &state)
{
  ndt::type tp = ndt::type::make<T>();
  while (state.KeepRunning()) {
    nd::empty(state.range_x(), tp);
  }
}
BENCHMARK_TEMPLATE(BM_Array_1DEmpty, int)->Range(2, 512);

template <typename T>
static void BM_Array_2DEmpty(benchmark::State &state)
{
  ndt::type tp = ndt::type::make<T>();
  while (state.KeepRunning()) {
    nd::empty(state.range_x(), state.range_y(), tp);
  }
}
BENCHMARK_TEMPLATE(BM_Array_2DEmpty, int)->RangePair(2, 512, 2, 512);
