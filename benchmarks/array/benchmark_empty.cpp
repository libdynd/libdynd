//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <benchmark/benchmark.h>

#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

template <typename T>
static void BM_UniquePtr(benchmark::State &state) {
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(std::make_unique<T>());
  }
}

BENCHMARK_TEMPLATE(BM_UniquePtr, int);
BENCHMARK_TEMPLATE(BM_UniquePtr, long long);
BENCHMARK_TEMPLATE(BM_UniquePtr, float);
BENCHMARK_TEMPLATE(BM_UniquePtr, double);

template <typename T>
static void BM_Array_BuiltinEmpty(benchmark::State &state) {
  const ndt::type &tp = ndt::make_type<T>();
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(nd::empty(tp));
  }
}
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, int);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, long);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, float);
BENCHMARK_TEMPLATE(BM_Array_BuiltinEmpty, double);

/*
template <typename T>
static void BM_Array_1DEmpty(benchmark::State &state)
{
  ndt::type tp = ndt::make_type<T>();
  while (state.KeepRunning()) {
    nd::empty(state.range_x(), tp);
  }
}
BENCHMARK_TEMPLATE(BM_Array_1DEmpty, int)->Range(2, 512);

template <typename T>
static void BM_Array_2DEmpty(benchmark::State &state)
{
  ndt::type tp = ndt::make_type<T>();
  while (state.KeepRunning()) {
    nd::empty(state.range_x(), state.range_y(), tp);
  }
}
BENCHMARK_TEMPLATE(BM_Array_2DEmpty, int)->RangePair(2, 512, 2, 512);
*/
