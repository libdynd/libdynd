//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <benchmark/benchmark.h>

#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

template <typename T>
static void BM_Func_Random_Uniform(benchmark::State &state)
{
  ndt::type dst_tp = ndt::make_fixed_dim(100000, ndt::type::make<T>());
  while (state.KeepRunning()) {
    nd::random::uniform(kwds("dst_tp", dst_tp));
  }
}

BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, int32_t);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, int64_t);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, float);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, double);
