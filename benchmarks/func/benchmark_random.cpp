//
// Copyright (C) 2011-14 DyND Developers
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
  T a = 0;
  T b = 1;

  ndt::type tp = ndt::make_fixed_dim(10000, ndt::make_type<T>());
  std::cout << tp << std::endl;
  while (state.KeepRunning()) {
    nd::random::uniform(kwds("a", a, "b", b, "tp", tp));
  }
}

BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, int32_t);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, int64_t);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, float);
BENCHMARK_TEMPLATE(BM_Func_Random_Uniform, double);