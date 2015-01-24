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

static void BM_Func_Arithmetic_Add(benchmark::State &state)
{
  ndt::type dst_tp = ndt::make_fixed_dim(100000, ndt::make_type<double>());

  nd::array a = nd::random::uniform(kwds("dst_tp", ndt::make_fixed_dim(1000, ndt::make_type<double>())));
  nd::array b = nd::random::uniform(kwds("dst_tp", ndt::make_fixed_dim(1000, ndt::make_type<double>())));
  while (state.KeepRunning()) {
    a + b;
  }
}

BENCHMARK(BM_Func_Arithmetic_Add);