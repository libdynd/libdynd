//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <benchmark/benchmark.h>

#include <dynd/func/apply.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

int func(int x, int y) {
  return x + y;
}

/*
static void BM_Func_Call(benchmark::State &state)
{
  int a = 10;
  int b = 11;
  int c;
  while (state.KeepRunning()) {
    c = func(a, b);
  }
}

BENCHMARK(BM_Func_Call);
*/

static void BM_Func_Apply_Function(benchmark::State &state)
{
  nd::callable af = nd::functional::apply<decltype(&func), &func>();

  nd::array a = 10;
  nd::array b = 11;
  nd::array c = nd::empty(af.get_type()->get_return_type());
  while (state.KeepRunning()) {
    af(a, b, kwds("dst", c));
  }
}

BENCHMARK(BM_Func_Apply_Function);

static void BM_Func_Apply_Callable(benchmark::State &state)
{
  nd::callable af = nd::functional::apply(&func);

  nd::array a = 10;
  nd::array b = 11;
  nd::array c = nd::empty(af.get_type()->get_return_type());
  while (state.KeepRunning()) {
    af(a, b, kwds("dst", c));
  }
}

BENCHMARK(BM_Func_Apply_Callable);
