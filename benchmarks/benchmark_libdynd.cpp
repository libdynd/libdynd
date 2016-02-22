//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <dynd/config.hpp>

#include <benchmark/benchmark.h>

using namespace std;
using namespace dynd;

int main(int argc, char **argv)
{
  libdynd_init();
  atexit(&libdynd_cleanup);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
