//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/array_range.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/callable.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

struct callable0 {
  int operator()(int x, int y) { return x + y; }
};

int main()
{
  dynd::libdynd_init();
  atexit(&dynd::libdynd_cleanup);

  nd::array a, b, c;

  // a = 1;
  // b = 2;
  a = {1, 2, 3};
  b = {3, 5, 2};

#ifdef DYND_CUDA
  cout << "moving to CUDA device..." << endl;

  nd::callable af = nd::functional::apply<kernel_request_cuda_device, callable0>();
  std::cout << "af: " << af(nd::array(1).to_cuda_device(), nd::array(2).to_cuda_device()) << std::endl;

  a = a.to_cuda_device();
  b = b.to_cuda_device();
#endif

  cout << "a: " << a << endl;
  cout << "b: " << b << endl;
  c = a + b;
  cout << "c: " << c << endl;

  return 0;
}
