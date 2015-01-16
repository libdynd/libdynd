//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/array_range.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
struct callable0 {
  DYND_CUDA_HOST_DEVICE int operator()(int x, int y) { return x + y; }
};
} // unnamed namespace

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
  nd::arrfunc af = nd::functional::apply<kernel_request_cuda_device, callable0>();
  //  std::cout << af(nd::array(1).to_cuda_device(),
  //  nd::array(2).to_cuda_device(), nd::array(3).to_cuda_device(),
  //    nd::array(4).to_cuda_device()) << std::endl;

  cout << "moving to CUDA device..." << endl;
  a = a.to_cuda_device();
  b = b.to_cuda_device();
#endif

  cout << "a: " << a << endl;
  cout << "b: " << b << endl;
  c = a + b;
  cout << "c: " << c << endl;

  return 0;
}
