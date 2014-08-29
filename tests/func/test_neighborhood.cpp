//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(Neighborhood, Sum) {
  // Start with a float32 reduction arrfunc
  nd::arrfunc reduction_kernel =
      kernels::make_builtin_sum_reduction_arrfunc(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction arrfunc
  bool reduction_dimflags[2] = {true, true};
  nd::arrfunc nh_op = lift_reduction_arrfunc(
      reduction_kernel, ndt::type("strided * strided * float32"), nd::array(),
      false, 2, reduction_dimflags, true, true, false, nd::array());

  intptr_t nh_shape[2] = {3, 3};
  intptr_t nh_centre[2] = {1, 1};
  nd::arrfunc naf = make_neighborhood2d_arrfunc(nh_op, nh_shape, nh_centre);

  nd::array a =
      parse_json("4 * 4 * float32",
                 "[[1, 2, 3, 4], [1, 5, 0, 2], [0, 1, -1, 1], [1, 2, 1, 0]]");
  //nd::array mask =
  //    parse_json("3 * 3 * float32", "[[0, 1, 0], [1, 1, 1], [0, 1, 0]]");
  nd::array b = nd::empty<float[4][4]>();
  b.vals() = 0;

  naf.call_out(a.view(ndt::type("strided * strided * float32")), b);
  cout << a << endl;
  cout << b << endl;
}
