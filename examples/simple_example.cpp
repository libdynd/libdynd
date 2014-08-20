//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>

using namespace std;
using namespace dynd;

double myfunc(int x, float y)
{
  return x + 2.5 * y;
}

int main()
{
  try {
    nd::arrfunc af = nd::make_functor_arrfunc(myfunc);
    cout << af(3, 1.1f) << endl;
  }
  catch (const std::exception &e)
  {
    cout << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
