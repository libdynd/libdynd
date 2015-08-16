//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/multidispatch.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/kernels/sum_kernel.hpp>
#include <dynd/func/reduction.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::sum::make()
{
  auto children = callable::make_all<
      sum_kernel, type_id_sequence<int32_type_id, int64_type_id>>(0);

  return functional::reduction(functional::multidispatch(
      ndt::type("(Any) -> Any"),
      [children](const ndt::type & DYND_UNUSED(dst_tp),
                 intptr_t DYND_UNUSED(nsrc),
                 const ndt::type * src_tp) mutable->callable &
  {
        callable &child = children[src_tp[0].get_dtype().get_type_id()];
        if (child.is_null()) {
          throw runtime_error("no suitable child found for nd::sum");
        }
        return child;
      },
      0));
}

struct nd::sum nd::sum;
