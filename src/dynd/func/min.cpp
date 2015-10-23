//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/min_kernel.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/func/min.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::min::make()
{
  auto children = callable::make_all<min_kernel, arithmetic_type_ids>();

  return functional::reduction(
      functional::multidispatch(ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()),
                                [children](const ndt::type & DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                           const ndt::type * src_tp) mutable->callable &
  {
                                  callable &child = children[src_tp[0].get_type_id()];
                                  if (child.is_null()) {
                                    throw runtime_error("no suitable child found for nd::min");
                                  }

                                  return child;
                                },
                                data_size_max(children)));
}

DYND_API struct nd::min nd::min;
