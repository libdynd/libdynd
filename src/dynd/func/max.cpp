//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/max.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/max_kernel.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::max::make()
{
  auto children = callable::make_all<max_kernel, arithmetic_ids>();

  return functional::reduction(
      functional::dispatch(ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()),
                           [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                      const ndt::type *src_tp) mutable -> callable & {
                             callable &child = children[src_tp[0].get_id()];
                             if (child.is_null()) {
                               throw runtime_error("no suitable child found for nd::sum");
                             }

                             return child;
                           }));
}

DYND_DEFAULT_DECLFUNC_GET(nd::max)

DYND_API struct nd::max nd::max;
