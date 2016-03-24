//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/sum_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class sum_callable : public base_instantiable_callable<sum_kernel<Arg0ID>> {
  public:
    sum_callable()
        : base_instantiable_callable<sum_kernel<Arg0ID>>(
              ndt::callable_type::make(ndt::make_type<typename nd::sum_kernel<Arg0ID>::dst_type>(), ndt::type(Arg0ID)))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
