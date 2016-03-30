//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/min_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class min_callable : public default_instantiable_callable<min_kernel<Arg0ID>> {
  public:
    min_callable()
        : default_instantiable_callable<min_kernel<Arg0ID>>(
              ndt::callable_type::make(ndt::make_type<typename nd::min_kernel<Arg0ID>::dst_type>(), ndt::type(Arg0ID)))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
