//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/less_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class less_callable : public default_instantiable_callable<less_kernel<Arg0ID, Arg1ID>> {
  public:
    less_callable()
        : default_instantiable_callable<less_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)}))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
