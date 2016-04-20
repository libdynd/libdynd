//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/greater_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class greater_callable : public default_instantiable_callable<greater_kernel<Arg0ID, Arg1ID>> {
  public:
    greater_callable()
        : default_instantiable_callable<greater_kernel<Arg0ID, Arg1ID>>(
              ndt::make_type<ndt::callable_type>(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)})) {}
  };

} // namespace dynd::nd
} // namespace dynd
