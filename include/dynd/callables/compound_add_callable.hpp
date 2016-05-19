//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/compound_add_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename ReturnType, typename Arg0Type>
  class compound_add_callable : public default_instantiable_callable<compound_add_kernel<ReturnType, Arg0Type>> {
  public:
    compound_add_callable()
        : default_instantiable_callable<compound_add_kernel<ReturnType, Arg0Type>>(ndt::make_type<ndt::callable_type>(
              ndt::type(type_id_of<ReturnType>::value), {ndt::type(type_id_of<Arg0Type>::value)})) {}
  };

} // namespace dynd::nd
} // namespace dynd
