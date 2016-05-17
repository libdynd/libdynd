//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/conj_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  class conj_callable : public default_instantiable_callable<conj_kernel<Arg0Type>> {
  public:
    conj_callable()
        : default_instantiable_callable<conj_kernel<Arg0Type>>(
              ndt::make_type<ndt::callable_type>(ndt::make_type<Arg0Type>(), {ndt::make_type<Arg0Type>()})) {}
  };

} // namespace dynd::nd
} // namespace dynd
