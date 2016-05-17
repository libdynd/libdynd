//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/sum_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  class sum_callable : public default_instantiable_callable<sum_kernel<Arg0Type>> {
  public:
    sum_callable()
        : default_instantiable_callable<sum_kernel<Arg0Type>>(ndt::make_type<ndt::callable_type>(
              ndt::make_type<typename nd::sum_kernel<Arg0Type>::dst_type>(), {ndt::make_type<Arg0Type>()})) {}
  };

} // namespace dynd::nd
} // namespace dynd
