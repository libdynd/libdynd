//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/imag_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class imag_callable : public default_instantiable_callable<imag_kernel<Arg0ID>> {
  public:
    imag_callable()
        : default_instantiable_callable<imag_kernel<Arg0ID>>(
              ndt::make_type<ndt::callable_type>(ndt::make_type<typename nd::imag_kernel<Arg0ID>::real_type>(),
                                                 {ndt::make_type<typename nd::imag_kernel<Arg0ID>::complex_type>()})) {}
  };

} // namespace dynd::nd
} // namespace dynd
