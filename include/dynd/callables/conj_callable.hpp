//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/conj_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class conj_callable : public base_instantiable_callable<conj_kernel<Arg0ID>> {
  public:
    conj_callable()
        : base_instantiable_callable<conj_kernel<Arg0ID>>(
              ndt::callable_type::make(ndt::make_type<typename nd::conj_kernel<Arg0ID>::complex_type>(),
                                       {ndt::make_type<typename nd::conj_kernel<Arg0ID>::complex_type>()}))

    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
