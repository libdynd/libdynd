//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/compound_div_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  class compound_div_callable : public default_instantiable_callable<compound_div_kernel_t<DstTypeID, Src0TypeID>> {
  public:
    compound_div_callable()
        : default_instantiable_callable<compound_div_kernel_t<DstTypeID, Src0TypeID>>(
              ndt::make_type<ndt::callable_type>(ndt::type(DstTypeID), {ndt::type(Src0TypeID)})) {}
  };

} // namespace dynd::nd
} // namespace dynd
