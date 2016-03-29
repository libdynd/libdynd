//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/compound_div_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  class compound_div_callable : public base_instantiable_callable<compound_div_kernel_t<DstTypeID, Src0TypeID>> {
  public:
    compound_div_callable()
        : base_instantiable_callable<compound_div_kernel_t<DstTypeID, Src0TypeID>>(
              ndt::callable_type::make(ndt::type(DstTypeID), ndt::type(Src0TypeID)))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd