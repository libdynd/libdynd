//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/compound_add_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  class compound_add_callable : public default_instantiable_callable<compound_add_kernel_t<DstTypeID, Src0TypeID>> {
  public:
    compound_add_callable()
        : default_instantiable_callable<compound_add_kernel_t<DstTypeID, Src0TypeID>>(
              ndt::callable_type::make(ndt::type(DstTypeID), ndt::type(Src0TypeID)))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
