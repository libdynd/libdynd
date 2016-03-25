//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/view_kernel.hpp>

namespace dynd {
namespace nd {

  class view_callable : public base_instantiable_callable<view_kernel> {
  public:
    view_callable() : base_instantiable_callable<view_kernel>(ndt::type("(Any) -> Any")) {}

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }
  };

} // namespace dynd::nd
} // namespace dynd
