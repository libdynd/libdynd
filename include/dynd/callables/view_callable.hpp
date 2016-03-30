//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/view_kernel.hpp>

namespace dynd {
namespace nd {

  class view_callable : public default_instantiable_callable<view_kernel> {
  public:
    view_callable() : default_instantiable_callable<view_kernel>(ndt::type("(Any) -> Any")) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), callable_graph &DYND_UNUSED(g), ndt::type &dst_tp,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t DYND_UNUSED(nkwd),
                     const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }
  };

} // namespace dynd::nd
} // namespace dynd
