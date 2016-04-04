//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {

  template <typename KernelType>
  class default_instantiable_callable : public base_callable {
  protected:
    using base_callable::base_callable;

  public:
    default_instantiable_callable(const ndt::type &tp) : base_callable(tp) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.push_back([](call_node *&node, kernel_builder *ckb, kernel_request_t kernreq,
                      const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                      const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<KernelType>(kernreq);
        node = next(node);
      });

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
