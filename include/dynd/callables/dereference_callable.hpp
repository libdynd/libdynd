//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/dereference_kernel.hpp>

namespace dynd {
namespace nd {

  class dereference_callable : public base_callable {
  public:
    dereference_callable() : base_callable(ndt::type("(pointer[Any]) -> Any")) {}

    array alloc(const ndt::type *dst_tp) const { return empty_shell(*dst_tp); }

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg, const ndt::type &DYND_UNUSED(dst_tp),
                      size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return src_tp[0].extended<ndt::pointer_type>()->get_target_type();
    }

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      dst_tp = src_tp[0].extended<ndt::pointer_type>()->get_target_type();
    }

    void instantiate(call_node *DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<dereference_kernel>(kernreq, dst_tp);
    }
  };

} // namespace dynd::nd
} // namespace dynd
