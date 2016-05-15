//
// Copyright (C) 2011-16 DyND Developers
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

    array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(
          [](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
             const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
             const char *const *DYND_UNUSED(src_arrmeta)) { kb.emplace_back<dereference_kernel>(kernreq); });

      return src_tp[0].extended<ndt::pointer_type>()->get_target_type();
    }
  };

} // namespace dynd::nd
} // namespace dynd
