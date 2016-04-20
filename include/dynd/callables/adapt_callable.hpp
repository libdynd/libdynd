//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/adapt_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class adapt_callable : public base_callable {
      ndt::type m_value_tp;
      callable m_forward;

    public:
      adapt_callable(const ndt::type &value_tp, const callable &forward)
          : base_callable(ndt::make_type<ndt::callable_type>(value_tp, {ndt::type("Any")})), m_value_tp(value_tp),
            m_forward(forward) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        return dst_tp;
      }

      /*
            void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb,
                             const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                             intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                             const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                             intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                             const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
              ckb->emplace_back<adapt_kernel>(kernreq, m_value_tp, m_forward);
              node = next(node);
            }
      */
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
