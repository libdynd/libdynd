//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/mean_kernel.hpp>

namespace dynd {
namespace nd {

  class mean_callable : public base_callable {
    ndt::type m_tp;

  public:
    mean_callable(const ndt::type &tp)
        : base_callable(ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::any_kind_type>(),
                                                           {ndt::make_type<ndt::any_kind_type>()})),
          m_tp(tp) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return dst_tp;
    }

    /*
        void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                         const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
       *src_arrmeta,
                         kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                         const std::map<std::string, ndt::type> &tp_vars) {
          intptr_t mean_offset = ckb->size();
          ckb->emplace_back<mean_kernel>(kernreq, src_tp[0].get_size(src_arrmeta[0]));
          node = next(node);

          nd::sum->instantiate(node, reinterpret_cast<data_type *>(data)->sum_data, ckb, dst_tp, dst_arrmeta, nsrc,
       src_tp,
                               src_arrmeta, kernreq, nkwd, kwds, tp_vars);

          mean_kernel *self = ckb->get_at<mean_kernel>(mean_offset);
          self->compound_div_offset = ckb->size();
          nd::compound_div->instantiate(node, reinterpret_cast<data_type *>(data)->compound_div_data, ckb, dst_tp,
                                        dst_arrmeta, 1, &m_tp, NULL, kernreq, nkwd, kwds, tp_vars);

          delete reinterpret_cast<data_type *>(data);
        }
    */
  };

} // namespace dynd::nd
} // namespace dynd
