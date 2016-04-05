//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/compose_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class compose_callable : public base_callable {
      callable m_first;
      callable m_second;
      ndt::type m_buffer_tp;

    public:
      compose_callable(const ndt::type &tp, const callable &first, const callable &second, const ndt::type &buffer_tp)
          : base_callable(tp), m_first(first), m_second(second), m_buffer_tp(buffer_tp) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                        const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        cg.push_back([buffer_tp = m_buffer_tp](kernel_builder * ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                                               size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          intptr_t ckb_offset = ckb->size();

          intptr_t root_ckb_offset = ckb_offset;
          ckb->emplace_back<compose_kernel>(kernreq, buffer_tp);

          ckb_offset = ckb->size();
          compose_kernel *self = ckb->get_at<compose_kernel>(root_ckb_offset);
          ckb->instantiate(kernreq | kernel_request_data_only, self->buffer_arrmeta.get(), 1, src_arrmeta);

          ckb_offset = ckb->size();
          self = ckb->get_at<compose_kernel>(root_ckb_offset);
          self->second_offset = ckb_offset - root_ckb_offset;
          const char *buffer_arrmeta = self->buffer_arrmeta.get();
          ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, 1, &buffer_arrmeta);
          ckb_offset = ckb->size();
        });

        m_first->resolve(this, nullptr, cg, m_buffer_tp, 1, src_tp, nkwd, kwds, tp_vars);
        m_second->resolve(this, nullptr, cg, dst_tp, 1, &m_buffer_tp, nkwd, kwds, tp_vars);

        return dst_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
