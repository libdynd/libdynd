//
// Copyright (C) 2011-15 DyND Developers
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
          : base_callable(tp), m_first(first), m_second(second), m_buffer_tp(buffer_tp)
      {
      }

      /**
       * Instantiate the chaining of callables ``first`` and ``second``, using ``buffer_tp`` as the intermediate type,
       * without creating a temporary chained callable.
       */
      void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                       intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();

        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<compose_kernel>(kernreq, m_buffer_tp);
        ckb_offset = ckb->size();
        compose_kernel *self = ckb->get_at<compose_kernel>(root_ckb_offset);
        m_first->instantiate(data, ckb, m_buffer_tp, self->buffer_arrmeta.get(), 1, src_tp, src_arrmeta,
                             kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        self = ckb->get_at<compose_kernel>(root_ckb_offset);
        self->second_offset = ckb_offset - root_ckb_offset;
        const char *buffer_arrmeta = self->buffer_arrmeta.get();
        m_second->instantiate(data, ckb, dst_tp, dst_arrmeta, 1, &m_buffer_tp, &buffer_arrmeta,
                              kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
