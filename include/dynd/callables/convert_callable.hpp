//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/convert_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class convert_callable : public base_callable {
      callable m_child;

    public:
      convert_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(call_node *DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *ckb,
                       const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t ckb_offset = ckb->size();
        callable &af = m_child;
        const std::vector<ndt::type> &src_tp_for_af = af.get_type()->get_pos_types();

        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<convert_kernel>(kernreq, nsrc);
        ckb_offset = ckb->size();
        std::vector<const char *> buffered_arrmeta(nsrc);
        convert_kernel *self = ckb->get_at<convert_kernel>(root_ckb_offset);
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (src_tp[i] == src_tp_for_af[i]) {
            buffered_arrmeta[i] = src_arrmeta[i];
          } else {
            self->m_bufs[i].allocate(src_tp_for_af[i]);
            buffered_arrmeta[i] = self->m_bufs[i].get_arrmeta();
          }
        }
        // Instantiate the callable being buffered
        af.get()->instantiate(nullptr, NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp_for_af.data(), &buffered_arrmeta[0],
                              kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(kernel_prefix));
        self = reinterpret_cast<kernel_builder *>(ckb)->get_at<convert_kernel>(root_ckb_offset);
        // Instantiate assignments for all the buffered operands
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (!self->m_bufs[i].is_null()) {
            self->m_src_buf_ck_offsets[i] = ckb_offset - root_ckb_offset;
            nd::array error_mode = eval::default_eval_context.errmode;
            assign->instantiate(nullptr, NULL, ckb, src_tp_for_af[i], self->m_bufs[i].get_arrmeta(), 1, src_tp + i,
                                src_arrmeta + i, kernreq | kernel_request_data_only, 1, &error_mode, tp_vars);
            ckb_offset = ckb->size();
            reinterpret_cast<kernel_builder *>(ckb)->reserve(ckb_offset + sizeof(kernel_prefix));
            if (i < nsrc - 1) {
              self = reinterpret_cast<kernel_builder *>(ckb)->get_at<convert_kernel>(root_ckb_offset);
            }
          }
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
