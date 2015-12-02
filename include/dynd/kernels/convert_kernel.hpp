//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/buffer_storage.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Instantiate an callable, adding buffers for any inputs where the types
     * don't match.
     */
    struct convert_kernel : base_kernel<convert_kernel> {
      typedef callable static_data_type;

      intptr_t m_nsrc;
      std::vector<intptr_t> m_src_buf_ck_offsets;
      std::vector<buffer_storage> m_bufs;

      convert_kernel(intptr_t narg) : m_nsrc(narg), m_src_buf_ck_offsets(m_nsrc), m_bufs(m_nsrc) {}

      void single(char *dst, char *const *src)
      {
        std::vector<char *> buf_src(m_nsrc);
        for (intptr_t i = 0; i < m_nsrc; ++i) {
          if (!m_bufs[i].is_null()) {
            m_bufs[i].reset_arrmeta();
            ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
            ck->single(m_bufs[i].get_storage(), &src[i]);
            buf_src[i] = m_bufs[i].get_storage();
          }
          else {
            buf_src[i] = src[i];
          }
        }
        ckernel_prefix *child = get_child();
        child->single(dst, &buf_src[0]);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        std::vector<char *> buf_src(m_nsrc);
        std::vector<intptr_t> buf_stride(m_nsrc);
        ckernel_prefix *child = get_child();
        expr_strided_t child_fn = child->get_function<expr_strided_t>();

        for (intptr_t i = 0; i < m_nsrc; ++i) {
          if (!m_bufs[i].is_null()) {
            buf_src[i] = m_bufs[i].get_storage();
            buf_stride[i] = m_bufs[i].get_stride();
          }
          else {
            buf_src[i] = src[i];
            buf_stride[i] = src_stride[i];
          }
        }

        while (count > 0) {
          size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
          for (intptr_t i = 0; i < m_nsrc; ++i) {
            if (!m_bufs[i].is_null()) {
              m_bufs[i].reset_arrmeta();
              ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
              expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
              ck_fn(ck, m_bufs[i].get_storage(), m_bufs[i].get_stride(), &src[i], &src_stride[i], chunk_size);
            }
          }
          child_fn(child, dst, dst_stride, &buf_src[0], &buf_stride[0], chunk_size);
          for (intptr_t i = 0; i < m_nsrc; ++i) {
            if (!m_bufs[i].is_null()) {
              m_bufs[i].reset_arrmeta();
              ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
              expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
              ck_fn(ck, m_bufs[i].get_storage(), buf_stride[i], &src[i], &src_stride[i], chunk_size);
            }
            else {
              buf_src[i] += chunk_size * buf_stride[i];
            }
          }
          count -= chunk_size;
        }
      }

      static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const ndt::type *src_tp_for_af,
                                  const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx)
      {
        callable &af = *reinterpret_cast<callable *>(static_data);

        intptr_t root_ckb_offset = ckb_offset;
        convert_kernel *self = convert_kernel::make(ckb, kernreq, ckb_offset, nsrc);
        std::vector<const char *> buffered_arrmeta(nsrc);
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (src_tp[i] == src_tp_for_af[i]) {
            buffered_arrmeta[i] = src_arrmeta[i];
          }
          else {
            self->m_bufs[i].allocate(src_tp_for_af[i]);
            buffered_arrmeta[i] = self->m_bufs[i].get_arrmeta();
          }
        }
        // Instantiate the callable being buffered
        ckb_offset = af.get()->instantiate(af.get()->static_data(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                           src_tp_for_af, &buffered_arrmeta[0], kernreq, ectx, 0, NULL,
                                           std::map<std::string, ndt::type>());
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<convert_kernel>(root_ckb_offset);
        // Instantiate assignments for all the buffered operands
        for (intptr_t i = 0; i < nsrc; ++i) {
          if (!self->m_bufs[i].is_null()) {
            self->m_src_buf_ck_offsets[i] = ckb_offset - root_ckb_offset;
            ckb_offset = make_assignment_kernel(ckb, ckb_offset, src_tp_for_af[i], self->m_bufs[i].get_arrmeta(),
                                                src_tp[i], src_arrmeta[i], kernreq, ectx);
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
            if (i < nsrc - 1) {
              self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                         ->get_at<convert_kernel>(root_ckb_offset);
            }
          }
        }

        return ckb_offset;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
