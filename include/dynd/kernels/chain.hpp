//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct instantiate_chain_data {
      nd::arrfunc first;
      nd::arrfunc second;
      ndt::type buf_tp;
    };

    /**
     * A ckernel for chaining two other ckernels, using temporary buffers
     * dynamically allocated on the heap.
     */
    struct unary_heap_chain_ck
        : base_kernel<unary_heap_chain_ck, kernel_request_host, 1> {
      // The offset to the second child ckernel
      intptr_t m_second_offset;
      ndt::type m_buf_tp;
      arrmeta_holder m_buf_arrmeta;
      std::vector<intptr_t> m_buf_shape;

      void single(char *dst, char *const *src);

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count);

      void destruct_children();

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  char *data, void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars);

      static void
      resolve_dst_type(const arrfunc_type_data *self,
                       const arrfunc_type *self_tp, char *data,
                       ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars);
    };
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd