//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct rolling_arrfunc_data {
      intptr_t window_size;
      // The window op
      arrfunc window_op;
    };

    struct strided_rolling_ck
        : base_kernel<strided_rolling_ck, kernel_request_host, 1> {
      intptr_t m_window_size;
      intptr_t m_dim_size, m_dst_stride, m_src_stride;
      size_t m_window_op_offset;
      arrmeta_holder m_src_winop_meta;

      void single(char *dst, char *const *src);

      void destruct_children();
    };

    struct var_rolling_ck
        : base_kernel<var_rolling_ck, kernel_request_host, 1> {
      intptr_t m_window_size;
      intptr_t m_src_stride, m_src_offset;
      ndt::type m_dst_tp;
      const char *m_dst_meta;
      size_t m_window_op_offset;

      void single(char *dst, char *const *src);

      void destruct_children();
    };

    struct rolling_ck : base_virtual_kernel<rolling_ck> {
      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const ndt::arrfunc_type *self_tp, size_t data_size,
                  char *data, void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars);

      static void
      resolve_dst_type(const arrfunc_type_data *af_self,
                       const ndt::arrfunc_type *af_tp, size_t data_size,
                       char *data, ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const nd::array &kwds,
                       const std::map<nd::string, ndt::type> &tp_vars);
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd