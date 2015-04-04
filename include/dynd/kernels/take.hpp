//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/virtual.hpp>

namespace dynd {
namespace nd {

  struct masked_take_ck : expr_ck<masked_take_ck, kernel_request_host, 2> {
    ndt::type m_dst_tp;
    const char *m_dst_meta;
    intptr_t m_dim_size, m_src0_stride, m_mask_stride;

    void single(char *dst, char *const *src);

    void destruct_children();

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);
  };

  /**
   * CKernel which does an indexed take operation. The child ckernel
   * should be a single unary operation.
   */
  struct indexed_take_ck : expr_ck<indexed_take_ck, kernel_request_host, 2> {
    intptr_t m_dst_dim_size, m_dst_stride, m_index_stride;
    intptr_t m_src0_dim_size, m_src0_stride;

    void single(char *dst, char *const *src);

    void destruct_children();

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);
  };

  struct take_ck : virtual_ck<take_ck> {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);

    static void
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     char *data, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, const nd::array &kwds,
                     const std::map<nd::string, ndt::type> &tp_vars);
  };

} // namespace dynd::nd
} // namespace dynd