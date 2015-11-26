//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {

  struct DYND_API masked_take_ck : base_kernel<masked_take_ck, 2> {
    ndt::type m_dst_tp;
    const char *m_dst_meta;
    intptr_t m_dim_size, m_src0_stride, m_mask_stride;

    ~masked_take_ck()
    {
      get_child()->destroy();
    }

    void single(char *dst, char *const *src);

    static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars);
  };

  /**
   * CKernel which does an indexed take operation. The child ckernel
   * should be a single unary operation.
   */
  struct DYND_API indexed_take_ck : base_kernel<indexed_take_ck, 2> {
    intptr_t m_dst_dim_size, m_dst_stride, m_index_stride;
    intptr_t m_src0_dim_size, m_src0_stride;

    ~indexed_take_ck()
    {
      get_child()->destroy();
    }

    void single(char *dst, char *const *src);

    static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars);
  };

  struct DYND_API take_ck : base_virtual_kernel<take_ck> {
    static void resolve_dst_type(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                 const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                 const std::map<std::string, ndt::type> &tp_vars);

    static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars);
  };

} // namespace dynd::nd
} // namespace dynd
