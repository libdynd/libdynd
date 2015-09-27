//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  inline void incref(char **data)
  {
    std::cout << (reinterpret_cast<decltype(array_preamble::data) *>(data)->ref == NULL) << std::endl;
    //    memory_block_incref(reinterpret_cast<decltype(array_preamble::m_data) *>(data)->m_data_reference);
  }

  struct view_kernel : base_kernel<view_kernel> {
    static const bool has_metadata_single = true;
    static const size_t data_size = 0;

    size_t metadata_size;

    view_kernel(size_t metadata_size) : metadata_size(metadata_size)
    {
    }

    void metadata_single(char *DYND_UNUSED(dst_metadata), char **DYND_UNUSED(dst),
                         char *const *DYND_UNUSED(src_metadata), char **const *DYND_UNUSED(src))
    {
      //      std::memcpy(dst_metadata, src_metadata[0], metadata_size);
      //    *dst = *src[0];

      //      refcpy(dst, src[0]);
      //      incref(src[0]);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
                                intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset, src_tp[0].get_arrmeta_size());
      return ckb_offset;
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }
  };

} // namespace dynd::nd
} // namespace dynd
