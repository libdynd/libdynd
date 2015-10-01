//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  inline void refcpy(char *dst, char *src)
  {
    dst -= sizeof(array_preamble);
    src -= sizeof(array_preamble);

    if (reinterpret_cast<array_preamble *>(src)->data.ref == NULL) {
      reinterpret_cast<array_preamble *>(dst)->data.ref = &reinterpret_cast<array_preamble *>(src)->m_memblockdata;
    } else {
      reinterpret_cast<array_preamble *>(dst)->data.ref = reinterpret_cast<array_preamble *>(src)->data.ref;
    }
  }

  inline void incref(char *metadata)
  {
    memory_block_incref(reinterpret_cast<array_preamble *>(metadata - sizeof(array_preamble))->data.ref);
  }

  struct view_kernel : base_kernel<view_kernel> {
    static const size_t data_size = 0;

    size_t metadata_size;

    view_kernel(size_t metadata_size) : metadata_size(metadata_size)
    {
    }

    void metadata_single(char *dst_metadata, char **dst, char *const *src_metadata, char **const *src)
    {
      std::memcpy(dst_metadata, src_metadata[0],
                  metadata_size); // need to use the type virtual function instead of this
      *dst = *src[0];

      refcpy(dst_metadata, src_metadata[0]);
      incref(dst_metadata);
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
