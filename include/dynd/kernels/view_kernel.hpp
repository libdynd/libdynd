//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct view_kernel : base_kernel<view_kernel> {
    static const size_t data_size = 0;

    void metadata_single(array *dst, array *const *src)
    {
      const ndt::type &dst_tp = dst->get_type();
      if (!dst_tp.is_builtin()) {
        dst_tp.extended()->arrmeta_copy_construct(dst->metadata(), src[0]->metadata(), *src[0]);
      }
      dst->get()->ptr = src[0]->get()->ptr;

      dst->get()->ref = src[0]->get()->owner();
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
