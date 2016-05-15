//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct view_kernel : base_kernel<view_kernel> {
    void call(const array *DYND_UNUSED(dst), const array *DYND_UNUSED(src)) {
      throw std::runtime_error("view kernel is not implemented");
      /*
            const ndt::type &dst_tp = dst->get_type();
            *const_cast<array *>(dst) =
                make_array(dst_tp, src[0]->get_data(), src[0]->get_owner() ? src[0]->get_owner() : src[0],
                           read_access_flag | write_access_flag);
            if (!dst_tp.is_builtin()) {
              dst_tp.extended()->arrmeta_copy_construct(dst->get()->metadata(), src[0]->metadata(), src[0]);
            }
      */
    }
  };

} // namespace dynd::nd
} // namespace dynd
