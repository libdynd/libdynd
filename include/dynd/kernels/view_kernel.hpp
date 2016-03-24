//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct view_kernel : base_kernel<view_kernel> {
    void call(const array *dst, const array *src)
    {
      const ndt::type &dst_tp = dst->get_type();
      if (!dst_tp.is_builtin()) {
        dst_tp.extended()->arrmeta_copy_construct(dst->get()->metadata(), src[0]->metadata(),
                                                  intrusive_ptr<memory_block_data>(src[0].get(), true));
      }
      dst->get()->data = src[0]->data;

      dst->get()->owner = src[0]->owner ? src[0]->owner : intrusive_ptr<memory_block_data>(src[0].get(), true);
    }
  };

} // namespace dynd::nd
} // namespace dynd
