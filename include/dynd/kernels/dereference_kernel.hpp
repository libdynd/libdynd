//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  struct dereference_kernel : base_strided_kernel<dereference_kernel, 1> {
    void call(const array *dst, const array *src) {
      const ndt::type &dst_tp = dst->get_type();
      if (!dst_tp.is_builtin()) {
        dst_tp.extended()->arrmeta_copy_construct((*dst)->metadata(), src[0]->metadata() + sizeof(pointer_type_arrmeta),
                                                  src[0]);
      }
      (*dst)->data = *reinterpret_cast<char **>(src[0]->data) +
                     reinterpret_cast<const pointer_type_arrmeta *>(src[0]->metadata())->offset;
      (*dst)->set_owner(src[0].get_data_memblock());
    }
  };

} // namespace dynd::nd
} // namespace dynd
