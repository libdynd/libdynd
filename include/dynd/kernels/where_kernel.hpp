//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  struct where_kernel : base_strided_kernel<where_kernel, 1> {
    intrusive_ptr<memory_block_data> dst_memory_block;

    where_kernel(const intrusive_ptr<memory_block_data> &dst_memory_block) : dst_memory_block(dst_memory_block) {}

    void single(char *dst, char *const *DYND_UNUSED(src)) {
      std::cout << "where_kernel::single" << std::endl;

      if (reinterpret_cast<ndt::var_dim_type::data_type *>(dst)->size == 0) {
        reinterpret_cast<ndt::var_dim_type::data_type *>(dst)->begin = dst_memory_block->alloc(10);
        reinterpret_cast<ndt::var_dim_type::data_type *>(dst)->size = 10;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
