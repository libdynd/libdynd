//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  struct where_kernel : base_strided_kernel<where_kernel, 2> {
    size_t &it;
    intptr_t ret_stride;
    memory_block dst_memory_block;
    size_t ret_element_size;
    size_t capacity;

    where_kernel(char *data, intptr_t ret_stride, const memory_block &dst_memory_block)
        : it(*reinterpret_cast<size_t *>(data)), ret_stride(ret_stride), dst_memory_block(dst_memory_block),
          ret_element_size(sizeof(intptr_t)), capacity(0) {}

    void single(char *ret, char *const *src) {
      bool child_ret;
      get_child()->single(reinterpret_cast<char *>(&child_ret), src);

      if (child_ret) {
        const state &src1 = *reinterpret_cast<state *>(src[1]);

        size_t &size = reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->size;
        if (size == 0) {
          capacity = 1;
          reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->begin = dst_memory_block->alloc(capacity);
        } else if (size == capacity) {
          capacity = size + 1;
          reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->begin =
              dst_memory_block->resize(reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->begin, capacity);
        }
        ++size;

        intptr_t *index = reinterpret_cast<intptr_t *>(reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->begin +
                                                       (size - 1) * ret_stride);
        *index = src1.index[0];
      }
    }

    size_t &begin() {
      it = 0;
      return it;
    }
  };

} // namespace dynd::nd
} // namespace dynd
