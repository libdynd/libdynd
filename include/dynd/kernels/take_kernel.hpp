//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/shape_tools.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/assignment.hpp>

namespace dynd {
namespace nd {

  struct DYND_API masked_take_ck : base_strided_kernel<masked_take_ck, 2> {
    const char *m_dst_meta;
    intptr_t m_dim_size, m_src0_stride, m_mask_stride;

    ~masked_take_ck() { get_child()->destroy(); }

    void single(char *dst, char *const *src) {
      kernel_prefix *child = get_child();
      kernel_strided_t child_fn = child->get_function<kernel_strided_t>();
      char *src0 = src[0];
      char *mask = src[1];
      intptr_t dim_size = m_dim_size, src0_stride = m_src0_stride, mask_stride = m_mask_stride;
      // Start with the dst matching the dim size. (Maybe better to
      // do smaller? This means no resize required in the loop.)
      ndt::var_dim_type::data_type *vdd = reinterpret_cast<ndt::var_dim_type::data_type *>(dst);
      vdd->begin = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->blockref->alloc(dim_size);
      vdd->size = dim_size;
      char *dst_ptr = vdd->begin;
      intptr_t dst_stride = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->stride;
      intptr_t dst_count = 0;
      intptr_t i = 0;
      while (i < dim_size) {
        // Run of false
        for (; i < dim_size && *mask == 0; src0 += src0_stride, mask += mask_stride, ++i) {
        }
        // Run of true
        intptr_t i_saved = i;
        for (; i < dim_size && *mask != 0; mask += mask_stride, ++i) {
        }
        // Copy the run of true
        if (i > i_saved) {
          intptr_t run_count = i - i_saved;
          child_fn(child, dst_ptr, dst_stride, &src0, &src0_stride, run_count);
          dst_ptr += run_count * dst_stride;
          src0 += run_count * src0_stride;
          dst_count += run_count;
        }
      }

      vdd->begin = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)
                       ->blockref->resize(vdd->begin, dst_count);
      vdd->size = dst_count;
    }
  };

  /**
   * CKernel which does an indexed take operation. The child ckernel
   * should be a single unary operation.
   */
  struct DYND_API indexed_take_ck : base_strided_kernel<indexed_take_ck, 2> {
    intptr_t m_dst_dim_size, m_dst_stride, m_index_stride;
    intptr_t m_src0_dim_size, m_src0_stride;

    ~indexed_take_ck() { get_child()->destroy(); }

    void single(char *dst, char *const *src) {
      kernel_prefix *child = get_child();
      kernel_single_t child_fn = child->get_function<kernel_single_t>();
      char *src0 = src[0];
      const char *index = src[1];
      intptr_t dst_dim_size = m_dst_dim_size, src0_dim_size = m_src0_dim_size, dst_stride = m_dst_stride,
               src0_stride = m_src0_stride, index_stride = m_index_stride;
      for (intptr_t i = 0; i < dst_dim_size; ++i) {
        intptr_t ix = *reinterpret_cast<const intptr_t *>(index);
        // Handle Python-style negative index, bounds checking
        ix = apply_single_index(ix, src0_dim_size, NULL);
        // Copy one element at a time
        char *child_src0 = src0 + ix * src0_stride;
        child_fn(child, dst, &child_src0);
        dst += dst_stride;
        index += index_stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
