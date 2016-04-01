//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Generic expr kernel + destructor for a strided dimension with
     * a fixed number of src operands.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, size_t N>
    struct elwise_kernel;

    template <size_t N>
    struct elwise_kernel<fixed_dim_id, fixed_dim_id, N>
        : base_strided_kernel<elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N> {
      typedef elwise_kernel self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N];

      elwise_kernel(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        opchild(child, dst, m_dst_stride, src, m_src_stride, m_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        char *src_loop[N];
        for (int j = 0; j != N; ++j) {
          src_loop[j] = src[j];
        }

        for (size_t i = 0; i < count; i += 1) {
          opchild(child, dst, m_dst_stride, src_loop, m_src_stride, m_size);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }
    };

    template <>
    struct elwise_kernel<fixed_dim_id, fixed_dim_id, 0>
        : base_strided_kernel<elwise_kernel<fixed_dim_id, fixed_dim_id, 0>, 0> {
      typedef elwise_kernel self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      elwise_kernel(intptr_t size, intptr_t dst_stride, const intptr_t *DYND_UNUSED(src_stride))
          : m_size(size), m_dst_stride(dst_stride)
      {
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();
        opchild(child, dst, m_dst_stride, src, NULL, m_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        for (size_t i = 0; i < count; i += 1) {
          opchild(child, dst, m_dst_stride, NULL, NULL, m_size);
          dst += dst_stride;
        }
      }
    };

    /**
     * Generic expr kernel + destructor for a strided/var dimensions with
     * a fixed number of src operands, outputing to a strided dimension.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <size_t N>
    struct elwise_kernel<fixed_dim_id, var_dim_id, N>
        : base_strided_kernel<elwise_kernel<fixed_dim_id, var_dim_id, N>, N> {
      typedef elwise_kernel self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N], m_src_offset[N];
      bool m_is_src_var[N];

      elwise_kernel(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride, const intptr_t *src_offset,
                    const bool *is_src_var)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (size_t i = 0; i < N; ++i) {
          if (m_is_src_var[i]) {
            ndt::var_dim_type::data_type *vddd = reinterpret_cast<ndt::var_dim_type::data_type *>(src[i]);
            modified_src[i] = vddd->begin + m_src_offset[i];
            if (vddd->size == 1) {
              modified_src_stride[i] = 0;
            }
            else if (vddd->size == static_cast<size_t>(dim_size)) {
              modified_src_stride[i] = m_src_stride[i];
            }
            else {
              throw broadcast_error(dim_size, vddd->size, "strided", "var");
            }
          }
          else {
            // strided dimensions were fully broadcast in the kernel factory
            modified_src[i] = src[i];
            modified_src_stride[i] = m_src_stride[i];
          }
        }
        opchild(child, dst, m_dst_stride, modified_src, modified_src_stride, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
          single(dst, src_loop);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }
    };

    template <>
    struct elwise_kernel<fixed_dim_id, var_dim_id, 0>
        : base_strided_kernel<elwise_kernel<fixed_dim_id, var_dim_id, 0>, 0> {
      typedef elwise_kernel self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      elwise_kernel(intptr_t size, intptr_t dst_stride, const intptr_t *DYND_UNUSED(src_stride),
                    const intptr_t *DYND_UNUSED(src_offset), const bool *DYND_UNUSED(is_src_var))
          : m_size(size), m_dst_stride(dst_stride)
      {
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        opchild(child, dst, m_dst_stride, NULL, NULL, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i) {
          single(dst, NULL);
          dst += dst_stride;
        }
      }
    };

    /**
     * Generic expr kernel + destructor for a strided/var dimensions with
     * a fixed number of src operands, outputing to a var dimension.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <size_t N>
    struct elwise_kernel<var_dim_id, fixed_dim_id, N>
        : base_strided_kernel<elwise_kernel<var_dim_id, fixed_dim_id, N>, N> {
      typedef elwise_kernel self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset, m_src_stride[N], m_src_offset[N], m_src_size[N];
      bool m_is_src_var[N];

      elwise_kernel(memory_block_data *dst_memblock, size_t dst_target_alignment, intptr_t dst_stride,
                    intptr_t dst_offset, const intptr_t *src_stride, const intptr_t *src_offset,
                    const intptr_t *src_size, const bool *is_src_var)
          : m_dst_memblock(dst_memblock), m_dst_target_alignment(dst_target_alignment), m_dst_stride(dst_stride),
            m_dst_offset(dst_offset)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_src_size, src_size, sizeof(m_src_size));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        ndt::var_dim_type::data_type *dst_vddd = reinterpret_cast<ndt::var_dim_type::data_type *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        if (dst_vddd->begin != NULL) {
          // If the destination already has allocated data, broadcast to that
          // data
          modified_dst = dst_vddd->begin + m_dst_offset;
          // Broadcast all the inputs to the existing destination dimension size
          dim_size = dst_vddd->size;
          for (size_t i = 0; i < N; ++i) {
            if (m_is_src_var[i]) {
              ndt::var_dim_type::data_type *vddd = reinterpret_cast<ndt::var_dim_type::data_type *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              }
              else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            }
            else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              }
              else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, m_src_size[i], "var", "strided");
              }
            }
          }
        }
        else {
          if (m_dst_offset != 0) {
            throw std::runtime_error("Cannot assign to an uninitialized dynd var_dim "
                                     "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          for (size_t i = 0; i < N; ++i) {
            if (m_is_src_var[i]) {
              ndt::var_dim_type::data_type *vddd = reinterpret_cast<ndt::var_dim_type::data_type *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              }
              else if (dim_size == 1) {
                dim_size = vddd->size;
                modified_src_stride[i] = m_src_stride[i];
              }
              else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            }
            else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              }
              else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else if (dim_size == 1) {
                dim_size = m_src_size[i];
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, m_src_size[i], "var", "strided");
              }
            }
          }
          // Allocate the output array data
          dst_vddd->begin = m_dst_memblock->alloc(dim_size);
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        }
        else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(child, modified_dst, modified_dst_stride, modified_src, modified_src_stride, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
          single(dst, src_loop);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }
    };

    template <>
    struct elwise_kernel<var_dim_id, fixed_dim_id, 0>
        : base_strided_kernel<elwise_kernel<var_dim_id, fixed_dim_id, 0>, 0> {
      typedef elwise_kernel self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset;

      elwise_kernel(memory_block_data *dst_memblock, size_t dst_target_alignment, intptr_t dst_stride,
                    intptr_t dst_offset, const intptr_t *DYND_UNUSED(src_stride),
                    const intptr_t *DYND_UNUSED(src_offset), const intptr_t *DYND_UNUSED(src_size),
                    const bool *DYND_UNUSED(is_src_var))
          : m_dst_memblock(dst_memblock), m_dst_target_alignment(dst_target_alignment), m_dst_stride(dst_stride),
            m_dst_offset(dst_offset)
      {
      }

      ~elwise_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        kernel_prefix *child = this->get_child();
        kernel_strided_t opchild = child->get_function<kernel_strided_t>();

        ndt::var_dim_type::data_type *dst_vddd = reinterpret_cast<ndt::var_dim_type::data_type *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        if (dst_vddd->begin != NULL) {
          // If the destination already has allocated data, broadcast to that
          // data
          modified_dst = dst_vddd->begin + m_dst_offset;
          // Broadcast all the inputs to the existing destination dimension size
          dim_size = dst_vddd->size;
        }
        else {
          if (m_dst_offset != 0) {
            throw std::runtime_error("Cannot assign to an uninitialized dynd var_dim "
                                     "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          // Allocate the output
          dst_vddd->begin = m_dst_memblock->alloc(dim_size);
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        }
        else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(child, modified_dst, modified_dst_stride, NULL, NULL, dim_size);
      }
    };

    template <size_t N>
    struct elwise_kernel<var_dim_id, var_dim_id, N> : elwise_kernel<var_dim_id, fixed_dim_id, N> {
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
