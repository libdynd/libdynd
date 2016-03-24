//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/convert_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * A kernel for chaining two other kernels, using temporary buffers
     * dynamically allocated on the heap.
     */
    // All methods are inlined, so this does not need to be declared DYND_API.
    struct compose_kernel : base_strided_kernel<compose_kernel, 1> {
      intptr_t second_offset; // The offset to the second child kernel
      ndt::type buffer_tp;
      arrmeta_holder buffer_arrmeta;
      std::vector<intptr_t> buffer_shape;

      compose_kernel(const ndt::type &buffer_tp) : buffer_tp(buffer_tp)
      {
        arrmeta_holder(this->buffer_tp).swap(buffer_arrmeta);
        buffer_arrmeta.arrmeta_default_construct(true);
        buffer_shape.push_back(DYND_BUFFER_CHUNK_SIZE);
      }

      ~compose_kernel()
      {
        // The first child ckernel
        get_child()->destroy();
        // The second child ckernel
        get_child(second_offset)->destroy();
      }

      void single(char *dst, char *const *src)
      {
        // Allocate a temporary buffer on the heap
        array buffer = empty(buffer_tp);
        char *buffer_data = buffer.data();

        kernel_prefix *first = get_child();
        kernel_single_t first_func = first->get_function<kernel_single_t>();

        kernel_prefix *second = get_child(second_offset);
        kernel_single_t second_func = second->get_function<kernel_single_t>();

        first_func(first, buffer_data, src);
        second_func(second, dst, &buffer_data);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Allocate a temporary buffer on the heap
        array buffer = empty(buffer_shape[0], buffer_tp);
        char *buffer_data = buffer.data();
        intptr_t buffer_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(buffer.get()->metadata())->stride;

        kernel_prefix *first = get_child();
        kernel_strided_t first_func = first->get_function<kernel_strided_t>();

        kernel_prefix *second = get_child(second_offset);
        kernel_strided_t second_func = second->get_function<kernel_strided_t>();

        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];

        size_t chunk_size = std::min(count, static_cast<size_t>(DYND_BUFFER_CHUNK_SIZE));
        first_func(first, buffer_data, buffer_stride, &src0, src_stride, chunk_size);
        second_func(second, dst, dst_stride, &buffer_data, &buffer_stride, chunk_size);
        count -= chunk_size;
        while (count) {
          src0 += chunk_size * src0_stride;
          dst += chunk_size * dst_stride;
          reset_strided_buffer_array(buffer);
          chunk_size = std::min(count, static_cast<size_t>(DYND_BUFFER_CHUNK_SIZE));
          first_func(first, buffer_data, buffer_stride, &src0, src_stride, chunk_size);
          second_func(second, dst, dst_stride, &buffer_data, &buffer_stride, chunk_size);
          count -= chunk_size;
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
