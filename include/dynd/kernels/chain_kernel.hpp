//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * A kernel for chaining two other kernels, using temporary buffers
     * dynamically allocated on the heap.
     */
    struct chain_kernel : base_kernel<chain_kernel, kernel_request_host, 1> {
      struct static_data {
        callable first;
        callable second;
        ndt::type buffer_tp;

        static_data(const callable &first, const callable &second,
                    const ndt::type &buffer_tp)
            : first(first), second(second), buffer_tp(buffer_tp)
        {
        }
      };

      intptr_t second_offset; // The offset to the second child kernel
      ndt::type buffer_tp;
      arrmeta_holder buffer_arrmeta;
      std::vector<intptr_t> buffer_shape;

      chain_kernel(const ndt::type &buffer_tp) : buffer_tp(buffer_tp)
      {
        arrmeta_holder(this->buffer_tp).swap(buffer_arrmeta);
        buffer_arrmeta.arrmeta_default_construct(true);
        buffer_shape.push_back(DYND_BUFFER_CHUNK_SIZE);
      }

      void single(char *dst, char *const *src)
      {
        // Allocate a temporary buffer on the heap
        array buffer = empty(buffer_tp);
        char *buffer_data = buffer.get_readwrite_originptr();

        ckernel_prefix *first = get_child_ckernel();
        expr_single_t first_func = first->get_function<expr_single_t>();

        ckernel_prefix *second = get_child_ckernel(second_offset);
        expr_single_t second_func = second->get_function<expr_single_t>();

        first_func(buffer_data, src, first);
        second_func(dst, &buffer_data, second);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        // Allocate a temporary buffer on the heap
        array buffer = empty(buffer_shape[0], buffer_tp);
        char *buffer_data = buffer.get_readwrite_originptr();
        intptr_t buffer_stride =
            reinterpret_cast<const fixed_dim_type_arrmeta *>(
                buffer.get_arrmeta())->stride;

        ckernel_prefix *first = get_child_ckernel();
        expr_strided_t first_func = first->get_function<expr_strided_t>();

        ckernel_prefix *second = get_child_ckernel(second_offset);
        expr_strided_t second_func = second->get_function<expr_strided_t>();

        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];

        size_t chunk_size =
            std::min(count, static_cast<size_t>(DYND_BUFFER_CHUNK_SIZE));
        first_func(buffer_data, buffer_stride, &src0, src_stride, chunk_size,
                   first);
        second_func(dst, dst_stride, &buffer_data, &buffer_stride, chunk_size,
                    second);
        count -= chunk_size;
        while (count) {
          src0 += chunk_size * src0_stride;
          dst += chunk_size * dst_stride;
          reset_strided_buffer_array(buffer);
          chunk_size =
              std::min(count, static_cast<size_t>(DYND_BUFFER_CHUNK_SIZE));
          first_func(buffer_data, buffer_stride, &src0, src_stride, chunk_size,
                     first);
          second_func(dst, dst_stride, &buffer_data, &buffer_stride, chunk_size,
                      second);
          count -= chunk_size;
        }
      }

      void destruct_children()
      {
        // The first child ckernel
        get_child_ckernel()->destroy();
        // The second child ckernel
        destroy_child_ckernel(second_offset);
      }

      static void
      resolve_dst_type(char *static_data, size_t data_size, char *data,
                       ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const dynd::nd::array &kwds,
                       const std::map<nd::string, ndt::type> &tp_vars);

      static intptr_t
      instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                  intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars);
    };
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd