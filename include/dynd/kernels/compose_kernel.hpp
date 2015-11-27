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
    // All methods are inlined, so this does not need to be declared DYND_API.
    struct compose_kernel : base_kernel<compose_kernel, 1> {
      struct static_data {
        callable first;
        callable second;
        ndt::type buffer_tp;

        static_data(const callable &first, const callable &second, const ndt::type &buffer_tp)
            : first(first), second(second), buffer_tp(buffer_tp)
        {
        }
      };

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

        ckernel_prefix *first = get_child();
        expr_single_t first_func = first->get_function<expr_single_t>();

        ckernel_prefix *second = get_child(second_offset);
        expr_single_t second_func = second->get_function<expr_single_t>();

        first_func(first, buffer_data, src);
        second_func(second, dst, &buffer_data);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Allocate a temporary buffer on the heap
        array buffer = empty(buffer_shape[0], buffer_tp);
        char *buffer_data = buffer.data();
        intptr_t buffer_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(buffer.get()->metadata())->stride;

        ckernel_prefix *first = get_child();
        expr_strided_t first_func = first->get_function<expr_strided_t>();

        ckernel_prefix *second = get_child(second_offset);
        expr_strided_t second_func = second->get_function<expr_strided_t>();

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

      static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                   intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &tp_vars)
      {
        dst_tp = ndt::substitute(dst_tp, tp_vars, true);
      }

      /**
       * Instantiate the chaining of callables ``first`` and ``second``, using
       * ``buffer_tp`` as the intermediate type, without creating a temporary
       * chained
       * callable.
       */
      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        const struct static_data *static_data_x = reinterpret_cast<struct static_data *>(static_data);

        base_callable *first = const_cast<base_callable *>(static_data_x->first.get());
        base_callable *second = const_cast<base_callable *>(static_data_x->second.get());

        const ndt::type &buffer_tp = static_data_x->buffer_tp;

        intptr_t root_ckb_offset = ckb_offset;
        compose_kernel *self = make(ckb, kernreq, ckb_offset, static_data_x->buffer_tp);
        ckb_offset =
            first->instantiate(first->static_data(), data, ckb, ckb_offset, buffer_tp, self->buffer_arrmeta.get(), 1,
                               src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
        self = get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), root_ckb_offset);
        self->second_offset = ckb_offset - root_ckb_offset;
        const char *buffer_arrmeta = self->buffer_arrmeta.get();
        return second->instantiate(second->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, 1, &buffer_tp,
                                   &buffer_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      }
    };
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
