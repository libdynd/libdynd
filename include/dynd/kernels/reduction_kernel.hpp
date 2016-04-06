//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/assignment.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/constant_kernel.hpp>
#include <dynd/kernels/reduction_kernel_prefix.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename SelfType>
    struct base_reduction_kernel : reduction_kernel_prefix {
      /**
       * Returns the child kernel immediately following this one.
       */
      kernel_prefix *get_child() { return kernel_prefix::get_child(sizeof(SelfType)); }

      kernel_prefix *get_child(intptr_t offset) {
        return kernel_prefix::get_child(kernel_builder::aligned_size(offset));
      }

      reduction_kernel_prefix *get_reduction_child() {
        return reinterpret_cast<reduction_kernel_prefix *>(this->get_child());
      }

      void call(array *dst, const array *src) {
        char *src_data[1];
        for (size_t i = 0; i < 1; ++i) {
          src_data[i] = const_cast<char *>(src[i].cdata());
        }
        reinterpret_cast<SelfType *>(this)->single_first(const_cast<char *>(dst->cdata()), src_data);
      }

      static void call_wrapper(kernel_prefix *self, array *dst, array *src) {
        reinterpret_cast<SelfType *>(self)->call(dst, src);
      }

      template <typename... ArgTypes>
      static void init(SelfType *self, kernel_request_t kernreq, ArgTypes &&... args) {
        new (self) SelfType(std::forward<ArgTypes>(args)...);

        self->destructor = SelfType::destruct;
        // Get the function pointer for the first_call
        switch (kernreq) {
        case kernel_request_call:
          self->set_first_call_function(SelfType::call_wrapper);
          break;
        case kernel_request_single:
          self->set_first_call_function(SelfType::single_first_wrapper);
          break;
        case kernel_request_strided:
          self->set_first_call_function(SelfType::strided_first_wrapper);
          break;
        default:
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
          throw std::runtime_error(ss.str());
        }
        // The function pointer for followup accumulation calls
        self->set_followup_call_function(SelfType::strided_followup_wrapper);
      }

      static void destruct(kernel_prefix *self) { reinterpret_cast<SelfType *>(self)->~SelfType(); }

      constexpr size_t size() const { return sizeof(SelfType); }

      static void single_first_wrapper(kernel_prefix *self, char *dst, char *const *src) {
        return reinterpret_cast<SelfType *>(self)->single_first(dst, src);
      }

      static void strided_first_wrapper(kernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                        const intptr_t *src_stride, size_t count)

      {
        return reinterpret_cast<SelfType *>(self)->strided_first(dst, dst_stride, src, src_stride, count);
      }

      static void strided_followup_wrapper(kernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                           const intptr_t *src_stride, size_t count)

      {
        return reinterpret_cast<SelfType *>(self)->strided_followup(dst, dst_stride, src, src_stride, count);
      }
    };

    /**
     * STRIDED INITIAL REDUCTION DIMENSION
     * This ckernel handles one dimension of the reduction processing,
     * where:
     *  - It's a reduction dimension, so dst_stride is zero.
     *  - It's an initial dimension, there are additional dimensions
     *    being processed by its child kernels.
     *  - The source data is strided.
     *
     * Requirements:
     *  - The child first_call function must be *single*.
     *  - The child followup_call function must be *strided*.
     *
     */
    template <type_id_t Src0TypeID, bool Broadcast, bool Inner>
    struct reduction_kernel;

    template <>
    struct reduction_kernel<fixed_dim_id, false, false>
        : base_reduction_kernel<reduction_kernel<fixed_dim_id, false, false>> {
      std::intptr_t src0_element_size;
      std::intptr_t src0_element_stride;

      reduction_kernel(std::intptr_t src0_element_size, std::intptr_t src_stride)
          : src0_element_size(src0_element_size), src0_element_stride(src_stride) {}

      ~reduction_kernel() { get_child()->destroy(); }

      void single_first(char *dst, char *const *src) {
        reduction_kernel_prefix *child = get_reduction_child();
        // The first call at the "dst" address
        child->single_first(dst, src);
        if (src0_element_size > 1) {
          // All the followup calls at the "dst" address
          char *src_second = src[0] + src0_element_stride;
          child->strided_followup(dst, 0, &src_second, &src0_element_stride, src0_element_size - 1);
        }
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
        reduction_kernel_prefix *child = get_reduction_child();

        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we have one "first", followed by many
          // "followup" calls
          child->single_first(dst, &src0);
          if (src0_element_size > 1) {
            char *inner_src_second = src0 + src0_element_stride;
            child->strided_followup(dst, 0, &inner_src_second, &src0_element_stride, src0_element_size - 1);
          }
          src0 += src0_stride;
          for (std::size_t i = 1; i != count; ++i) {
            child->strided_followup(dst, 0, &src0, &src0_element_stride, src0_element_size);
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop is
          // "first"
          for (size_t i = 0; i != count; ++i) {
            child->single_first(dst, &src0);
            if (src0_element_size > 1) {
              char *inner_src_second = src0 + src0_element_stride;
              child->strided_followup(dst, 0, &inner_src_second, &src0_element_stride, src0_element_size - 1);
            }
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                            size_t count) {
        reduction_kernel_prefix *child = get_reduction_child();

        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          child->strided_followup(dst, 0, &src0, &src0_element_stride, src0_element_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    /**
     * STRIDED INNER REDUCTION DIMENSION
     * This ckernel handles one dimension of the reduction processing,
     * where:
     *  - It's a reduction dimension, so dst_stride is zero.
     *  - It's an inner dimension, calling the reduction kernel directly.
     *  - The source data is strided.
     *
     * Requirements:
     *  - The child destination initialization kernel must be *single*.
     *  - The child reduction kernel must be *strided*.
     *
     */
    template <>
    struct reduction_kernel<fixed_dim_id, false, true>
        : base_reduction_kernel<reduction_kernel<fixed_dim_id, false, true>> {
      // The code assumes that size >= 1
      intptr_t size_first;
      intptr_t src_stride_first;
      intptr_t _size;
      intptr_t src_stride;
      size_t init_offset;

      ~reduction_kernel() {
        get_child()->destroy();
        get_child(init_offset)->destroy();
      }

      void single_first(char *dst, char *const *src) {
        char *src0 = src[0];

        // Initialize the dst values
        get_child(init_offset)->single(dst, src);
        src0 += src_stride_first;

        // Do the reduction
        get_child()->strided(dst, 0, &src0, &src_stride, size_first);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
        kernel_prefix *init_child = get_child(init_offset);
        kernel_prefix *reduction_child = get_child();

        char *src0 = src[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          init_child->single(dst, &src0);
          src0 += src_stride_first;

          reduction_child->strided(dst, 0, &src0, &this->src_stride, size_first);

          for (std::size_t i = 1; i != count; ++i) {
            reduction_child->strided(dst, 0, &src0, &this->src_stride, size_first);
            dst += dst_stride;
            src0 += src_stride[0];
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop has to
          // initialize then reduce
          for (size_t i = 0; i != count; ++i) {
            init_child->single(dst, &src0);

            char *inner_child_src = src0 + src_stride_first;
            reduction_child->strided(dst, 0, &inner_child_src, &this->src_stride, size_first);
            dst += dst_stride;
            src0 += src_stride[0];
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                            size_t count) {
        kernel_prefix *reduce_child = get_child();

        // No initialization, all reduction
        char *src0 = src[0];
        for (size_t i = 0; i != count; ++i) {
          reduce_child->strided(dst, 0, &src0, &this->src_stride, _size);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }
    };

    template <>
    struct reduction_kernel<var_dim_id, false, true>
        : base_reduction_kernel<reduction_kernel<var_dim_id, false, true>> {
      intptr_t src0_inner_stride;
      intptr_t src0_inner_stride_first;
      intptr_t init_offset;

      reduction_kernel(std::intptr_t src0_inner_stride, bool with_identity = false)
          : src0_inner_stride(src0_inner_stride) {
        if (with_identity) {
          src0_inner_stride_first = 0;
        } else {
          src0_inner_stride_first = src0_inner_stride;
        }
      }

      ~reduction_kernel() {
        get_child(init_offset)->destroy();
        get_child()->destroy();
      }

      void single_first(char *dst, char *const *src) {
        size_t inner_size = reinterpret_cast<ndt::var_dim_type::data_type *>(src[0])->size;
        if (src0_inner_stride_first != 0) {
          --inner_size;
        }

        char *src0_data = reinterpret_cast<ndt::var_dim_type::data_type *>(src[0])->begin;
        get_child(init_offset)->single(dst, &src0_data);
        src0_data += src0_inner_stride_first;

        get_child()->strided(dst, 0, &src0_data, &src0_inner_stride, inner_size);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
        kernel_prefix *init_child = get_child(init_offset);
        kernel_prefix *reduction_child = get_child();

        char *src0 = src[0];
        for (size_t i = 0; i != count; ++i) {
          char *src0_data = reinterpret_cast<ndt::var_dim_type::data_type *>(src0)->begin;
          init_child->single(dst, &src0_data);

          size_t inner_size = reinterpret_cast<ndt::var_dim_type::data_type *>(src0)->size;
          if (src0_inner_stride_first != 0) {
            --inner_size;
          }

          src0_data += src0_inner_stride_first;
          reduction_child->strided(dst, 0, &src0_data, &src0_inner_stride,
                                   reinterpret_cast<ndt::var_dim_type::data_type *>(src0)->size - 1);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t size) {
        kernel_prefix *child = get_child();

        char *src0 = src[0];
        for (size_t i = 0; i != size; ++i) {
          child->strided(dst, 0, &reinterpret_cast<ndt::var_dim_type::data_type *>(src0)->begin, &src0_inner_stride,
                         reinterpret_cast<ndt::var_dim_type::data_type *>(src0)->size);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }
    };

    /**
     * STRIDED INITIAL BROADCAST DIMENSION
     * This ckernel handles one dimension of the reduction processing,
     * where:
     *  - It's a broadcast dimension, so dst_stride is not zero.
     *  - It's an initial dimension, there are additional dimensions
     *    being processed after this one.
     *  - The source data is strided.
     *
     * Requirements:
     *  - The child first_call function must be *strided*.
     *  - The child followup_call function must be *strided*.
     *
     */
    template <>
    struct reduction_kernel<fixed_dim_id, true, false>
        : base_reduction_kernel<reduction_kernel<fixed_dim_id, true, false>> {
      intptr_t _size;
      intptr_t dst_stride, src_stride;

      reduction_kernel(std::intptr_t size, std::intptr_t dst_stride, std::intptr_t src_stride)
          : _size(size), dst_stride(dst_stride), src_stride(src_stride) {}

      ~reduction_kernel() { get_child()->destroy(); }

      void single_first(char *dst, char *const *src) {
        reduction_kernel_prefix *child = get_reduction_child();
        child->strided_first(dst, dst_stride, src, &src_stride, _size);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
        reduction_kernel_prefix *echild = reinterpret_cast<reduction_kernel_prefix *>(this->get_child());
        kernel_strided_t opchild_first_call = echild->get_first_call_function<kernel_strided_t>();
        kernel_strided_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = this->_size;
        intptr_t inner_dst_stride = this->dst_stride;
        intptr_t inner_src_stride = this->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we have one "first", followed by many
          // "followup"
          // calls
          opchild_first_call(echild, dst, inner_dst_stride, &src0, &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            opchild_followup_call(echild, dst, inner_dst_stride, &src0, &inner_src_stride, inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop is
          // "first"
          for (size_t i = 0; i != count; ++i) {
            opchild_first_call(echild, dst, inner_dst_stride, &src0, &inner_src_stride, inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                            size_t count) {
        reduction_kernel_prefix *reduction_child = this->get_reduction_child();

        char *src0 = src[0];
        for (size_t i = 0; i != count; ++i) {
          reduction_child->strided_followup(dst, this->dst_stride, &src0, &this->src_stride, this->_size);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }
    };

    /**
     * STRIDED INNER BROADCAST DIMENSION
     * This ckernel handles one dimension of the reduction processing,
     * where:
     *  - It's a broadcast dimension, so dst_stride is not zero.
     *  - It's an inner dimension, calling the reduction kernel directly.
     *  - The source data is strided.
     *
     * Requirements:
     *  - The child reduction kernel must be *strided*.
     *  - The child destination initialization kernel must be *strided*.
     *
     */
    template <>
    struct reduction_kernel<fixed_dim_id, true, true>
        : base_reduction_kernel<reduction_kernel<fixed_dim_id, true, true>> {
      // The code assumes that size >= 1
      intptr_t _size;
      intptr_t dst_stride, src_stride;
      size_t dst_init_kernel_offset;

      intptr_t size_first;
      intptr_t dst_stride_first;
      intptr_t src_stride_first;

      reduction_kernel(intptr_t dst_stride, intptr_t src_stride) : dst_stride(dst_stride), src_stride(src_stride) {}

      ~reduction_kernel() {
        // The reduction kernel
        get_child()->destroy();
        // The destination initialization kernel
        get_child(dst_init_kernel_offset)->destroy();
      }

      void single_first(char *dst, char *const *src) {
        // Initialize the dst values
        get_child(dst_init_kernel_offset)->strided(dst, dst_stride, src, &src_stride_first, _size);
        if (src_stride_first == 0) {
          // Then do the accumulation
          get_child()->strided(dst, dst_stride, src, &src_stride, _size);
        }
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
        kernel_prefix *init_child = get_child(dst_init_kernel_offset);
        kernel_prefix *reduction_child = get_child();

        intptr_t inner_size = this->_size;
        intptr_t inner_dst_stride = this->dst_stride;
        intptr_t inner_src_stride = this->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          init_child->strided(dst, inner_dst_stride, &src0, &this->src_stride_first, inner_size);
          dst += dst_stride_first;
          src0 += src_stride_first;
          for (size_t i = 1; i != count; ++i) {
            reduction_child->strided(dst, inner_dst_stride, &src0, &inner_src_stride, inner_size);
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, every iteration is an initialization
          for (size_t i = 0; i != count; ++i) {
            init_child->strided(dst, inner_dst_stride, &src0, &src_stride_first, _size);
            if (src_stride_first == 0) {
              reduction_child->strided(dst, inner_dst_stride, &src0, &inner_src_stride, _size);
            }

            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                            size_t count) {
        kernel_prefix *echild_reduce = get_child();
        // No initialization, all reduction
        kernel_strided_t opchild_reduce = echild_reduce->get_function<kernel_strided_t>();
        intptr_t inner_size = this->_size;
        intptr_t inner_dst_stride = this->dst_stride;
        intptr_t inner_src_stride = this->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          opchild_reduce(echild_reduce, dst, inner_dst_stride, &src0, &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
