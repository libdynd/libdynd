//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct reduction_ckernel_prefix : ckernel_prefix {
      // This function pointer is for all the calls of the function
      // on a given destination data address after the "first call".
      expr_strided_t followup_call_function;

      template <typename T>
      T get_first_call_function() const
      {
        return get_function<T>();
      }

      template <typename T>
      void set_first_call_function(T fnptr)
      {
        set_function<T>(fnptr);
      }

      expr_strided_t get_followup_call_function() const
      {
        return followup_call_function;
      }

      void set_followup_call_function(expr_strided_t fnptr)
      {
        followup_call_function = fnptr;
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
    struct strided_initial_reduction_kernel_extra
        : nd::base_kernel<strided_initial_reduction_kernel_extra,
                          kernel_request_host, 1, reduction_ckernel_prefix> {
      typedef strided_initial_reduction_kernel_extra self_type;

      // The code assumes that size >= 1
      intptr_t size;
      intptr_t src_stride;

      static void single_first(ckernel_prefix *extra, char *dst,
                               char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        // The first call at the "dst" address
        expr_single_t opchild_first_call =
            echild->get_first_call_function<expr_single_t>();
        opchild_first_call(echild, dst, src);
        if (e->size > 1) {
          // All the followup calls at the "dst" address
          expr_strided_t opchild = echild->get_followup_call_function();
          char *src_second = src[0] + e->src_stride;
          opchild(echild, dst, 0, &src_second, &e->src_stride, e->size - 1);
        }
      }

      static void strided_first(ckernel_prefix *extra, char *dst,
                                intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        expr_strided_t opchild_followup_call =
            echild->get_followup_call_function();
        expr_single_t opchild_first_call =
            echild->get_first_call_function<expr_single_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we have one "first", followed by many
          // "followup"
          // calls
          opchild_first_call(echild, dst, &src0);
          if (inner_size > 1) {
            char *inner_src_second = src0 + inner_src_stride;
            opchild_followup_call(echild, dst, 0, &inner_src_second,
                                  &inner_src_stride, inner_size - 1);
          }
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            opchild_followup_call(echild, dst, 0, &src0, &inner_src_stride,
                                  inner_size);
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop is "first"
          for (size_t i = 0; i != count; ++i) {
            opchild_first_call(echild, dst, &src0);
            if (inner_size > 1) {
              char *inner_src_second = src0 + inner_src_stride;
              opchild_followup_call(echild, dst, 0, &inner_src_second,
                                    &inner_src_stride, inner_size - 1);
            }
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_followup(ckernel_prefix *extra, char *dst,
                                   intptr_t dst_stride, char *const *src,
                                   const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        expr_strided_t opchild_followup_call =
            echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          opchild_followup_call(echild, dst, 0, &src0, &inner_src_stride,
                                inner_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }

      void destruct_children()
      {
        get_child_ckernel()->destroy();
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
    struct strided_initial_broadcast_kernel_extra
        : nd::base_kernel<strided_initial_broadcast_kernel_extra,
                          kernel_request_host, 1, reduction_ckernel_prefix> {
      typedef strided_initial_broadcast_kernel_extra self_type;

      // The code assumes that size >= 1
      intptr_t size;
      intptr_t dst_stride, src_stride;

      static void single_first(ckernel_prefix *extra, char *dst,
                               char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        expr_strided_t opchild_first_call =
            echild->get_first_call_function<expr_strided_t>();
        opchild_first_call(echild, dst, e->dst_stride, src, &e->src_stride,
                           e->size);
      }

      static void strided_first(ckernel_prefix *extra, char *dst,
                                intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        expr_strided_t opchild_first_call =
            echild->get_first_call_function<expr_strided_t>();
        expr_strided_t opchild_followup_call =
            echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we have one "first", followed by many
          // "followup"
          // calls
          opchild_first_call(echild, dst, inner_dst_stride, &src0,
                             &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            opchild_followup_call(echild, dst, inner_dst_stride, &src0,
                                  &inner_src_stride, inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop is "first"
          for (size_t i = 0; i != count; ++i) {
            opchild_first_call(echild, dst, inner_dst_stride, &src0,
                               &inner_src_stride, inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_followup(ckernel_prefix *extra, char *dst,
                                   intptr_t dst_stride, char *const *src,
                                   const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        reduction_ckernel_prefix *echild =
            reinterpret_cast<reduction_ckernel_prefix *>(
                e->get_child_ckernel());
        expr_strided_t opchild_followup_call =
            echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          opchild_followup_call(echild, dst, inner_dst_stride, &src0,
                                &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }

      void destruct_children()
      {
        get_child_ckernel()->destroy();
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
    struct strided_inner_reduction_kernel_extra
        : nd::base_kernel<strided_inner_reduction_kernel_extra,
                          kernel_request_host, 1, reduction_ckernel_prefix> {
      typedef strided_inner_reduction_kernel_extra self_type;

      // The code assumes that size >= 1
      intptr_t size;
      intptr_t src_stride;
      size_t dst_init_kernel_offset;
      // For the case with a reduction identity
      const char *ident_data;
      memory_block_data *ident_ref;

      static void single_first(ckernel_prefix *extra, char *dst,
                               char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce = e->get_child_ckernel();
        ckernel_prefix *echild_dst_init =
            e->get_child_ckernel(e->dst_init_kernel_offset);
        // The first call to initialize the "dst" value
        expr_single_t opchild_dst_init =
            echild_dst_init->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        opchild_dst_init(echild_dst_init, dst, src);
        if (e->size > 1) {
          // All the followup calls to accumulate at the "dst" address
          char *child_src = src[0] + e->src_stride;
          opchild_reduce(echild_reduce, dst, 0, &child_src, &e->src_stride,
                         e->size - 1);
        }
      }

      static void single_first_with_ident(ckernel_prefix *extra, char *dst,
                                          char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        // The first call to initialize the "dst" value
        expr_single_t opchild_ident =
            echild_ident->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        opchild_ident(echild_ident, dst,
                      const_cast<char *const *>(&e->ident_data));
        // All the followup calls to accumulate at the "dst" address
        opchild_reduce(echild_reduce, dst, 0, src, &e->src_stride, e->size);
      }

      static void strided_first(ckernel_prefix *extra, char *dst,
                                intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_single_t opchild_dst_init =
            echild_dst_init->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          opchild_dst_init(echild_dst_init, dst, &src0);
          if (inner_size > 1) {
            char *inner_child_src = src0 + inner_src_stride;
            opchild_reduce(echild_reduce, dst, 0, &inner_child_src,
                           &inner_src_stride, inner_size - 1);
          }
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            opchild_reduce(echild_reduce, dst, 0, &src0, &inner_src_stride,
                           inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop has to
          // initialize then reduce
          for (size_t i = 0; i != count; ++i) {
            opchild_dst_init(echild_dst_init, dst, &src0);
            if (inner_size > 1) {
              char *inner_child_src = src0 + inner_src_stride;
              opchild_reduce(echild_reduce, dst, 0, &inner_child_src,
                             &inner_src_stride, inner_size - 1);
            }
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_first_with_ident(ckernel_prefix *extra, char *dst,
                                           intptr_t dst_stride,
                                           char *const *src,
                                           const intptr_t *src_stride,
                                           size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_single_t opchild_ident =
            echild_ident->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        const char *ident_data = e->ident_data;
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          opchild_ident(echild_ident, dst,
                        const_cast<char *const *>(&ident_data));
          for (intptr_t i = 0; i < (intptr_t)count; ++i) {
            opchild_reduce(echild_reduce, dst, 0, &src0, &inner_src_stride,
                           inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop has to
          // initialize then reduce
          for (size_t i = 0; i != count; ++i) {
            opchild_ident(echild_ident, dst,
                          const_cast<char *const *>(&ident_data));
            opchild_reduce(echild_reduce, dst, 0, &src0, &inner_src_stride,
                           inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_followup(ckernel_prefix *extra, char *dst,
                                   intptr_t dst_stride, char *const *src,
                                   const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        // No initialization, all reduction
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          opchild_reduce(echild_reduce, dst, 0, &src0, &inner_src_stride,
                         inner_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }

      void destruct_children()
      {
        if (ident_ref != NULL) {
          memory_block_decref(ident_ref);
        }
        // The reduction kernel
        get_child_ckernel()->destroy();
        // The destination initialization kernel
        destroy_child_ckernel(dst_init_kernel_offset);
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
    struct strided_inner_broadcast_kernel_extra
        : nd::base_kernel<strided_inner_broadcast_kernel_extra,
                          kernel_request_host, 1, reduction_ckernel_prefix> {
      typedef strided_inner_broadcast_kernel_extra self_type;

      // The code assumes that size >= 1
      intptr_t size;
      intptr_t dst_stride, src_stride;
      size_t dst_init_kernel_offset;
      // For the case with a reduction identity
      const char *ident_data;
      memory_block_data *ident_ref;

      static void single_first(ckernel_prefix *extra, char *dst,
                               char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_strided_t opchild_dst_init =
            echild_dst_init->get_function<expr_strided_t>();
        // All we do is initialize the dst values
        opchild_dst_init(echild_dst_init, dst, e->dst_stride, src,
                         &e->src_stride, e->size);
      }

      static void single_first_with_ident(ckernel_prefix *extra, char *dst,
                                          char *const *src)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        expr_strided_t opchild_ident =
            echild_ident->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        // First initialize the dst values (TODO: Do we want to do
        // initialize/reduce
        // in
        // blocks instead of just one then the other?)
        intptr_t zero_stride = 0;
        opchild_ident(echild_ident, dst, e->dst_stride,
                      const_cast<char *const *>(&e->ident_data), &zero_stride,
                      e->size);
        // Then do the accumulation
        opchild_reduce(echild_reduce, dst, e->dst_stride, src, &e->src_stride,
                       e->size);
      }

      static void strided_first(ckernel_prefix *extra, char *dst,
                                intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        expr_strided_t opchild_dst_init =
            echild_dst_init->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          opchild_dst_init(echild_dst_init, dst, inner_dst_stride, &src0,
                           &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            opchild_reduce(echild_reduce, dst, inner_dst_stride, &src0,
                           &inner_src_stride, inner_size);
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, every iteration is an initialization
          for (size_t i = 0; i != count; ++i) {
            opchild_dst_init(echild_dst_init, dst, inner_dst_stride, &src0,
                             &inner_src_stride, e->size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_first_with_ident(ckernel_prefix *extra, char *dst,
                                           intptr_t dst_stride,
                                           char *const *src,
                                           const intptr_t *src_stride,
                                           size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        expr_strided_t opchild_ident =
            echild_ident->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we initialize "dst" once, then do many
          // accumulations
          intptr_t zero_stride = 0;
          opchild_ident(echild_ident, dst, inner_dst_stride,
                        const_cast<char *const *>(&e->ident_data), &zero_stride,
                        e->size);
          for (intptr_t i = 0; i < (intptr_t)count; ++i) {
            opchild_reduce(echild_reduce, dst, inner_dst_stride, &src0,
                           &inner_src_stride, inner_size);
            src0 += src0_stride;
          }
        } else {
          intptr_t zero_stride = 0;
          // With a non-zero stride, every iteration is an initialization
          for (size_t i = 0; i != count; ++i) {
            opchild_ident(echild_ident, dst, inner_dst_stride,
                          const_cast<char *const *>(&e->ident_data),
                          &zero_stride, inner_size);
            opchild_reduce(echild_reduce, dst, inner_dst_stride, &src0,
                           &inner_src_stride, inner_size);
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      static void strided_followup(ckernel_prefix *extra, char *dst,
                                   intptr_t dst_stride, char *const *src,
                                   const intptr_t *src_stride, size_t count)
      {
        self_type *e = reinterpret_cast<self_type *>(extra);
        ckernel_prefix *echild_reduce =
            extra->get_child_ckernel(sizeof(self_type));
        // No initialization, all reduction
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          opchild_reduce(echild_reduce, dst, inner_dst_stride, &src0,
                         &inner_src_stride, inner_size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }

      void destruct_children()
      {
        if (ident_ref != NULL) {
          memory_block_decref(ident_ref);
        }
        // The reduction kernel
        get_child_ckernel()->destroy();
        // The destination initialization kernel
        destroy_child_ckernel(dst_init_kernel_offset);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
