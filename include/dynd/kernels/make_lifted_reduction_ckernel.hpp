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

void check_dst_initialization(const ndt::callable_type *dst_initialization_tp,
                              const ndt::type &dst_tp, const ndt::type &src_tp)
{
  if (dst_initialization_tp->get_return_type() != dst_tp) {
    std::stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "dst type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization_tp->get_pos_type(0) != src_tp) {
    std::stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "src type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
}

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

    template <typename SelfType, int N>
    struct base_reduction_kernel {
      typedef SelfType self_type;
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

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being reduced, and is not
       * the final dimension before the accumulation operation.
       */
      static size_t instantiate(char *DYND_UNUSED(static_data), void *ckb,
                                intptr_t ckb_offset, intptr_t src_stride,
                                intptr_t src_size, kernel_request_t kernreq)
      {
        nd::functional::strided_initial_reduction_kernel_extra *e =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->alloc_ck<
                      nd::functional::strided_initial_reduction_kernel_extra>(
                      ckb_offset);
        e->destructor =
            &nd::functional::strided_initial_reduction_kernel_extra::destruct;
        // Get the function pointer for the first_call
        if (kernreq == kernel_request_single) {
          e->set_first_call_function(
              &nd::functional::strided_initial_reduction_kernel_extra::
                   single_first);
        } else if (kernreq == kernel_request_strided) {
          e->set_first_call_function(
              &nd::functional::strided_initial_reduction_kernel_extra::
                   strided_first);
        } else {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: unrecognized request "
             << (int)kernreq;
          throw std::runtime_error(ss.str());
        }
        // The function pointer for followup accumulation calls
        e->set_followup_call_function(
            &nd::functional::strided_initial_reduction_kernel_extra::
                 strided_followup);
        // The striding parameters
        e->src_stride = src_stride;
        e->size = src_size;
        return ckb_offset;
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

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being broadcast, and is not
       * the final dimension before the accumulation operation.
       */
      static size_t instantiate(void *ckb, intptr_t ckb_offset,
                                intptr_t dst_stride, intptr_t src_stride,
                                intptr_t src_size, kernel_request_t kernreq)
      {
        nd::functional::strided_initial_broadcast_kernel_extra *e =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->alloc_ck<
                      nd::functional::strided_initial_broadcast_kernel_extra>(
                      ckb_offset);
        e->destructor =
            &nd::functional::strided_initial_broadcast_kernel_extra::destruct;
        // Get the function pointer for the first_call
        if (kernreq == kernel_request_single) {
          e->set_first_call_function(
              &nd::functional::strided_initial_broadcast_kernel_extra::
                   single_first);
        } else if (kernreq == kernel_request_strided) {
          e->set_first_call_function(
              &nd::functional::strided_initial_broadcast_kernel_extra::
                   strided_first);
        } else {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: unrecognized request "
             << (int)kernreq;
          throw std::runtime_error(ss.str());
        }
        // The function pointer for followup accumulation calls
        e->set_followup_call_function(
            &nd::functional::strided_initial_broadcast_kernel_extra::
                 strided_followup);
        // The striding parameters
        e->dst_stride = dst_stride;
        e->src_stride = src_stride;
        e->size = src_size;
        return ckb_offset;
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

      struct stored_data_type {
        callable child_elwise_reduction;
        callable child_dst_initialization;
        array reduction_identity;
        ndt::type data_types[2];
        intptr_t reduction_ndim;
        bool associative, commutative, right_associative;
        shortvector<bool> reduction_dimflags;
      };

      typedef std::shared_ptr<stored_data_type> static_data_type;

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

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being reduced, and is
       * the final dimension before the accumulation operation.
       *
       * If dst_initialization is NULL, an assignment kernel is used.
       */
      static size_t
      instantiate(char *_static_data, void *ckb, intptr_t ckb_offset,
                  intptr_t src_stride, intptr_t src_size,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  const ndt::type &src_tp, const char *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx)
      {
        static_data_type &static_data =
            *reinterpret_cast<static_data_type *>(_static_data);

        callable_type_data *elwise_reduction =
            static_data->child_elwise_reduction.get();
        const ndt::callable_type *elwise_reduction_tp =
            static_data->child_elwise_reduction.get_type();
        callable_type_data *dst_initialization =
            static_data->child_dst_initialization.get();
        const ndt::callable_type *dst_initialization_tp =
            static_data->child_dst_initialization.get_type();
        const array &reduction_identity = static_data->reduction_identity;

        intptr_t root_ckb_offset = ckb_offset;
        nd::functional::strided_inner_reduction_kernel_extra *e =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->alloc_ck<
                      nd::functional::strided_inner_reduction_kernel_extra>(
                      ckb_offset);
        e->destructor =
            &nd::functional::strided_inner_reduction_kernel_extra::destruct;
        // Cannot have both a dst_initialization kernel and a reduction identity
        if (dst_initialization != NULL && !reduction_identity.is_null()) {
          throw std::invalid_argument(
              "make_lifted_reduction_ckernel: cannot specify"
              " both a dst_initialization kernel and a reduction_identity");
        }
        if (reduction_identity.is_null()) {
          // Get the function pointer for the first_call, for the case with
          // no reduction identity
          if (kernreq == kernel_request_single) {
            e->set_first_call_function(
                &nd::functional::strided_inner_reduction_kernel_extra::
                     single_first);
          } else if (kernreq == kernel_request_strided) {
            e->set_first_call_function(
                &nd::functional::strided_inner_reduction_kernel_extra::
                     strided_first);
          } else {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: unrecognized request "
               << (int)kernreq;
            throw std::runtime_error(ss.str());
          }
        } else {
          // Get the function pointer for the first_call, for the case with
          // a reduction identity
          if (kernreq == kernel_request_single) {
            e->set_first_call_function(
                &nd::functional::strided_inner_reduction_kernel_extra::
                     single_first_with_ident);
          } else if (kernreq == kernel_request_strided) {
            e->set_first_call_function(
                &nd::functional::strided_inner_reduction_kernel_extra::
                     strided_first_with_ident);
          } else {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: unrecognized request "
               << (int)kernreq;
            throw std::runtime_error(ss.str());
          }
          if (reduction_identity.get_type() != dst_tp) {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: reduction identity type ";
            ss << reduction_identity.get_type() << " does not match dst type ";
            ss << dst_tp;
            throw std::runtime_error(ss.str());
          }
          e->ident_data = reduction_identity.get_readonly_originptr();
          e->ident_ref = reduction_identity.get_memblock().release();
        }
        // The function pointer for followup accumulation calls
        e->set_followup_call_function(
            &nd::functional::strided_inner_reduction_kernel_extra::
                 strided_followup);
        // The striding parameters
        e->src_stride = src_stride;
        e->size = src_size;
        // Validate that the provided callables are unary operations,
        // and have the correct types
        if (elwise_reduction_tp->get_npos() != 1 &&
            elwise_reduction_tp->get_npos() != 2) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "funcproto must be unary or a binary expr with all equal types";
          throw std::runtime_error(ss.str());
        }
        if (elwise_reduction_tp->get_return_type() != dst_tp) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "dst type is " << elwise_reduction_tp->get_return_type();
          ss << ", expected " << dst_tp;
          throw type_error(ss.str());
        }
        if (elwise_reduction_tp->get_pos_type(0) != src_tp) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "src type is " << elwise_reduction_tp->get_return_type();
          ss << ", expected " << src_tp;
          throw type_error(ss.str());
        }
        if (dst_initialization != NULL) {
          check_dst_initialization(dst_initialization_tp, dst_tp, src_tp);
        }
        if (elwise_reduction_tp->get_npos() == 2) {
          ckb_offset = wrap_binary_as_unary_reduction_ckernel(
              ckb, ckb_offset, static_data->right_associative,
              kernel_request_strided);
          ndt::type src_tp_doubled[2] = {src_tp, src_tp};
          const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
          ckb_offset = elwise_reduction->instantiate(
              elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), src_tp_doubled,
              src_arrmeta_doubled, kernel_request_strided, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        } else {
          ckb_offset = elwise_reduction->instantiate(
              elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp,
              &src_arrmeta, kernel_request_strided, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        }
        // Make sure there's capacity for the next ckernel
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset + sizeof(ckernel_prefix));
        // Need to retrieve 'e' again because it may have moved
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<nd::functional::strided_inner_reduction_kernel_extra>(
                      root_ckb_offset);
        e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
        if (dst_initialization != NULL) {
          ckb_offset = dst_initialization->instantiate(
              dst_initialization->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp,
              &src_arrmeta, kernel_request_single, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        } else if (reduction_identity.is_null()) {
          ckb_offset = make_assignment_kernel(ckb, ckb_offset, dst_tp,
                                              dst_arrmeta, src_tp, src_arrmeta,
                                              kernel_request_single, ectx);
        } else {
          ckb_offset = make_assignment_kernel(
              ckb, ckb_offset, dst_tp, dst_arrmeta,
              reduction_identity.get_type(), reduction_identity.get_arrmeta(),
              kernel_request_single, ectx);
        }

        return ckb_offset;
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

      struct stored_data_type {
        callable child_elwise_reduction;
        callable child_dst_initialization;
        array reduction_identity;
        ndt::type data_types[2];
        intptr_t reduction_ndim;
        bool associative, commutative, right_associative;
        shortvector<bool> reduction_dimflags;
      };

      typedef std::shared_ptr<stored_data_type> static_data_type;

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

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being broadcast, and is
       * the final dimension before the accumulation operation.
       */
      static size_t
      instantiate(char *_static_data, void *ckb, intptr_t ckb_offset,
                  intptr_t dst_stride, intptr_t src_stride, intptr_t src_size,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  const ndt::type &src_tp, const char *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx)
      {
        static_data_type &static_data =
            *reinterpret_cast<static_data_type *>(_static_data);

        callable_type_data *elwise_reduction =
            static_data->child_elwise_reduction.get();
        const ndt::callable_type *elwise_reduction_tp =
            static_data->child_elwise_reduction.get_type();
        callable_type_data *dst_initialization =
            static_data->child_dst_initialization.get();
        const ndt::callable_type *dst_initialization_tp =
            static_data->child_dst_initialization.get_type();
        const array &reduction_identity = static_data->reduction_identity;

        intptr_t root_ckb_offset = ckb_offset;
        nd::functional::strided_inner_broadcast_kernel_extra *e =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->alloc_ck<
                      nd::functional::strided_inner_broadcast_kernel_extra>(
                      ckb_offset);
        e->destructor =
            &nd::functional::strided_inner_broadcast_kernel_extra::destruct;
        // Cannot have both a dst_initialization kernel and a reduction identity
        if (dst_initialization != NULL && !reduction_identity.is_null()) {
          throw std::invalid_argument(
              "make_lifted_reduction_ckernel: cannot specify"
              " both a dst_initialization kernel and a reduction_identity");
        }
        if (reduction_identity.is_null()) {
          // Get the function pointer for the first_call, for the case with
          // no reduction identity
          if (kernreq == kernel_request_single) {
            e->set_first_call_function(
                &nd::functional::strided_inner_broadcast_kernel_extra::
                     single_first);
          } else if (kernreq == kernel_request_strided) {
            e->set_first_call_function(
                &nd::functional::strided_inner_broadcast_kernel_extra::
                     strided_first);
          } else {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: unrecognized request "
               << (int)kernreq;
            throw std::runtime_error(ss.str());
          }
        } else {
          // Get the function pointer for the first_call, for the case with
          // a reduction identity
          if (kernreq == kernel_request_single) {
            e->set_first_call_function(
                &nd::functional::strided_inner_broadcast_kernel_extra::
                     single_first_with_ident);
          } else if (kernreq == kernel_request_strided) {
            e->set_first_call_function(
                &nd::functional::strided_inner_broadcast_kernel_extra::
                     strided_first_with_ident);
          } else {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: unrecognized request "
               << (int)kernreq;
            throw std::runtime_error(ss.str());
          }
          if (reduction_identity.get_type() != dst_tp) {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: reduction identity type ";
            ss << reduction_identity.get_type() << " does not match dst type ";
            ss << dst_tp;
            throw std::runtime_error(ss.str());
          }
          e->ident_data = reduction_identity.get_readonly_originptr();
          e->ident_ref = reduction_identity.get_memblock().release();
        }
        // The function pointer for followup accumulation calls
        e->set_followup_call_function(
            &nd::functional::strided_inner_broadcast_kernel_extra::
                 strided_followup);
        // The striding parameters
        e->dst_stride = dst_stride;
        e->src_stride = src_stride;
        e->size = src_size;
        // Validate that the provided callables are unary operations,
        // and have the correct types
        if (elwise_reduction_tp->get_npos() != 1 &&
            elwise_reduction_tp->get_npos() != 2) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "funcproto must be unary or a binary expr with all equal types";
          throw std::runtime_error(ss.str());
        }
        if (elwise_reduction_tp->get_return_type() != dst_tp) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "dst type is " << elwise_reduction_tp->get_return_type();
          ss << ", expected " << dst_tp;
          throw type_error(ss.str());
        }
        if (elwise_reduction_tp->get_pos_type(0) != src_tp) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
          ss << "src type is " << elwise_reduction_tp->get_return_type();
          ss << ", expected " << src_tp;
          throw type_error(ss.str());
        }
        if (dst_initialization != NULL) {
          check_dst_initialization(dst_initialization_tp, dst_tp, src_tp);
        }
        if (elwise_reduction_tp->get_npos() == 2) {
          ckb_offset = wrap_binary_as_unary_reduction_ckernel(
              ckb, ckb_offset, static_data->right_associative,
              kernel_request_strided);
          ndt::type src_tp_doubled[2] = {src_tp, src_tp};
          const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
          ckb_offset = elwise_reduction->instantiate(
              elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), src_tp_doubled,
              src_arrmeta_doubled, kernel_request_strided, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        } else {
          ckb_offset = elwise_reduction->instantiate(
              elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp,
              &src_arrmeta, kernel_request_strided, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        }
        // Make sure there's capacity for the next ckernel
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset + sizeof(ckernel_prefix));
        // Need to retrieve 'e' again because it may have moved
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<nd::functional::strided_inner_broadcast_kernel_extra>(
                      root_ckb_offset);
        e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
        if (dst_initialization != NULL) {
          ckb_offset = dst_initialization->instantiate(
              dst_initialization->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, elwise_reduction_tp->get_npos(), &src_tp,
              &src_arrmeta, kernel_request_strided, ectx, nd::array(),
              std::map<std::string, ndt::type>());
        } else if (reduction_identity.is_null()) {
          ckb_offset = make_assignment_kernel(ckb, ckb_offset, dst_tp,
                                              dst_arrmeta, src_tp, src_arrmeta,
                                              kernel_request_strided, ectx);
        } else {
          ckb_offset = make_assignment_kernel(
              ckb, ckb_offset, dst_tp, dst_arrmeta,
              reduction_identity.get_type(), reduction_identity.get_arrmeta(),
              kernel_request_strided, ectx);
        }

        return ckb_offset;
      }
    };

    struct reduction_kernel {
      struct stored_data_type {
        callable child_elwise_reduction;
        callable child_dst_initialization;
        array reduction_identity;
        ndt::type data_types[2];
        intptr_t reduction_ndim;
        bool associative, commutative, right_associative;
        shortvector<bool> reduction_dimflags;
      };

      typedef std::shared_ptr<stored_data_type> static_data_type;

      static intptr_t
      instantiate(char *_static_data, size_t DYND_UNUSED(data_size),
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const array &kwds,
                  const std::map<std::string, ndt::type> &tp_vars)
      {
        static_data_type &static_data =
            *reinterpret_cast<static_data_type *>(_static_data);

        callable &elwise_reduction = static_data->child_elwise_reduction;
        const ndt::callable_type *elwise_reduction_tp =
            elwise_reduction.get_type();

        callable &dst_initialization = static_data->child_dst_initialization;

        // Count the number of dimensions being reduced
        intptr_t reducedim_count = 0;
        for (intptr_t i = 0; i < static_data->reduction_ndim; ++i) {
          reducedim_count += static_data->reduction_dimflags[i];
        }
        if (reducedim_count == 0) {
          if (static_data->reduction_ndim == 0) {
            // If there are no dimensions to reduce, it's
            // just a dst_initialization operation, so create
            // that ckernel directly
            if (!dst_initialization.is_null()) {
              return dst_initialization.get()->instantiate(
                  dst_initialization.get()->static_data, 0, NULL, ckb,
                  ckb_offset, dst_tp, dst_arrmeta,
                  elwise_reduction_tp->get_npos(), src_tp, src_arrmeta, kernreq,
                  ectx, kwds, tp_vars);
            } else if (static_data->reduction_identity.is_null()) {
              return make_assignment_kernel(ckb, ckb_offset, dst_tp,
                                            dst_arrmeta, src_tp[0],
                                            src_arrmeta[0], kernreq, ectx);
            } else {
              // Create the kernel which copies the identity and then
              // does one reduction
              return strided_inner_reduction_kernel_extra::instantiate(
                  _static_data, ckb, ckb_offset, 0, 1, dst_tp, dst_arrmeta,
                  src_tp[0], src_arrmeta[0], kernreq, ectx);
            }
          }
          throw std::runtime_error(
              "make_lifted_reduction_ckernel: no dimensions were "
              "flagged for reduction");
        }

        if (!(reducedim_count == 1 ||
              (static_data->associative && static_data->commutative))) {
          throw std::runtime_error(
              "make_lifted_reduction_ckernel: for reducing "
              "along multiple dimensions,"
              " the reduction function must be both "
              "associative and commutative");
        }
        if (static_data->right_associative) {
          throw std::runtime_error(
              "make_lifted_reduction_ckernel: right_associative is "
              "not yet supported");
        }

        ndt::type dst_el_tp = elwise_reduction_tp->get_return_type();
        ndt::type src_el_tp = elwise_reduction_tp->get_pos_type(0);

        // This is the number of dimensions being processed by the reduction
        if (static_data->reduction_ndim !=
            src_tp->get_ndim() - src_el_tp.get_ndim()) {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: wrong number of reduction "
                "dimensions, ";
          ss << "requested " << static_data->reduction_ndim
             << ", but types have ";
          ss << (src_tp[0].get_ndim() - src_el_tp.get_ndim());
          ss << " lifting from " << src_el_tp << " to " << src_tp;
          throw std::runtime_error(ss.str());
        }
        // Determine whether reduced dimensions are being kept or not
        bool keep_dims;
        if (static_data->reduction_ndim ==
            dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
          keep_dims = true;
        } else if (static_data->reduction_ndim - reducedim_count ==
                   dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
          keep_dims = false;
        } else {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: The number of dimensions "
                "flagged for "
                "reduction, ";
          ss << reducedim_count
             << ", is not consistent with the destination type ";
          ss << "reducing " << dst_tp << " with element " << dst_el_tp;
          throw std::runtime_error(ss.str());
        }

        ndt::type dst_i_tp = dst_tp, src_i_tp = src_tp[0];

        for (intptr_t i = 0; i < static_data->reduction_ndim; ++i) {
          intptr_t dst_stride, dst_size, src_stride, src_size;
          // Get the striding parameters for the source dimension
          if (!src_i_tp.get_as_strided(
                   src_arrmeta[0], &src_size, &src_stride, &src_i_tp,
                   const_cast<const char **>(src_arrmeta))) {
            std::stringstream ss;
            ss << "make_lifted_reduction_ckernel: type " << src_i_tp
               << " not supported as source";
            throw type_error(ss.str());
          }
          if (static_data->reduction_dimflags[i]) {
            // This dimension is being reduced
            if (src_size == 0 && static_data->reduction_identity.is_null()) {
              // If the size of the src is 0, a reduction identity is required
              // to get
              // a value
              std::stringstream ss;
              ss << "cannot reduce a zero-sized dimension (axis ";
              ss << i << " of " << src_i_tp << ") because the operation";
              ss << " has no identity";
              throw std::invalid_argument(ss.str());
            }
            if (keep_dims) {
              // If the dimensions are being kept, the output should be a
              // a strided dimension of size one
              if (dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                          &dst_i_tp, &dst_arrmeta)) {
                if (dst_size != 1 || dst_stride != 0) {
                  std::stringstream ss;
                  ss << "make_lifted_reduction_ckernel: destination of a "
                        "reduction "
                        "dimension ";
                  ss << "must have size 1, not size" << dst_size << "/stride "
                     << dst_stride;
                  ss << " in type " << dst_i_tp;
                  throw type_error(ss.str());
                }
              } else {
                std::stringstream ss;
                ss << "make_lifted_reduction_ckernel: type " << dst_i_tp;
                ss << " not supported the destination of a dimension being "
                      "reduced";
                throw type_error(ss.str());
              }
            }
            if (i < static_data->reduction_ndim - 1) {
              // An initial dimension being reduced
              ckb_offset = strided_initial_reduction_kernel_extra::instantiate(
                  _static_data, ckb, ckb_offset, src_stride, src_size, kernreq);
              // The next request should be single, as that's the kind of
              // ckernel the 'first_call' should be in this case
              kernreq = kernel_request_single;
            } else {
              // The innermost dimension being reduced
              return strided_inner_reduction_kernel_extra::instantiate(
                  _static_data, ckb, ckb_offset, src_stride, src_size, dst_i_tp,
                  dst_arrmeta, src_i_tp, src_arrmeta[0], kernreq, ectx);
            }
          } else {
            // This dimension is being broadcast, not reduced
            if (!dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                         &dst_i_tp, &dst_arrmeta)) {
              std::stringstream ss;
              ss << "make_lifted_reduction_ckernel: type " << dst_i_tp
                 << " not supported as destination";
              throw type_error(ss.str());
            }
            if (dst_size != src_size) {
              std::stringstream ss;
              ss << "make_lifted_reduction_ckernel: the dst dimension size "
                 << dst_size;
              ss << " must equal the src dimension size " << src_size
                 << " for broadcast dimensions";
              throw std::runtime_error(ss.str());
            }
            if (i < static_data->reduction_ndim - 1) {
              // An initial dimension being broadcast
              ckb_offset = strided_initial_broadcast_kernel_extra::instantiate(
                  ckb, ckb_offset, dst_stride, src_stride, src_size, kernreq);
              // The next request should be strided, as that's the kind of
              // ckernel the 'first_call' should be in this case
              kernreq = kernel_request_strided;
            } else {
              // The innermost dimension being broadcast
              return strided_inner_broadcast_kernel_extra::instantiate(
                  _static_data, ckb, ckb_offset, dst_stride, src_stride,
                  src_size, dst_i_tp, dst_arrmeta, src_i_tp, src_arrmeta[0],
                  kernreq, ectx);
            }
          }
        }

        throw std::runtime_error(
            "make_lifted_reduction_ckernel: internal error, "
            "should have returned in the loop");
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
