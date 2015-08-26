//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/gfunc/call_callable.hpp>
#include <dynd/func/constant.hpp>
#include <dynd/kernels/constant_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct reduction_kernel_prefix : ckernel_prefix {
      struct static_data_type {
        callable child;

        callable_property properties;

        static_data_type(const callable &child) : child(child), properties(commutative | left_associative)
        {
        }
      };

      struct data_type {
        array identity;
        std::size_t ndim;          // total number of dimensions being processed
        std::intptr_t reduce_ndim; // number of dimensions being reduced
        int32 *axes;
        bool keepdims;

        ndt::type child_src_tp;
      };

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
        function = reinterpret_cast<void *>(fnptr);
      }

      expr_strided_t get_followup_call_function() const
      {
        return followup_call_function;
      }

      void set_followup_call_function(expr_strided_t fnptr)
      {
        followup_call_function = fnptr;
      }

      void single_first(char *dst, char *const *src)
      {
        (*reinterpret_cast<expr_single_t>(function))(this, dst, src);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        (*reinterpret_cast<expr_strided_t>(function))(this, dst, dst_stride, src, src_stride, count);
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        (*reinterpret_cast<expr_strided_t>(followup_call_function))(this, dst, dst_stride, src, src_stride, count);
      }
    };

    template <typename SelfType>
    struct base_reduction_kernel : kernel_prefix_wrapper<reduction_kernel_prefix, SelfType> {
      typedef kernel_prefix_wrapper<reduction_kernel_prefix, SelfType> wrapper_type;

      reduction_kernel_prefix *get_reduction_child()
      {
        return reinterpret_cast<reduction_kernel_prefix *>(this->get_child());
      }

      template <typename... A>
      static SelfType *init(reduction_kernel_prefix *prefix, kernel_request_t kernreq, A &&... args)
      {
        SelfType *self = wrapper_type::init(prefix, kernreq, std::forward<A>(args)...);
        // Get the function pointer for the first_call
        if (kernreq == kernel_request_single) {
          prefix->set_first_call_function(&SelfType::single_first_wrapper);
        } else if (kernreq == kernel_request_strided) {
          prefix->set_first_call_function(&SelfType::strided_first_wrapper);
        } else {
          std::stringstream ss;
          ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
          throw std::runtime_error(ss.str());
        }
        // The function pointer for followup accumulation calls
        prefix->set_followup_call_function(&SelfType::strided_followup_wrapper);

        return self;
      }

      static void single_first_wrapper(ckernel_prefix *self, char *dst, char *const *src)
      {
        return reinterpret_cast<SelfType *>(self)->single_first(dst, src);
      }

      static void strided_first_wrapper(ckernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                        const intptr_t *src_stride, size_t count)

      {
        return reinterpret_cast<SelfType *>(self)->strided_first(dst, dst_stride, src, src_stride, count);
      }

      static void strided_followup_wrapper(ckernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
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
    template <type_id_t Src0TypeID>
    struct reduction_kernel;

    template <>
    struct reduction_kernel<fixed_dim_type_id> : base_reduction_kernel<reduction_kernel<fixed_dim_type_id>> {
      intptr_t size;
      intptr_t src_stride;

      reduction_kernel(std::intptr_t size, std::intptr_t src_stride) : size(size), src_stride(src_stride)
      {
      }

      ~reduction_kernel()
      {
        get_child()->destroy();
      }

      void single_first(char *dst, char *const *src)
      {
        reduction_kernel_prefix *child = get_reduction_child();
        // The first call at the "dst" address
        child->single_first(dst, src);
        if (size > 1) {
          // All the followup calls at the "dst" address
          char *src_second = src[0] + src_stride;
          child->strided_followup(dst, 0, &src_second, &src_stride, size - 1);
        }
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        reduction_kernel_prefix *child = get_reduction_child();
        intptr_t inner_size = size;
        intptr_t inner_src_stride = this->src_stride;
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
          // With a zero stride, we have one "first", followed by many
          // "followup" calls
          child->single_first(dst, &src0);
          if (inner_size > 1) {
            char *inner_src_second = src0 + inner_src_stride;
            child->strided_followup(dst, 0, &inner_src_second, &inner_src_stride, inner_size - 1);
          }
          src0 += src0_stride;
          for (intptr_t i = 1; i < (intptr_t)count; ++i) {
            child->strided_followup(dst, 0, &src0, &inner_src_stride, inner_size);
            src0 += src0_stride;
          }
        } else {
          // With a non-zero stride, each iteration of the outer loop is
          // "first"
          for (size_t i = 0; i != count; ++i) {
            child->single_first(dst, &src0);
            if (inner_size > 1) {
              char *inner_src_second = src0 + inner_src_stride;
              child->strided_followup(dst, 0, &inner_src_second, &inner_src_stride, inner_size - 1);
            }
            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        reduction_kernel_prefix *child = get_reduction_child();
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          child->strided_followup(dst, 0, &src0, &this->src_stride, size);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being reduced, and is not
       * the final dimension before the accumulation operation.
       */
      static intptr_t instantiate(char *DYND_UNUSED(static_data), std::size_t DYND_UNUSED(data_size),
                                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                  const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
                                  intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        intptr_t src_size = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
        intptr_t src_stride = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_stride(src_arrmeta[0]);

        make(ckb, kernreq, ckb_offset, src_size, src_stride);
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
    template <type_id_t Src0TypeID>
    struct inner_reduction_kernel;

    template <>
    struct inner_reduction_kernel<fixed_dim_type_id> : base_reduction_kernel<
                                                           inner_reduction_kernel<fixed_dim_type_id>> {
      typedef inner_reduction_kernel self_type;

      // The code assumes that size >= 1
      intptr_t size_first;
      intptr_t src_stride_first;
      intptr_t size;
      intptr_t src_stride;
      size_t init_offset;

      ~inner_reduction_kernel()
      {
        get_child()->destroy();
        get_child(init_offset)->destroy();
      }

      void single_first(char *dst, char *const *src)
      {
        char *src0 = src[0];

        // Initialize the dst values
        get_child(init_offset)->single(dst, src);
        src0 += src_stride_first;

        // Do the reduction
        get_child()->strided(dst, 0, &src0, &src_stride, size_first);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        ckernel_prefix *init_child = get_child(init_offset);
        ckernel_prefix *reduction_child = get_child();

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

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        ckernel_prefix *reduce_child = get_child();

        // No initialization, all reduction
        char *src0 = src[0];
        for (size_t i = 0; i != count; ++i) {
          reduce_child->strided(dst, 0, &src0, &this->src_stride, size);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being reduced, and is
       * the final dimension before the accumulation operation.
       */
      static size_t instantiate(static_data_type *static_data, std::size_t DYND_UNUSED(data_size), data_type *data,
                                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_init_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        ndt::type src_tp[1];

        intptr_t src_size, src_stride;
        if (src_init_tp[0].is_scalar()) {
          src_tp[0] = src_init_tp[0];
          src_size = 1;
          src_stride = 0;
        } else {
          src_tp[0] = src_init_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
          src_size = src_init_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
          src_stride = src_init_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_stride(src_arrmeta[0]);
        }

        callable_type_data *elwise_reduction = static_data->child.get();
        const ndt::callable_type *elwise_reduction_tp = static_data->child.get_type();
        const array &identity = data->identity;

        intptr_t root_ckb_offset = ckb_offset;

        inner_reduction_kernel *e = make(ckb, kernreq, ckb_offset);
        // The striding parameters
        e->src_stride = src_stride;
        e->size = src_size;
        if (identity.is_null()) {
          e->size_first = e->size - 1;
          e->src_stride_first = e->src_stride;
        } else {
          e->size_first = e->size;
          e->src_stride_first = 0;
        }

        ckb_offset =
            elwise_reduction->instantiate(elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                          elwise_reduction_tp->get_npos(), src_tp, src_arrmeta, kernel_request_strided,
                                          ectx, 0, NULL, std::map<std::string, ndt::type>());
        // Need to retrieve 'e' again because it may have moved
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<inner_reduction_kernel>(root_ckb_offset);
        e->init_offset = ckb_offset - root_ckb_offset;
        if (identity.is_null()) {
          ckb_offset = make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                              kernel_request_single, ectx);
        } else {
          ckb_offset = functional::constant_kernel::instantiate(
              reinterpret_cast<char *>(const_cast<nd::array *>(&identity)), 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, 1, src_tp, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
        }

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

    template <type_id_t Src0TypeID, bool Inner = false>
    struct reduction_broadcast_kernel;

    template <>
    struct reduction_broadcast_kernel<
        fixed_dim_type_id, false> : base_reduction_kernel<reduction_broadcast_kernel<fixed_dim_type_id, false>> {
      intptr_t size;
      intptr_t dst_stride, src_stride;

      reduction_broadcast_kernel(std::intptr_t size, std::intptr_t dst_stride, std::intptr_t src_stride)
          : size(size), dst_stride(dst_stride), src_stride(src_stride)
      {
      }

      ~reduction_broadcast_kernel()
      {
        get_child()->destroy();
      }

      void single_first(char *dst, char *const *src)
      {
        reduction_kernel_prefix *child = get_reduction_child();
        child->strided_first(dst, dst_stride, src, &src_stride, size);
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        reduction_kernel_prefix *echild = reinterpret_cast<reduction_kernel_prefix *>(this->get_child());
        expr_strided_t opchild_first_call = echild->get_first_call_function<expr_strided_t>();
        expr_strided_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = this->size;
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

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        reduction_kernel_prefix *reduction_child = this->get_reduction_child();

        char *src0 = src[0];
        for (size_t i = 0; i != count; ++i) {
          reduction_child->strided_followup(dst, this->dst_stride, &src0, &this->src_stride, this->size);
          dst += dst_stride;
          src0 += src_stride[0];
        }
      }

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being broadcast, and is not
       * the final dimension before the accumulation operation.
       */
      static size_t instantiate(char *DYND_UNUSED(static_data), std::size_t DYND_UNUSED(data_size),
                                char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        intptr_t src_size = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
        intptr_t src_stride = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_stride(src_arrmeta[0]);

        intptr_t dst_stride = dst_tp.extended<ndt::fixed_dim_type>()->get_fixed_stride(dst_arrmeta);

        make(ckb, kernreq, ckb_offset, src_size, dst_stride, src_stride);
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
    struct inner_reduction_broadcast_kernel : base_reduction_kernel<inner_reduction_broadcast_kernel> {

      // The code assumes that size >= 1
      intptr_t size;
      intptr_t dst_stride, src_stride;
      size_t dst_init_kernel_offset;

      intptr_t size_first;
      intptr_t dst_stride_first;
      intptr_t src_stride_first;

      inner_reduction_broadcast_kernel(intptr_t dst_stride, intptr_t src_stride)
          : dst_stride(dst_stride), src_stride(src_stride)
      {
      }

      ~inner_reduction_broadcast_kernel()
      {
        // The reduction kernel
        get_child()->destroy();
        // The destination initialization kernel
        get_child(dst_init_kernel_offset)->destroy();
      }

      void single_first(char *dst, char *const *src)
      {
        // Initialize the dst values
        get_child(dst_init_kernel_offset)->strided(dst, dst_stride, src, &src_stride_first, size);
        if (src_stride_first == 0) {
          // Then do the accumulation
          get_child()->strided(dst, dst_stride, src, &src_stride, size);
        }
      }

      void strided_first(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        ckernel_prefix *init_child = get_child(dst_init_kernel_offset);
        ckernel_prefix *reduction_child = get_child();

        intptr_t inner_size = this->size;
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
            init_child->strided(dst, inner_dst_stride, &src0, &src_stride_first, size);
            if (src_stride_first == 0) {
              reduction_child->strided(dst, inner_dst_stride, &src0, &inner_src_stride, size);
            }

            dst += dst_stride;
            src0 += src0_stride;
          }
        }
      }

      void strided_followup(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        ckernel_prefix *echild_reduce = get_child();
        // No initialization, all reduction
        expr_strided_t opchild_reduce = echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = this->size;
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

      /**
       * Adds a ckernel layer for processing one dimension of the reduction.
       * This is for a strided dimension which is being broadcast, and is
       * the final dimension before the accumulation operation.
       */
      static size_t instantiate(static_data_type *static_data, std::size_t DYND_UNUSED(data_size), data_type *data,
                                void *ckb, intptr_t ckb_offset, const ndt::type &dst_i_tp, const char *dst_arrmeta,
                                intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const ndt::type &src_child_tp = src_tp[0].extended<ndt::base_dim_type>()->get_element_type();
        const ndt::type &dst_tp = dst_i_tp.extended<ndt::fixed_dim_type>()->get_element_type();

        intptr_t src_size = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
        intptr_t src_stride = src_tp[0].extended<ndt::fixed_dim_type>()->get_fixed_stride(src_arrmeta[0]);
        intptr_t dst_stride = dst_tp.extended<ndt::fixed_dim_type>()->get_fixed_stride(dst_arrmeta);

        callable_type_data *elwise_reduction = static_data->child.get();
        const array &identity = data->identity;

        intptr_t root_ckb_offset = ckb_offset;
        inner_reduction_broadcast_kernel *e = make(ckb, kernreq, ckb_offset, dst_stride, src_stride);

        // The striding parameters
        e->size = src_size;

        ckb_offset = elwise_reduction->instantiate(elwise_reduction->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
                                                   dst_arrmeta, nsrc, &src_child_tp, src_arrmeta,
                                                   kernel_request_strided, ectx, nkwd - 3, kwds + 3, tp_vars);
        // Make sure there's capacity for the next ckernel
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
        // Need to retrieve 'e' again because it may have moved
        e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                ->get_at<inner_reduction_broadcast_kernel>(root_ckb_offset);
        e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
        if (identity.is_null()) {
          e->size_first = e->size - 1;
          e->dst_stride_first = e->dst_stride;
          e->src_stride_first = e->src_stride;

          ckb_offset = make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_child_tp, src_arrmeta[0],
                                              kernel_request_strided, ectx);
        } else {
          e->size_first = e->size;
          e->dst_stride_first = 0;
          e->src_stride_first = 0;

          ckb_offset = functional::constant_kernel::instantiate(
              reinterpret_cast<char *>(const_cast<nd::array *>(&identity)), 0, NULL, ckb, ckb_offset, dst_tp,
              dst_arrmeta, nsrc, src_tp, src_arrmeta, kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        }

        return ckb_offset;
      }
    };

    struct reduction_virtual_kernel : reduction_kernel_prefix {
      static void data_init(static_data_type *static_data, std::size_t DYND_UNUSED(data_size), data_type *data,
                            const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                            const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        new (data) data_type();

        const array &identity = kwds[1];
        if (!identity.is_missing()) {
          data->identity = identity;
        }

        if (kwds[0].is_missing()) {
          data->reduce_ndim = src_tp[0].get_ndim() - static_data->child.get_type()->get_return_type().get_ndim();
          data->axes = NULL;
        } else {
          data->reduce_ndim = kwds[0].get_dim_size();
          data->axes = const_cast<int *>(reinterpret_cast<const int *>(kwds[0].get_readonly_originptr()));
        }

        if (kwds[2].is_missing()) {
          data->keepdims = false;
        } else {
          data->keepdims = kwds[2].as<bool>();
        }

        const ndt::type &child_dst_tp = static_data->child.get_type()->get_return_type();
        if (!dst_tp.is_symbolic()) {
          data->ndim = src_tp[0].get_ndim() - child_dst_tp.get_ndim();
        }

        if (static_data->child.get()->data_size != 0) {
          static_data->child.get()->data_init(
              static_data->child.get()->static_data, static_data->child.get()->data_size,
              reinterpret_cast<char *>(data) + sizeof(data_type), child_dst_tp, nsrc, src_tp, nkwd - 3, kwds, tp_vars);
        }
      }

      static void resolve_dst_type(static_data_type *static_data, std::size_t DYND_UNUSED(data_size), data_type *data,
                                   ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                                   const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        ndt::type child_dst_tp = static_data->child.get_type()->get_return_type();
        if (child_dst_tp.is_symbolic()) {
          data->child_src_tp = src_tp[0].get_type_at_dimension(NULL, data->reduce_ndim);
          static_data->child.get()->resolve_dst_type(static_data->child.get()->static_data, 0, NULL, child_dst_tp, nsrc,
                                                     &data->child_src_tp, nkwd, kwds, tp_vars);
        }

        // check that the child_dst_tp and the child_src_tp are the same here

        dst_tp = child_dst_tp;
        data->ndim = src_tp[0].get_ndim() - dst_tp.get_ndim();

        //        std::vector<const ndt::type *> element_tp =
        //          src_tp[0].extended<ndt::base_dim_type>()->get_element_types(
        //            data->ndim);
        //  for (auto tp : element_tp) {
        //          std::cout << *tp << std::endl;
        //}

        for (intptr_t i = data->ndim - 1, j = data->reduce_ndim - 1; i >= 0; --i) {
          if (data->axes == NULL || (j >= 0 && i == data->axes[j])) {
            if (data->keepdims) {
              dst_tp = ndt::make_fixed_dim(1, dst_tp);
            }
            --j;
          } else {
            ndt::type dim_tp = src_tp[0].get_type_at_dimension(NULL, i);
            dst_tp = dim_tp.extended<ndt::base_dim_type>()->with_element_type(dst_tp);
          }
        }
      }

      static intptr_t instantiate(static_data_type *static_data, std::size_t data_size, data_type *data, void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        if (data->reduce_ndim == 0) {
          if (data->ndim == 0) {
            // If there are no dimensions to reduce, it's
            // just a dst_initialization operation, so create
            // that ckernel directly
            if (data->identity.is_null()) {
              return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq,
                                            ectx);
            } else {
              // Create the kernel which copies the identity and then
              // does one reduction
              return inner_reduction_kernel<fixed_dim_type_id>::instantiate(
                  static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                  kernreq, ectx, nkwd, kwds, tp_vars);
            }
          }
          throw std::runtime_error("make_lifted_reduction_ckernel: no dimensions were "
                                   "flagged for reduction");
        }

        if (!(data->reduce_ndim == 1 ||
              (static_data->properties & left_associative && static_data->properties & commutative))) {
          throw std::runtime_error("make_lifted_reduction_ckernel: for reducing "
                                   "along multiple dimensions,"
                                   " the reduction function must be both "
                                   "associative and commutative");
        }
        if (static_data->properties & right_associative) {
          throw std::runtime_error("make_lifted_reduction_ckernel: right_associative is "
                                   "not yet supported");
        }

        ndt::type dst_i_tp = dst_tp, src_i_tp = src_tp[0];
        const char *src_i_arrmeta = src_arrmeta[0];
        const char *dst_i_arrmeta = dst_arrmeta;
        for (intptr_t i = 0, j = 0; i < static_cast<intptr_t>(data->ndim); ++i) {
          if ((data->axes == NULL) || (j < data->reduce_ndim && i == data->axes[j])) {
            // This dimension is being reduced
            if (static_cast<size_t>(i) < data->ndim - 1) {
              // An initial dimension being reduced
              ckb_offset = reduction_kernel<fixed_dim_type_id>::instantiate(
                  reinterpret_cast<char *>(static_data), data_size, reinterpret_cast<char *>(data), ckb, ckb_offset,
                  dst_tp, dst_arrmeta, nsrc, &src_i_tp, &src_i_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
              // The next request should be single, as that's the kind of
              // ckernel the 'first_call' should be in this case
              kernreq = kernel_request_single;
            } else {
              // The innermost dimension being reduced
              return inner_reduction_kernel<fixed_dim_type_id>::instantiate(
                  static_data, data_size, data, ckb, ckb_offset, dst_i_tp, dst_arrmeta, nsrc, &src_i_tp, &src_i_arrmeta,
                  kernreq, ectx, nkwd, kwds, tp_vars);
            }
            ++j;
          } else {
            // This dimension is being broadcast, not reduced
            if (static_cast<size_t>(i) < data->ndim - 1) {
              // An initial dimension being broadcast
              ckb_offset = reduction_broadcast_kernel<fixed_dim_type_id>::instantiate(
                  reinterpret_cast<char *>(static_data), data_size, reinterpret_cast<char *>(data), ckb, ckb_offset,
                  dst_i_tp, dst_i_arrmeta, nsrc, &src_i_tp, &src_i_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
              // The next request should be strided, as that's the kind of
              // ckernel the 'first_call' should be in this case
              kernreq = kernel_request_strided;
              dst_i_tp = dst_i_tp.extended<ndt::base_dim_type>()->get_element_type();
              dst_i_arrmeta += sizeof(size_stride_t);
            } else {
              // The innermost dimension being broadcast
              return inner_reduction_broadcast_kernel::instantiate(static_data, data_size, data, ckb, ckb_offset,
                                                                   dst_i_tp, dst_i_arrmeta, nsrc, &src_i_tp,
                                                                   &src_i_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
            }
          }

          src_i_tp = src_i_tp.extended<ndt::base_dim_type>()->get_element_type();
          src_i_arrmeta += sizeof(size_stride_t);
          if (data->keepdims) {
            dst_i_tp = dst_i_tp.extended<ndt::base_dim_type>()->get_element_type();
            dst_i_arrmeta += sizeof(size_stride_t);
          }
        }

        throw std::runtime_error("make_lifted_reduction_ckernel: internal error, "
                                 "should have returned in the loop");
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
