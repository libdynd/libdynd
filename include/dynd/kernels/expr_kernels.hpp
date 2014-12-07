//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {

namespace kernels {
  /**
   * A CRTP (curiously recurring template pattern) base class to help
   * create ckernels.
   */
  template <class CKT, kernel_request_t kernreq, int Nsrc>
  struct expr_ck;

#define EXPR_CK(KERNREQ, ...)                                                  \
  template <class CKT, int Nsrc>                                               \
  struct expr_ck<CKT, KERNREQ, Nsrc> : public general_ck<CKT, KERNREQ> {       \
    typedef CKT self_type;                                                     \
    typedef general_ck<CKT, KERNREQ> parent_type;                              \
                                                                               \
    /** Initializes just the base.function member. */                          \
    __VA_ARGS__ void init_kernfunc(kernel_request_t kernreq)                   \
    {                                                                          \
      switch (kernreq) {                                                       \
      case kernel_request_single:                                              \
        this->base.template set_function<expr_single_t>(                       \
            &self_type::single_wrapper);                                       \
        break;                                                                 \
      case kernel_request_strided:                                             \
        this->base.template set_function<expr_strided_t>(                      \
            &self_type::strided_wrapper);                                      \
        break;                                                                 \
      default:                                                                 \
        DYND_HOST_THROW(std::invalid_argument,                                 \
                        "expr ckernel init: unrecognized ckernel request " +   \
                            std::to_string(kernreq));                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void single_wrapper(char *dst, char *const *src,        \
                                           ckernel_prefix *rawself)            \
    {                                                                          \
      return parent_type::get_self(rawself)->single(dst, src);                 \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void strided_wrapper(char *dst, intptr_t dst_stride,    \
                                            char *const *src,                  \
                                            const intptr_t *src_stride,        \
                                            size_t count,                      \
                                            ckernel_prefix *rawself)           \
    {                                                                          \
      return parent_type::get_self(rawself)                                    \
          ->strided(dst, dst_stride, src, src_stride, count);                  \
    }                                                                          \
                                                                               \
    /** Default strided implementation calls single repeatedly. */             \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *src, \
                             const intptr_t *src_stride, size_t count)         \
    {                                                                          \
      self_type *self = parent_type::get_self(&this->base);                    \
      char *src_copy[Nsrc ? Nsrc : 1];                                         \
      memcpy(src_copy, src, Nsrc ? sizeof(src_copy) : 0);                      \
      for (size_t i = 0; i != count; ++i) {                                    \
        self->single(dst, src_copy);                                           \
        dst += dst_stride;                                                     \
        for (int j = 0; j < Nsrc; ++j) {                                       \
          src_copy[j] += src_stride[j];                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

  EXPR_CK(kernel_request_host)

#ifdef __CUDACC__

  EXPR_CK(kernel_request_cuda_device, __device__)
  EXPR_CK(kernel_request_host | kernel_request_cuda_device, __host__ __device__)

#endif

#undef EXPR_CK

} // namespace kernels

class expr_kernel_generator;

/**
 * Evaluates any expression types in the array of
 * source types, passing the result non-expression
 * types on to the handler to build the rest of the
 * kernel.
 */
size_t make_expression_type_expr_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, size_t src_count, const ndt::type *src_dt,
    const char **src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const expr_kernel_generator *handler);

} // namespace dynd
