//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <typename...>
  struct kernel_prefix_wrapper;

  template <typename PrefixType>
  struct kernel_prefix_wrapper<PrefixType> : PrefixType {
  };

  template <typename SelfType, typename PrefixType>
  struct kernel_prefix_wrapper<SelfType,
                               PrefixType> : kernel_prefix_wrapper<PrefixType> {
    DYND_CUDA_HOST_DEVICE static SelfType *get_self(ckernel_prefix *rawself)
    {
      return reinterpret_cast<SelfType *>(rawself);
    }

    DYND_CUDA_HOST_DEVICE static const SelfType *
    get_self(const ckernel_prefix *rawself)
    {
      return reinterpret_cast<const SelfType *>(rawself);
    }

    template <typename CKBT>
    static SelfType *get_self(CKBT *ckb, intptr_t ckb_offset)
    {
      return ckb->template get_at<SelfType>(ckb_offset);
    }

    /**  Returns the child ckernel immediately following this one. */
    DYND_CUDA_HOST_DEVICE ckernel_prefix *get_child(intptr_t offset =
                                                        sizeof(SelfType))
    {
      return ckernel_prefix::get_child(offset);
    }

    template <typename CKBT>
    static SelfType *reserve(CKBT *ckb, intptr_t ckb_offset,
                             size_t requested_capacity)
    {
      ckb->reserve(requested_capacity);
      return get_self(ckb, ckb_offset);
    }

    static SelfType *reserve(void *ckb, kernel_request_t kernreq,
                             intptr_t ckb_offset, size_t requested_capacity)
    {
      switch (kernreq & kernel_request_memory) {
      case kernel_request_host:
        return reserve(
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
            ckb_offset, requested_capacity);
      default:
        throw std::invalid_argument("unrecognized ckernel request");
      }
    }

    /**
     * Creates the ckernel, and increments ``inckb_offset``
     * to the position after it.
     */
    template <typename CKBT, typename... A>
    static SelfType *make(CKBT *ckb, kernel_request_t kernreq,
                          intptr_t &inout_ckb_offset, A &&... args)
    {
      intptr_t ckb_offset = inout_ckb_offset;
      inc_ckb_offset<SelfType>(inout_ckb_offset);
      ckb->reserve(inout_ckb_offset);
      ckernel_prefix *rawself =
          ckb->template get_at<ckernel_prefix>(ckb_offset);
      return ckb->template init<SelfType>(rawself, kernreq,
                                          std::forward<A>(args)...);
    }

    template <typename... A>
    static SelfType *make(void *ckb, kernel_request_t kernreq,
                          intptr_t &inout_ckb_offset, A &&... args);

    /**                                                                        \
     * Initializes an instance of this ckernel in-place according to the       \
     * kernel request. This calls the constructor in-place, and initializes    \
     * the base function and destructor.                                       \
     */
    template <typename... A>
    static SelfType *init(ckernel_prefix *rawself,
                          kernel_request_t DYND_UNUSED(kernreq), A &&... args)
    {
      /* Alignment requirement of the type. */
      static_assert(static_cast<size_t>(scalar_align_of<SelfType>::value) <=
                        static_cast<size_t>(scalar_align_of<uint64_t>::value),
                    "ckernel types require alignment <= 64 bits");

      /* Call the constructor in-place. */
      SelfType *self = new (rawself) SelfType(args...);
      /* Double check that the C++ struct layout is as we expect. */
      if (self != get_self(rawself)) {
        DYND_HOST_THROW(std::runtime_error,
                        "internal ckernel error: struct layout is not valid");
      }
      self->destructor = &SelfType::destruct;

      return self;
    }

    /**
     * The ckernel destructor function, which is placed in
     * the ckernel_prefix destructor.
     */
    static void destruct(ckernel_prefix *rawself)
    {
      SelfType *self = get_self(rawself);
      /* If there are any child kernels, a child class must implement */
      /* this to destroy them. */
      self->~SelfType();
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
        const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      SelfType::make(ckb, kernreq, ckb_offset);
      return ckb_offset;
    }
  };

  template <typename SelfType, typename PrefixType>
  template <typename... A>
  SelfType *kernel_prefix_wrapper<SelfType, PrefixType>::make(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    // Disallow requests from a different memory space
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return SelfType::make(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
    default:
      throw std::invalid_argument(
          "unrecognized ckernel request for the wrong memory space");
    }
  }

  /**
   * Some common shared implementation details of a CRTP
   * (curiously recurring template pattern) base class to help
   * create kernels.
   *
   * For most ckernels, the structure is not known beyond that
   * the ckernel_prefix is at the beginning. In some, such
   * as the reduction ckernel, more is known, in which case
   * CKP may be overriden.
   */
  template <typename T, int... N>
  struct base_kernel;

/**
 * This is a helper macro for this header file. It's the memory kernel requests
 * (kernel_request_host is the only one without CUDA enabled) to appropriate
 * function qualifiers in the variadic arguments, which tell e.g. the CUDA
 * compiler to build the functions for the GPU device.
 *
 * The classes it generates are the base classes to use for defining ckernels
 * with a single and strided kernel function.
 */
#define BASE_KERNEL(KERNREQ, ...)                                              \
  template <typename SelfType>                                                 \
  struct base_kernel<SelfType> : kernel_prefix_wrapper<SelfType,               \
                                                       ckernel_prefix> {       \
    typedef SelfType self_type;                                                \
    typedef kernel_prefix_wrapper<SelfType, ckernel_prefix> parent_type;       \
                                                                               \
    /** Initializes just the ckernel_prefix function member. */                \
    template <typename... A>                                                   \
    static SelfType *init(ckernel_prefix *rawself, kernel_request_t kernreq,   \
                          A &&... args)                                        \
    {                                                                          \
      SelfType *self =                                                         \
          parent_type::init(rawself, kernreq, std::forward<A>(args)...);       \
      switch (kernreq) {                                                       \
      case kernel_request_single:                                              \
        self->function = reinterpret_cast<void *>(&SelfType::single_wrapper);  \
        break;                                                                 \
      case kernel_request_strided:                                             \
        self->function = reinterpret_cast<void *>(&SelfType::strided_wrapper); \
        break;                                                                 \
      default:                                                                 \
        DYND_HOST_THROW(std::invalid_argument,                                 \
                        "expr ckernel init: unrecognized ckernel request " +   \
                            std::to_string(kernreq));                          \
      }                                                                        \
                                                                               \
      return self;                                                             \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void single_wrapper(ckernel_prefix *rawself, char *dst, \
                                           char *const *src)                   \
    {                                                                          \
      return parent_type::get_self(rawself)->single(dst, src);                 \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void strided_wrapper(ckernel_prefix *rawself,           \
                                            char *dst, intptr_t dst_stride,    \
                                            char *const *src,                  \
                                            const intptr_t *src_stride,        \
                                            size_t count)                      \
    {                                                                          \
      return parent_type::get_self(rawself)                                    \
          ->strided(dst, dst_stride, src, src_stride, count);                  \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename SelfType>                                                 \
  struct base_kernel<SelfType, 0> : base_kernel<SelfType> {                    \
    typedef SelfType self_type;                                                \
    typedef base_kernel<SelfType> parent_type;                                 \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_UNUSED(src),                    \
                             const intptr_t *DYND_UNUSED(src_stride),          \
                             size_t count)                                     \
    {                                                                          \
      SelfType *self = parent_type::get_self(this);                            \
      for (size_t i = 0; i != count; ++i) {                                    \
        self->single(dst, NULL);                                               \
        dst += dst_stride;                                                     \
      }                                                                        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename SelfType, int N>                                          \
  struct base_kernel<SelfType, N> : base_kernel<SelfType> {                    \
    static_assert(N > 0, "N must be greater or equal to 0");                   \
                                                                               \
    typedef SelfType self_type;                                                \
    typedef base_kernel<SelfType> parent_type;                                 \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *src, \
                             const intptr_t *src_stride, size_t count)         \
    {                                                                          \
      self_type *self = parent_type::get_self(this);                           \
      char *src_copy[N];                                                       \
      memcpy(src_copy, src, sizeof(src_copy));                                 \
      for (size_t i = 0; i != count; ++i) {                                    \
        self->single(dst, src_copy);                                           \
        dst += dst_stride;                                                     \
        for (int j = 0; j < N; ++j) {                                          \
          src_copy[j] += src_stride[j];                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

  BASE_KERNEL(kernel_request_host);

#undef BASE_KERNEL

  typedef void *(*make_t)(void *, kernel_request_t, intptr_t &);

  template <typename T>
  void *make(void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset)
  {
    return T::make(ckb, kernreq, inout_ckb_offset);
  }

} // namespace dynd::nd

template <typename VariadicType,
          template <type_id_t, type_id_t, VariadicType...> class T>
struct bind {
  template <type_id_t TypeID0, type_id_t TypeID1>
  using type = T<TypeID0, TypeID1>;
};

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
