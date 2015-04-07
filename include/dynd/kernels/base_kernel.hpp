//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/arrfunc_type.hpp>

namespace dynd {
namespace nd {

  /**
   * Some common shared implementation details of a CRTP
   * (curiously recurring template pattern) base class to help
   * create kernels.
   */
  template <typename T, kernel_request_t kernreq, int N>
  struct base_kernel;

#define BASE_KERNEL(KERNREQ, ...)                                              \
  template <typename T>                                                        \
  struct base_kernel<T, KERNREQ, -1> {                                         \
    typedef T self_type;                                                       \
                                                                               \
    ckernel_prefix base;                                                       \
                                                                               \
    DYND_CUDA_HOST_DEVICE static self_type *get_self(ckernel_prefix *rawself)  \
    {                                                                          \
      return reinterpret_cast<self_type *>(rawself);                           \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE static const self_type *                             \
    get_self(const ckernel_prefix *rawself)                                    \
    {                                                                          \
      return reinterpret_cast<const self_type *>(rawself);                     \
    }                                                                          \
                                                                               \
    template <typename CKBT>                                                   \
    static self_type *get_self(CKBT *ckb, intptr_t ckb_offset)                 \
    {                                                                          \
      return ckb->template get_at<self_type>(ckb_offset);                      \
    }                                                                          \
                                                                               \
    /** \                                                                      \
     * Creates the ckernel, and increments ``inckb_offset`` \                  \
     * to the position after it. \                                             \
     */                                                                        \
    template <typename CKBT, typename... A>                                    \
    static self_type *create(CKBT *ckb, kernel_request_t kernreq,              \
                             intptr_t &inout_ckb_offset, A &&... args)         \
    {                                                                          \
      intptr_t ckb_offset = inout_ckb_offset;                                  \
      inc_ckb_offset<self_type>(inout_ckb_offset);                             \
      ckb->ensure_capacity(inout_ckb_offset);                                  \
      ckernel_prefix *rawself =                                                \
          ckb->template get_at<ckernel_prefix>(ckb_offset);                    \
      return ckb->template init<self_type>(rawself, kernreq,                   \
                                           std::forward<A>(args)...);          \
    }                                                                          \
                                                                               \
    template <typename... A>                                                   \
    static self_type *create(void *ckb, kernel_request_t kernreq,              \
                             intptr_t &inout_ckb_offset, A &&... args);        \
                                                                               \
    /** \                                                                      \
     * Creates the ckernel, and increments ``inckb_offset`` \                  \
     * to the position after it. \                                             \
     */                                                                        \
    template <typename CKBT, typename... A>                                    \
    static self_type *create_leaf(CKBT *ckb, kernel_request_t kernreq,         \
                                  intptr_t &inout_ckb_offset, A &&... args)    \
    {                                                                          \
      intptr_t ckb_offset = inout_ckb_offset;                                  \
      inc_ckb_offset<self_type>(inout_ckb_offset);                             \
      ckb->ensure_capacity_leaf(inout_ckb_offset);                             \
      ckernel_prefix *rawself =                                                \
          ckb->template get_at<ckernel_prefix>(ckb_offset);                    \
      return ckb->template init<self_type>(rawself, kernreq,                   \
                                           std::forward<A>(args)...);          \
    }                                                                          \
                                                                               \
    template <typename... A>                                                   \
    static self_type *create_leaf(void *ckb, kernel_request_t kernreq,         \
                                  intptr_t &inout_ckb_offset, A &&... args);   \
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
    /** \                                                                      \
     * Initializes an instance of this ckernel in-place according to the \     \
     * kernel request. This calls the constructor in-place, and initializes \  \
     * the base function and destructor. \                                     \
     */                                                                        \
    template <typename... A>                                                   \
    __VA_ARGS__ static self_type *init(ckernel_prefix *rawself,                \
                                       kernel_request_t kernreq, A &&... args) \
    {                                                                          \
      /* Alignment requirement of the type. */                                 \
      static_assert((size_t)scalar_align_of<self_type>::value <=               \
                        (size_t)scalar_align_of<uint64_t>::value,              \
                    "ckernel types require alignment <= 64 bits");             \
                                                                               \
      /* Call the constructor in-place. */                                     \
      self_type *self = new (rawself) self_type(args...);                      \
      /* Double check that the C++ struct layout is as we expect. */           \
      if (self != get_self(rawself)) {                                         \
        DYND_HOST_THROW(std::runtime_error,                                    \
                        "internal ckernel error: struct layout is not valid"); \
      }                                                                        \
      self->base.destructor = &self_type::destruct;                            \
      /* A child class must implement this to fill in self->base.function. */  \
      self->init_kernfunc(kernreq);                                            \
      return self;                                                             \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void single_wrapper(char *dst, char *const *src,        \
                                           ckernel_prefix *rawself)            \
    {                                                                          \
      return get_self(rawself)->single(dst, src);                              \
    }                                                                          \
                                                                               \
    __VA_ARGS__ static void strided_wrapper(char *dst, intptr_t dst_stride,    \
                                            char *const *src,                  \
                                            const intptr_t *src_stride,        \
                                            size_t count,                      \
                                            ckernel_prefix *rawself)           \
    {                                                                          \
      return get_self(rawself)                                                 \
          ->strided(dst, dst_stride, src, src_stride, count);                  \
    }                                                                          \
                                                                               \
    /** \                                                                      \
     * The ckernel destructor function, which is placed in \                   \
     * base.destructor. \                                                      \
     */                                                                        \
    __VA_ARGS__ static void destruct(ckernel_prefix *rawself)                  \
    {                                                                          \
      self_type *self = get_self(rawself);                                     \
      /* If there are any child kernels, a child class must implement this to  \
       * \                                                                     \
       * destroy them. */                                                      \
      self->destruct_children();                                               \
      self->~self_type();                                                      \
    }                                                                          \
                                                                               \
    /** \                                                                      \
     * Default implementation of destruct_children does nothing. \             \
     */                                                                        \
    __VA_ARGS__ void destruct_children() {}                                    \
                                                                               \
    /** \                                                                      \
     * Returns the child ckernel immediately following this one. \             \
     */                                                                        \
    __VA_ARGS__ ckernel_prefix *get_child_ckernel()                            \
    {                                                                          \
      return get_child_ckernel(sizeof(self_type));                             \
    }                                                                          \
                                                                               \
    /** \                                                                      \
     * Returns the child ckernel at the specified offset. \                    \
     */                                                                        \
    __VA_ARGS__ ckernel_prefix *get_child_ckernel(intptr_t offset)             \
    {                                                                          \
      return base.get_child_ckernel(ckernel_prefix::align_offset(offset));     \
    }                                                                          \
                                                                               \
    static intptr_t instantiate(                                               \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),       \
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),  \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *DYND_UNUSED(src_tp),                                  \
        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, \
        const eval::eval_context *DYND_UNUSED(ectx),                           \
        const nd::array &DYND_UNUSED(kwds),                                    \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset);                             \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static void resolve_dst_type(                                              \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *self_tp, char *DYND_UNUSED(data),                  \
        ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),                         \
        const ndt::type *DYND_UNUSED(src_tp),                                  \
        const dynd::nd::array &DYND_UNUSED(kwds),                              \
        const std::map<dynd::nd::string, ndt::type> &DYND_UNUSED(tp_vars))     \
    {                                                                          \
      dst_tp = self_tp->get_return_type();                                     \
    }                                                                          \
                                                                               \
    static void resolve_option_values(                                         \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), char *DYND_UNUSED(data),     \
        intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),      \
        nd::array &DYND_UNUSED(kwds),                                          \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T>                                                        \
  struct base_kernel<T, KERNREQ, 0> : base_kernel<T, KERNREQ, -1> {            \
    typedef T self_type;                                                       \
    typedef base_kernel<T, KERNREQ, -1> parent_type;                           \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_UNUSED(src),                    \
                             const intptr_t *DYND_UNUSED(src_stride),          \
                             size_t count)                                     \
    {                                                                          \
      self_type *self = parent_type::get_self(&this->base);                    \
      for (size_t i = 0; i != count; ++i) {                                    \
        self->single(dst, NULL);                                               \
        dst += dst_stride;                                                     \
      }                                                                        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T, int N>                                                 \
  struct base_kernel<T, KERNREQ, N> : base_kernel<T, KERNREQ, -1> {            \
    typedef T self_type;                                                       \
    typedef base_kernel<T, KERNREQ, -1> parent_type;                           \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *src, \
                             const intptr_t *src_stride, size_t count)         \
    {                                                                          \
      self_type *self = parent_type::get_self(&this->base);                    \
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

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_host, -1>::self_type *
  base_kernel<T, kernel_request_host, -1>::create(void *ckb,
                                                  kernel_request_t kernreq,
                                                  intptr_t &inout_ckb_offset,
                                                  A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_host, -1>::self_type *
  base_kernel<T, kernel_request_host, -1>::create_leaf(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#ifdef __CUDACC__

  BASE_KERNEL(kernel_request_cuda_device, __device__);

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_cuda_device, -1>::self_type *
  base_kernel<T, kernel_request_cuda_device, -1>::create(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_cuda_device:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_cuda_device, -1>::self_type *
  base_kernel<T, kernel_request_cuda_device, -1>::create_leaf(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_cuda_device:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#endif

#ifdef DYND_CUDA

  BASE_KERNEL(kernel_request_cuda_host_device, DYND_CUDA_HOST_DEVICE);

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_cuda_host_device, -1>::self_type *
  base_kernel<T, kernel_request_cuda_host_device, -1>::create(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
#ifdef __CUDACC__
    case kernel_request_cuda_device:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
#endif
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename T>
  template <typename... A>
  typename base_kernel<T, kernel_request_cuda_host_device, -1>::self_type *
  base_kernel<T, kernel_request_cuda_host_device, -1>::create_leaf(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
#ifdef __CUDACC__
    case kernel_request_cuda_device:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
#endif
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#endif

#undef BASE_KERNEL

  typedef void *(*create_t)(void *, kernel_request_t, intptr_t &);

  template <typename T>
  void *create(void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset)
  {
    return T::create(ckb, kernreq, inout_ckb_offset);
  }

} // namespace dynd::nd

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
