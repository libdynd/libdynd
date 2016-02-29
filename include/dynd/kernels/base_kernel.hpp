//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <typeinfo>

#include <dynd/array.hpp>
#include <dynd/kernels/kernel_prefix.hpp>

namespace dynd {
namespace nd {

  /**
   * This is a helper macro for this header file. It's the memory kernel requests
   * (kernel_request_host is the only one without CUDA enabled) to appropriate
   * function qualifiers in the variadic arguments, which tell e.g. the CUDA
   * compiler to build the functions for the GPU device.
   *
   * The classes it generates are the base classes to use for defining ckernels
   * with a single and strided kernel function.
   */
  template <typename SelfType>
  struct base_kernel : kernel_prefix {
    /**
     * Returns the child kernel immediately following this one.
     */
    kernel_prefix *get_child(intptr_t offset) { return kernel_prefix::get_child(kernel_builder::aligned_size(offset)); }

    /**
     * Returns the child kernel immediately following this one.
     */
    kernel_prefix *get_child() { return kernel_prefix::get_child(reinterpret_cast<SelfType *>(this)->size()); }

    template <size_t I>
    std::enable_if_t<I == 0, kernel_prefix *> get_child()
    {
      return get_child();
    }

    template <size_t I>
    std::enable_if_t<(I > 0), kernel_prefix *> get_child()
    {
      const size_t *offsets = this->get_offsets();
      return kernel_prefix::get_child(offsets[I - 1]);
    }

    constexpr size_t size() const { return sizeof(SelfType); }

    /** Initializes just the kernel_prefix function member. */
    template <typename... ArgTypes>
    static void init(SelfType *self, kernel_request_t kernreq, ArgTypes &&... args)
    {
      new (self) SelfType(std::forward<ArgTypes>(args)...);

      self->destructor = SelfType::destruct;
      switch (kernreq) {
      case kernel_request_call:
        self->function = reinterpret_cast<void *>(SelfType::call_wrapper);
        break;
      case kernel_request_single:
        self->function = reinterpret_cast<void *>(SelfType::single_wrapper);
        break;
      case kernel_request_strided:
        self->function = reinterpret_cast<void *>(SelfType::strided_wrapper);
        break;
      default:
        DYND_HOST_THROW(std::invalid_argument,
                        "expr ckernel init: unrecognized ckernel request " + std::to_string(kernreq));
      }
    }

    /**
     * The ckernel destructor function, which is placed in
     * the kernel_prefix destructor.
     */
    static void destruct(kernel_prefix *self) { reinterpret_cast<SelfType *>(self)->~SelfType(); }

    void call(array *DYND_UNUSED(dst), const array *DYND_UNUSED(src))
    {
      std::stringstream ss;
      ss << "void call(array *dst, const array *src) is not implemented in " << typeid(SelfType).name();
      throw std::runtime_error(ss.str());
    }

    static void call_wrapper(kernel_prefix *self, array *dst, const array *src)
    {
      reinterpret_cast<SelfType *>(self)->call(dst, src);
    }

    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      std::stringstream ss;
      ss << "void single(char *dst, char *const *src) is not implemented in " << typeid(SelfType).name();
      throw std::runtime_error(ss.str());
    }

    static void DYND_EMIT_LLVM(single_wrapper)(kernel_prefix *self, char *dst, char *const *src)
    {
      reinterpret_cast<SelfType *>(self)->single(dst, src);
    }

    static void strided_wrapper(kernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
    {
      reinterpret_cast<SelfType *>(self)->strided(dst, dst_stride, src, src_stride, count);
    }

    static const volatile char *DYND_USED(ir);

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                            const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<SelfType>(kernreq);
    }
  };

  template <typename SelfType>
  const volatile char *DYND_USED(base_kernel<SelfType>::ir) = NULL;

  template <typename SelfType, size_t N>
  struct base_strided_kernel : base_kernel<SelfType> {
    void call(array *dst, const array *src)
    {
      char *src_data[N];
      for (size_t i = 0; i < N; ++i) {
        src_data[i] = const_cast<char *>(src[i].cdata());
      }
      reinterpret_cast<SelfType *>(this)->single(const_cast<char *>(dst->cdata()), src_data);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      char *src_copy[N];
      memcpy(src_copy, src, sizeof(src_copy));
      for (size_t i = 0; i != count; ++i) {
        reinterpret_cast<SelfType *>(this)->single(dst, src_copy);
        dst += dst_stride;
        for (size_t j = 0; j < N; ++j) {
          src_copy[j] += src_stride[j];
        }
      }
    }
  };

  template <typename SelfType>
  struct base_strided_kernel<SelfType, 0> : base_kernel<SelfType> {
    void call(array *dst, const array *DYND_UNUSED(src))
    {
      reinterpret_cast<SelfType *>(this)->single(const_cast<char *>(dst->cdata()), nullptr);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src), const intptr_t *DYND_UNUSED(src_stride),
                 size_t count)
    {
      for (size_t i = 0; i != count; ++i) {
        reinterpret_cast<SelfType *>(this)->single(dst, NULL);
        dst += dst_stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
