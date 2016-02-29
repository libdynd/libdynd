//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename SelfType, size_t... N>
  struct base_strided_kernel;

  template <typename SelfType>
  struct base_strided_kernel<SelfType> : base_kernel<SelfType> {
    static void strided_wrapper(kernel_prefix *self, char *dst, intptr_t dst_stride, char *const *src,
                                const intptr_t *src_stride, size_t count)
    {
      reinterpret_cast<SelfType *>(self)->strided(dst, dst_stride, src, src_stride, count);
    }

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
  };

  template <typename SelfType, size_t N>
  struct base_strided_kernel<SelfType, N> : base_strided_kernel<SelfType> {
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
  struct base_strided_kernel<SelfType, 0> : base_strided_kernel<SelfType> {
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
