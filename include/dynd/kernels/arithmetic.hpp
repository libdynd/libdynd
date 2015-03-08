#pragma once

#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {
namespace nd {

#define ARITHMETIC_OPERATOR(NAME, SYMBOL)                                      \
  template <typename A0, typename A1>                                          \
  struct NAME##_ck                                                             \
      : expr_ck<NAME##_ck<A0, A1>, kernel_request_cuda_host_device, 2> {       \
    typedef decltype(std::declval<A0>() SYMBOL std::declval<A1>()) R;          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)             \
    {                                                                          \
      *reinterpret_cast<R *>(dst) = *reinterpret_cast<A0 *>(src[0]) SYMBOL *   \
                                    reinterpret_cast<A1 *>(src[1]);            \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void strided(R *__restrict dst,                      \
                                       const A0 *__restrict src0,              \
                                       const A1 *__restrict src1,              \
                                       size_t count)                           \
    {                                                                          \
      for (size_t i = DYND_THREAD_ID(0); i < count;                            \
           i += DYND_THREAD_COUNT(0)) {                                        \
        dst[i] = src0[i] SYMBOL src1[i];                                       \
      }                                                                        \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE void strided(char *__restrict dst,                   \
                                       intptr_t dst_stride,                    \
                                       char *__restrict const *src,            \
                                       const intptr_t *__restrict src_stride,  \
                                       size_t count)                           \
    {                                                                          \
      if (dst_stride == sizeof(R) && src_stride[0] == sizeof(A0) &&            \
          src_stride[1] == sizeof(A1)) {                                       \
        strided(reinterpret_cast<R *>(dst),                                    \
                reinterpret_cast<const A0 *>(src[0]),                          \
                reinterpret_cast<const A1 *>(src[1]), count);                  \
      } else {                                                                 \
        const char *__restrict src0 = src[0], *__restrict src1 = src[1];       \
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];     \
        for (size_t i = 0; i != count; ++i) {                                  \
          *reinterpret_cast<R *>(dst) =                                        \
              *reinterpret_cast<const A0 *>(src0) SYMBOL *                     \
              reinterpret_cast<const A1 *>(src1);                              \
          dst += dst_stride;                                                   \
          src0 += src0_stride;                                                 \
          src1 += src1_stride;                                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };

  ARITHMETIC_OPERATOR(add, +);
  ARITHMETIC_OPERATOR(sub, -);
  ARITHMETIC_OPERATOR(mul, *);
  ARITHMETIC_OPERATOR(div, / );

#undef ARITHMETIC_OPERATOR

} // namespace nd
} // namespace dynd

#undef ARITHMETIC_OPERATOR