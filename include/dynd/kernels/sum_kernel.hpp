//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Src0TypeID>
  struct sum_kernel
      : base_kernel<sum_kernel<Src0TypeID>, kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;
    typedef src0_type dst_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<dst_type *>(dst) +
          *reinterpret_cast<const src0_type *>(src[0]);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      if (dst_stride == 0) {
        dst_type s(0);
        for (size_t i = 0; i < count; ++i) {
          s = s + *reinterpret_cast<src0_type *>(src0);
          src0 += src0_stride;
        }
        *reinterpret_cast<dst_type *>(dst) =
            static_cast<src0_type>(*reinterpret_cast<dst_type *>(dst) + s);
      } else {
        for (size_t i = 0; i < count; ++i) {
          *reinterpret_cast<dst_type *>(dst) =
              *reinterpret_cast<dst_type *>(dst) +
              *reinterpret_cast<src0_type *>(src0);
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::sum_kernel<Src0TypeID>> {
    static type make()
    {
      return callable_type::make(type(Src0TypeID), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
