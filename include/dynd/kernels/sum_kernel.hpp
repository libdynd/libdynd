//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Src0TypeID>
  struct DYND_API sum_kernel : base_strided_kernel<sum_kernel<Src0TypeID>, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;
    //    typedef decltype(std::declval<src0_type>() +
    //                     std::declval<src0_type>()) dst_type;
    typedef src0_type dst_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<dst_type *>(dst) + *reinterpret_cast<src0_type *>(src[0]);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      for (size_t i = 0; i < count; ++i) {
        *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<dst_type *>(dst) + *reinterpret_cast<src0_type *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID>
  struct traits<nd::sum_kernel<Src0TypeID>> {
    static type equivalent()
    {
      return callable_type::make(ndt::make_type<typename nd::sum_kernel<Src0TypeID>::dst_type>(), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
