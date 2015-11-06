//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Src0TypeID>
  struct min_kernel : base_kernel<min_kernel<Src0TypeID>, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;
    typedef src0_type dst_type;

    static const std::size_t data_size = 0;

    void single(char *dst, char *const *src)
    {
      if (*reinterpret_cast<src0_type *>(src[0]) < *reinterpret_cast<dst_type *>(dst)) {
        *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<src0_type *>(src[0]);
      }
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {

      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      for (size_t i = 0; i < count; ++i) {
        if (*reinterpret_cast<src0_type *>(src0) < *reinterpret_cast<dst_type *>(dst)) {
          *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<src0_type *>(src0);
        }
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <>
  struct min_kernel<complex_float32_type_id> : base_kernel<min_kernel<complex_float32_type_id>, 1> {
    typedef complex<float> src0_type;
    typedef src0_type dst_type;

    static const std::size_t data_size = 0;

    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      throw std::runtime_error("nd::min is not implemented for complex types");
    }
  };

  template <>
  struct min_kernel<complex_float64_type_id> : base_kernel<min_kernel<complex_float64_type_id>, 1> {
    typedef complex<double> src0_type;
    typedef src0_type dst_type;

    static const std::size_t data_size = 0;

    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      throw std::runtime_error("nd::min is not implemented for complex types");
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::min_kernel<Src0TypeID>> {
    static type make()
    {
      return callable_type::make(ndt::type::make<typename nd::min_kernel<Src0TypeID>::dst_type>(), type(Src0TypeID));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
