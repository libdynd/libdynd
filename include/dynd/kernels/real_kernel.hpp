//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t RealTypeID>
  struct real_kernel : base_kernel<real_kernel<RealTypeID>, 1> {
    typedef typename type_of<RealTypeID>::type real_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<real_type *>(dst) = reinterpret_cast<complex<real_type> *>(src[0])->real();
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::real_kernel<float32_type_id>> {
    static type make() { return type("(complex[float32]) -> float32"); }
  };

  template <>
  struct type::equivalent<nd::real_kernel<float64_type_id>> {
    static type make() { return type("(complex[float64]) -> float64"); }
  };

} // namespace dynd::ndt
} // namespace dynd
