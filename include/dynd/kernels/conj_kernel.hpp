//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t RealTypeID>
  struct conj_kernel : base_kernel<conj_kernel<RealTypeID>, 1> {
    typedef complex<typename type_of<RealTypeID>::type> complex_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<complex_type *>(dst) = conj(*reinterpret_cast<complex_type *>(src[0]));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t RealTypeID>
  struct traits<nd::conj_kernel<RealTypeID>> {
    static type equivalent()
    {
      return callable_type::make(make_type<typename nd::conj_kernel<RealTypeID>::complex_type>(),
                                 {make_type<typename nd::conj_kernel<RealTypeID>::complex_type>()});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
