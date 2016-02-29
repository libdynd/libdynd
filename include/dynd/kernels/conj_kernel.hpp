//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct conj_kernel : base_strided_kernel<conj_kernel<Arg0ID>, 1> {
    typedef typename type_of<Arg0ID>::type complex_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<complex_type *>(dst) = dynd::conj(*reinterpret_cast<complex_type *>(src[0]));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<nd::conj_kernel<Arg0ID>> {
    static type equivalent()
    {
      return callable_type::make(make_type<typename nd::conj_kernel<Arg0ID>::complex_type>(),
                                 {make_type<typename nd::conj_kernel<Arg0ID>::complex_type>()});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
