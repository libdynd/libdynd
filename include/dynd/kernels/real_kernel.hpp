//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct real_kernel : base_strided_kernel<real_kernel<Arg0ID>, 1> {
    typedef typename type_of<Arg0ID>::type complex_type;
    typedef typename complex_type::value_type real_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<real_type *>(dst) = reinterpret_cast<complex_type *>(src[0])->real();
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<nd::real_kernel<Arg0ID>> {
    static type equivalent()
    {
      return callable_type::make(make_type<typename nd::real_kernel<Arg0ID>::real_type>(),
                                 {make_type<typename nd::real_kernel<Arg0ID>::complex_type>()});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
