//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  struct imag_kernel : base_strided_kernel<imag_kernel<Arg0ID>, 1> {
    typedef typename type_of<Arg0ID>::type complex_type;
    typedef typename complex_type::value_type real_type;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<real_type *>(dst) = reinterpret_cast<complex_type *>(src[0])->imag();
    }
  };

} // namespace dynd::nd
} // namespace dynd
