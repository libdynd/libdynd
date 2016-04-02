//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

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
} // namespace dynd
