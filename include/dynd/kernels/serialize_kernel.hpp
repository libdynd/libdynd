//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/scalar_kind_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  struct serialize_kernel;

  template <>
  struct serialize_kernel<ndt::scalar_kind_type> : base_strided_kernel<serialize_kernel<ndt::scalar_kind_type>, 1> {
    size_t data_size;

    serialize_kernel(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src) { reinterpret_cast<bytes *>(dst)->append(src[0], data_size); }
  };

} // namespace dynd::nd
} // namespace dynd
