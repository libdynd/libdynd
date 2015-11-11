//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/asarray.hpp>

namespace dynd {
namespace nd {

  /**
   * Helper function which provides access to a 1D array's values as a strided
   * array, either accessing the values directly if possible or via a temporary
   * copy.
   */
  template <typename T, typename F>
  void with_1d_stride(const nd::array &a, F &&f)
  {
    static_assert(is_dynd_scalar<T>::value, "T must have the same representation in DyND and C++");
    nd::array b = nd::asarray(a, ndt::type::make<T[]>());
    auto ss = reinterpret_cast<const size_stride_t *>(b.get()->metadata());
    f(ss->dim_size, ss->stride / sizeof(T), reinterpret_cast<const T *>(b.data()));
  }

} // namespace nd
} // namespace dynd
