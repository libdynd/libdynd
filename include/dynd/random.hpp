//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/dynd_complex.hpp>

namespace dynd { namespace nd {

  /**
   * Primitive function to construct an nd::array with each element initialized
   * to a random value. This is used only for testing right now, and it should
   * be completely redone at some point.
   */
  nd::array rand(const ndt::type &tp);

  inline nd::array rand(intptr_t dim0, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, tp));
  }

  inline nd::array rand(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, ndt::make_fixed_dim(dim1, tp)));
  }

  inline nd::array rand(intptr_t dim0, intptr_t dim1, intptr_t dim2,
                        const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(
        dim0, ndt::make_fixed_dim(dim1, ndt::make_fixed_dim(dim2, tp))));
  }

}} // namespace dynd::nd
