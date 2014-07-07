//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>
#include <dynd/types/dynd_complex.hpp>

namespace dynd { namespace nd {

/**
 * Primitive function to construct an nd::array with each element initialized
 * to a random value. This is used only for testing right now, and it should be 
 * completely redone at some point. Variable dimensions are supported. Only a dtype
 * of double is currently supported.
 */
nd::array typed_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp);

inline nd::array dtyped_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
  if (ndim > 0) {
    intptr_t i = ndim - 1;
    ndt::type rtp =
        shape[i] >= 0 ? ndt::make_strided_dim(tp) : ndt::make_var_dim(tp);
    while (i-- > 0) {
      rtp = shape[i] >= 0 ? ndt::make_strided_dim(rtp) : ndt::make_var_dim(rtp);
    }
    return typed_rand(ndim, shape, rtp);
  } else {
    return typed_rand(ndim, shape, tp);
  }
}

inline nd::array dtyped_rand(intptr_t dim0, const ndt::type &tp) {
    intptr_t dims[1] = {dim0};

    return dtyped_rand(1, dims, tp);
}
inline nd::array dtyped_rand(intptr_t dim0, intptr_t dim1, const ndt::type &tp) {
    intptr_t dims[2] = {dim0, dim1};

    return dtyped_rand(2, dims, tp);
}
inline nd::array dtyped_rand(intptr_t dim0, intptr_t dim1, intptr_t dim2, const ndt::type &tp) {
    intptr_t dims[3] = {dim0, dim1, dim2};

    return dtyped_rand(3, dims, tp);
}

}} // namespace dynd::nd
