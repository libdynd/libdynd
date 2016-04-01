//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <chrono>
#include <memory>
#include <random>

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace random {

    extern DYND_API callable uniform;

  } // namespace dynd::nd::random

  inline array rand(const ndt::type &tp) { return random::uniform({}, {{"dst_tp", tp}}); }

  inline array rand(intptr_t dim0, const ndt::type &tp) { return rand(ndt::make_fixed_dim(dim0, tp)); }

  inline array rand(intptr_t dim0, intptr_t dim1, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, ndt::make_fixed_dim(dim1, tp)));
  }

  inline array rand(intptr_t dim0, intptr_t dim1, intptr_t dim2, const ndt::type &tp)
  {
    return rand(ndt::make_fixed_dim(dim0, ndt::make_fixed_dim(dim1, ndt::make_fixed_dim(dim2, tp))));
  }

} // namespace dynd::nd
} // namespace dynd
