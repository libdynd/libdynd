//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/adapt_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct adapt_kernel : base_kernel<adapt_kernel> {
      const ndt::type &value_tp;
      const callable &forward;

      adapt_kernel(const ndt::type &value_tp, const callable &forward) : value_tp(value_tp), forward(forward) {}

      void call(array *dst, const array *src)
      {
        *dst = src[0].replace_dtype(ndt::make_type<ndt::adapt_type>(value_tp, src[0].get_dtype(), forward, callable()));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
