//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/adapt_kernel.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/outer.hpp>
#include <dynd/func/permute.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    inline callable adapt(const ndt::type &value_tp, const callable &forward)
    {
      return callable::make<adapt_kernel>(ndt::callable_type::make(value_tp, {ndt::type("Any")}),
                                          adapt_kernel::static_data_type{value_tp, forward});
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
