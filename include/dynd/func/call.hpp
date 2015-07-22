//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>
#include <dynd/kernels/call_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename T>
    callable call(const ndt::type &child_tp)
    {
      return callable::make<call_kernel<T>>(child_tp, 0);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd