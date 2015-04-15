//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/call_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename T>
    arrfunc call(const ndt::type &child_tp, const T &child)
    {
      return arrfunc::make<call_kernel<T>>(child_tp, &child, 0);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd