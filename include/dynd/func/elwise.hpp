//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Lifts the provided ckernel, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     *
     * \param child  The callable being lifted
     */
    DYND_API callable elwise(const callable &child);

    DYND_API callable elwise(const ndt::type &self_tp, const callable &child);

    DYND_API ndt::type elwise_make_type(const ndt::callable_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
