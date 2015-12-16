//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/static_data_callable.hpp>

namespace dynd {
namespace nd {

  template <typename DispatcherType>
  struct dispatcher_callable : static_data_callable<DispatcherType> {
    using static_data_callable<DispatcherType>::static_data_callable;

    callable &overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp)
    {
      return this->static_data(ret_tp, narg, arg_tp);
    }
  };

} // namespace dynd::nd
} // namespace dynd
