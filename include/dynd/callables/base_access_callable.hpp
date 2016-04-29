//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {

  class base_access_callable : public base_callable {
  public:
    base_access_callable(const ndt::type &res_tp, const ndt::type &arg0_tp)
        : base_callable(
              ndt::make_type<ndt::callable_type>(res_tp, {arg0_tp}, {{ndt::make_type<std::string>(), "name"}})) {}

    virtual const std::vector<std::string> &get_names(const ndt::type &arg0_tp) const = 0;
  };

} // namespace dynd::nd
} // namespace dynd
