//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API callable field_access;

  extern DYND_API nd::callable make_field_access_kernel(const ndt::type &dt, const std::string &name);

} // namespace dynd::nd
} // namespace dynd
