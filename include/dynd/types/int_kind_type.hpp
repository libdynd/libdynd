//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/scalar_kind_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API int_kind_type;

  template <>
  struct base_of<int_kind_type> {
    typedef scalar_kind_type type;
  };

} // namespace dynd::ndt
} // namespace dynd
