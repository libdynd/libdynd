//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/outer.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    arrfunc outer(const arrfunc &child);

    ndt::type outer_make_type(const arrfunc_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd