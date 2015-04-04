//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Returns an arrfunc which chains the two arrfuncs together.
     * The buffer used to connect them is made out of the provided ``buf_tp``.
     */
    arrfunc chain(const arrfunc &first, const arrfunc &second,
                  const ndt::type &buf_tp = ndt::type());

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
