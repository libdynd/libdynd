//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/substitute_typevars.hpp>

namespace dynd {
namespace nd {

  /**
   * This is defining some boilerplate functions for defining arrfuncs.
   * Child classes of this need to implement instantiate, and possibly
   * override the resolve* functions.
   *
   * For example, elwise and the multidispatch arrfuncs do this.
   *
   * TODO: Come up with a good name for this. We have dissatisfaction with
   *       the usage of the words "virtual" and "kernel" here, as well
   *       as with the "arrfunc" name generally. One possibility will be
   *       to rename "arrfunc" to "callable" once the current callable goes
   *       away, and a name for "base_virtual_kernel" might be
   *       "base_callable_definition".
   */
  template <typename T>
  struct base_virtual_kernel {
    static void
    resolve_dst_type(const arrfunc_type_data *DYND_UNUSED(self),
                     const ndt::arrfunc_type *self_tp,
                     const char *DYND_UNUSED(static_data),
                     size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                     ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp),
                     const dynd::nd::array &DYND_UNUSED(kwds),
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      dst_tp = ndt::substitute(self_tp->get_return_type(), tp_vars, true);
    }
  };

} // namespace dynd::nd
} // namespace dynd