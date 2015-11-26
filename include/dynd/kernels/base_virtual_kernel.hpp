//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/callable_type.hpp>
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
    struct single_wrapper {
      static void func(ckernel_prefix *DYND_UNUSED(self), char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {}

      static const char *ir;
    };

    static char *data_init(char *DYND_UNUSED(static_data), const ndt::type &DYND_UNUSED(dst_tp),
                           intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd),
                           const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      return NULL;
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &tp_vars)
    {
      dst_tp = ndt::substitute(dst_tp, tp_vars, true);
    }
  };

  template <typename SelfType>
  const char *base_virtual_kernel<SelfType>::single_wrapper::ir = NULL;

} // namespace dynd::nd
} // namespace dynd
