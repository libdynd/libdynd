//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/arrfunc_type.hpp>
#include <dynd/types/substitute_typevars.hpp>

namespace dynd {
namespace nd {

  template <typename T>
  struct base_virtual_kernel {
    static void
    resolve_dst_type(const arrfunc_type_data *DYND_UNUSED(self),
                     const arrfunc_type *self_tp, char *DYND_UNUSED(data),
                     ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp),
                     const dynd::nd::array &DYND_UNUSED(kwds),
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      dst_tp = ndt::substitute(self_tp->get_return_type(), tp_vars, true);
    }

    static void resolve_option_values(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(self_tp), char *DYND_UNUSED(data),
        intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
        nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd