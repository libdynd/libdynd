//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {
namespace nd {

  template <typename CKT>
  struct virtual_ck {
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