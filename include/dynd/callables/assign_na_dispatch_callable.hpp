//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_dispatch_callable.hpp>

namespace dynd {
namespace nd {

  class assign_na_dispatch_callable : public base_dispatch_callable {
    dispatcher<1, callable> m_dispatcher;
    dispatcher<1, callable> m_dim_dispatcher;

  public:
    assign_na_dispatch_callable(const ndt::type &tp, const dispatcher<1, callable> &dispatcher,
                                const dynd::dispatcher<1, callable> &dim_dispatcher)
        : base_dispatch_callable(tp), m_dispatcher(dispatcher), m_dim_dispatcher(dim_dispatcher) {}

    const callable &specialize(const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                               const ndt::type *DYND_UNUSED(src_tp)) {
      if (dst_tp.get_id() == option_id) {
        return m_dispatcher(dst_tp.extended<ndt::option_type>()->get_value_type().get_id());
      } else {
        return m_dim_dispatcher(dst_tp.get_id());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
