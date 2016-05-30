//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_dispatch_callable.hpp>

namespace dynd {
namespace nd {

  class is_na_dispatch_callable : public base_dispatch_callable {
    dispatcher<1, callable> m_dispatcher;
    dispatcher<1, callable> m_dim_dispatcher;

  public:
    is_na_dispatch_callable(const ndt::type &tp, const dispatcher<1, callable> &dispatcher,
                            const dynd::dispatcher<1, callable> &dim_dispatcher)
        : base_dispatch_callable(tp), m_dispatcher(dispatcher), m_dim_dispatcher(dim_dispatcher) {}

    const callable &specialize(const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp) {
      if (src_tp[0].get_id() == option_id) {
        return m_dispatcher(dst_tp, nsrc, src_tp);
      } else {
        return m_dim_dispatcher(dst_tp, nsrc, src_tp);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
