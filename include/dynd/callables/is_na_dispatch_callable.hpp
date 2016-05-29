//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_dispatch_callable.hpp>

namespace dynd {
namespace nd {

  template <std::vector<ndt::type> (*Func)(const ndt::type &, size_t, const ndt::type *)>
  class is_na_dispatch_callable : public base_dispatch_callable {
    dispatcher<Func, 1, callable> m_dispatcher;
    dispatcher<Func, 1, callable> m_dim_dispatcher;

  public:
    is_na_dispatch_callable(const ndt::type &tp, const dispatcher<Func, 1, callable> &dispatcher,
                            const dynd::dispatcher<Func, 1, callable> &dim_dispatcher)
        : base_dispatch_callable(tp), m_dispatcher(dispatcher), m_dim_dispatcher(dim_dispatcher) {}

    const callable &specialize(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                               const ndt::type *src_tp) {
      if (src_tp[0].get_id() == option_id) {
        return m_dispatcher(src_tp[0].extended<ndt::option_type>()->get_value_type());
      } else {
        return m_dim_dispatcher(src_tp[0]);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
