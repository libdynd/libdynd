//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_dispatch_callable.hpp>

namespace dynd {
namespace nd {

  class is_na_dispatch_callable : public base_dispatch_callable {
    dispatcher<callable> m_dispatcher;
    dispatcher<callable> m_dim_dispatcher;

  public:
    is_na_dispatch_callable(const ndt::type &tp, const dispatcher<callable> &dispatcher,
                            const dynd::dispatcher<callable> &dim_dispatcher)
        : base_dispatch_callable(tp), m_dispatcher(dispatcher), m_dim_dispatcher(dim_dispatcher)
    {
    }

    const callable &specialize(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                               const ndt::type *src_tp)
    {
      if (src_tp[0].get_id() == option_id) {
        return m_dispatcher(src_tp[0].extended<ndt::option_type>()->get_value_type().get_id());
      }
      else {
        return m_dim_dispatcher(src_tp[0].get_id());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
