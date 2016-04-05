//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/call_graph.hpp>

namespace dynd {
namespace nd {

  class base_dispatch_callable : public base_callable {
  public:
    using base_callable::base_callable;

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      return child->resolve(this, nullptr, cg, dst_tp.is_symbolic() ? child.get_ret_type() : dst_tp, nsrc, src_tp, nkwd,
                            kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
