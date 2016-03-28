//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  struct call_frame {
    callable func;
    char *data;
    ndt::type dst_tp;
    size_t nsrc;
    std::vector<ndt::type> src_tp;
    kernel_request_t kernreq;

    call_frame(const callable &func, const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp,
               kernel_request_t kernreq)
        : func(func), data(nullptr), dst_tp(dst_tp), nsrc(nsrc), src_tp(nsrc), kernreq(kernreq)
    {
      for (size_t i = 0; i < nsrc; ++i) {
        this->src_tp[i] = src_tp[i];
      }
    }
  };

  class call_stack {
  public:
    std::vector<call_frame> m_stack;

    ndt::type &res_type() { return m_stack.back().dst_tp; }

    size_t narg() { return m_stack.back().src_tp.size(); }

    const ndt::type *arg_types() { return m_stack.back().src_tp.data(); }

    kernel_request_t kernreq() { return m_stack.back().kernreq; }

    void push_back(const callable &func, const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp,
                   kernel_request_t kernreq)
    {
      m_stack.emplace_back(func, dst_tp, nsrc, src_tp, kernreq);
    }

    decltype(auto) begin() { return m_stack.begin(); }

    decltype(auto) end() { return m_stack.end(); }
  };

} // namespace dynd::nd
} // namespace dynd
