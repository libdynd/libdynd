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
    ndt::type dst_tp;
    intptr_t dst_arrmeta_offset;
    size_t nsrc;
    std::vector<ndt::type> src_tp;
    std::vector<intptr_t> src_arrmeta_offsets;
    kernel_request_t kernreq;
    char data[100];

    call_frame(const callable &func, const ndt::type &dst_tp, intptr_t dst_arrmeta_offset, size_t nsrc,
               const ndt::type *src_tp, const intptr_t *src_arrmeta_offsets, kernel_request_t kernreq)
        : func(func), dst_tp(dst_tp), dst_arrmeta_offset(dst_arrmeta_offset), nsrc(nsrc), src_tp(nsrc),
          src_arrmeta_offsets(nsrc), kernreq(kernreq)
    {
      for (size_t i = 0; i < nsrc; ++i) {
        this->src_tp[i] = src_tp[i];
        this->src_arrmeta_offsets[i] = src_arrmeta_offsets[i];
      }
    }
  };

  class call_stack {
  public:
    std::vector<call_frame> m_stack;

    const callable &func() { return m_stack.back().func; }

    ndt::type &res_type() { return m_stack.back().dst_tp; }

    intptr_t res_metadata_offset() { return m_stack.back().dst_arrmeta_offset; }

    size_t narg() { return m_stack.back().src_tp.size(); }

    const ndt::type *arg_types() { return m_stack.back().src_tp.data(); }

    const intptr_t *arg_metadata_offsets() { return m_stack.back().src_arrmeta_offsets.data(); }

    kernel_request_t kernreq() { return m_stack.back().kernreq; }

    char *data() { return m_stack.back().data; }

    const callable &parent() { return m_stack[m_stack.size() - 2].func; }

    void push_back(const callable &func, const ndt::type &dst_tp, intptr_t dst_arrmeta_offset, size_t nsrc,
                   const ndt::type *src_tp, const intptr_t *src_arrmeta_offsets, kernel_request_t kernreq)
    {
      m_stack.emplace_back(func, dst_tp, dst_arrmeta_offset, nsrc, src_tp, src_arrmeta_offsets, kernreq);
    }

    template <typename DataType>
    void push_back_data(DataType data)
    {
      new (this->data()) DataType(data);
    }

    decltype(auto) begin() { return m_stack.begin(); }

    decltype(auto) end() { return m_stack.end(); }
  };

} // namespace dynd::nd
} // namespace dynd
