//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/state_kernel.hpp>
#include <dynd/types/iteration_type.hpp>

namespace dynd {
namespace nd {

  template <size_t NArg>
  class state_callable : public base_callable {
    callable m_child;

  public:
    state_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *dst_arrmeta,
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        size_t self_offset = kb.size();
        kb.emplace_back<state_kernel<NArg>>(kernreq);
        state_kernel<NArg> *self = kb.get_at<state_kernel<NArg>>(self_offset);

        const char *child_src_arrmeta[NArg + 1];
        for (size_t i = 0; i < NArg; ++i) {
          child_src_arrmeta[i] = src_arrmeta[i];
        }
        child_src_arrmeta[NArg] = nullptr;

        kb(kernreq | kernel_request_data_only, reinterpret_cast<char *>(&self->st), dst_arrmeta, NArg + 1,
           child_src_arrmeta);
      });

      ndt::type child_tp[NArg + 1];
      for (size_t i = 0; i < NArg; ++i) {
        child_tp[i] = src_tp[i];
      }
      child_tp[NArg] = ndt::make_type<ndt::iteration_type>();

      return m_child->resolve(this, nullptr, cg, dst_tp, NArg + 1, child_tp, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
