//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/forward_na_kernel.hpp>

namespace dynd {
namespace nd {

  typedef intptr_t index_t;

  template <index_t... I>
  class forward_na_callable : public base_callable {
    callable m_child;

    using kernel_type = forward_na_kernel<I...>;

  public:
    forward_na_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

    ndt::type resolve(base_callable *caller, char *DYND_UNUSED(data), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *dst_arrmeta,
                         size_t nsrc, const char *const *src_arrmeta) {
        size_t self_offset = kb.size();
	//using kernel_type = forward_na_kernel<I...>;
        kb.emplace_back<kernel_type>(kernreq);

        kb(kernel_request_single, nullptr, dst_arrmeta, nsrc, src_arrmeta);

	auto indices = std::array<index_t, sizeof...(I)>({I...}); 
        for (intptr_t i : indices) {
          size_t is_na_offset = kb.size() - self_offset;
          kb(kernel_request_single, nullptr, nullptr, 1, src_arrmeta + i);
          kb.get_at<kernel_type>(self_offset)->is_na_offset[i] = is_na_offset;
        }

        size_t assign_na_offset = kb.size() - self_offset;
        kb(kernel_request_single, nullptr, nullptr, 0, nullptr);
        kb.get_at<kernel_type>(self_offset)->assign_na_offset = assign_na_offset;
      });

      ndt::type src_value_tp[2];
      for (intptr_t i = 0; i < 2; ++i) {
        src_value_tp[i] = src_tp[i];
      }
      for (intptr_t i : std::array<index_t, sizeof...(I)>({I...})) {
        src_value_tp[i] = src_value_tp[i].extended<ndt::option_type>()->get_value_type();
      }

      base_callable *child;
      if (m_child.is_null()) {
        child = caller;
      } else {
        child = m_child.get();
      }

      ndt::type res_value_tp =
          child->resolve(this, nullptr, cg, dst_tp.is_symbolic() ? child->get_ret_type() : dst_tp, 2, src_value_tp,
                         nkwd, kwds, tp_vars);

      for (index_t i : std::array<index_t, sizeof...(I)>({I...})) {
        is_na->resolve(this, nullptr, cg, ndt::make_type<bool>(), 1, src_tp + i, 0, nullptr, tp_vars);
      }

      return assign_na->resolve(this, nullptr, cg, ndt::make_type<ndt::option_type>(res_value_tp), 0, nullptr, nkwd,
                                kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
