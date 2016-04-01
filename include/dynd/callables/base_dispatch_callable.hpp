//
// Copyright (C) 2011-15 DyND Developers
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
    base_dispatch_callable(const ndt::type &tp) : base_callable(tp) { m_abstract = true; }

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      return child->resolve(this, nullptr, cg, dst_tp.is_symbolic() ? child.get_ret_type() : dst_tp, nsrc, src_tp, nkwd,
                            kwds, tp_vars);
    }

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &g, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      if (!child->is_abstract()) {
        g.emplace_back(child.get());
      }

      dst_tp = child.get_type()->get_return_type();
      child->new_resolve(this, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    char *data_init(const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                    const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();

      return child->data_init(child_dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        throw std::runtime_error("no suitable child for multidispatch");
      }

      if (dst_tp.is_symbolic()) {
        dst_tp = child.get_type()->get_return_type();
      }
      child->resolve_dst_type(data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void instantiate(call_node *DYND_UNUSED(node), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        std::stringstream ss;
        ss << "no suitable child for multidispatch for types " << src_tp[0] << ", and " << dst_tp << "\n";
        throw std::runtime_error(ss.str());
      }
      child->instantiate(nullptr, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds,
                         tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
