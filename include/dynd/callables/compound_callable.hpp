//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/compound_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class left_compound_callable : public base_callable {
    public:
      callable m_child;

      left_compound_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.push_back([](call_node *&node, kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                        intptr_t nsrc, const char *const *src_arrmeta) {
          ckb->emplace_back<left_compound_kernel>(kernreq);
          node = next(node);

          const char *child_src_arrmeta[2] = {dst_arrmeta, src_arrmeta[0]};
          node->instantiate(node, ckb, kernreq | kernel_request_data_only, dst_arrmeta, nsrc + 1, child_src_arrmeta);
        });

        ndt::type child_src_tp[2] = {dst_tp, src_tp[0]};
        m_child->resolve(this, nullptr, cg, dst_tp, nsrc + 1, child_src_tp, nkwd, kwds, tp_vars);

        return dst_tp;
      }
    };

    class right_compound_callable : public base_callable {
    public:
      callable m_child;

      right_compound_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.push_back([](call_node *&node, kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                        intptr_t nsrc, const char *const *src_arrmeta) {
          ckb->emplace_back<right_compound_kernel>(kernreq);
          node = next(node);

          const char *child_src_arrmeta[2] = {src_arrmeta[0], dst_arrmeta};
          node->instantiate(node, ckb, kernreq | kernel_request_data_only, dst_arrmeta, nsrc + 1, child_src_arrmeta);
        });

        ndt::type child_src_tp[2] = {src_tp[0], dst_tp};
        m_child->resolve(this, nullptr, cg, dst_tp, nsrc + 1, child_src_tp, nkwd, kwds, tp_vars);

        return dst_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
