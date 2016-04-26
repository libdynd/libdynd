//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/callables/outer_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class outer_entry_callable : public base_callable {
      class dispatch_callable : public base_callable {
        callable m_children[4];

      public:
        dispatch_callable()
            : base_callable(ndt::type()),
              m_children{callable(), make_callable<outer_callable<1>>(), make_callable<outer_callable<2>>(),
                         make_callable<outer_callable<3>>()} {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &dst_tp,
                          size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &tp_vars) {
          return m_children[nsrc]->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }
      };

      struct data_type {
        base_callable *child;
        size_t i;
      };

      static callable dispatch_child;

      callable m_child;

    public:
      outer_entry_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        data_type data{m_child.get(), 0};

        return dispatch_child->resolve(this, reinterpret_cast<char *>(&data), cg, dst_tp, nsrc, src_tp, nkwd, kwds,
                                       tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
