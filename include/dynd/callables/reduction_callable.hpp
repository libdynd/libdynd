//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/base_reduction_callable.hpp>
#include <dynd/kernels/reduction_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class reduction_dispatch_callable : public base_callable {
      callable m_child;

    public:
      reduction_dispatch_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      typedef typename base_reduction_callable::data_type new_data_type;

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &dst_tp,
                        size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        new_data_type new_data;
        if (data == nullptr) {
          new_data.child = m_child;
          if (kwds[0].is_na()) {
            new_data.naxis = src_tp[0].get_ndim() - m_child.get_type()->get_return_type().get_ndim();
            new_data.axes = NULL;
          } else {
            new_data.naxis = kwds[0].get_dim_size();
            new_data.axes = reinterpret_cast<const int *>(kwds[0].cdata());
          }

          new_data.identity = kwds[1];

          if (kwds[2].is_na()) {
            new_data.keepdims = false;
          } else {
            new_data.keepdims = kwds[2].as<bool>();
          }

          intptr_t ndim = src_tp[0].get_ndim() - m_child->get_ret_type().get_ndim();
          new_data.ndim = ndim;
          new_data.axis = 0;

          data = reinterpret_cast<char *>(&new_data);
        }

        if (src_tp[0].get_id() == fixed_dim_id) {
          static callable f = make_callable<reduction_callable<fixed_dim_id>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else {
          static callable f = make_callable<reduction_callable<var_dim_id>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
