//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class base_reduction_callable : public base_callable {
    public:
      struct data_type {
        callable child;
        bool keepdims;
        size_t naxis;
        const int *axes;
        int axis;
        int ndim;
      };

      base_reduction_callable() : base_callable(ndt::type()) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &res_tp,
                        size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        callable &child = reinterpret_cast<data_type *>(data)->child;
        const ndt::type &child_ret_tp = child.get_ret_type();

        bool reduce = reinterpret_cast<data_type *>(data)->axes == NULL;
        for (size_t i = 0; i < reinterpret_cast<data_type *>(data)->naxis && !reduce; ++i) {
          if (reinterpret_cast<data_type *>(data)->axes[i] == reinterpret_cast<data_type *>(data)->axis) {
            reduce = true;
          }
        }

        ndt::type arg_element_tp[1];
        if (reduce) {
          arg_element_tp[0] = src_tp[0].extended<ndt::base_dim_type>()->get_element_type();
        } else {
          arg_element_tp[0] = src_tp[0];
        }
        ++reinterpret_cast<data_type *>(data)->axis;

        ndt::type ret_element_tp;
        if (reinterpret_cast<data_type *>(data)->axis == reinterpret_cast<data_type *>(data)->ndim) {
          ret_element_tp =
              child->resolve(this, nullptr, cg, child_ret_tp, nsrc, arg_element_tp, nkwd - 3, kwds + 3, tp_vars);
        } else {
          ret_element_tp = this->resolve(this, data, cg, res_tp, nsrc, arg_element_tp, nkwd, kwds, tp_vars);
        }

        if (reduce) {
          if (reinterpret_cast<data_type *>(data)->keepdims) {
            return ndt::make_type<ndt::fixed_dim_type>(1, ret_element_tp);
          }

          return ret_element_tp;
        }

        return src_tp[0].extended<ndt::base_dim_type>()->with_element_type(ret_element_tp);
      }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb), const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                       const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                       kernel_request_t DYND_UNUSED(kernreq), intptr_t DYND_UNUSED(nkwd),
                       const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
