//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <size_t N>
    class base_elwise_callable : public base_callable {
    protected:
      struct codata_type {
        base_callable *child;
        bool res_ignore;
        bool state;
        size_t ndim;
        bool first;
      };

      struct data_type {
        std::array<bool, N> arg_broadcast;
        std::array<bool, N> arg_var;
        intptr_t res_alignment;
        size_t ndim;
      };

    public:
      base_elwise_callable() : base_callable(ndt::type()) {}

      virtual void subresolve(call_graph &cg, const char *data) = 0;

      virtual ndt::type with_return_type(intptr_t ret_size, const ndt::type &ret_element_tp) = 0;

      ndt::type resolve(base_callable *caller, char *codata, call_graph &cg, const ndt::type &res_tp,
                        size_t DYND_UNUSED(narg), const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        data_type data;
        data.ndim = reinterpret_cast<codata_type *>(codata)->ndim;
        //        bool res_ignore = reinterpret_cast<codata_type *>(codata)->res_ignore;

        base_callable *child = reinterpret_cast<codata_type *>(codata)->child;
        const ndt::type &child_ret_tp = child->get_return_type();
        const std::vector<ndt::type> &child_arg_tp = child->get_argument_types();

        std::array<intptr_t, N> arg_size;
        std::array<intptr_t, N> arg_ndim;
        intptr_t max_ndim = reinterpret_cast<codata_type *>(codata)->ndim;
        for (size_t i = 0; i < N; ++i) {
          arg_ndim[i] = arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim();
          if (arg_ndim[i] == 0) {
            arg_size[i] = 1;
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_dim_size();
          }
        }

        for (size_t i = 0; i < N; ++i) {
          data.arg_var[i] = arg_tp[i].get_id() == var_dim_id;
        }

        bool res_variadic = res_tp.is_variadic();
        intptr_t res_size;
        ndt::type res_element_tp;
        if (res_variadic) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (arg_ndim[i] == max_ndim && arg_size[i] != -1) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          res_size = res_tp.extended<ndt::base_dim_type>()->get_dim_size();
          res_element_tp = res_tp.extended<ndt::base_dim_type>()->get_element_type();
        }

        std::array<ndt::type, N> arg_element_tp;
        for (size_t i = 0; i < N; ++i) {
          if (arg_ndim[i] == max_ndim) {
            data.arg_broadcast[i] = false;
            if (arg_size[i] != -1 && res_size != -1 && res_size != arg_size[i] && arg_size[i] != 1) {
              throw std::runtime_error("broadcast error 1");
            }
            arg_element_tp[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          } else {
            data.arg_broadcast[i] = true;
            arg_element_tp[i] = arg_tp[i];
          }
        }

        subresolve(cg, reinterpret_cast<char *>(&data));

        ndt::type resolved_ret_tp;
        if (--reinterpret_cast<codata_type *>(codata)->ndim > 0) {
          resolved_ret_tp = with_return_type(res_size, caller->resolve(this, codata, cg, res_element_tp, N,
                                                                       arg_element_tp.data(), nkwd, kwds, tp_vars));
        } else {
          resolved_ret_tp =
              with_return_type(res_size, child->resolve(this, nullptr, cg, res_variadic ? child_ret_tp : res_element_tp,
                                                        N, arg_element_tp.data(), nkwd, kwds, tp_vars));
        }

        if (resolved_ret_tp.get_id() == var_dim_id) {
          data.res_alignment = resolved_ret_tp.extended<ndt::var_dim_type>()->get_target_alignment();
        }

        return resolved_ret_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
