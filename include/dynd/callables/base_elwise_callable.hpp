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
        bool res_ignore;
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
        bool res_ignore = reinterpret_cast<codata_type *>(codata)->res_ignore;
        data.res_ignore = reinterpret_cast<codata_type *>(codata)->res_ignore;

        base_callable *child = reinterpret_cast<codata_type *>(codata)->child;
        const ndt::type &child_ret_tp = child->get_return_type();
        const std::vector<ndt::type> &child_arg_tp = child->get_argument_types();

        std::array<intptr_t, N> arg_size;
        std::array<ndt::type, N> arg_element_tp;
        intptr_t max_ndim = reinterpret_cast<codata_type *>(codata)->ndim;
        for (size_t i = 0; i < N; ++i) {
          data.arg_broadcast[i] = (arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim()) < max_ndim;
          if (data.arg_broadcast[i]) {
            arg_size[i] = 1;
            arg_element_tp[i] = arg_tp[i];
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_dim_size();
            arg_element_tp[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          }
        }

        for (size_t i = 0; i < N; ++i) {
          data.arg_var[i] = arg_tp[i].get_id() == var_dim_id;
        }

        intptr_t res_size;
        ndt::type res_element_tp;
        if (res_ignore) {
          res_size = 1;
          res_element_tp = res_tp;
        } else if (res_tp.is_variadic()) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (!data.arg_broadcast[i] && arg_size[i] != -1) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          res_size = res_tp.extended<ndt::base_dim_type>()->get_dim_size();
          res_element_tp = res_tp.extended<ndt::base_dim_type>()->get_element_type();
        }

        // if not ignore res
        if (!res_ignore) {
          for (size_t i = 0; i < N; ++i) {
            if (!data.arg_broadcast[i]) {
              if (arg_size[i] != -1 && res_size != -1 && res_size != arg_size[i] && arg_size[i] != 1) {
                throw std::runtime_error("broadcast error 1");
              }
            }
          }
        }

        subresolve(cg, reinterpret_cast<char *>(&data));

        if (--reinterpret_cast<codata_type *>(codata)->ndim > 0) {
          res_element_tp =
              caller->resolve(this, codata, cg, res_element_tp, N, arg_element_tp.data(), nkwd, kwds, tp_vars);
        } else {
          res_element_tp = child->resolve(this, nullptr, cg, res_tp.is_variadic() ? child_ret_tp : res_element_tp, N,
                                          arg_element_tp.data(), nkwd, kwds, tp_vars);
        }

        if (res_ignore) {
          return res_element_tp;
        }

        return with_return_type(res_size, res_element_tp);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
