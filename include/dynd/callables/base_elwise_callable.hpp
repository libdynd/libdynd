//
// Copyright (C) 2011-16 DyND Developers
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

    template <size_t N>
    class base_elwise_callable : public base_callable {
    protected:
      struct data_type {
        callable &child;
      };

      struct node_type : call_node {
        using call_node::call_node;

        std::array<bool, N> arg_broadcast;
        std::array<bool, N> arg_var;
        intptr_t res_alignment;
      };

    public:
      base_elwise_callable() : base_callable(ndt::type(), sizeof(node_type)) {}

      virtual ndt::type with_ret_type(intptr_t ret_size, const ndt::type &ret_element_tp) = 0;

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &res_tp,
                        size_t DYND_UNUSED(narg), const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        node_type *node = cg.emplace_back<node_type>(this);

        callable &child = reinterpret_cast<data_type *>(data)->child;
        const ndt::type &child_ret_tp = child.get_ret_type();
        const std::vector<ndt::type> &child_arg_tp = child.get_arg_types();

        std::array<intptr_t, N> arg_size;
        std::array<intptr_t, N> arg_ndim;
        intptr_t max_ndim = 0;
        for (size_t i = 0; i < N; ++i) {
          arg_ndim[i] = arg_tp[i].get_ndim() - child_arg_tp[i].get_ndim();
          if (arg_ndim[i] == 0) {
            arg_size[i] = 1;
          } else {
            arg_size[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_dim_size();
            if (arg_ndim[i] > max_ndim) {
              max_ndim = arg_ndim[i];
            }
          }
        }

        for (size_t i = 0; i < N; ++i) {
          node->arg_var[i] = arg_tp[i].get_id() == var_dim_id;
        }

        bool res_variadic = res_tp.is_variadic();
        intptr_t res_size;
        ndt::type res_element_tp;
        intptr_t ret_ndim = res_tp.get_ndim() - child_ret_tp.get_ndim();
        if (res_variadic) {
          res_size = 1;
          for (size_t i = 0; i < N && res_size == 1; ++i) {
            if (arg_ndim[i] == max_ndim && arg_size[i] != -1) {
              res_size = arg_size[i];
            }
          }
          res_element_tp = res_tp;
        } else {
          if (ret_ndim > max_ndim) {
            max_ndim = ret_ndim;
          } else if (res_tp.get_ndim() - child_ret_tp.get_ndim() < max_ndim) {
            throw std::runtime_error("broadcast error");
          }
          res_size = res_tp.extended<ndt::base_dim_type>()->get_dim_size();
          res_element_tp = res_tp.extended<ndt::base_dim_type>()->get_element_type();
        }

        std::array<ndt::type, N> arg_element_tp;
        bool callback = ret_ndim > 1;
        for (size_t i = 0; i < N; ++i) {
          if (arg_ndim[i] == max_ndim) {
            node->arg_broadcast[i] = false;
            if (arg_size[i] != -1 && res_size != -1 && res_size != arg_size[i] && arg_size[i] != 1) {
              throw std::runtime_error("broadcast error");
            }
            arg_element_tp[i] = arg_tp[i].extended<ndt::base_dim_type>()->get_element_type();
          } else {
            node->arg_broadcast[i] = true;
            arg_element_tp[i] = arg_tp[i];
          }
          if (arg_element_tp[i].get_ndim() != child_arg_tp[i].get_ndim()) {
            callback = true;
          }
        }

        ndt::type resolved_ret_tp;
        if (callback) {
          resolved_ret_tp = with_ret_type(res_size, caller->resolve(this, nullptr, cg, res_element_tp, N,
                                                                    arg_element_tp.data(), nkwd, kwds, tp_vars));
        } else {
          resolved_ret_tp = with_ret_type(res_size, child->resolve(this, nullptr, cg,
                                                                   res_variadic ? child.get_ret_type() : res_element_tp,
                                                                   N, arg_element_tp.data(), nkwd, kwds, tp_vars));
        }

        if (resolved_ret_tp.get_id() == var_dim_id) {
          node->res_alignment = resolved_ret_tp.extended<ndt::var_dim_type>()->get_target_alignment();
        }

        return resolved_ret_tp;
      }

      void instantiate(call_node *&DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                       const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                       intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                       const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                       intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                       const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
