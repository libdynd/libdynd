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

    typedef void (*callable_reduction_instantiate_t)(callable &self, callable &child, char *data, kernel_builder *ckb,
                                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                                     const ndt::type *src_tp, const char *const *src_arrmeta,
                                                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                                                     const std::map<std::string, ndt::type> &tp_vars);

    class reduction_dispatch_callable : public base_callable {
      callable m_child;

    public:
      typedef reduction_data_type data_type;

      reduction_dispatch_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      typedef typename base_reduction_callable::data_type new_data_type;

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &dst_tp, size_t nsrc,
                        const ndt::type *src_tp, size_t nkwd, const array *kwds,
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

          intptr_t ndim = src_tp[0].get_ndim() - m_child.get_ret_type().get_ndim();
          new_data.ndim = ndim;
          new_data.axis = 0;

          data = reinterpret_cast<char *>(&new_data);
        }

        static callable f = make_callable<reduction_callable<fixed_dim_id>>();
        return f->resolve(caller, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        static const callable_reduction_instantiate_t table[2][2][2] = {
            {{reduction_kernel<fixed_dim_id, false, false>::instantiate,
              reduction_kernel<fixed_dim_id, false, true>::instantiate},
             {reduction_kernel<fixed_dim_id, true, false>::instantiate,
              reduction_kernel<fixed_dim_id, true, true>::instantiate}},
            {{NULL, reduction_kernel<var_dim_id, false, true>::instantiate}, {NULL, NULL}}};

        if (reinterpret_cast<data_type *>(data)->ndim == 0) {
          callable &child = m_child;
          child.get()->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   (reinterpret_cast<data_type *>(data)->stored_ndim == 0) ? kernel_request_single
                                                                                           : kernel_request_strided,
                                   nkwd - 3, kwds + 3, tp_vars);

          reinterpret_cast<data_type *>(data)->init_offset = ckb->size();

          // if identity is NULL, assign the first element to the output
          // otherwise, assign the identity to the output
          // init()

          // f(Ret, Identity)
          // f(Ret, Arg0_0)
          // f(Ret, Arg0_1)
          if (reinterpret_cast<data_type *>(data)->identity.is_null()) {
            nd::array error_mode = eval::default_eval_context.errmode;
            nd::assign->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, 1, src_tp, src_arrmeta, kernreq, 1,
                                    &error_mode, std::map<std::string, ndt::type>());
            return;
          }

          nd::callable constant = functional::constant(reinterpret_cast<data_type *>(data)->identity);
          constant->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds,
                                tp_vars);
          return;
        }

        callable f = callable(this, true);
        table[src_tp[0].get_id() - fixed_dim_id][reinterpret_cast<data_type *>(data)
                                                     ->is_broadcast()][reinterpret_cast<data_type *>(data)->is_inner()](
            f, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
