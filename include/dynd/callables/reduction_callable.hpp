//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/reduction_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    typedef void (*callable_reduction_instantiate_t)(callable &self, callable &child, char *data, kernel_builder *ckb,
                                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                                     const ndt::type *src_tp, const char *const *src_arrmeta,
                                                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                                                     const std::map<std::string, ndt::type> &tp_vars);

    class reduction_callable : public base_callable {
      callable m_child;

    public:
      typedef reduction_data_type data_type;

      reduction_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(call_graph &DYND_UNUSED(cg), const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                               const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                               const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        return dst_tp;
      }

      char *data_init(const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
        char *data = reinterpret_cast<char *>(new data_type());

        const array &identity = kwds[1];
        if (!identity.is_na()) {
          reinterpret_cast<data_type *>(data)->identity = identity;
        }

        if (kwds[0].is_na()) {
          reinterpret_cast<data_type *>(data)->naxis =
              src_tp[0].get_ndim() - m_child.get_type()->get_return_type().get_ndim();
          reinterpret_cast<data_type *>(data)->axes = NULL;
        } else {
          reinterpret_cast<data_type *>(data)->naxis = kwds[0].get_dim_size();
          reinterpret_cast<data_type *>(data)->axes = reinterpret_cast<const int *>(kwds[0].cdata());
        }

        if (kwds[2].is_na()) {
          reinterpret_cast<data_type *>(data)->keepdims = false;
        } else {
          reinterpret_cast<data_type *>(data)->keepdims = kwds[2].as<bool>();
        }

        const ndt::type &child_dst_tp = m_child.get_type()->get_return_type();

        if (!dst_tp.is_symbolic()) {
          reinterpret_cast<data_type *>(data)->ndim = src_tp[0].get_ndim() - child_dst_tp.get_ndim();
          reinterpret_cast<data_type *>(data)->stored_ndim = reinterpret_cast<data_type *>(data)->ndim;
        }

        ndt::type child_src_tp = src_tp[0].get_type_at_dimension(NULL, reinterpret_cast<data_type *>(data)->naxis);
        reinterpret_cast<data_type *>(data)->child_data =
            m_child->data_init(child_dst_tp, nsrc, &child_src_tp, nkwd - 3, kwds, tp_vars);

        return data;
      }

      void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                            const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        ndt::type child_dst_tp = m_child.get_type()->get_return_type();
        if (child_dst_tp.is_symbolic()) {
          ndt::type child_src_tp = src_tp[0].get_type_at_dimension(NULL, reinterpret_cast<data_type *>(data)->naxis);
          m_child.get()->resolve_dst_type(NULL, child_dst_tp, nsrc, &child_src_tp, nkwd, kwds, tp_vars);
        }

        // check that the child_dst_tp and the child_src_tp are the same here

        dst_tp = child_dst_tp;
        reinterpret_cast<data_type *>(data)->ndim = src_tp[0].get_ndim() - dst_tp.get_ndim();
        reinterpret_cast<data_type *>(data)->stored_ndim = reinterpret_cast<data_type *>(data)->ndim;

        for (intptr_t i = reinterpret_cast<data_type *>(data)->ndim - 1,
                      j = reinterpret_cast<data_type *>(data)->naxis - 1;
             i >= 0; --i) {
          if (reinterpret_cast<data_type *>(data)->axes == NULL ||
              (j >= 0 && i == reinterpret_cast<data_type *>(data)->axes[j])) {
            if (reinterpret_cast<data_type *>(data)->keepdims) {
              dst_tp = ndt::make_fixed_dim(1, dst_tp);
            }
            --j;
          } else {
            ndt::type dim_tp = src_tp[0].get_type_at_dimension(NULL, i);
            dst_tp = dim_tp.extended<ndt::base_dim_type>()->with_element_type(dst_tp);
          }
        }
      }

      void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                       const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                       const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        static const callable_reduction_instantiate_t table[2][2][2] = {
            {{reduction_kernel<fixed_dim_id, false, false>::instantiate,
              reduction_kernel<fixed_dim_id, false, true>::instantiate},
             {reduction_kernel<fixed_dim_id, true, false>::instantiate,
              reduction_kernel<fixed_dim_id, true, true>::instantiate}},
            {{NULL, reduction_kernel<var_dim_id, false, true>::instantiate}, {NULL, NULL}}};

        if (reinterpret_cast<data_type *>(data)->ndim == 0) {
          callable &child = m_child;
          child.get()->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
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
            make_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq,
                                   &eval::default_eval_context);
            return;
          }

          nd::callable constant = functional::constant(reinterpret_cast<data_type *>(data)->identity);
          constant->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds,
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
