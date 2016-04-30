//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/outer_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <size_t NArg>
    class outer_callable : public base_callable {
      struct data_type {
        base_callable *child;
        size_t i;
      };

    public:
      outer_callable() : base_callable(ndt::type()) {}

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &dst_tp,
                        size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        base_callable *child = reinterpret_cast<data_type *>(data)->child;
        size_t &i = reinterpret_cast<data_type *>(data)->i;

        cg.emplace_back([i](kernel_builder &kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta,
                            size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          kb.emplace_back<outer_kernel<NArg>>(kernreq, i, dst_arrmeta, src_arrmeta);

          const char *src_element_arrmeta[NArg];
          for (size_t j = 0; j < i; ++j) {
            src_element_arrmeta[j] = src_arrmeta[j];
          }
          src_element_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
          for (size_t j = i + 1; j < NArg; ++j) {
            src_element_arrmeta[j] = src_arrmeta[j];
          }

          kb(kernel_request_strided, data, dst_arrmeta + sizeof(size_stride_t), NArg, src_element_arrmeta);
        });

        ndt::type arg_element_tp[NArg];
        for (size_t j = 0; j < i; ++j) {
          arg_element_tp[j] = src_tp[j];
        }
        arg_element_tp[i] = src_tp[i].extended<ndt::base_dim_type>()->get_element_type();
        for (size_t j = i + 1; j < NArg; ++j) {
          arg_element_tp[j] = src_tp[j];
        }

        size_t j = i;
        while (i < NArg && arg_element_tp[i].is_scalar()) {
          ++i;
        }

        ndt::type ret_element_tp;
        if (i < NArg) {
          ret_element_tp = caller->resolve(this, data, cg, dst_tp, NArg, arg_element_tp, nkwd, kwds, tp_vars);
        } else {
          ret_element_tp =
              child->resolve(this, nullptr, cg, child->get_ret_type(), NArg, arg_element_tp, nkwd, kwds, tp_vars);
        }

        return src_tp[j].extended<ndt::base_dim_type>()->with_element_type(ret_element_tp);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
