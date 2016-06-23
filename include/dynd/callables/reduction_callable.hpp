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
      callable m_identity;
      callable m_child;

    public:
      reduction_dispatch_callable(const ndt::type &tp, const callable &identity, const callable &child)
          : base_callable(tp), m_identity(identity), m_child(child) {}

      typedef typename base_reduction_callable::data_type new_data_type;

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &dst_tp,
                        size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        new_data_type new_data;
        if (data == nullptr) {
          new_data.identity = m_identity;
          new_data.child = m_child;
          if (kwds[0].is_na()) {
            new_data.naxis = src_tp[0].get_ndim() - m_child->get_ret_type().get_ndim();
            new_data.axes = NULL;
          } else {
            new_data.naxis = kwds[0].get_dim_size();
            new_data.axes = reinterpret_cast<const int *>(kwds[0].cdata());
          }

          if (kwds[1].is_na()) {
            new_data.keepdims = false;
          } else {
            new_data.keepdims = kwds[1].as<bool>();
          }

          intptr_t ndim = src_tp[0].get_ndim() - m_child->get_ret_type().get_ndim();
          new_data.ndim = ndim;
          new_data.axis = 0;

          data = reinterpret_cast<char *>(&new_data);
        }

        if (src_tp[0].is_scalar()) {
          cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                             const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
            intptr_t root_ckb_offset = kb.size();
            switch (nsrc) {
            case 1:
              kb.emplace_back<scalar_reduction_kernel<1>>(kernreq);
              break;
            case 2:
              kb.emplace_back<scalar_reduction_kernel<2>>(kernreq);
              break;
            default:
              throw std::runtime_error("unsupported number of arguments");
            }

            kb(kernel_request_single, nullptr, dst_arrmeta, nsrc, src_arrmeta);

            intptr_t init_offset = kb.size();
            kb(kernel_request_single, nullptr, dst_arrmeta, nsrc, nullptr);

            switch (nsrc) {
            case 1: {
              scalar_reduction_kernel<1> *e = kb.get_at<scalar_reduction_kernel<1>>(root_ckb_offset);
              e->init_offset = init_offset - root_ckb_offset;
              break;
            }
            case 2: {
              scalar_reduction_kernel<2> *e = kb.get_at<scalar_reduction_kernel<2>>(root_ckb_offset);
              e->init_offset = init_offset - root_ckb_offset;
              break;
            }
            default:
              throw std::runtime_error("unsupported number of arguments");
            };

          });

          ndt::type ret_tp = m_child->resolve(this, nullptr, cg, dst_tp, nsrc, src_tp, nkwd - 2, kwds + 2, tp_vars);

          m_identity->resolve(this, nullptr, cg, ret_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

          return ret_tp;
        }

        if (src_tp[0].get_id() == fixed_dim_id) {
          static callable f[2] = {make_callable<reduction_callable<fixed_dim_id, 1>>(),
                                  make_callable<reduction_callable<fixed_dim_id, 2>>()};
          return f[nsrc - 1]->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else {
          static callable f = make_callable<reduction_callable<var_dim_id, 1>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
