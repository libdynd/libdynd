//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/elwise_callable.hpp>
#include <dynd/types/dim_fragment_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <size_t N>
    class elwise_dispatch_callable : public base_callable {
    public:
      struct data_type {
        base_callable *child;
        bool res_ignore;
        bool state;
        size_t ndim;
        bool first;
      };

      elwise_dispatch_callable() : base_callable(ndt::type()) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *data, call_graph &cg, const ndt::type &dst_tp,
                        size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp =
            reinterpret_cast<data_type *>(data)->child->get_type().template extended<ndt::callable_type>();
        bool first = reinterpret_cast<data_type *>(data)->first;
        reinterpret_cast<data_type *>(data)->first = false;
        bool state = reinterpret_cast<data_type *>(data)->state;

        bool dst_variadic = dst_tp.is_variadic();
        bool all_same = true;
        if (!dst_tp.is_symbolic()) {
          all_same = dst_tp.get_ndim() == child_tp->get_return_type().get_ndim();
        }
        for (size_t i = 0; i < nsrc; ++i) {
          if (src_tp[i].get_ndim() != child_tp->get_pos_type(i).get_ndim()) {
            all_same = false;
            break;
          }
        }

        if (all_same) {
          return reinterpret_cast<data_type *>(data)->child->resolve(
              this, reinterpret_cast<char *>(&data), cg,
              dst_tp.is_symbolic() ? reinterpret_cast<data_type *>(data)->child->get_return_type() : dst_tp, nsrc,
              src_tp, nkwd, kwds, tp_vars);
        }

        // Do a pass through the src types to classify them
        bool src_all_strided = true, src_all_strided_or_var = true;
        for (size_t i = 0; i < nsrc; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          switch (src_tp[i].get_id()) {
          case fixed_dim_id:
            break;
          case var_dim_id:
            src_all_strided = false;
            break;
          default:
            // If it's a scalar, allow it to broadcast like
            // a strided dimension
            if (src_ndim > 0) {
              src_all_strided_or_var = false;
            }
            break;
          }
        }

        bool var_broadcast = !src_all_strided;
        for (size_t i = 0; i < nsrc; ++i) {
          var_broadcast &= src_tp[i].get_id() == var_dim_id ||
                           (src_tp[i].get_id() == fixed_dim_id &&
                            src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size() == 1);
        }

        if ((dst_variadic || (dst_tp.get_id() == fixed_dim_id)) && src_all_strided) {
          static callable f = make_callable<elwise_callable<fixed_dim_id, fixed_dim_id, no_traits, N>>();
          static callable g = make_callable<elwise_callable<fixed_dim_id, fixed_dim_id, state_traits, N>>();
          if (!first && state) {
            return g->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
          } else {
            return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
          }
        } else if (((dst_variadic) || dst_tp.get_id() == var_dim_id) && (var_broadcast || src_all_strided)) {
          static callable f = make_callable<elwise_callable<var_dim_id, fixed_dim_id, no_traits, N>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else if (src_all_strided_or_var) {
          static callable f = make_callable<elwise_callable<fixed_dim_id, var_dim_id, no_traits, N>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }

        std::stringstream ss;
        ss << "Cannot process lifted elwise expression from (";
        for (size_t i = 0; i < nsrc; ++i) {
          ss << src_tp[i];
          if (i != nsrc - 1) {
            ss << ", ";
          }
        }
        ss << ") to " << dst_tp;
        throw std::runtime_error(ss.str());
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
