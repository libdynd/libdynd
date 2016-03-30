//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/elwise_callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <size_t N>
    class elwise_dispatch_callable : public base_callable {
      callable m_child;

    public:
      elwise_dispatch_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child)
      {
        m_abstract = true;
      }

      void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &g, ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, size_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars)
      {
        std::cout << "elwise_dispatch_callable::new_resolve" << std::endl;
        //        m_child->new_resolve(stack, nkwd, kwds, tp_vars);

        resolve_dst_type(nullptr, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

        const ndt::callable_type *child_tp = m_child.get_type();

        // Check if no lifting is required
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        if (dst_ndim == 0) {
          intptr_t i = 0;
          for (; i < nsrc; ++i) {
            intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
            if (src_ndim != 0) {
              break;
            }
          }
          if (i == nsrc) {
            if (!m_child->is_abstract()) {
              g.emplace_back(m_child.get());
            }

            // No dimensions to lift, call the elementwise instantiate directly
            return m_child->new_resolve(this, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
          }
          else {
            intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
            std::stringstream ss;
            ss << "Trying to broadcast " << src_ndim << " dimensions of " << src_tp[i] << " into 0 dimensions of "
               << dst_tp << ", the destination dimension count must be greater. The "
                            "element "
                            "callable type is \""
               << ndt::type(child_tp, true) << "\"";
            throw broadcast_error(ss.str());
          }
        }

        // Do a pass through the src types to classify them
        bool src_all_strided = true, src_all_strided_or_var = true;
        for (intptr_t i = 0; i < nsrc; ++i) {
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

        // Call to some special-case functions based on the
        // destination type
        switch (dst_tp.get_id()) {
        case fixed_dim_id:
          if (src_all_strided) {
            callable f = make_callable<elwise_callable<fixed_dim_id, fixed_dim_id, N>>(m_child);
            if (!f->is_abstract()) {
              g.emplace_back(f.get());
            }

            f->new_resolve(this, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

            //            elwise_callable<fixed_dim_id, fixed_dim_id, N>::new_resolve(stack, nkwd, kwds, tp_vars);
            return;
          }
          else if (src_all_strided_or_var) {
            throw std::runtime_error("fixed_dim_id, var_dim_id");
            //            elwise_callable<fixed_dim_id, var_dim_id, N>::elwise_instantiate(
            //              self, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
            //              kwds, tp_vars);
            return;
          }
          else {
            // TODO
          }
          break;
        case var_dim_id:
          if (src_all_strided_or_var) {
            callable f = make_callable<elwise_callable<var_dim_id, fixed_dim_id, N>>(m_child);
            if (!f->is_abstract()) {
              g.emplace_back(f.get());
            }

            f->new_resolve(this, g, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
            //            elwise_callable<var_dim_id, fixed_dim_id, N>::elwise_instantiate(
            //              self, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
            //              kwds, tp_vars);
            return;
          }
          else {
            // TODO
          }
          break;
        default:
          break;
        }

        std::stringstream ss;
        ss << "Cannot process lifted elwise expression from (";
        for (intptr_t i = 0; i < nsrc; ++i) {
          ss << src_tp[i];
          if (i != nsrc - 1) {
            ss << ", ";
          }
        }
        ss << ") to " << dst_tp;
        throw std::runtime_error(ss.str());
      }

      void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                            intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const callable &child = m_child;
        const ndt::callable_type *child_af_tp = m_child.get_type();

        intptr_t ndim = 0;
        // First get the type for the child callable
        ndt::type child_dst_tp;
        std::vector<ndt::type> child_src_tp(nsrc);
        for (intptr_t i = 0; i < nsrc; ++i) {
          intptr_t child_ndim_i = child_af_tp->get_pos_type(i).get_ndim();
          if (child_ndim_i < src_tp[i].get_ndim()) {
            child_src_tp[i] = src_tp[i].get_dtype(child_ndim_i);
            ndim = std::max(ndim, src_tp[i].get_ndim() - child_ndim_i);
          }
          else {
            child_src_tp[i] = src_tp[i];
          }
        }

        child_dst_tp = child_af_tp->get_return_type();
        if (child_dst_tp.is_symbolic()) {
          child->resolve_dst_type(NULL, child_dst_tp, nsrc, child_src_tp.empty() ? NULL : child_src_tp.data(), nkwd,
                                  kwds, tp_vars);
        }

        // ...
        //        new (data) ndt::type(child_dst_tp);

        if (nsrc == 0) {
          dst_tp =
              tp_vars.at("Dims").extended<ndt::dim_fragment_type>()->apply_to_dtype(child_dst_tp.without_memory_type());
          if (child_dst_tp.get_base_id() == memory_id) {
            dst_tp = child_dst_tp.extended<ndt::base_memory_type>()->with_replaced_storage_type(dst_tp);
          }

          return;
        }

        // Then build the type for the rest of the dimensions
        if (ndim > 0) {
          dimvector shape(ndim), tmp_shape(ndim);
          for (intptr_t i = 0; i < ndim; ++i) {
            shape[i] = -1;
          }
          for (intptr_t i = 0; i < nsrc; ++i) {
            intptr_t ndim_i = src_tp[i].get_ndim() - child_af_tp->get_pos_type(i).get_ndim();
            if (ndim_i > 0) {
              ndt::type tp = src_tp[i].without_memory_type();
              intptr_t *shape_i = shape.get() + (ndim - ndim_i);
              intptr_t shape_at_j;
              for (intptr_t j = 0; j < ndim_i; ++j) {
                switch (tp.get_id()) {
                case fixed_dim_id:
                  shape_at_j = tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
                  if (shape_i[j] < 0 || shape_i[j] == 1) {
                    if (shape_at_j != 1) {
                      shape_i[j] = shape_at_j;
                    }
                  }
                  else if (shape_i[j] != shape_at_j && shape_at_j != 1) {
                    throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
                  }
                  break;
                case var_dim_id:
                  break;
                default:
                  throw broadcast_error(ndim, shape.get(), ndim_i, shape_i);
                }
                tp = tp.get_dtype(tp.get_ndim() - 1);
              }
            }
          }

          ndt::type tp = child_dst_tp.without_memory_type();
          for (intptr_t i = ndim - 1; i >= 0; --i) {
            if (shape[i] == -1) {
              tp = ndt::var_dim_type::make(tp);
            }
            else {
              tp = ndt::make_fixed_dim(shape[i], tp);
            }
          }
          if (child_dst_tp.get_base_id() == memory_id) {
            child_dst_tp = child_dst_tp.extended<ndt::base_memory_type>()->with_replaced_storage_type(tp);
          }
          else {
            child_dst_tp = tp;
          }
        }
        dst_tp = child_dst_tp;
      }

      void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                       const ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                       intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)

      {
        callable self = callable(this, true);
        callable &child = m_child;
        const ndt::callable_type *child_tp = m_child.get_type();

        // Check if no lifting is required
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        if (dst_ndim == 0) {
          intptr_t i = 0;
          for (; i < nsrc; ++i) {
            intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
            if (src_ndim != 0) {
              break;
            }
          }
          if (i == nsrc) {
            // No dimensions to lift, call the elementwise instantiate directly
            return child.get()->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                                            kwds, tp_vars);
          }
          else {
            intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
            std::stringstream ss;
            ss << "Trying to broadcast " << src_ndim << " dimensions of " << src_tp[i] << " into 0 dimensions of "
               << dst_tp << ", the destination dimension count must be greater. The "
                            "element "
                            "callable type is \""
               << ndt::type(child_tp, true) << "\"";
            throw broadcast_error(ss.str());
          }
        }

        // Do a pass through the src types to classify them
        bool src_all_strided = true, src_all_strided_or_var = true;
        for (intptr_t i = 0; i < nsrc; ++i) {
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

        // Call to some special-case functions based on the
        // destination type
        switch (dst_tp.get_id()) {
        case fixed_dim_id:
          if (src_all_strided) {
            elwise_callable<fixed_dim_id, fixed_dim_id, N>::elwise_instantiate(
                self, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
            return;
          }
          else if (src_all_strided_or_var) {
            elwise_callable<fixed_dim_id, var_dim_id, N>::elwise_instantiate(
                self, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
            return;
          }
          else {
            // TODO
          }
          break;
        case var_dim_id:
          if (src_all_strided_or_var) {
            elwise_callable<var_dim_id, fixed_dim_id, N>::elwise_instantiate(
                self, m_child, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
            return;
          }
          else {
            // TODO
          }
          break;
        default:
          break;
        }

        std::stringstream ss;
        ss << "Cannot process lifted elwise expression from (";
        for (intptr_t i = 0; i < nsrc; ++i) {
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
