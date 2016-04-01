//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/functional.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <int N>
    class outer_callable : public base_callable {
      callable m_child;

    public:
      outer_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        ndt::type tp = m_child->resolve(this, nullptr, cg, dst_tp.is_symbolic() ? m_child.get_ret_type() : dst_tp, nsrc,
                                        src_tp, nkwd, kwds, tp_vars);
        for (intptr_t i = nsrc - 1; i >= 0; --i) {
          if (!src_tp[i].is_scalar()) {
            tp = src_tp[i].with_replaced_dtype(tp);
          }
        }

        return tp;
      }

      void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                            intptr_t nkwd, const dynd::nd::array *kwds,
                            const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = m_child.get_type();

        if (child_tp->get_return_type().is_symbolic()) {
          m_child->resolve_dst_type(NULL, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else {
          dst_tp = ndt::substitute(child_tp->get_return_type(), tp_vars, false);
        }

        ndt::type tp = dst_tp.without_memory_type();
        for (intptr_t i = nsrc - 1; i >= 0; --i) {
          if (!src_tp[i].without_memory_type().is_scalar()) {
            tp = src_tp[i].without_memory_type().with_replaced_dtype(tp);
          }
        }
        if (dst_tp.get_base_id() == memory_id) {
          dst_tp = dst_tp.extended<ndt::base_memory_type>()->with_replaced_storage_type(tp);
        } else {
          dst_tp = tp;
        }
      }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                       intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t ndim = 0;
        for (intptr_t i = 0; i < nsrc; ++i) {
          ndim += src_tp[i].get_ndim();
        }

        std::vector<ndt::type> new_src_tp(nsrc);
        std::vector<const char *> new_src_arrmeta;

        arrmeta_holder *new_src_arrmeta_holder = new arrmeta_holder[nsrc];
        for (intptr_t i = 0, j = 0; i < nsrc; ++i) {
          ndt::type old_tp = src_tp[i];
          new_src_tp[i] = old_tp.with_new_axis(0, j);
          new_src_tp[i] = new_src_tp[i].with_new_axis(new_src_tp[i].get_ndim(), ndim - new_src_tp[i].get_ndim());
          ndt::type new_tp = new_src_tp[i];

          new (new_src_arrmeta_holder + i) arrmeta_holder(new_tp);
          char *new_arrmeta = new_src_arrmeta_holder[i].get();

          intptr_t k;
          for (k = 0; k < j; ++k) {
            size_stride_t *smd = reinterpret_cast<size_stride_t *>(new_arrmeta);
            smd->dim_size = 1;
            smd->stride = 0;
            new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
          }
          j += old_tp.get_ndim();
          for (; old_tp.get_ndim(); ++k) {
            if (new_tp.get_base_id() == memory_id) {
              new_tp.extended<ndt::base_memory_type>()
                  ->get_element_type()
                  .extended<ndt::base_dim_type>()
                  ->arrmeta_copy_construct_onedim(new_arrmeta, src_arrmeta[i], intrusive_ptr<memory_block_data>());
            } else {
              new_tp.extended<ndt::base_dim_type>()->arrmeta_copy_construct_onedim(new_arrmeta, src_arrmeta[i],
                                                                                   intrusive_ptr<memory_block_data>());
            }
            old_tp = old_tp.get_type_at_dimension(const_cast<char **>(src_arrmeta + i), 1);
            new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
          }
          for (; new_tp.get_ndim();) {
            size_stride_t *smd = reinterpret_cast<size_stride_t *>(new_arrmeta);
            smd->dim_size = 1;
            smd->stride = 0;
            new_tp = new_tp.get_type_at_dimension(&new_arrmeta, 1);
          }

          new_src_arrmeta.push_back(new_src_arrmeta_holder[i].get());
        }

        callable f = elwise(m_child);
        f->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, new_src_tp.data(), new_src_arrmeta.data(), kernreq, nkwd,
                       kwds, tp_vars);

        delete[] new_src_arrmeta_holder;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
