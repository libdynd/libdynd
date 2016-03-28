//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/elwise_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, int N>
    class elwise_callable;

    template <size_t N>
    class elwise_callable<fixed_dim_id, fixed_dim_id, N> {
    public:
      static void instantiate(callable &self, callable &child, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_id() == typevar_constructed_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        std::array<intptr_t, N> src_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size;
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          }
          else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                            &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            finished &= src_ndim == 1;
          }
          else {
            std::stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                  "or fixed dim, got "
               << src_tp[i];
            throw std::runtime_error(ss.str());
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

    template <size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, N> {
    public:
      static void instantiate(callable &self, callable &child, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        std::array<intptr_t, N> src_stride, src_offset;
        std::array<bool, N> is_src_var;
        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_size;
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          // The src[i] strided parameters
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          }
          else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                            &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          }
          else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, N>>(kernreq, size, dst_stride, src_stride.data(),
                                                                      src_offset.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

    template <size_t N>
    class elwise_callable<var_dim_id, fixed_dim_id, N> {
    public:
      static void instantiate(callable &self, callable &child, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();
        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type);
        child_dst_tp = dst_vdd->get_element_type();

        std::array<intptr_t, N> src_stride, src_offset, src_size;
        std::array<bool, N> is_src_var;

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          // The src[i] strided parameters
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          }
          else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i], &src_stride[i], &child_src_tp[i],
                                            &child_src_arrmeta[i])) {
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          }
          else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
            kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
            src_stride.data(), src_offset.data(), src_size.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
