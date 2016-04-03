//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/base_elwise_callable.hpp>
#include <dynd/kernels/elwise_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, size_t N>
    class elwise_callable;

    template <size_t N>
    class elwise_callable<fixed_dim_id, fixed_dim_id, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::node_type node_type;

    public:
      ndt::type with_ret_type(intptr_t ret_size, const ndt::type &ret_element_tp) {
        return ndt::make_type<ndt::fixed_dim_type>(ret_size, ret_element_tp);
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride;
        for (size_t i = 0; i < N; ++i) {
          if (reinterpret_cast<node_type *>(node)->arg_broadcast[i]) {
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
          } else {
            src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        node = next(node);
        node->callee->instantiate(node, data, ckb, dst_tp, dst_arrmeta + sizeof(size_stride_t), N, src_tp,
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
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
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            finished &= src_ndim == 1;
          } else {
            std::stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                  "or fixed dim, got "
               << src_tp[i];
            throw std::runtime_error(ss.str());
          }
        }

        call_node *node = NULL;

        ckb->emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(node, data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(node, NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

    // src is either fixed or var
    template <size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::node_type node_type;

    public:
      ndt::type with_ret_type(intptr_t ret_size, const ndt::type &ret_element_tp) {
        if (ret_size == 1) {
          return ndt::make_type<ndt::var_dim_type>(ret_element_tp);
        }

        return ndt::make_type<ndt::fixed_dim_type>(ret_size, ret_element_tp);
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t dst_size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride, src_offset;
        for (size_t i = 0; i < N; ++i) {
          if (reinterpret_cast<node_type *>(node)->arg_var[i]) {
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
          } else {
            if (reinterpret_cast<node_type *>(node)->arg_broadcast[i]) {
              src_offset[i] = 0;
              src_stride[i] = 0;
              child_src_arrmeta[i] = src_arrmeta[i];
            } else {
              src_offset[i] = 0;
              src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
              child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            }
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, N>>(
            kernreq, dst_size, dst_stride, src_stride.data(), src_offset.data(),
            reinterpret_cast<node_type *>(node)->arg_var.data());

        node = next(node);
        node->callee->instantiate(node, data, ckb, dst_tp, dst_arrmeta + sizeof(size_stride_t), N, src_tp,
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
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
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
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

        call_node *node = NULL;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(node, data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(node, NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

    template <size_t N>
    class elwise_callable<var_dim_id, fixed_dim_id, N> : public base_elwise_callable<N> {
      typedef typename base_elwise_callable<N>::node_type node_type;

    public:
      ndt::type with_ret_type(intptr_t DYND_UNUSED(ret_size), const ndt::type &ret_element_tp) {
        return ndt::make_type<ndt::var_dim_type>(ret_element_tp);
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride;
        std::array<intptr_t, N> src_offset;
        std::array<intptr_t, N> src_size;
        for (size_t i = 0; i < N; ++i) {
          if (reinterpret_cast<node_type *>(node)->arg_var[i]) {
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            //            src_size[i] = -1;
          } else {
            if (reinterpret_cast<node_type *>(node)->arg_broadcast[i]) {
              src_stride[i] = 0;
              src_offset[i] = 0;
              child_src_arrmeta[i] = src_arrmeta[i];
              src_size[i] = 1;
            } else {
              src_offset[i] = 0;
              src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
              src_size[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->dim_size;
            }
          }
        }

        ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
            kernreq, dst_md->blockref.get(), reinterpret_cast<node_type *>(node)->res_alignment, dst_md->stride,
            dst_md->offset, src_stride.data(), src_offset.data(), src_size.data(),
            reinterpret_cast<node_type *>(node)->arg_var.data());

        node = next(node);
        node->callee->instantiate(node, data, ckb, dst_tp, dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), N,
                                  src_tp, child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
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
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i], &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
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

        call_node *node = NULL;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(node, data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(node, NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
