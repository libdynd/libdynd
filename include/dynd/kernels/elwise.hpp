//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Generic expr kernel + destructor for a strided dimension with
     * a fixed number of src operands.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, int N>
    struct elwise_ck;

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <int N>
    struct elwise_virtual_ck : base_virtual_kernel<elwise_virtual_ck<N>> {
      static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc,
                                   const ndt::type *src_tp, intptr_t nkwd, const dynd::nd::array *kwds,
                                   const std::map<std::string, ndt::type> &tp_vars)
      {
        const base_callable *child_af = reinterpret_cast<callable *>(static_data)->get();
        const ndt::callable_type *child_af_tp = reinterpret_cast<callable *>(static_data)->get_type();

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
          child_af->resolve_dst_type(const_cast<char *>(child_af->static_data()), NULL, child_dst_tp, nsrc,
                                     child_src_tp.empty() ? NULL : child_src_tp.data(), nkwd, kwds, tp_vars);
        }

        // ...
        //        new (data) ndt::type(child_dst_tp);

        if (nsrc == 0) {
          dst_tp =
              tp_vars.at("Dims").extended<ndt::dim_fragment_type>()->apply_to_dtype(child_dst_tp.without_memory_type());
          if (child_dst_tp.get_kind() == memory_kind) {
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
                switch (tp.get_type_id()) {
                case fixed_dim_type_id:
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
                case var_dim_type_id:
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
          if (child_dst_tp.get_kind() == memory_kind) {
            child_dst_tp = child_dst_tp.extended<ndt::base_memory_type>()->with_replaced_storage_type(tp);
          }
          else {
            child_dst_tp = tp;
          }
        }
        dst_tp = child_dst_tp;
      }

      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta,
                                  dynd::kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                  const dynd::nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)

      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = reinterpret_cast<callable *>(static_data)->get_type();

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
            return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
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
          switch (src_tp[i].get_type_id()) {
          case fixed_dim_type_id:
            break;
          case var_dim_type_id:
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
        switch (dst_tp.get_type_id()) {
        case fixed_dim_type_id:
          if (src_all_strided) {
            return elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>::instantiate(
                static_data, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
                kwds, tp_vars);
          }
          else if (src_all_strided_or_var) {
            return elwise_ck<fixed_dim_type_id, var_dim_type_id, N>::instantiate(
                static_data, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
                kwds, tp_vars);
          }
          else {
            // TODO
          }
          break;
        case var_dim_type_id:
          if (src_all_strided_or_var) {
            return elwise_ck<var_dim_type_id, fixed_dim_type_id, N>::instantiate(
                static_data, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
                kwds, tp_vars);
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

    template <int N>
    struct elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>
        : base_kernel<elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>, N> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N];

      DYND_CUDA_HOST_DEVICE elwise_ck(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
      }

      DYND_CUDA_HOST_DEVICE ~elwise_ck() { this->get_child()->destroy(); }

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        opchild(child, dst, m_dst_stride, src, m_src_stride, m_size);
      }

      DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride,
                                         size_t count)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        char *src_loop[N];
        for (int j = 0; j != N; ++j) {
          src_loop[j] = src[j];
        }

        for (size_t i = 0; i < count; i += 1) {
          opchild(child, dst, m_dst_stride, src_loop, m_src_stride, m_size);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_type_id() == typevar_constructed_type_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        intptr_t size, dst_stride, src_stride[N];
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
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

        self_type::make(ckb, kernreq, ckb_offset, size, dst_stride, dynd::detail::make_array_wrapper<N>(src_stride));
        kernreq = (kernreq & kernel_request_memory) | kernel_request_strided;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<N>::instantiate(
              static_data, data, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp,
              child_src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta, kernreq, ectx, nkwd,
                                        kwds, tp_vars);
      }
    };

    // int N, int K

    template <>
    struct elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 0>
        : base_kernel<elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 0>, 0> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      DYND_CUDA_HOST_DEVICE elwise_ck(intptr_t size, intptr_t dst_stride) : m_size(size), m_dst_stride(dst_stride) {}

      DYND_CUDA_HOST_DEVICE ~elwise_ck() { this->get_child()->destroy(); }

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();
        opchild(child, dst, m_dst_stride, src, NULL, m_size);
      }

      DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                                         const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        for (size_t i = 0; i < count; i += 1) {
          opchild(child, dst, m_dst_stride, NULL, NULL, m_size);
          dst += dst_stride;
        }
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }
        else if (child_tp->get_return_type().get_type_id() == typevar_constructed_type_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        self_type::make(ckb, kernreq, ckb_offset, size, dst_stride);
        kernreq = (kernreq & kernel_request_memory) | kernel_request_strided;

        bool finished = dst_ndim == 1;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<0>::instantiate(static_data, data, ckb, ckb_offset, child_dst_tp,
                                                                   child_dst_arrmeta, nsrc, NULL, NULL, kernreq, ectx,
                                                                   nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, NULL, NULL, kernreq, ectx, nkwd, kwds, tp_vars);
      }
    };

    /**
     * Generic expr kernel + destructor for a strided/var dimensions with
     * a fixed number of src operands, outputing to a strided dimension.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <int N>
    struct elwise_ck<fixed_dim_type_id, var_dim_type_id, N>
        : base_kernel<elwise_ck<fixed_dim_type_id, var_dim_type_id, N>, N> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N], m_src_offset[N];
      bool m_is_src_var[N];

      elwise_ck(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride, const intptr_t *src_offset,
                const bool *is_src_var)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      ~elwise_ck() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (int i = 0; i < N; ++i) {
          if (m_is_src_var[i]) {
            var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
            modified_src[i] = vddd->begin + m_src_offset[i];
            if (vddd->size == 1) {
              modified_src_stride[i] = 0;
            }
            else if (vddd->size == static_cast<size_t>(dim_size)) {
              modified_src_stride[i] = m_src_stride[i];
            }
            else {
              throw broadcast_error(dim_size, vddd->size, "strided", "var");
            }
          }
          else {
            // strided dimensions were fully broadcast in the kernel factory
            modified_src[i] = src[i];
            modified_src_stride[i] = m_src_stride[i];
          }
        }
        opchild(child, dst, m_dst_stride, modified_src, modified_src_stride, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
          single(dst, src_loop);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        intptr_t src_stride[N], src_offset[N];
        bool is_src_var[N];
        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
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
            const var_dim_type_arrmeta *src_md = reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        self_type::make(ckb, kernreq, ckb_offset, size, dst_stride, src_stride, src_offset, is_src_var);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<N>::instantiate(
              static_data, data, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp,
              child_src_arrmeta, kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                                        kernel_request_strided, ectx, nkwd, kwds, tp_vars);
      }
    };

    template <>
    struct elwise_ck<fixed_dim_type_id, var_dim_type_id, 0>
        : base_kernel<elwise_ck<fixed_dim_type_id, var_dim_type_id, 0>, 0> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      elwise_ck(intptr_t size, intptr_t dst_stride) : m_size(size), m_dst_stride(dst_stride) {}

      ~elwise_ck() { this->get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        opchild(child, dst, m_dst_stride, NULL, NULL, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i) {
          single(dst, NULL);
          dst += dst_stride;
        }
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        self_type::make(ckb, kernreq, ckb_offset, size, dst_stride);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<0>::instantiate(static_data, data, ckb, ckb_offset, child_dst_tp,
                                                                   child_dst_arrmeta, nsrc, NULL, NULL,
                                                                   kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, NULL, NULL, kernel_request_strided, ectx, nkwd, kwds,
                                        tp_vars);
      }
    };

    /**
     * Generic expr kernel + destructor for a strided/var dimensions with
     * a fixed number of src operands, outputing to a var dimension.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <int N>
    struct elwise_ck<var_dim_type_id, fixed_dim_type_id, N>
        : base_kernel<elwise_ck<var_dim_type_id, fixed_dim_type_id, N>, N> {
      typedef elwise_ck self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset, m_src_stride[N], m_src_offset[N], m_src_size[N];
      bool m_is_src_var[N];

      elwise_ck(memory_block_data *dst_memblock, size_t dst_target_alignment, intptr_t dst_stride, intptr_t dst_offset,
                const intptr_t *src_stride, const intptr_t *src_offset, const intptr_t *src_size,
                const bool *is_src_var)
          : m_dst_memblock(dst_memblock), m_dst_target_alignment(dst_target_alignment), m_dst_stride(dst_stride),
            m_dst_offset(dst_offset)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_src_size, src_size, sizeof(m_src_size));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      ~elwise_ck() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        var_dim_type_data *dst_vddd = reinterpret_cast<var_dim_type_data *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        if (dst_vddd->begin != NULL) {
          // If the destination already has allocated data, broadcast to that
          // data
          modified_dst = dst_vddd->begin + m_dst_offset;
          // Broadcast all the inputs to the existing destination dimension size
          dim_size = dst_vddd->size;
          for (int i = 0; i < N; ++i) {
            if (m_is_src_var[i]) {
              var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              }
              else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            }
            else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              }
              else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, m_src_size[i], "var", "strided");
              }
            }
          }
        }
        else {
          if (m_dst_offset != 0) {
            throw std::runtime_error("Cannot assign to an uninitialized dynd var_dim "
                                     "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          for (int i = 0; i < N; ++i) {
            if (m_is_src_var[i]) {
              var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              }
              else if (dim_size == 1) {
                dim_size = vddd->size;
                modified_src_stride[i] = m_src_stride[i];
              }
              else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            }
            else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              }
              else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              }
              else if (dim_size == 1) {
                dim_size = m_src_size[i];
                modified_src_stride[i] = m_src_stride[i];
              }
              else {
                throw broadcast_error(dim_size, m_src_size[i], "var", "strided");
              }
            }
          }
          // Allocate the output
          memory_block_data *memblock = m_dst_memblock;
          if (memblock->m_type == objectarray_memory_block_type) {
            memory_block_data::api *allocator = memblock->get_api();

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          }
          else {
            memory_block_data::api *allocator = memblock->get_api();

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          }
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        }
        else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(child, modified_dst, modified_dst_stride, modified_src, modified_src_stride, dim_size);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
          single(dst, src_loop);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();
        const var_dim_type_arrmeta *dst_md = reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
        child_dst_tp = dst_vdd->get_element_type();

        intptr_t src_stride[N], src_offset[N], src_size[N];
        bool is_src_var[N];

        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
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
            const var_dim_type_arrmeta *src_md = reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        self_type::make(ckb, kernreq, ckb_offset, dst_md->blockref.get(), dst_vdd->get_target_alignment(),
                        dst_md->stride, dst_md->offset, src_stride, src_offset, src_size, is_src_var);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<N>::instantiate(
              static_data, data, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp,
              child_src_arrmeta, kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                                        kernel_request_strided, ectx, nkwd, kwds, tp_vars);
      }
    };

    template <>
    struct elwise_ck<var_dim_type_id, fixed_dim_type_id, 0>
        : base_kernel<elwise_ck<var_dim_type_id, fixed_dim_type_id, 0>, 0> {
      typedef elwise_ck self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset;

      elwise_ck(memory_block_data *dst_memblock, size_t dst_target_alignment, intptr_t dst_stride, intptr_t dst_offset)
          : m_dst_memblock(dst_memblock), m_dst_target_alignment(dst_target_alignment), m_dst_stride(dst_stride),
            m_dst_offset(dst_offset)
      {
      }

      ~elwise_ck() { this->get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        ckernel_prefix *child = this->get_child();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        var_dim_type_data *dst_vddd = reinterpret_cast<var_dim_type_data *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        if (dst_vddd->begin != NULL) {
          // If the destination already has allocated data, broadcast to that
          // data
          modified_dst = dst_vddd->begin + m_dst_offset;
          // Broadcast all the inputs to the existing destination dimension size
          dim_size = dst_vddd->size;
        }
        else {
          if (m_dst_offset != 0) {
            throw std::runtime_error("Cannot assign to an uninitialized dynd var_dim "
                                     "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          // Allocate the output
          memory_block_data *memblock = m_dst_memblock;
          if (memblock->m_type == objectarray_memory_block_type) {
            memory_block_data::api *allocator = memblock->get_api();

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          }
          else {
            memory_block_data::api *allocator = memblock->get_api();

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          }
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        }
        else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(child, modified_dst, modified_dst_stride, NULL, NULL, dim_size);
      }

      static size_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
      {
        callable &child = *reinterpret_cast<callable *>(static_data);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();
        const var_dim_type_arrmeta *dst_md = reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
        child_dst_tp = dst_vdd->get_element_type();

        bool finished = dst_ndim == 1;

        self_type::make(ckb, kernreq, ckb_offset, dst_md->blockref.get(), dst_vdd->get_target_alignment(),
                        dst_md->stride, dst_md->offset);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck<0>::instantiate(static_data, data, ckb, ckb_offset, child_dst_tp,
                                                                   child_dst_arrmeta, nsrc, NULL, NULL,
                                                                   kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child.get()->instantiate(child.get()->static_data(), NULL, ckb, ckb_offset, child_dst_tp,
                                        child_dst_arrmeta, nsrc, NULL, NULL, kernel_request_strided, ectx, nkwd, kwds,
                                        tp_vars);
      }
    };

    template <int N>
    struct elwise_ck<var_dim_type_id, var_dim_type_id, N> : elwise_ck<var_dim_type_id, fixed_dim_type_id, N> {
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
