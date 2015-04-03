//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/virtual.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    // elwise_virtual_ck does multidispatch to the templated elwise_ck with the
    // right number of args

    struct elwise_virtual_ck : virtual_ck<elwise_virtual_ck> {
      static void resolve_option_values(
          const arrfunc_type_data *self, const arrfunc_type *self_tp,
          char *data, intptr_t nsrc, const ndt::type *src_tp, nd::array &kwds,
          const std::map<nd::string, ndt::type> &tp_vars);

      static int
      resolve_dst_type(const arrfunc_type_data *self,
                       const arrfunc_type *self_tp, char *data, intptr_t nsrc,
                       const ndt::type *src_tp, int throw_on_error,
                       ndt::type &dst_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars);

      static int resolve_dst_type_with_child(
          const arrfunc_type_data *child, const arrfunc_type *child_tp,
          char *data, intptr_t nsrc, const ndt::type *src_tp,
          int throw_on_error, ndt::type &dst_tp, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars);

      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const arrfunc_type *DYND_UNUSED(self_tp), char *data,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  dynd::kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars);

      static intptr_t instantiate_with_child(
          const arrfunc_type_data *child, const arrfunc_type *child_tp,
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
          const eval::eval_context *ectx, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars);

      template <type_id_t dst_dim_id, type_id_t src_dim_id>
      static intptr_t instantiate_with_child(
          const arrfunc_type_data *child, const arrfunc_type *child_tp,
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, kernel_request_t kernreq,
          const eval::eval_context *ectx, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars);
    };

    /**
     * Generic expr kernel + destructor for a strided dimension with
     * a fixed number of src operands.
     * This requires that the child kernel be created with the
     * kernel_request_strided type of kernel.
     */
    template <type_id_t dst_dim_type_id, type_id_t src_dim_type_id, int N>
    struct elwise_ck;

    template <int N>
    struct elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>
        : nd::expr_ck<elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>,
                      kernel_request_cuda_host_device, N> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N];

      DYND_CUDA_HOST_DEVICE elwise_ck(intptr_t size, intptr_t dst_stride,
                                      const intptr_t *src_stride)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
      }

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        opchild(dst, m_dst_stride, src, m_src_stride, m_size, child);
      }

      DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride,
                                         char *const *src,
                                         const intptr_t *src_stride,
                                         size_t count)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        char *src_loop[N];
        for (int j = 0; j != N; ++j) {
          src_loop[j] = src[j];
        }

        for (size_t i = 0; i < count; i += 1) {
          opchild(dst, m_dst_stride, src_loop, m_src_stride, m_size, child);
          dst += dst_stride;
          for (int j = 0; j != N; ++j) {
            src_loop[j] += src_stride[j];
          }
        }
      }

      DYND_CUDA_HOST_DEVICE static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_type_id() ==
                typevar_constructed_type_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        intptr_t size, dst_stride, src_stride[N];
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride,
                                   &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type " << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
          intptr_t src_ndim =
              src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size;
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                              &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i],
                                    src_arrmeta[i]);
            }
            finished &= src_ndim == 1;
          } else {
            std::stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                  "or fixed dim, got " << src_tp[i];
            throw std::runtime_error(ss.str());
          }
        }

        self_type::create(ckb, kernreq, ckb_offset, size, dst_stride,
                          dynd::detail::make_array_wrapper<N>(src_stride));
        kernreq = (kernreq & kernel_request_memory) | kernel_request_strided;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta, kernreq,
                                  ectx, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta, kernreq,
                                  ectx, kwds, tp_vars);
      }
    };

    template <>
    struct elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 0>
        : nd::expr_ck<elwise_ck<fixed_dim_type_id, fixed_dim_type_id, 0>,
                      kernel_request_cuda_host_device, 0> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      DYND_CUDA_HOST_DEVICE elwise_ck(intptr_t size, intptr_t dst_stride)
          : m_size(size), m_dst_stride(dst_stride)
      {
      }

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();
        opchild(dst, m_dst_stride, src, NULL, m_size, child);
      }

      DYND_CUDA_HOST_DEVICE void
      strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
              const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        for (size_t i = 0; i < count; i += 1) {
          opchild(dst, m_dst_stride, NULL, NULL, m_size, child);
          dst += dst_stride;
        }
      }

      DYND_CUDA_HOST_DEVICE static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                  const char *const *DYND_UNUSED(src_arrmeta),
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        } else if (child_tp->get_return_type().get_type_id() ==
                   typevar_constructed_type_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride,
                                   &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type " << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        self_type::create(ckb, kernreq, ckb_offset, size, dst_stride);
        kernreq = (kernreq & kernel_request_memory) | kernel_request_strided;

        bool finished = dst_ndim == 1;

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernreq, ectx, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernreq, ectx, kwds, tp_vars);
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
        : nd::expr_ck<elwise_ck<fixed_dim_type_id, var_dim_type_id, N>,
                      kernel_request_host, N> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride, m_src_stride[N], m_src_offset[N];
      bool m_is_src_var[N];

      elwise_ck(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride,
                const intptr_t *src_offset, const bool *is_src_var)
          : m_size(size), m_dst_stride(dst_stride)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (int i = 0; i < N; ++i) {
          if (m_is_src_var[i]) {
            var_dim_type_data *vddd =
                reinterpret_cast<var_dim_type_data *>(src[i]);
            modified_src[i] = vddd->begin + m_src_offset[i];
            if (vddd->size == 1) {
              modified_src_stride[i] = 0;
            } else if (vddd->size == static_cast<size_t>(dim_size)) {
              modified_src_stride[i] = m_src_stride[i];
            } else {
              throw broadcast_error(dim_size, vddd->size, "strided", "var");
            }
          } else {
            // strided dimensions were fully broadcast in the kernel factory
            modified_src[i] = src[i];
            modified_src_stride[i] = m_src_stride[i];
          }
        }
        opchild(dst, m_dst_stride, modified_src, modified_src_stride, dim_size,
                child);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
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

      static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride,
                                   &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type " << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        intptr_t src_stride[N], src_offset[N];
        bool is_src_var[N];
        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
          intptr_t src_size;
          intptr_t src_ndim =
              src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          // The src[i] strided parameters
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                              &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i],
                                    src_arrmeta[i]);
            }
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const var_dim_type *vdd =
                static_cast<const var_dim_type *>(src_tp[i].extended());
            const var_dim_type_arrmeta *src_md =
                reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] =
                src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        self_type::create(ckb, kernreq, ckb_offset, size, dst_stride,
                          src_stride, src_offset, is_src_var);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta,
                                  kernel_request_strided, ectx, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta,
                                  kernel_request_strided, ectx, kwds, tp_vars);
      }
    };

    template <>
    struct elwise_ck<fixed_dim_type_id, var_dim_type_id, 0>
        : nd::expr_ck<elwise_ck<fixed_dim_type_id, var_dim_type_id, 0>,
                      kernel_request_host, 0> {
      typedef elwise_ck self_type;

      intptr_t m_size;
      intptr_t m_dst_stride;

      elwise_ck(intptr_t size, intptr_t dst_stride)
          : m_size(size), m_dst_stride(dst_stride)
      {
      }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = m_size;
        opchild(dst, m_dst_stride, NULL, NULL, dim_size, child);
      }

      void strided(char *dst, intptr_t dst_stride,
                   char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i) {
          single(dst, NULL);
          dst += dst_stride;
        }
      }

      static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                  const char *const *DYND_UNUSED(src_arrmeta),
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride,
                                   &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type " << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        self_type::create(ckb, kernreq, ckb_offset, size, dst_stride);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernel_request_strided, ectx, kwds,
                                  tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernel_request_strided, ectx, kwds,
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
        : nd::expr_ck<elwise_ck<var_dim_type_id, fixed_dim_type_id, N>,
                      kernel_request_host, N> {
      typedef elwise_ck self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset, m_src_stride[N], m_src_offset[N],
          m_src_size[N];
      bool m_is_src_var[N];

      elwise_ck(memory_block_data *dst_memblock, size_t dst_target_alignment,
                intptr_t dst_stride, intptr_t dst_offset,
                const intptr_t *src_stride, const intptr_t *src_offset,
                const intptr_t *src_size, const bool *is_src_var)
          : m_dst_memblock(dst_memblock),
            m_dst_target_alignment(dst_target_alignment),
            m_dst_stride(dst_stride), m_dst_offset(dst_offset)
      {
        memcpy(m_src_stride, src_stride, sizeof(m_src_stride));
        memcpy(m_src_offset, src_offset, sizeof(m_src_offset));
        memcpy(m_src_size, src_size, sizeof(m_src_size));
        memcpy(m_is_src_var, is_src_var, sizeof(m_is_src_var));
      }

      void single(char *dst, char *const *src)
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        var_dim_type_data *dst_vddd =
            reinterpret_cast<var_dim_type_data *>(dst);
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
              var_dim_type_data *vddd =
                  reinterpret_cast<var_dim_type_data *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              } else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              } else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            } else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              } else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              } else {
                throw broadcast_error(dim_size, m_src_size[i], "var",
                                      "strided");
              }
            }
          }
        } else {
          if (m_dst_offset != 0) {
            throw std::runtime_error(
                "Cannot assign to an uninitialized dynd var_dim "
                "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          for (int i = 0; i < N; ++i) {
            if (m_is_src_var[i]) {
              var_dim_type_data *vddd =
                  reinterpret_cast<var_dim_type_data *>(src[i]);
              modified_src[i] = vddd->begin + m_src_offset[i];
              if (vddd->size == 1) {
                modified_src_stride[i] = 0;
              } else if (dim_size == 1) {
                dim_size = vddd->size;
                modified_src_stride[i] = m_src_stride[i];
              } else if (vddd->size == static_cast<size_t>(dim_size)) {
                modified_src_stride[i] = m_src_stride[i];
              } else {
                throw broadcast_error(dim_size, vddd->size, "var", "var");
              }
            } else {
              modified_src[i] = src[i];
              if (m_src_size[i] == 1) {
                modified_src_stride[i] = 0;
              } else if (m_src_size[i] == dim_size) {
                modified_src_stride[i] = m_src_stride[i];
              } else if (dim_size == 1) {
                dim_size = m_src_size[i];
                modified_src_stride[i] = m_src_stride[i];
              } else {
                throw broadcast_error(dim_size, m_src_size[i], "var",
                                      "strided");
              }
            }
          }
          // Allocate the output
          memory_block_data *memblock = m_dst_memblock;
          if (memblock->m_type == objectarray_memory_block_type) {
            memory_block_objectarray_allocator_api *allocator =
                get_memory_block_objectarray_allocator_api(memblock);

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          } else {
            memory_block_pod_allocator_api *allocator =
                get_memory_block_pod_allocator_api(memblock);

            // Allocate the output array data
            char *dst_end = NULL;
            allocator->allocate(memblock, dim_size * m_dst_stride,
                                m_dst_target_alignment, &dst_vddd->begin,
                                &dst_end);
          }
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        } else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(modified_dst, modified_dst_stride, modified_src,
                modified_src_stride, dim_size, child);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
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

      static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        const char *child_src_arrmeta[N];
        ndt::type child_dst_tp;
        ndt::type child_src_tp[N];

        // The dst var parameters
        const var_dim_type *dst_vdd = dst_tp.extended<var_dim_type>();
        const var_dim_type_arrmeta *dst_md =
            reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
        child_dst_tp = dst_vdd->get_element_type();

        intptr_t src_stride[N], src_offset[N], src_size[N];
        bool is_src_var[N];

        bool finished = dst_ndim == 1;
        for (int i = 0; i < N; ++i) {
          // The src[i] strided parameters
          intptr_t src_ndim =
              src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i],
                                              &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const var_dim_type *vdd =
                static_cast<const var_dim_type *>(src_tp[i].extended());
            const var_dim_type_arrmeta *src_md =
                reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] =
                src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        self_type::create(ckb, kernreq, ckb_offset, dst_md->blockref,
                          dst_vdd->get_target_alignment(), dst_md->stride,
                          dst_md->offset, src_stride, src_offset, src_size,
                          is_src_var);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta,
                                  kernel_request_strided, ectx, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc,
                                  child_src_tp, child_src_arrmeta,
                                  kernel_request_strided, ectx, kwds, tp_vars);
      }
    };

    template <>
    struct elwise_ck<var_dim_type_id, fixed_dim_type_id, 0>
        : nd::expr_ck<elwise_ck<var_dim_type_id, fixed_dim_type_id, 0>,
                      kernel_request_host, 0> {
      typedef elwise_ck self_type;

      memory_block_data *m_dst_memblock;
      size_t m_dst_target_alignment;
      intptr_t m_dst_stride, m_dst_offset;

      elwise_ck(memory_block_data *dst_memblock, size_t dst_target_alignment,
                intptr_t dst_stride, intptr_t dst_offset)
          : m_dst_memblock(dst_memblock),
            m_dst_target_alignment(dst_target_alignment),
            m_dst_stride(dst_stride), m_dst_offset(dst_offset)
      {
      }

      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        ckernel_prefix *child = this->get_child_ckernel();
        expr_strided_t opchild = child->get_function<expr_strided_t>();

        var_dim_type_data *dst_vddd =
            reinterpret_cast<var_dim_type_data *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        if (dst_vddd->begin != NULL) {
          // If the destination already has allocated data, broadcast to that
          // data
          modified_dst = dst_vddd->begin + m_dst_offset;
          // Broadcast all the inputs to the existing destination dimension size
          dim_size = dst_vddd->size;
        } else {
          if (m_dst_offset != 0) {
            throw std::runtime_error(
                "Cannot assign to an uninitialized dynd var_dim "
                "which has a non-zero offset");
          }
          // Broadcast all the inputs together to get the destination size
          dim_size = 1;
          // Allocate the output
          memory_block_data *memblock = m_dst_memblock;
          if (memblock->m_type == objectarray_memory_block_type) {
            memory_block_objectarray_allocator_api *allocator =
                get_memory_block_objectarray_allocator_api(memblock);

            // Allocate the output array data
            dst_vddd->begin = allocator->allocate(memblock, dim_size);
          } else {
            memory_block_pod_allocator_api *allocator =
                get_memory_block_pod_allocator_api(memblock);

            // Allocate the output array data
            char *dst_end = NULL;
            allocator->allocate(memblock, dim_size * m_dst_stride,
                                m_dst_target_alignment, &dst_vddd->begin,
                                &dst_end);
          }
          modified_dst = dst_vddd->begin;
          dst_vddd->size = dim_size;
        }
        if (dim_size <= 1) {
          modified_dst_stride = 0;
        } else {
          modified_dst_stride = m_dst_stride;
        }
        opchild(modified_dst, modified_dst_stride, NULL, NULL, dim_size, child);
      }

      static void destruct(ckernel_prefix *self)
      {
        self->destroy_child_ckernel(sizeof(self_type));
      }

      static size_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                  const char *const *DYND_UNUSED(src_arrmeta),
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        ndt::type child_dst_tp;

        // The dst var parameters
        const var_dim_type *dst_vdd = dst_tp.extended<var_dim_type>();
        const var_dim_type_arrmeta *dst_md =
            reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
        child_dst_tp = dst_vdd->get_element_type();

        bool finished = dst_ndim == 1;

        self_type::create(ckb, kernreq, ckb_offset, dst_md->blockref,
                          dst_vdd->get_target_alignment(), dst_md->stride,
                          dst_md->offset);

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return nd::functional::elwise_virtual_ck::instantiate_with_child(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernel_request_strided, ectx, kwds,
                                  tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(child, child_tp, NULL, ckb, ckb_offset,
                                  child_dst_tp, child_dst_arrmeta, nsrc, NULL,
                                  NULL, kernel_request_strided, ectx, kwds,
                                  tp_vars);
      }
    };

    template <int N>
    struct elwise_ck<var_dim_type_id, var_dim_type_id, N>
        : elwise_ck<var_dim_type_id, fixed_dim_type_id, N> {
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd