//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/cuda_kernels.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

namespace dynd {
namespace decl {
  namespace nd {
    class elwise : public lift<elwise> {
    public:
      static int resolve_dst_type(const arrfunc_type_data *child,
                                  const arrfunc_type *child_tp, intptr_t nsrc,
                                  const ndt::type *src_tp, int throw_on_error,
                                  ndt::type &dst_tp,
                                  const dynd::nd::array &kwds);

      /**
       * Lifts the provided ckernel, broadcasting it as necessary to execute
       * across the additional dimensions in the ``lifted_types`` array.
       *
       * This version is for 'expr' ckernels.
       *
       * \param child  The arrfunc being lifted
       * \param ckb  The ckernel_builder into which to place the ckernel.
       * \param ckb_offset  Where within the ckernel_builder to place the
       *ckernel.
       * \param dst_tp  The destination type to lift to.
       * \param dst_arrmeta  The destination arrmeta to lift to.
       * \param src_tp  The source types to lift to.
       * \param src_arrmeta  The source arrmetas to lift to.
       * \param kernreq  Either kernel_request_single or kernel_request_strided,
       *                 as required by the caller.
       * \param ectx  The evaluation context.
       */
      static intptr_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta,
                  dynd::kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds);

      template <type_id_t dst_dim_id, type_id_t src_dim_id>
      static intptr_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds);

      /** Prepends "Dims..." to all the types in the proto */
      static dynd::ndt::type make_lifted_type(const arrfunc_type *child_tp);
    };
  } // namespace nd
} // namespace decl

namespace nd {
  extern decl::nd::elwise elwise;
} // namespace nd

namespace kernels {
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
      : expr_ck<elwise_ck<fixed_dim_type_id, fixed_dim_type_id, N>,
                kernel_request_cuda_host_device, N> {
    typedef elwise_ck self_type;

    intptr_t size;
    intptr_t dst_stride, src_stride[N];

    DYND_CUDA_HOST_DEVICE elwise_ck(intptr_t size, intptr_t dst_stride,
                                    const intptr_t *src_stride)
        : size(size), dst_stride(dst_stride)
    {
      memcpy(this->src_stride, src_stride, sizeof(this->src_stride));
    }

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();
      opchild(dst, this->dst_stride, src, this->src_stride, this->size, child);
    }

    DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride,
                                       char *const *src,
                                       const intptr_t *src_stride, size_t count)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();
      intptr_t inner_size = this->size, inner_dst_stride = this->dst_stride;
      const intptr_t *inner_src_stride = this->src_stride;
      char *src_loop[N];
      memcpy(src_loop, src, sizeof(src_loop));
      for (size_t i = 0; i != count; ++i) {
        opchild(dst, inner_dst_stride, src_loop, inner_src_stride, inner_size,
                child);
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
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds)
    {
      intptr_t dst_ndim = dst_tp.get_ndim();
      if (!child_tp->get_return_type().is_symbolic()) {
        dst_ndim -= child_tp->get_return_type().get_ndim();
      }

      const char *child_dst_arrmeta;
      const char *child_src_arrmeta[N];
      ndt::type child_dst_tp;
      ndt::type child_src_tp[N];

      intptr_t size, dst_stride, src_stride[N];
      if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp,
                                 &child_dst_arrmeta)) {
        stringstream ss;
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
          stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                "or fixed dim, got " << src_tp[i];
          throw runtime_error(ss.str());
        }
      }

      self_type::create(ckb, kernreq, ckb_offset, size, dst_stride,
                        kernels::array_wrapper<intptr_t, N>(src_stride));
      kernreq = (kernreq & kernel_request_memory) | kernel_request_strided;

      // If there are still dimensions to broadcast, recursively lift more
      if (!finished) {
        return nd::elwise.instantiate(
            child, child_tp, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta,
            child_src_tp, child_src_arrmeta, kernreq, ectx, kwds);
      }

      // Instantiate the elementwise handler
      return child->instantiate(child, child_tp, ckb, ckb_offset, child_dst_tp,
                                child_dst_arrmeta, child_src_tp,
                                child_src_arrmeta, kernreq, ectx, kwds);
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
      : expr_ck<elwise_ck<fixed_dim_type_id, var_dim_type_id, N>,
                kernel_request_host, N> {
    typedef elwise_ck self_type;

    intptr_t size;
    intptr_t dst_stride, src_stride[N], src_offset[N];
    bool is_src_var[N];

    elwise_ck(intptr_t size, intptr_t dst_stride, const intptr_t *src_stride,
              const intptr_t *src_offset, const bool *is_src_var)
        : size(size), dst_stride(dst_stride)
    {
      memcpy(this->src_stride, src_stride, sizeof(this->src_stride));
      memcpy(this->src_offset, src_offset, sizeof(this->src_offset));
      memcpy(this->is_src_var, is_src_var, sizeof(this->is_src_var));
    }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();

      // Broadcast all the src 'var' dimensions to dst
      intptr_t dim_size = this->size;
      char *modified_src[N];
      intptr_t modified_src_stride[N];
      for (int i = 0; i < N; ++i) {
        if (this->is_src_var[i]) {
          var_dim_type_data *vddd =
              reinterpret_cast<var_dim_type_data *>(src[i]);
          modified_src[i] = vddd->begin + this->src_offset[i];
          if (vddd->size == 1) {
            modified_src_stride[i] = 0;
          } else if (vddd->size == static_cast<size_t>(dim_size)) {
            modified_src_stride[i] = this->src_stride[i];
          } else {
            throw broadcast_error(dim_size, vddd->size, "strided", "var");
          }
        } else {
          // strided dimensions were fully broadcast in the kernel factory
          modified_src[i] = src[i];
          modified_src_stride[i] = this->src_stride[i];
        }
      }
      opchild(dst, this->dst_stride, modified_src, modified_src_stride,
              dim_size, child);
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
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds)
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
      if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp,
                                 &child_dst_arrmeta)) {
        stringstream ss;
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
          child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
          child_src_tp[i] = vdd->get_element_type();
          finished &= src_ndim == 1;
        }
      }

      self_type::create(ckb, kernreq, ckb_offset, size, dst_stride, src_stride,
                        src_offset, is_src_var);

      // If there are still dimensions to broadcast, recursively lift more
      if (!finished) {
        return nd::elwise.instantiate(child, child_tp, ckb, ckb_offset,
                                      child_dst_tp, child_dst_arrmeta,
                                      child_src_tp, child_src_arrmeta,
                                      kernel_request_strided, ectx, kwds);
      }
      // Instantiate the elementwise handler
      return child->instantiate(
          child, child_tp, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta,
          child_src_tp, child_src_arrmeta, kernel_request_strided, ectx, kwds);
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
      : expr_ck<elwise_ck<var_dim_type_id, fixed_dim_type_id, N>,
                kernel_request_host, N> {
    typedef elwise_ck self_type;

    memory_block_data *dst_memblock;
    size_t dst_target_alignment;
    intptr_t dst_stride, dst_offset, src_stride[N], src_offset[N], src_size[N];
    bool is_src_var[N];

    elwise_ck(memory_block_data *dst_memblock, size_t dst_target_alignment,
              intptr_t dst_stride, intptr_t dst_offset,
              const intptr_t *src_stride, const intptr_t *src_offset,
              const intptr_t *src_size, const bool *is_src_var)
        : dst_memblock(dst_memblock),
          dst_target_alignment(dst_target_alignment), dst_stride(dst_stride),
          dst_offset(dst_offset)
    {
      memcpy(this->src_stride, src_stride, sizeof(this->src_stride));
      memcpy(this->src_offset, src_offset, sizeof(this->src_offset));
      memcpy(this->src_size, src_size, sizeof(this->src_size));
      memcpy(this->is_src_var, is_src_var, sizeof(this->is_src_var));
    }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();

      var_dim_type_data *dst_vddd = reinterpret_cast<var_dim_type_data *>(dst);
      char *modified_dst;
      intptr_t modified_dst_stride = 0;
      intptr_t dim_size;
      char *modified_src[N];
      intptr_t modified_src_stride[N];
      if (dst_vddd->begin != NULL) {
        // If the destination already has allocated data, broadcast to that data
        modified_dst = dst_vddd->begin + this->dst_offset;
        // Broadcast all the inputs to the existing destination dimension size
        dim_size = dst_vddd->size;
        for (int i = 0; i < N; ++i) {
          if (this->is_src_var[i]) {
            var_dim_type_data *vddd =
                reinterpret_cast<var_dim_type_data *>(src[i]);
            modified_src[i] = vddd->begin + this->src_offset[i];
            if (vddd->size == 1) {
              modified_src_stride[i] = 0;
            } else if (vddd->size == static_cast<size_t>(dim_size)) {
              modified_src_stride[i] = this->src_stride[i];
            } else {
              throw broadcast_error(dim_size, vddd->size, "var", "var");
            }
          } else {
            modified_src[i] = src[i];
            if (this->src_size[i] == 1) {
              modified_src_stride[i] = 0;
            } else if (this->src_size[i] == dim_size) {
              modified_src_stride[i] = this->src_stride[i];
            } else {
              throw broadcast_error(dim_size, this->src_size[i], "var",
                                    "strided");
            }
          }
        }
      } else {
        if (this->dst_offset != 0) {
          throw runtime_error("Cannot assign to an uninitialized dynd var_dim "
                              "which has a non-zero offset");
        }
        // Broadcast all the inputs together to get the destination size
        dim_size = 1;
        for (int i = 0; i < N; ++i) {
          if (this->is_src_var[i]) {
            var_dim_type_data *vddd =
                reinterpret_cast<var_dim_type_data *>(src[i]);
            modified_src[i] = vddd->begin + this->src_offset[i];
            if (vddd->size == 1) {
              modified_src_stride[i] = 0;
            } else if (dim_size == 1) {
              dim_size = vddd->size;
              modified_src_stride[i] = this->src_stride[i];
            } else if (vddd->size == static_cast<size_t>(dim_size)) {
              modified_src_stride[i] = this->src_stride[i];
            } else {
              throw broadcast_error(dim_size, vddd->size, "var", "var");
            }
          } else {
            modified_src[i] = src[i];
            if (this->src_size[i] == 1) {
              modified_src_stride[i] = 0;
            } else if (this->src_size[i] == dim_size) {
              modified_src_stride[i] = this->src_stride[i];
            } else if (dim_size == 1) {
              dim_size = this->src_size[i];
              modified_src_stride[i] = this->src_stride[i];
            } else {
              throw broadcast_error(dim_size, this->src_size[i], "var",
                                    "strided");
            }
          }
        }
        // Allocate the output
        memory_block_data *memblock = this->dst_memblock;
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
          allocator->allocate(memblock, dim_size * this->dst_stride,
                              this->dst_target_alignment, &dst_vddd->begin,
                              &dst_end);
        }
        modified_dst = dst_vddd->begin;
        dst_vddd->size = dim_size;
      }
      if (dim_size <= 1) {
        modified_dst_stride = 0;
      } else {
        modified_dst_stride = this->dst_stride;
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
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds)
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
          child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
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
        return nd::elwise.instantiate(child, child_tp, ckb, ckb_offset,
                                      child_dst_tp, child_dst_arrmeta,
                                      child_src_tp, child_src_arrmeta,
                                      kernel_request_strided, ectx, kwds);
      }
      // All the types matched, so instantiate the elementwise handler
      return child->instantiate(
          child, child_tp, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta,
          child_src_tp, child_src_arrmeta, kernel_request_strided, ectx, kwds);
    }
  };

  template <int N>
  struct elwise_ck<var_dim_type_id, var_dim_type_id, N>
      : elwise_ck<var_dim_type_id, fixed_dim_type_id, N> {
  };
} // namespace kernels
} // namespace dynd
