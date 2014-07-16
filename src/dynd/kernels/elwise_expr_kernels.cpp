//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// DEPRECATED in favor of make_lifted_ckernel

#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

////////////////////////////////////////////////////////////////////
// make_elwise_strided_dimension_expr_kernel

/**
 * Generic expr kernel + destructor for a strided dimension with
 * a fixed number of src operands.
 * This requires that the child kernel be created with the
 * kernel_request_strided type of kernel.
 */
template<int N>
struct strided_expr_kernel_extra {
    typedef strided_expr_kernel_extra extra_type;

    ckernel_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride[N];

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        intptr_t inner_size = e->size, inner_dst_stride = e->dst_stride;
        const intptr_t *inner_src_stride = e->src_stride;
        const char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
            opchild(dst, inner_dst_stride, src_loop, inner_src_stride, inner_size, echild);
            dst += dst_stride;
            for (int j = 0; j != N; ++j) {
                src_loop[j] += src_stride[j];
            }
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        self->destroy_child_ckernel(sizeof(extra_type));
    }
};

template <int N>
static size_t make_elwise_strided_dimension_expr_kernel_for_N(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, size_t DYND_UNUSED(src_count),
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const expr_kernel_generator *elwise_handler)
{
  intptr_t undim = dst_tp.get_ndim();
  const char *dst_child_arrmeta;
  const char *src_child_arrmeta[N];
  ndt::type dst_child_dt;
  ndt::type src_child_dt[N];

  strided_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_expr_kernel_extra<N> >(ckb_offset);
  e->base.template set_expr_function<strided_expr_kernel_extra<N> >(kernreq);
  e->base.destructor = strided_expr_kernel_extra<N>::destruct;
  // The dst strided parameters
  if (!dst_tp.get_as_strided(dst_arrmeta, &e->size, &e->dst_stride,
                                 &dst_child_dt, &dst_child_arrmeta)) {
    throw type_error("make_elwise_strided_dimension_expr_kernel: dst was not "
                     "strided as expected");
  }
  for (int i = 0; i < N; ++i) {
    intptr_t src_size;
    // The src[i] strided parameters
    if (src_tp[i].get_ndim() < undim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      src_child_arrmeta[i] = src_arrmeta[i];
      src_child_dt[i] = src_tp[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                        &e->src_stride[i], &src_child_dt[i],
                                        &src_child_arrmeta[i])) {
      // Check for a broadcasting error
      if (src_size != 1 && e->size != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
      }
    } else {
      throw type_error("make_elwise_strided_dimension_expr_kernel: src was "
                       "not strided as expected");
    }
  }
  return elwise_handler->make_expr_kernel(
      ckb, ckb_offset, dst_child_dt, dst_child_arrmeta, N, src_child_dt,
      src_child_arrmeta, kernel_request_strided, ectx);
}

inline static size_t make_elwise_strided_dimension_expr_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t src_count, const ndt::type *src_tp, const char *const*src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_dimension_expr_kernel_for_N<1>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 2:
            return make_elwise_strided_dimension_expr_kernel_for_N<2>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 3:
            return make_elwise_strided_dimension_expr_kernel_for_N<3>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 4:
            return make_elwise_strided_dimension_expr_kernel_for_N<4>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 5:
            return make_elwise_strided_dimension_expr_kernel_for_N<5>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 6:
            return make_elwise_strided_dimension_expr_kernel_for_N<6>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_dimension_expr_kernel with src_count > 6 not implemented yet");
    }
}

////////////////////////////////////////////////////////////////////
// make_elwise_strided_or_var_to_strided_dimension_expr_kernel

/**
 * Generic expr kernel + destructor for a strided/var dimensions with
 * a fixed number of src operands, outputing to a strided dimension.
 * This requires that the child kernel be created with the
 * kernel_request_strided type of kernel.
 */
template<int N>
struct strided_or_var_to_strided_expr_kernel_extra {
    typedef strided_or_var_to_strided_expr_kernel_extra extra_type;

    ckernel_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride[N], src_offset[N];
    bool is_src_var[N];

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = e->size;
        const char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (int i = 0; i < N; ++i) {
            if (e->is_src_var[i]) {
                const var_dim_type_data *vddd = reinterpret_cast<const var_dim_type_data *>(src[i]);
                modified_src[i] = vddd->begin + e->src_offset[i];
                if (vddd->size == 1) {
                    modified_src_stride[i] = 0;
                } else if (vddd->size == static_cast<size_t>(dim_size)) {
                    modified_src_stride[i] = e->src_stride[i];
                } else {
                    throw broadcast_error(dim_size, vddd->size, "strided dim", "var dim");
                }
            } else {
                // strided dimensions were fully broadcast in the kernel factory
                modified_src[i] = src[i];
                modified_src_stride[i] = e->src_stride[i];
            }
        }
        opchild(dst, e->dst_stride, modified_src, modified_src_stride, dim_size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        const char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
            single(dst, src_loop, extra);
            dst += dst_stride;
            for (int j = 0; j != N; ++j) {
                src_loop[j] += src_stride[j];
            }
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        self->destroy_child_ckernel(sizeof(extra_type));
    }
};

template <int N>
static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, size_t DYND_UNUSED(src_count),
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const expr_kernel_generator *elwise_handler)
{
  intptr_t undim = dst_tp.get_ndim();
  const char *dst_child_arrmeta;
  const char *src_child_arrmeta[N];
  ndt::type dst_child_dt;
  ndt::type src_child_dt[N];

  strided_or_var_to_strided_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_or_var_to_strided_expr_kernel_extra<N> >(
          ckb_offset);
  e->base.template set_expr_function<
      strided_or_var_to_strided_expr_kernel_extra<N> >(kernreq);
  e->base.destructor =
      &strided_or_var_to_strided_expr_kernel_extra<N>::destruct;
  // The dst strided parameters
  if (!dst_tp.get_as_strided(dst_arrmeta, &e->size, &e->dst_stride,
                             &dst_child_dt, &dst_child_arrmeta)) {
    throw type_error("make_elwise_strided_dimension_expr_kernel: dst was not "
                     "strided as expected");
  }

  for (int i = 0; i < N; ++i) {
    intptr_t src_size;
    // The src[i] strided parameters
    if (src_tp[i].get_ndim() < undim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
      src_child_arrmeta[i] = src_arrmeta[i];
      src_child_dt[i] = src_tp[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                        &e->src_stride[i], &src_child_dt[i],
                                        &src_child_arrmeta[i])) {
      // Check for a broadcasting error
      if (src_size != 1 && e->size != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
      }
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
    } else {
      const var_dim_type *vdd =
          static_cast<const var_dim_type *>(src_tp[i].extended());
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
      e->src_stride[i] = src_md->stride;
      e->src_offset[i] = src_md->offset;
      e->is_src_var[i] = true;
      src_child_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
      src_child_dt[i] = vdd->get_element_type();
    }
  }
  return elwise_handler->make_expr_kernel(
      ckb, ckb_offset, dst_child_dt, dst_child_arrmeta, N, src_child_dt,
      src_child_arrmeta, kernel_request_strided, ectx);
}

static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t src_count, const ndt::type *src_tp, const char *const*src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<1>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 2:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<2>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 3:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<3>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 4:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<4>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 5:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<5>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        case 6:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<6>(
                            ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_count, src_tp, src_arrmeta,
                            kernreq, ectx,
                            elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_or_var_to_strided_dimension_expr_kernel with src_count > 6 not implemented yet");
    }
}

////////////////////////////////////////////////////////////////////
// make_elwise_strided_or_var_to_var_dimension_expr_kernel

/**
 * Generic expr kernel + destructor for a strided/var dimensions with
 * a fixed number of src operands, outputing to a var dimension.
 * This requires that the child kernel be created with the
 * kernel_request_strided type of kernel.
 */
template<int N>
struct strided_or_var_to_var_expr_kernel_extra {
    typedef strided_or_var_to_var_expr_kernel_extra extra_type;

    ckernel_prefix base;
    memory_block_data *dst_memblock;
    size_t dst_target_alignment;
    intptr_t dst_stride, dst_offset, src_stride[N], src_offset[N];
    bool is_src_var[N];

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        var_dim_type_data *dst_vddd = reinterpret_cast<var_dim_type_data *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        const char *modified_src[N];
        intptr_t modified_src_stride[N];
        if (dst_vddd->begin != NULL) {
            // If the destination already has allocated data, broadcast to that data
            modified_dst = dst_vddd->begin + e->dst_offset;
            // Broadcast all the inputs to the existing destination dimension size
            dim_size = dst_vddd->size;
            for (int i = 0; i < N; ++i) {
                if (e->is_src_var[i]) {
                    const var_dim_type_data *vddd = reinterpret_cast<const var_dim_type_data *>(src[i]);
                    modified_src[i] = vddd->begin + e->src_offset[i];
                    if (vddd->size == 1) {
                        modified_src_stride[i] = 0;
                    } else if (vddd->size == static_cast<size_t>(dim_size)) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, vddd->size, "var dim", "var dim");
                    }
                } else {
                    // strided dimensions are all size 1
                    modified_src[i] = src[i];
                    modified_src_stride[i] = e->src_stride[i];
                }
            }
        } else {
            if (e->dst_offset != 0) {
                throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
            }
            // Broadcast all the inputs together to get the destination size
            dim_size = 1;
            for (int i = 0; i < N; ++i) {
                if (e->is_src_var[i]) {
                    const var_dim_type_data *vddd = reinterpret_cast<const var_dim_type_data *>(src[i]);
                    modified_src[i] = vddd->begin + e->src_offset[i];
                    if (vddd->size == 1) {
                        modified_src_stride[i] = 0;
                    } else if (dim_size == 1) {
                        dim_size = vddd->size;
                        modified_src_stride[i] = e->src_stride[i];
                    } else if (vddd->size == static_cast<size_t>(dim_size)) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, vddd->size, "var dim", "var dim");
                    }
                } else {
                    // strided dimensions are all size 1
                    modified_src[i] = src[i];
                    modified_src_stride[i] = e->src_stride[i];
                }
            }
            // Allocate the output
            memory_block_data *memblock = e->dst_memblock;
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
                allocator->allocate(memblock, dim_size * e->dst_stride,
                            e->dst_target_alignment, &dst_vddd->begin, &dst_end);
            }
            modified_dst = dst_vddd->begin;
            dst_vddd->size = dim_size;
            if (dim_size <= 1) {
                modified_dst_stride = 0;
            } else {
                modified_dst_stride = e->dst_stride;
            }
        }
        opchild(modified_dst, modified_dst_stride, modified_src, modified_src_stride, dim_size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        const char *src_loop[N];
        memcpy(src_loop, src, sizeof(src_loop));
        for (size_t i = 0; i != count; ++i) {
            single(dst, src_loop, extra);
            dst += dst_stride;
            for (int j = 0; j != N; ++j) {
                src_loop[j] += src_stride[j];
            }
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        self->destroy_child_ckernel(sizeof(extra_type));
    }
};

template<int N>
static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t DYND_UNUSED(src_count), const ndt::type *src_tp, const char *const*src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
  intptr_t undim = dst_tp.get_ndim();
  const char *dst_child_arrmeta;
  const char *src_child_arrmeta[N];
  ndt::type dst_child_dt;
  ndt::type src_child_dt[N];

  strided_or_var_to_var_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_or_var_to_var_expr_kernel_extra<N> >(ckb_offset);
  e->base.template set_expr_function<strided_or_var_to_var_expr_kernel_extra<N> >(
      kernreq);
  e->base.destructor = &strided_or_var_to_var_expr_kernel_extra<N>::destruct;
  // The dst var parameters
  const var_dim_type *dst_vdd = dst_tp.tcast<var_dim_type>();
  const var_dim_type_arrmeta *dst_md =
      reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);
  e->dst_memblock = dst_md->blockref;
  e->dst_stride = dst_md->stride;
  e->dst_offset = dst_md->offset;
  e->dst_target_alignment = dst_vdd->get_target_alignment();
  dst_child_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
  dst_child_dt = dst_vdd->get_element_type();

  for (int i = 0; i < N; ++i) {
    intptr_t src_size;
    // The src[i] strided parameters
    if (src_tp[i].get_ndim() < undim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
      src_child_arrmeta[i] = src_arrmeta[i];
      src_child_dt[i] = src_tp[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                        &e->src_stride[i], &src_child_dt[i],
                                        &src_child_arrmeta[i])) {
      // Check for a broadcasting error (the strided dimension size must be 1,
      // otherwise the destination should be strided, not var)
      if (src_size != 1) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
      }
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
    } else {
      const var_dim_type *vdd =
          static_cast<const var_dim_type *>(src_tp[i].extended());
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
      e->src_stride[i] = src_md->stride;
      e->src_offset[i] = src_md->offset;
      e->is_src_var[i] = true;
      src_child_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
      src_child_dt[i] = vdd->get_element_type();
    }
  }
  return elwise_handler->make_expr_kernel(
      ckb, ckb_offset, dst_child_dt, dst_child_arrmeta, N, src_child_dt,
      src_child_arrmeta, kernel_request_strided, ectx);
}

static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, size_t src_count, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const expr_kernel_generator *elwise_handler)
{
  switch (src_count) {
  case 1:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<1>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  case 2:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<2>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  case 3:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<3>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  case 4:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<4>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  case 5:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<5>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  case 6:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<6>(
        ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
        kernreq, ectx, elwise_handler);
  default:
    throw runtime_error("make_elwise_strided_or_var_to_var_dimension_expr_"
                        "kernel with src_count > 6 not implemented yet");
  }
}

size_t dynd::make_elwise_dimension_expr_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, size_t src_count, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const expr_kernel_generator *elwise_handler)
{
  // Do a pass through the src types to classify them
  bool src_all_strided = true, src_all_strided_or_var = true;
  for (size_t i = 0; i != src_count; ++i) {
    switch (src_tp[i].get_type_id()) {
    case strided_dim_type_id:
    case fixed_dim_type_id:
    case cfixed_dim_type_id:
      break;
    case var_dim_type_id:
      src_all_strided = false;
      break;
    default:
      // If it's a scalar, allow it to broadcast like
      // a strided dimension
      if (src_tp[i].get_ndim() > 0) {
        src_all_strided_or_var = false;
      }
      break;
    }
  }

  // Call to some special-case functions based on the
  // destination type
  switch (dst_tp.get_type_id()) {
  case strided_dim_type_id:
  case fixed_dim_type_id:
  case cfixed_dim_type_id:
    if (src_all_strided) {
      return make_elwise_strided_dimension_expr_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
          kernreq, ectx, elwise_handler);
    } else if (src_all_strided_or_var) {
      return make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
          kernreq, ectx, elwise_handler);
    } else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return make_elwise_strided_or_var_to_var_dimension_expr_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_count, src_tp, src_arrmeta,
          kernreq, ectx, elwise_handler);
    } else {
      // TODO
    }
    break;
  case offset_dim_type_id:
    // TODO
    break;
  default:
    break;
  }

  stringstream ss;
  ss << "Cannot evaluate elwise expression from (";
  for (size_t i = 0; i != src_count; ++i) {
    ss << src_tp[i];
    if (i != src_count - 1) {
      ss << ", ";
    }
  }
  ss << ") to " << dst_tp;
  throw runtime_error(ss.str());
}
