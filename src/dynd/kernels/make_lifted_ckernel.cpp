//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;

////////////////////////////////////////////////////////////////////
// make_elwise_strided_dimension_expr_kernel

namespace {

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

    static void single(char *dst, char **src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    char **src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        intptr_t inner_size = e->size, inner_dst_stride = e->dst_stride;
        const intptr_t *inner_src_stride = e->src_stride;
        char *src_loop[N];
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

} // anonymous namespace

template <int N>
static size_t make_elwise_strided_dimension_expr_kernel_for_N(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    size_t DYND_UNUSED(src_count), const intptr_t *src_ndim,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  const char *child_dst_arrmeta;
  const char *child_src_arrmeta[N];
  ndt::type child_dst_tp;
  ndt::type child_src_tp[N];
  strided_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_expr_kernel_extra<N> >(ckb_offset);
  switch (kernreq) {
  case kernel_request_single:
    e->base.template set_function<expr_single_t>(
        &strided_expr_kernel_extra<N>::single);
    break;
  case kernel_request_strided:
    e->base.template set_function<expr_strided_t>(
        &strided_expr_kernel_extra<N>::strided);
    break;
  default: {
    stringstream ss;
    ss << "make_elwise_strided_dimension_expr_kernel: unrecognized request "
       << (int)kernreq;
    throw runtime_error(ss.str());
  }
  }
  e->base.destructor = strided_expr_kernel_extra<N>::destruct;
  if (!dst_tp.get_as_strided(dst_arrmeta, &e->size, &e->dst_stride,
                             &child_dst_tp, &child_dst_arrmeta)) {
    stringstream ss;
    ss << "make_elwise_strided_dimension_expr_kernel: error processing "
          "type " << dst_tp << " as strided";
    throw type_error(ss.str());
  }

  intptr_t child_src_ndim[N];
  bool finished = dst_ndim == 1;
  for (int i = 0; i < N; ++i) {
    intptr_t src_size;
    // The src[i] strided parameters
    if (src_ndim[i] < dst_ndim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      child_src_arrmeta[i] = src_arrmeta[i];
      child_src_tp[i] = src_tp[i];
      child_src_ndim[i] = src_ndim[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                        &e->src_stride[i], &child_src_tp[i],
                                        &child_src_arrmeta[i])) {
      // Check for a broadcasting error
      if (src_size != 1 && e->size != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
      }
      child_src_ndim[i] = src_ndim[i] - 1;
    } else {
      stringstream ss;
      ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
            "or fixed dim, got " << src_tp[i];
      throw runtime_error(ss.str());
    }
    finished = finished && child_src_ndim[i] == 0;
  }
  // If there are still dimensions to broadcast, recursively lift more
  if (!finished) {
    return make_lifted_expr_ckernel(
        elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_ndim - 1,
        child_dst_tp, child_dst_arrmeta, child_src_ndim, child_src_tp,
        child_src_arrmeta, kernel_request_strided, ectx);
  }
  // Instantiate the elementwise handler
  return elwise_handler->instantiate(
      elwise_handler, elwise_handler_tp, ckb, ckb_offset, child_dst_tp,
      child_dst_arrmeta, child_src_tp, child_src_arrmeta,
      kernel_request_strided, ectx, nd::array(), nd::array());
}

inline static size_t make_elwise_strided_dimension_expr_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta, size_t src_count,
    const intptr_t *src_ndim, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  switch (src_count) {
  case 1:
    return make_elwise_strided_dimension_expr_kernel_for_N<1>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 2:
    return make_elwise_strided_dimension_expr_kernel_for_N<2>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 3:
    return make_elwise_strided_dimension_expr_kernel_for_N<3>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 4:
    return make_elwise_strided_dimension_expr_kernel_for_N<4>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 5:
    return make_elwise_strided_dimension_expr_kernel_for_N<5>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 6:
    return make_elwise_strided_dimension_expr_kernel_for_N<6>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  default:
    throw runtime_error("make_elwise_strided_dimension_expr_kernel with "
                        "src_count > 6 not implemented yet");
  }
}

////////////////////////////////////////////////////////////////////
// make_elwise_strided_or_var_to_strided_dimension_expr_kernel

namespace {

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

    static void single(char *dst, char **src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = e->size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (int i = 0; i < N; ++i) {
            if (e->is_src_var[i]) {
                var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
                modified_src[i] = vddd->begin + e->src_offset[i];
                if (vddd->size == 1) {
                    modified_src_stride[i] = 0;
                } else if (vddd->size == static_cast<size_t>(dim_size)) {
                    modified_src_stride[i] = e->src_stride[i];
                } else {
                    throw broadcast_error(dim_size, vddd->size, "strided", "var");
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
                    char **src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        char *src_loop[N];
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

} // anonymous namespace

template <int N>
static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    size_t DYND_UNUSED(src_count), const intptr_t *src_ndim,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  const char *child_dst_arrmeta;
  const char *child_src_arrmeta[N];
  ndt::type child_dst_tp;
  ndt::type child_src_tp[N];

  strided_or_var_to_strided_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_or_var_to_strided_expr_kernel_extra<N> >(
          ckb_offset);
  switch (kernreq) {
  case kernel_request_single:
    e->base.template set_function<expr_single_t>(
        &strided_or_var_to_strided_expr_kernel_extra<N>::single);
    break;
  case kernel_request_strided:
    e->base.template set_function<expr_strided_t>(
        &strided_or_var_to_strided_expr_kernel_extra<N>::strided);
    break;
  default: {
    stringstream ss;
    ss << "make_elwise_strided_or_var_to_strided_dimension_expr_kernel: "
          "unrecognized request " << (int)kernreq;
    throw runtime_error(ss.str());
  }
  }
  e->base.destructor = strided_or_var_to_strided_expr_kernel_extra<N>::destruct;
  if (!dst_tp.get_as_strided(dst_arrmeta, &e->size, &e->dst_stride,
                             &child_dst_tp, &child_dst_arrmeta)) {
    stringstream ss;
    ss << "make_elwise_strided_dimension_expr_kernel: error processing "
          "type " << dst_tp << " as strided";
    throw type_error(ss.str());
  }

  intptr_t child_src_ndim[N];
  bool finished = dst_ndim == 1;
  for (int i = 0; i < N; ++i) {
    intptr_t src_size;
    // The src[i] strided parameters
    if (src_ndim[i] < dst_ndim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
      child_src_arrmeta[i] = src_arrmeta[i];
      child_src_tp[i] = src_tp[i];
      child_src_ndim[i] = src_ndim[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size,
                                        &e->src_stride[i], &child_src_tp[i],
                                        &child_src_arrmeta[i])) {
      // Check for a broadcasting error
      if (src_size != 1 && e->size != src_size) {
        throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
      }
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
      child_src_ndim[i] = src_ndim[i] - 1;
    } else {
      const var_dim_type *vdd =
          static_cast<const var_dim_type *>(src_tp[i].extended());
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
      e->src_stride[i] = src_md->stride;
      e->src_offset[i] = src_md->offset;
      e->is_src_var[i] = true;
      child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
      child_src_tp[i] = vdd->get_element_type();
      child_src_ndim[i] = src_ndim[i] - 1;
    }
    finished = finished && child_src_ndim[i] == 0;
  }
  // If there are still dimensions to broadcast, recursively lift more
  if (!finished) {
    return make_lifted_expr_ckernel(
        elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_ndim - 1, child_dst_tp,
        child_dst_arrmeta, child_src_ndim, child_src_tp, child_src_arrmeta,
        kernel_request_strided, ectx);
  }
  // Instantiate the elementwise handler
  return elwise_handler->instantiate(
      elwise_handler, elwise_handler_tp, ckb, ckb_offset, child_dst_tp,
      child_dst_arrmeta, child_src_tp, child_src_arrmeta,
      kernel_request_strided, ectx, nd::array(), nd::array());
}

static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta, size_t src_count,
    const intptr_t *src_ndim, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  switch (src_count) {
  case 1:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<1>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 2:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<2>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 3:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<3>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 4:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<4>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 5:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<5>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 6:
    return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<6>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  default:
    throw runtime_error("make_elwise_strided_or_var_to_strided_dimension_expr_"
                        "kernel with src_count > 6 not implemented yet");
  }
}

////////////////////////////////////////////////////////////////////
// make_elwise_strided_or_var_to_var_dimension_expr_kernel

namespace {

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
    intptr_t dst_stride, dst_offset, src_stride[N], src_offset[N], src_size[N];
    bool is_src_var[N];

    static void single(char *dst, char **src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = e->base.get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild = echild->get_function<expr_strided_t>();
        var_dim_type_data *dst_vddd = reinterpret_cast<var_dim_type_data *>(dst);
        char *modified_dst;
        intptr_t modified_dst_stride = 0;
        intptr_t dim_size;
        char *modified_src[N];
        intptr_t modified_src_stride[N];
        if (dst_vddd->begin != NULL) {
            // If the destination already has allocated data, broadcast to that data
            modified_dst = dst_vddd->begin + e->dst_offset;
            // Broadcast all the inputs to the existing destination dimension size
            dim_size = dst_vddd->size;
            for (int i = 0; i < N; ++i) {
                if (e->is_src_var[i]) {
                    var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
                    modified_src[i] = vddd->begin + e->src_offset[i];
                    if (vddd->size == 1) {
                        modified_src_stride[i] = 0;
                    } else if (vddd->size == static_cast<size_t>(dim_size)) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, vddd->size, "var", "var");
                    }
                } else {
                    modified_src[i] = src[i];
                    if (e->src_size[i] == 1) {
                        modified_src_stride[i] = 0;
                    } else if (e->src_size[i] == dim_size) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, e->src_size[i], "var", "strided");
                    }
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
                    var_dim_type_data *vddd = reinterpret_cast<var_dim_type_data *>(src[i]);
                    modified_src[i] = vddd->begin + e->src_offset[i];
                    if (vddd->size == 1) {
                        modified_src_stride[i] = 0;
                    } else if (dim_size == 1) {
                        dim_size = vddd->size;
                        modified_src_stride[i] = e->src_stride[i];
                    } else if (vddd->size == static_cast<size_t>(dim_size)) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, vddd->size, "var", "var");
                    }
                } else {
                    modified_src[i] = src[i];
                    if (e->src_size[i] == 1) {
                        modified_src_stride[i] = 0;
                    } else if (e->src_size[i] == dim_size) {
                        modified_src_stride[i] = e->src_stride[i];
                    } else if (dim_size == 1) {
                        dim_size = e->src_size[i];
                        modified_src_stride[i] = e->src_stride[i];
                    } else {
                        throw broadcast_error(dim_size, e->src_size[i], "var", "strided");
                    }
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
        }
        if (dim_size <= 1) {
            modified_dst_stride = 0;
        } else {
            modified_dst_stride = e->dst_stride;
        }
        opchild(modified_dst, modified_dst_stride, modified_src, modified_src_stride, dim_size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    char **src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        char *src_loop[N];
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

} // anonymous namespace

template <int N>
static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    size_t DYND_UNUSED(src_count), const intptr_t *src_ndim,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  const char *child_dst_arrmeta;
  const char *child_src_arrmeta[N];
  ndt::type child_dst_tp;
  ndt::type child_src_tp[N];

  strided_or_var_to_var_expr_kernel_extra<N> *e =
      ckb->alloc_ck<strided_or_var_to_var_expr_kernel_extra<N> >(ckb_offset);
  switch (kernreq) {
  case kernel_request_single:
    e->base.template set_function<expr_single_t>(
        &strided_or_var_to_var_expr_kernel_extra<N>::single);
    break;
  case kernel_request_strided:
    e->base.template set_function<expr_strided_t>(
        &strided_or_var_to_var_expr_kernel_extra<N>::strided);
    break;
  default: {
    stringstream ss;
    ss << "make_elwise_strided_or_var_to_var_dimension_expr_kernel: "
          "unrecognized request " << (int)kernreq;
    throw runtime_error(ss.str());
  }
  }
  e->base.destructor = strided_or_var_to_var_expr_kernel_extra<N>::destruct;
  // The dst var parameters
  const var_dim_type *dst_vdd = dst_tp.extended<var_dim_type>();
  const var_dim_type_arrmeta *dst_md =
      reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta);
  e->dst_memblock = dst_md->blockref;
  e->dst_stride = dst_md->stride;
  e->dst_offset = dst_md->offset;
  e->dst_target_alignment = dst_vdd->get_target_alignment();
  child_dst_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
  child_dst_tp = dst_vdd->get_element_type();

  intptr_t child_src_ndim[N];
  bool finished = dst_ndim == 1;
  for (int i = 0; i < N; ++i) {
    // The src[i] strided parameters
    if (src_ndim[i] < dst_ndim) {
      // This src value is getting broadcasted
      e->src_stride[i] = 0;
      e->src_offset[i] = 0;
      e->src_size[i] = 1;
      e->is_src_var[i] = false;
      child_src_arrmeta[i] = src_arrmeta[i];
      child_src_tp[i] = src_tp[i];
      child_src_ndim[i] = src_ndim[i];
    } else if (src_tp[i].get_as_strided(src_arrmeta[i], &e->src_size[i],
                                        &e->src_stride[i], &child_src_tp[i],
                                        &child_src_arrmeta[i])) {
      e->src_offset[i] = 0;
      e->is_src_var[i] = false;
      child_src_ndim[i] = src_ndim[i] - 1;
    } else {
      const var_dim_type *vdd =
          static_cast<const var_dim_type *>(src_tp[i].extended());
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[i]);
      e->src_stride[i] = src_md->stride;
      e->src_offset[i] = src_md->offset;
      e->is_src_var[i] = true;
      child_src_arrmeta[i] = src_arrmeta[i] + sizeof(var_dim_type_arrmeta);
      child_src_tp[i] = vdd->get_element_type();
      child_src_ndim[i] = src_ndim[i] - 1;
    }
    finished = finished && child_src_ndim[i] == 0;
  }
  // If there are still dimensions to broadcast, recursively lift more
  if (!finished) {
    return make_lifted_expr_ckernel(
        elwise_handler, elwise_handler_tp, ckb, ckb_offset, dst_ndim - 1,
        child_dst_tp, child_dst_arrmeta, child_src_ndim, child_src_tp,
        child_src_arrmeta, kernel_request_strided, ectx);
  }
  // All the types matched, so instantiate the elementwise handler
  return elwise_handler->instantiate(
      elwise_handler, elwise_handler_tp, ckb, ckb_offset, child_dst_tp,
      child_dst_arrmeta, child_src_tp, child_src_arrmeta,
      kernel_request_strided, ectx, nd::array(), nd::array());
}

static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, intptr_t dst_ndim,
    const ndt::type &dst_tp, const char *dst_arrmeta, size_t src_count,
    const intptr_t *src_ndim, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, const eval::eval_context *ectx)
{
  switch (src_count) {
  case 1:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<1>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 2:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<2>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 3:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<3>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 4:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<4>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 5:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<5>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  case 6:
    return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<6>(
        ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
        src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
  default:
    throw runtime_error("make_elwise_strided_or_var_to_var_dimension_expr_"
                        "kernel with src_count > 6 not implemented yet");
  }
}

size_t dynd::make_lifted_expr_ckernel(
    const arrfunc_type_data *elwise_handler,
    const arrfunc_type *elwise_handler_tp, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t dst_ndim, const ndt::type &dst_tp,
    const char *dst_arrmeta, const intptr_t *src_ndim, const ndt::type *src_tp,
    const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
    const eval::eval_context *ectx)

{
  intptr_t src_count = elwise_handler_tp->get_npos();

  // Check if no lifting is required
  if (dst_ndim == 0) {
    intptr_t i = 0;
    for (; i < src_count; ++i) {
      if (src_ndim[i] != 0) {
        break;
      }
    }
    if (i == src_count) {
      // No dimensions to lift, call the elementwise instantiate directly
      return elwise_handler->instantiate(elwise_handler, elwise_handler_tp, ckb,
                                         ckb_offset, dst_tp, dst_arrmeta,
                                         src_tp, src_arrmeta, kernreq, ectx,
                                         nd::array(), nd::array());
    }
    else {
      stringstream ss;
      ss << "Trying to broadcast " << src_ndim[i] << " dimensions of "
         << src_tp[i] << " into 0 dimensions of " << dst_tp
         << ", the destination dimension count must be greater";
      throw broadcast_error(ss.str());
    }
  }

  // Do a pass through the src types to classify them
  bool src_all_strided = true, src_all_strided_or_var = true;
  for (intptr_t i = 0; i < src_count; ++i) {
    switch (src_tp[i].get_type_id()) {
    case fixed_dim_type_id:
    case cfixed_dim_type_id:
      break;
    case var_dim_type_id:
      src_all_strided = false;
      break;
    default:
      // If it's a scalar, allow it to broadcast like
      // a strided dimension
      if (src_ndim[i] > 0) {
        src_all_strided_or_var = false;
      }
      break;
    }
  }

  // Call to some special-case functions based on the
  // destination type
  switch (dst_tp.get_type_id()) {
  case fixed_dim_type_id:
  case cfixed_dim_type_id:
    if (src_all_strided) {
      return make_elwise_strided_dimension_expr_kernel(
          ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
          src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
    }
    else if (src_all_strided_or_var) {
      return make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
          ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
          src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
    }
    else {
      // TODO
    }
    break;
  case var_dim_type_id:
    if (src_all_strided_or_var) {
      return make_elwise_strided_or_var_to_var_dimension_expr_kernel(
          ckb, ckb_offset, dst_ndim, dst_tp, dst_arrmeta, src_count, src_ndim,
          src_tp, src_arrmeta, kernreq, elwise_handler, elwise_handler_tp, ectx);
    }
    else {
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
  ss << "Cannot process lifted elwise expression from (";
  for (intptr_t i = 0; i < src_count; ++i) {
    ss << src_tp[i];
    if (i != src_count - 1) {
      ss << ", ";
    }
  }
  ss << ") to " << dst_tp;
  throw runtime_error(ss.str());
}
