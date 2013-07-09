//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>

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

    kernel_data_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride[N];

    static void single(char *dst, const char * const *src,
                    kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
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

    static void destruct(kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

template<int N>
static size_t make_elwise_strided_dimension_expr_kernel_for_N(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    size_t undim = dst_dt.get_undim();
    const char *dst_child_metadata;
    const char *src_child_metadata[N];
    ndt::type dst_child_dt;
    ndt::type src_child_dt[N];
 
    out->ensure_capacity(offset_out + sizeof(strided_expr_kernel_extra<N>));
    strided_expr_kernel_extra<N> *e = out->get_at<strided_expr_kernel_extra<N> >(offset_out);
    switch (kernreq) {
        case kernel_request_single:
            e->base.template set_function<expr_single_operation_t>(&strided_expr_kernel_extra<N>::single);
            break;
        case kernel_request_strided:
            e->base.template set_function<expr_strided_operation_t>(&strided_expr_kernel_extra<N>::strided);
            break;
        default: {
            stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: unrecognized request " << (int)kernreq;
            throw runtime_error(ss.str());
        }
    }
    e->base.destructor = strided_expr_kernel_extra<N>::destruct;
    // The dst strided parameters
    if (dst_dt.get_type_id() == strided_dim_type_id) {
        const strided_dim_dtype *sdd = static_cast<const strided_dim_dtype *>(dst_dt.extended());
        const strided_dim_dtype_metadata *dst_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(dst_metadata);
        e->size = dst_md->size;
        e->dst_stride = dst_md->stride;
        dst_child_metadata = dst_metadata + sizeof(strided_dim_dtype_metadata);
        dst_child_dt = sdd->get_element_type();
    } else {
        const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(dst_dt.extended());
        e->size = fdd->get_fixed_dim_size();
        e->dst_stride = fdd->get_fixed_stride();
        dst_child_metadata = dst_metadata;
        dst_child_dt = fdd->get_element_type();
    }
    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_dt[i].get_undim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = src_dt[i];
        } else if (src_dt[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_dtype *sdd = static_cast<const strided_dim_dtype *>(src_dt[i].extended());
            const strided_dim_dtype_metadata *src_md =
                            reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata[i]);
            // Check for a broadcasting error
            if (src_md->size != 1 && e->size != src_md->size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            src_child_metadata[i] = src_metadata[i] + sizeof(strided_dim_dtype_metadata);
            src_child_dt[i] = sdd->get_element_type();
        } else {
            const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(src_dt[i].extended());
            // Check for a broadcasting error
            if (fdd->get_fixed_dim_size() != 1 && (size_t)e->size != fdd->get_fixed_dim_size()) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = fdd->get_element_type();
        }
    }
    return elwise_handler->make_expr_kernel(
                    out, offset_out + sizeof(strided_expr_kernel_extra<N>),
                    dst_child_dt, dst_child_metadata,
                    N, src_child_dt, src_child_metadata,
                    kernel_request_strided, ectx);
}

inline static size_t make_elwise_strided_dimension_expr_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_dimension_expr_kernel_for_N<1>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 2:
            return make_elwise_strided_dimension_expr_kernel_for_N<2>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 3:
            return make_elwise_strided_dimension_expr_kernel_for_N<3>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 4:
            return make_elwise_strided_dimension_expr_kernel_for_N<4>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 5:
            return make_elwise_strided_dimension_expr_kernel_for_N<5>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 6:
            return make_elwise_strided_dimension_expr_kernel_for_N<6>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
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

    kernel_data_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride[N], src_offset[N];
    bool is_src_var[N];

    static void single(char *dst, const char * const *src,
                    kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
        // Broadcast all the src 'var' dimensions to dst
        intptr_t dim_size = e->size;
        const char *modified_src[N];
        intptr_t modified_src_stride[N];
        for (int i = 0; i < N; ++i) {
            if (e->is_src_var[i]) {
                const var_dim_dtype_data *vddd = reinterpret_cast<const var_dim_dtype_data *>(src[i]);
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
                    size_t count, kernel_data_prefix *extra)
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

    static void destruct(kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

template<int N>
static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    size_t undim = dst_dt.get_undim();
    const char *dst_child_metadata;
    const char *src_child_metadata[N];
    ndt::type dst_child_dt;
    ndt::type src_child_dt[N];
 
    out->ensure_capacity(offset_out + sizeof(strided_or_var_to_strided_expr_kernel_extra<N>));
    strided_or_var_to_strided_expr_kernel_extra<N> *e =
                    out->get_at<strided_or_var_to_strided_expr_kernel_extra<N> >(offset_out);
    switch (kernreq) {
        case kernel_request_single:
            e->base.template set_function<expr_single_operation_t>(&strided_or_var_to_strided_expr_kernel_extra<N>::single);
            break;
        case kernel_request_strided:
            e->base.template set_function<expr_strided_operation_t>(&strided_or_var_to_strided_expr_kernel_extra<N>::strided);
            break;
        default: {
            stringstream ss;
            ss << "make_elwise_strided_or_var_to_strided_dimension_expr_kernel: unrecognized request " << (int)kernreq;
            throw runtime_error(ss.str());
        }
    }
    e->base.destructor = strided_or_var_to_strided_expr_kernel_extra<N>::destruct;
    // The dst strided parameters
    if (dst_dt.get_type_id() == strided_dim_type_id) {
        const strided_dim_dtype *sdd = static_cast<const strided_dim_dtype *>(dst_dt.extended());
        const strided_dim_dtype_metadata *dst_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(dst_metadata);
        e->size = dst_md->size;
        e->dst_stride = dst_md->stride;
        dst_child_metadata = dst_metadata + sizeof(strided_dim_dtype_metadata);
        dst_child_dt = sdd->get_element_type();
    } else {
        const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(dst_dt.extended());
        e->size = fdd->get_fixed_dim_size();
        e->dst_stride = fdd->get_fixed_stride();
        dst_child_metadata = dst_metadata;
        dst_child_dt = fdd->get_element_type();
    }
    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_dt[i].get_undim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = src_dt[i];
        } else if (src_dt[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_dtype *sdd = static_cast<const strided_dim_dtype *>(src_dt[i].extended());
            const strided_dim_dtype_metadata *src_md =
                            reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata[i]);
            // Check for a broadcasting error
            if (src_md->size != 1 && e->size != src_md->size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i] + sizeof(strided_dim_dtype_metadata);
            src_child_dt[i] = sdd->get_element_type();
        } else if (src_dt[i].get_type_id() == fixed_dim_type_id) {
            const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(src_dt[i].extended());
            // Check for a broadcasting error
            if (fdd->get_fixed_dim_size() != 1 && (size_t)e->size != fdd->get_fixed_dim_size()) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = fdd->get_element_type();
        } else {
            const var_dim_dtype *vdd = static_cast<const var_dim_dtype *>(src_dt[i].extended());
            const var_dim_dtype_metadata *src_md =
                            reinterpret_cast<const var_dim_dtype_metadata *>(src_metadata[i]);
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = src_md->offset;
            e->is_src_var[i] = true;
            src_child_metadata[i] = src_metadata[i] + sizeof(var_dim_dtype_metadata);
            src_child_dt[i] = vdd->get_element_type();
        }
    }
    return elwise_handler->make_expr_kernel(
                    out, offset_out + sizeof(strided_or_var_to_strided_expr_kernel_extra<N>),
                    dst_child_dt, dst_child_metadata,
                    N, src_child_dt, src_child_metadata,
                    kernel_request_strided, ectx);
}

static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<1>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 2:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<2>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 3:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<3>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 4:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<4>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 5:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<5>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 6:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<6>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
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

    kernel_data_prefix base;
    memory_block_data *dst_memblock;
    size_t dst_target_alignment;
    intptr_t dst_stride, dst_offset, src_stride[N], src_offset[N];
    bool is_src_var[N];

    static void single(char *dst, const char * const *src,
                    kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
        var_dim_dtype_data *dst_vddd = reinterpret_cast<var_dim_dtype_data *>(dst);
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
                    const var_dim_dtype_data *vddd = reinterpret_cast<const var_dim_dtype_data *>(src[i]);
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
                    const var_dim_dtype_data *vddd = reinterpret_cast<const var_dim_dtype_data *>(src[i]);
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
                    size_t count, kernel_data_prefix *extra)
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

    static void destruct(kernel_data_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        kernel_data_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

template<int N>
static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    size_t undim = dst_dt.get_undim();
    const char *dst_child_metadata;
    const char *src_child_metadata[N];
    ndt::type dst_child_dt;
    ndt::type src_child_dt[N];
 
    out->ensure_capacity(offset_out + sizeof(strided_or_var_to_var_expr_kernel_extra<N>));
    strided_or_var_to_var_expr_kernel_extra<N> *e =
                    out->get_at<strided_or_var_to_var_expr_kernel_extra<N> >(offset_out);
    switch (kernreq) {
        case kernel_request_single:
            e->base.template set_function<expr_single_operation_t>(
                            &strided_or_var_to_var_expr_kernel_extra<N>::single);
            break;
        case kernel_request_strided:
            e->base.template set_function<expr_strided_operation_t>(
                            &strided_or_var_to_var_expr_kernel_extra<N>::strided);
            break;
        default: {
            stringstream ss;
            ss << "make_elwise_strided_or_var_to_var_dimension_expr_kernel: unrecognized request " << (int)kernreq;
            throw runtime_error(ss.str());
        }
    }
    e->base.destructor = strided_or_var_to_var_expr_kernel_extra<N>::destruct;
    // The dst var parameters
    const var_dim_dtype *dst_vdd = static_cast<const var_dim_dtype *>(dst_dt.extended());
    const var_dim_dtype_metadata *dst_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(dst_metadata);
    e->dst_memblock = dst_md->blockref;
    e->dst_stride = dst_md->stride;
    e->dst_offset = dst_md->offset;
    e->dst_target_alignment = dst_vdd->get_target_alignment();
    dst_child_metadata = dst_metadata + sizeof(var_dim_dtype_metadata);
    dst_child_dt = dst_vdd->get_element_type();

    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_dt[i].get_undim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = src_dt[i];
        } else if (src_dt[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_dtype *sdd = static_cast<const strided_dim_dtype *>(src_dt[i].extended());
            const strided_dim_dtype_metadata *src_md =
                            reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata[i]);
            // Check for a broadcasting error (the strided dimension size must be 1,
            // otherwise the destination should be strided, not var)
            if (src_md->size != 1) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i] + sizeof(strided_dim_dtype_metadata);
            src_child_dt[i] = sdd->get_element_type();
        } else if (src_dt[i].get_type_id() == fixed_dim_type_id) {
            const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(src_dt[i].extended());
            // Check for a broadcasting error (the strided dimension size must be 1,
            // otherwise the destination should be strided, not var)
            if (fdd->get_fixed_dim_size() != 1) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = fdd->get_element_type();
        } else {
            const var_dim_dtype *vdd = static_cast<const var_dim_dtype *>(src_dt[i].extended());
            const var_dim_dtype_metadata *src_md =
                            reinterpret_cast<const var_dim_dtype_metadata *>(src_metadata[i]);
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = src_md->offset;
            e->is_src_var[i] = true;
            src_child_metadata[i] = src_metadata[i] + sizeof(var_dim_dtype_metadata);
            src_child_dt[i] = vdd->get_element_type();
        }
    }
    return elwise_handler->make_expr_kernel(
                    out, offset_out + sizeof(strided_or_var_to_var_expr_kernel_extra<N>),
                    dst_child_dt, dst_child_metadata,
                    N, src_child_dt, src_child_metadata,
                    kernel_request_strided, ectx);
}

static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<1>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 2:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<2>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 3:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<3>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 4:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<4>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 5:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<5>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        case 6:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<6>(
                            out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_or_var_to_var_dimension_expr_kernel with src_count > 6 not implemented yet");
    }
}

size_t dynd::make_elwise_dimension_expr_kernel(hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    // Do a pass through the src dtypes to classify them
    bool src_all_strided = true, src_all_strided_or_var = true;
    for (size_t i = 0; i != src_count; ++i) {
        switch (src_dt[i].get_type_id()) {
            case strided_dim_type_id:
            case fixed_dim_type_id:
                break;
            case var_dim_type_id:
                src_all_strided = false;
                break;
            default:
                // If it's a scalar, allow it to broadcast like
                // a strided dimension
                if (src_dt[i].get_undim() > 0) {
                    src_all_strided_or_var = false;
                }
                break;
        }
    }

    // Call to some special-case functions based on the
    // destination dtype
    switch (dst_dt.get_type_id()) {
        case strided_dim_type_id:
        case fixed_dim_type_id:
            if (src_all_strided) {
                return make_elwise_strided_dimension_expr_kernel(
                                out, offset_out,
                                dst_dt, dst_metadata,
                                src_count, src_dt, src_metadata,
                                kernreq, ectx,
                                elwise_handler);
            } else if (src_all_strided_or_var) {
                return make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                                out, offset_out,
                                dst_dt, dst_metadata,
                                src_count, src_dt, src_metadata,
                                kernreq, ectx,
                                elwise_handler);
            } else {
                // TODO
            }
            break;
        case var_dim_type_id:
            if (src_all_strided_or_var) {
                return make_elwise_strided_or_var_to_var_dimension_expr_kernel(
                                out, offset_out,
                                dst_dt, dst_metadata,
                                src_count, src_dt, src_metadata,
                                kernreq, ectx,
                                elwise_handler);
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
        ss << src_dt[i];
        if (i != src_count - 1) {
            ss << ", ";
        }
    }
    ss << ") to " << dst_dt;
    throw runtime_error(ss.str());
}
