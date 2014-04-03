//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
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

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, echild);
    }

    static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
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

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

} // anonymous namespace

template<int N>
static size_t make_elwise_strided_dimension_expr_kernel_for_N(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    intptr_t undim = dst_tp.get_ndim();
    const char *child_metadata[N+1];
    ndt::type child_tp[N+1];
    intptr_t ckb_child_offset = ckb_offset + sizeof(strided_expr_kernel_extra<N>);
    
    out_ckb->ensure_capacity(ckb_offset + sizeof(strided_expr_kernel_extra<N>));
    strided_expr_kernel_extra<N> *e = out_ckb->get_at<strided_expr_kernel_extra<N> >(ckb_offset);
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
    if (dst_tp.get_type_id() == strided_dim_type_id) {
        const strided_dim_type *sdd = static_cast<const strided_dim_type *>(dst_tp.extended());
        const strided_dim_type_metadata *dst_md =
                        reinterpret_cast<const strided_dim_type_metadata *>(dst_metadata);
        e->size = dst_md->size;
        e->dst_stride = dst_md->stride;
        child_metadata[0] = dst_metadata + sizeof(strided_dim_type_metadata);
        child_tp[0] = sdd->get_element_type();
    } else {
        const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(dst_tp.extended());
        e->size = fdd->get_fixed_dim_size();
        e->dst_stride = fdd->get_fixed_stride();
        child_metadata[0] = dst_metadata;
        child_tp[0] = fdd->get_element_type();
    }
    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_tp[i].get_ndim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = src_tp[i];
        } else if (src_tp[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_type *sdd = static_cast<const strided_dim_type *>(src_tp[i].extended());
            const strided_dim_type_metadata *src_md =
                            reinterpret_cast<const strided_dim_type_metadata *>(src_metadata[i]);
            // Check for a broadcasting error
            if (src_md->size != 1 && e->size != src_md->size) {
                throw broadcast_error(dst_tp, dst_metadata, src_tp[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            child_metadata[i + 1] = src_metadata[i] + sizeof(strided_dim_type_metadata);
            child_tp[i + 1] = sdd->get_element_type();
        } else {
            const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(src_tp[i].extended());
            // Check for a broadcasting error
            if (fdd->get_fixed_dim_size() != 1 && (size_t)e->size != fdd->get_fixed_dim_size()) {
                throw broadcast_error(dst_tp, dst_metadata, src_tp[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = fdd->get_element_type();
        }
    }
    // If any of the types don't match, continue broadcasting the dimensions
    for (intptr_t i = 0; i < N + 1; ++i) {
        if (child_tp[i] != elwise_handler->data_dynd_types[i]) {
            return make_lifted_expr_ckernel(elwise_handler,
                            out_ckb, ckb_child_offset,
                            child_tp, child_metadata,
                            kernel_request_strided);
        }
    }
    // All the types matched, so instantiate the elementwise handler
    return elwise_handler->instantiate_func(
                    elwise_handler->data_ptr,
                    out_ckb, ckb_child_offset,
                    child_metadata, kernel_request_strided);
}

inline static size_t make_elwise_strided_dimension_expr_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_dimension_expr_kernel_for_N<1>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 2:
            return make_elwise_strided_dimension_expr_kernel_for_N<2>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 3:
            return make_elwise_strided_dimension_expr_kernel_for_N<3>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 4:
            return make_elwise_strided_dimension_expr_kernel_for_N<4>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 5:
            return make_elwise_strided_dimension_expr_kernel_for_N<5>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 6:
            return make_elwise_strided_dimension_expr_kernel_for_N<6>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_dimension_expr_kernel with src_count > 6 not implemented yet");
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

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
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

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

} // anonymous namespace

template<int N>
static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    intptr_t undim = dst_tp.get_ndim();
    const char *child_metadata[N+1];
    ndt::type child_tp[N+1];
    intptr_t ckb_child_offset = ckb_offset + sizeof(strided_or_var_to_strided_expr_kernel_extra<N>);
 
    out_ckb->ensure_capacity(ckb_offset + sizeof(strided_or_var_to_strided_expr_kernel_extra<N>));
    strided_or_var_to_strided_expr_kernel_extra<N> *e =
                    out_ckb->get_at<strided_or_var_to_strided_expr_kernel_extra<N> >(ckb_offset);
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
    if (dst_tp.get_type_id() == strided_dim_type_id) {
        const strided_dim_type *sdd = static_cast<const strided_dim_type *>(dst_tp.extended());
        const strided_dim_type_metadata *dst_md =
                        reinterpret_cast<const strided_dim_type_metadata *>(dst_metadata);
        e->size = dst_md->size;
        e->dst_stride = dst_md->stride;
        child_metadata[0] = dst_metadata + sizeof(strided_dim_type_metadata);
        child_tp[0] = sdd->get_element_type();
    } else {
        const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(dst_tp.extended());
        e->size = fdd->get_fixed_dim_size();
        e->dst_stride = fdd->get_fixed_stride();
        child_metadata[0] = dst_metadata;
        child_tp[0] = fdd->get_element_type();
    }
    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_tp[i].get_ndim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = src_tp[i];
        } else if (src_tp[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_type *sdd = static_cast<const strided_dim_type *>(src_tp[i].extended());
            const strided_dim_type_metadata *src_md =
                            reinterpret_cast<const strided_dim_type_metadata *>(src_metadata[i]);
            // Check for a broadcasting error
            if (src_md->size != 1 && e->size != src_md->size) {
                throw broadcast_error(dst_tp, dst_metadata, src_tp[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i] + sizeof(strided_dim_type_metadata);
            child_tp[i + 1] = sdd->get_element_type();
        } else if (src_tp[i].get_type_id() == fixed_dim_type_id) {
            const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(src_tp[i].extended());
            // Check for a broadcasting error
            if (fdd->get_fixed_dim_size() != 1 && (size_t)e->size != fdd->get_fixed_dim_size()) {
                throw broadcast_error(dst_tp, dst_metadata, src_tp[i], src_metadata[i]);
            }
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            e->src_offset[i] = 0;
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = fdd->get_element_type();
        } else {
            const var_dim_type *vdd = static_cast<const var_dim_type *>(src_tp[i].extended());
            const var_dim_type_metadata *src_md =
                            reinterpret_cast<const var_dim_type_metadata *>(src_metadata[i]);
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = src_md->offset;
            e->is_src_var[i] = true;
            child_metadata[i + 1] = src_metadata[i] + sizeof(var_dim_type_metadata);
            child_tp[i + 1] = vdd->get_element_type();
        }
    }
    // If any of the types don't match, continue broadcasting the dimensions
    for (intptr_t i = 0; i < N + 1; ++i) {
        if (child_tp[i] != elwise_handler->data_dynd_types[i]) {
            return make_lifted_expr_ckernel(elwise_handler,
                            out_ckb, ckb_child_offset,
                            child_tp, child_metadata,
                            kernel_request_strided);
        }
    }
    // All the types matched, so instantiate the elementwise handler
    return elwise_handler->instantiate_func(
                    elwise_handler->data_ptr,
                    out_ckb, ckb_child_offset,
                    child_metadata,
                    kernel_request_strided);
}

static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<1>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 2:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<2>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 3:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<3>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 4:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<4>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 5:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<5>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 6:
            return make_elwise_strided_or_var_to_strided_dimension_expr_kernel_for_N<6>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_or_var_to_strided_dimension_expr_kernel with src_count > 6 not implemented yet");
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

    static void single(char *dst, const char * const *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        expr_strided_operation_t opchild = echild->get_function<expr_strided_operation_t>();
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

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base;
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

} // anonymous namespace

template<int N>
static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    intptr_t undim = dst_tp.get_ndim();
    const char *child_metadata[N+1];
    ndt::type child_tp[N+1];
    intptr_t ckb_child_offset = ckb_offset + sizeof(strided_or_var_to_var_expr_kernel_extra<N>);
 
    out_ckb->ensure_capacity(ckb_child_offset);
    strided_or_var_to_var_expr_kernel_extra<N> *e =
                    out_ckb->get_at<strided_or_var_to_var_expr_kernel_extra<N> >(ckb_offset);
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
    const var_dim_type *dst_vdd = static_cast<const var_dim_type *>(dst_tp.extended());
    const var_dim_type_metadata *dst_md =
                    reinterpret_cast<const var_dim_type_metadata *>(dst_metadata);
    e->dst_memblock = dst_md->blockref;
    e->dst_stride = dst_md->stride;
    e->dst_offset = dst_md->offset;
    e->dst_target_alignment = dst_vdd->get_target_alignment();
    child_metadata[0] = dst_metadata + sizeof(var_dim_type_metadata);
    child_tp[0] = dst_vdd->get_element_type();

    for (int i = 0; i < N; ++i) {
        // The src[i] strided parameters
        if (src_tp[i].get_ndim() < undim) {
            // This src value is getting broadcasted
            e->src_stride[i] = 0;
            e->src_offset[i] = 0;
            e->src_size[i] = 1;
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = src_tp[i];
        } else if (src_tp[i].get_type_id() == strided_dim_type_id) {
            const strided_dim_type *sdd = static_cast<const strided_dim_type *>(src_tp[i].extended());
            const strided_dim_type_metadata *src_md =
                            reinterpret_cast<const strided_dim_type_metadata *>(src_metadata[i]);
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = 0;
            e->src_size[i] = src_md->size;
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i] + sizeof(strided_dim_type_metadata);
            child_tp[i + 1] = sdd->get_element_type();
        } else if (src_tp[i].get_type_id() == fixed_dim_type_id) {
            const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(src_tp[i].extended());
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            e->src_offset[i] = 0;
            e->src_size[i] = fdd->get_fixed_dim_size();
            e->is_src_var[i] = false;
            child_metadata[i + 1] = src_metadata[i];
            child_tp[i + 1] = fdd->get_element_type();
        } else {
            const var_dim_type *vdd = static_cast<const var_dim_type *>(src_tp[i].extended());
            const var_dim_type_metadata *src_md =
                            reinterpret_cast<const var_dim_type_metadata *>(src_metadata[i]);
            e->src_stride[i] = src_md->stride;
            e->src_offset[i] = src_md->offset;
            e->is_src_var[i] = true;
            child_metadata[i + 1] = src_metadata[i] + sizeof(var_dim_type_metadata);
            child_tp[i + 1] = vdd->get_element_type();
        }
    }
    // If any of the types don't match, continue broadcasting the dimensions
    for (intptr_t i = 0; i < N + 1; ++i) {
        if (child_tp[i] != elwise_handler->data_dynd_types[i]) {
            return make_lifted_expr_ckernel(elwise_handler,
                            out_ckb, ckb_child_offset,
                            child_tp, child_metadata,
                            kernel_request_strided);
        }
    }
    // All the types matched, so instantiate the elementwise handler
    return elwise_handler->instantiate_func(
                    elwise_handler->data_ptr,
                    out_ckb, ckb_child_offset,
                    child_metadata, kernel_request_strided);
}

static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_tp, const char *const*src_metadata,
                kernel_request_t kernreq,
                const ckernel_deferred *elwise_handler)
{
    switch (src_count) {
        case 1:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<1>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 2:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<2>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 3:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<3>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 4:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<4>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 5:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<5>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        case 6:
            return make_elwise_strided_or_var_to_var_dimension_expr_kernel_for_N<6>(
                            out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, elwise_handler);
        default:
            throw runtime_error("make_elwise_strided_or_var_to_var_dimension_expr_kernel with src_count > 6 not implemented yet");
    }
}

size_t dynd::make_lifted_expr_ckernel(const ckernel_deferred *elwise_handler,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const ndt::type *lifted_types,
                const char *const* dynd_metadata,
                dynd::kernel_request_t kernreq)
{
    const ndt::type& dst_tp = *lifted_types;
    const ndt::type *src_tp = lifted_types + 1;
    intptr_t src_count = elwise_handler->data_types_size - 1;
    const char *dst_metadata = *dynd_metadata;
    const char *const*src_metadata = dynd_metadata + 1;
    // Do a pass through the src types to classify them
    bool src_all_strided = true, src_all_strided_or_var = true;
    for (intptr_t i = 0; i < src_count; ++i) {
        switch (src_tp[i].get_type_id()) {
            case strided_dim_type_id:
            case fixed_dim_type_id:
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
            if (src_all_strided) {
                return make_elwise_strided_dimension_expr_kernel(
                                out_ckb, ckb_offset,
                                dst_tp, dst_metadata,
                                src_count, src_tp, src_metadata,
                                kernreq, elwise_handler);
            } else if (src_all_strided_or_var) {
                return make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                                out_ckb, ckb_offset,
                                dst_tp, dst_metadata,
                                src_count, src_tp, src_metadata,
                                kernreq, elwise_handler);
            } else {
                // TODO
            }
            break;
        case var_dim_type_id:
            if (src_all_strided_or_var) {
                return make_elwise_strided_or_var_to_var_dimension_expr_kernel(
                                out_ckb, ckb_offset,
                                dst_tp, dst_metadata,
                                src_count, src_tp, src_metadata,
                                kernreq, elwise_handler);
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

    // Check if no lifting is required
    if (dst_tp == elwise_handler->data_dynd_types[0]) {
        intptr_t i = 0;
        for (; i < src_count; ++i) {
            if (src_tp[i] != elwise_handler->data_dynd_types[i+1]) {
                break;
            }
        }
        if (i == src_count) {
            // All the types matched, call the elementwise instantiate directly
            return elwise_handler->instantiate_func(elwise_handler->data_ptr,
                            out_ckb, ckb_offset,
                            dynd_metadata, kernreq);
        }
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
