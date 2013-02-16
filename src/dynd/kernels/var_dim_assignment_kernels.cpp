//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>
#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/diagnostics.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/var_dim_assignment_kernels.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>

using namespace std;
using namespace dynd;

/////////////////////////////////////////
// broadcast to var array assignment

namespace {
    struct broadcast_to_var_assign_kernel_extra {
        typedef broadcast_to_var_assign_kernel_extra extra_type;

        kernel_data_prefix base;
        intptr_t dst_target_alignment;
        const var_dim_dtype_metadata *dst_md;

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            var_dim_dtype_data *dst_d = reinterpret_cast<var_dim_dtype_data *>(dst);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                // If we're writing to an empty array, have to allocate the output
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                // Allocate the output array data
                char *dst_end = NULL;
                allocator->allocate(e->dst_md->blockref, e->dst_md->stride,
                            e->dst_target_alignment, &dst_d->begin, &dst_end);
                dst_d->size = 1;
                // Copy a single input to the newly allocated element
                opchild(dst_d->begin, src, echild);
            } else {
                // We're broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                intptr_t size = dst_d->size, dst_stride = e->dst_md->stride;
                for (intptr_t i = 0; i < size; ++i, dst += dst_stride) {
                    opchild(dst, src, echild);
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
} // anonymous namespace

size_t dynd::make_broadcast_to_var_dim_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_var_dim_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (dst_var_dim_dt.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided destination dtype " << dst_var_dim_dt << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_dtype *dst_vad = static_cast<const var_dim_dtype *>(dst_var_dim_dt.extended());

    out->ensure_capacity(offset_out + sizeof(broadcast_to_var_assign_kernel_extra));
    broadcast_to_var_assign_kernel_extra *e = out->get_at<broadcast_to_var_assign_kernel_extra>(offset_out);
    const var_dim_dtype_metadata *dst_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(dst_metadata);
    e->base.set_function<unary_single_operation_t>(&broadcast_to_var_assign_kernel_extra::single);
    e->base.destructor = &broadcast_to_var_assign_kernel_extra::destruct;
    e->dst_target_alignment = dst_vad->get_element_dtype().get_alignment();
    e->dst_md = dst_md;
    return ::make_assignment_kernel(out, offset_out + sizeof(broadcast_to_var_assign_kernel_extra),
                    dst_vad->get_element_dtype(), dst_metadata + sizeof(var_dim_dtype_metadata),
                    src_dt, src_metadata,
                    errmode, ectx);
}

/////////////////////////////////////////
// var array to var array assignment

namespace {
    struct var_assign_kernel_extra {
        typedef var_assign_kernel_extra extra_type;

        kernel_data_prefix base;
        intptr_t dst_target_alignment;
        const var_dim_dtype_metadata *dst_md, *src_md;

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            var_dim_dtype_data *dst_d = reinterpret_cast<var_dim_dtype_data *>(dst);
            const var_dim_dtype_data *src_d = reinterpret_cast<const var_dim_dtype_data *>(src);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                // As a special case, allow uninitialized -> uninitialized assignment as a no-op
                if (src_d->begin != NULL) {
                    intptr_t dim_size = src_d->size;
                    intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_md->stride;
                    // If we're writing to an empty array, have to allocate the output
                    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                    // Allocate the output array data
                    char *dst_end = NULL;
                    allocator->allocate(e->dst_md->blockref, dim_size * dst_stride,
                                e->dst_target_alignment, &dst_d->begin, &dst_end);
                    dst_d->size = dim_size;
                    // Copy to the newly allocated element
                    dst = dst_d->begin;
                    src = src_d->begin + e->src_md->offset;
                    for (intptr_t i = 0; i < dim_size; ++i, dst += dst_stride, src += src_stride) {
                        opchild(dst, src, echild);
                    }
                }
            } else {
                if (src_d->begin == NULL) {
                    throw runtime_error("Cannot assign an uninitialized dynd var_dim to an initialized one");
                }
                intptr_t dst_dim_size = dst_d->size, src_dim_size = src_d->size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = src_dim_size != 1 ? e->src_md->stride : 0;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input var_dim sized " << src_dim_size << " to output var_dim sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                src = src_d->begin + e->src_md->offset;
                for (intptr_t i = 0; i < dst_dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
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
} // anonymous namespace

size_t dynd::make_var_dim_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_var_dim_dt, const char *dst_metadata,
                const dtype& src_var_dim_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (dst_var_dim_dt.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided destination dtype " << dst_var_dim_dt << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    if (src_var_dim_dt.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_broadcast_to_blockref_array_assignment_kernel: provided source dtype " << src_var_dim_dt << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_dtype *dst_vad = static_cast<const var_dim_dtype *>(dst_var_dim_dt.extended());
    const var_dim_dtype *src_vad = static_cast<const var_dim_dtype *>(src_var_dim_dt.extended());

    out->ensure_capacity(offset_out + sizeof(var_assign_kernel_extra));
    const var_dim_dtype_metadata *dst_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(dst_metadata);
    const var_dim_dtype_metadata *src_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(src_metadata);
    var_assign_kernel_extra *e = out->get_at<var_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&var_assign_kernel_extra::single);
    e->base.destructor = &var_assign_kernel_extra::destruct;
    e->dst_target_alignment = dst_vad->get_element_dtype().get_alignment();
    e->dst_md = dst_md;
    e->src_md = src_md;
    return ::make_assignment_kernel(out, offset_out + sizeof(var_assign_kernel_extra),
                    dst_vad->get_element_dtype(), dst_metadata + sizeof(var_dim_dtype_metadata),
                    src_vad->get_element_dtype(), src_metadata + sizeof(var_dim_dtype_metadata),
                    errmode, ectx);
}

/////////////////////////////////////////
// strided array to var array assignment

namespace {
    struct strided_to_var_assign_kernel_extra {
        typedef strided_to_var_assign_kernel_extra extra_type;

        kernel_data_prefix base;
        intptr_t dst_target_alignment;
        const var_dim_dtype_metadata *dst_md;
        intptr_t src_stride, src_dim_size;

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            var_dim_dtype_data *dst_d = reinterpret_cast<var_dim_dtype_data *>(dst);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                intptr_t dim_size = e->src_dim_size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_stride;
                // If we're writing to an empty array, have to allocate the output
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                // Allocate the output array data
                char *dst_end = NULL;
                allocator->allocate(e->dst_md->blockref, dim_size * dst_stride,
                            e->dst_target_alignment, &dst_d->begin, &dst_end);
                dst_d->size = dim_size;
                // Copy to the newly allocated element
                dst = dst_d->begin;
                for (intptr_t i = 0; i < dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
                }
            } else {
                intptr_t dst_dim_size = dst_d->size, src_dim_size = e->src_dim_size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_stride;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input strided array sized " << src_dim_size << " to output var_dim sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                for (intptr_t i = 0; i < dst_dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
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
} // anonymous namespace

size_t dynd::make_strided_to_var_dim_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_var_dim_dt, const char *dst_metadata,
                const dtype& src_strided_dim_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (dst_var_dim_dt.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_strided_to_var_dim_assignment_kernel: provided destination dtype " << dst_var_dim_dt << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_dtype *dst_vad = static_cast<const var_dim_dtype *>(dst_var_dim_dt.extended());

    out->ensure_capacity(offset_out + sizeof(strided_to_var_assign_kernel_extra));
    const var_dim_dtype_metadata *dst_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(dst_metadata);
    strided_to_var_assign_kernel_extra *e = out->get_at<strided_to_var_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&strided_to_var_assign_kernel_extra::single);
    e->base.destructor = &strided_to_var_assign_kernel_extra::destruct;
    e->dst_target_alignment = dst_vad->get_element_dtype().get_alignment();
    e->dst_md = dst_md;

    dtype src_element_dt;
    const char *src_element_metadata;
    if (src_strided_dim_dt.get_type_id() == strided_dim_type_id) {
        const strided_dim_dtype *src_sad = static_cast<const strided_dim_dtype *>(src_strided_dim_dt.extended());
        const strided_dim_dtype_metadata *src_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata);
        e->src_stride = src_md->stride;
        e->src_dim_size = src_md->size;
        src_element_dt = src_sad->get_element_dtype();
        src_element_metadata = src_metadata + sizeof(strided_dim_dtype_metadata);
    } else if (src_strided_dim_dt.get_type_id() == fixed_dim_type_id) {
        const fixed_dim_dtype *src_fad = static_cast<const fixed_dim_dtype *>(src_strided_dim_dt.extended());
        e->src_stride = src_fad->get_fixed_stride();
        e->src_dim_size = src_fad->get_fixed_dim_size();
        src_element_dt = src_fad->get_element_dtype();
        src_element_metadata = src_metadata;
    } else {
        stringstream ss;
        ss << "make_strided_to_var_dim_assignment_kernel: provided source dtype " << src_strided_dim_dt << " is not a strided_dim or fixed_array";
        throw runtime_error(ss.str());
    }

    return ::make_assignment_kernel(out, offset_out + sizeof(strided_to_var_assign_kernel_extra),
                    dst_vad->get_element_dtype(), dst_metadata + sizeof(var_dim_dtype_metadata),
                    src_element_dt, src_element_metadata,
                    errmode, ectx);
}

/////////////////////////////////////////
// var array to strided array assignment

namespace {
    struct var_to_strided_assign_kernel_extra {
        typedef var_to_strided_assign_kernel_extra extra_type;

        kernel_data_prefix base;
        intptr_t dst_stride, dst_dim_size;
        const var_dim_dtype_metadata *src_md;

        static void single(char *dst, const char *src,
                            kernel_data_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const var_dim_dtype_data *src_d = reinterpret_cast<const var_dim_dtype_data *>(src);
            kernel_data_prefix *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (src_d->begin == NULL) {
                throw runtime_error("Cannot assign an uninitialized dynd var array to a strided one");
            }

            intptr_t dst_dim_size = e->dst_dim_size, src_dim_size = src_d->size;
            intptr_t dst_stride = e->dst_stride, src_stride = src_dim_size != 1 ? e->src_md->stride : 0;
            // Check for a broadcasting error
            if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                stringstream ss;
                ss << "error broadcasting input var array sized " << src_dim_size << " to output strided array sized " << dst_dim_size;
                throw broadcast_error(ss.str());
            }
            // Copying/broadcasting elements
            src = src_d->begin + e->src_md->offset;
            for (intptr_t i = 0; i < dst_dim_size; ++i, dst += dst_stride, src += src_stride) {
                opchild(dst, src, echild);
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
} // anonymous namespace

size_t dynd::make_var_to_strided_dim_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_strided_dim_dt, const char *dst_metadata,
                const dtype& src_var_dim_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx)
{
    if (src_var_dim_dt.get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "make_var_to_strided_dim_assignment_kernel: provided source dtype " << src_var_dim_dt << " is not a var_dim";
        throw runtime_error(ss.str());
    }
    const var_dim_dtype *src_vad = static_cast<const var_dim_dtype *>(src_var_dim_dt.extended());

    out->ensure_capacity(offset_out + sizeof(var_to_strided_assign_kernel_extra));
    const var_dim_dtype_metadata *src_md =
                    reinterpret_cast<const var_dim_dtype_metadata *>(src_metadata);
    var_to_strided_assign_kernel_extra *e = out->get_at<var_to_strided_assign_kernel_extra>(offset_out);
    e->base.set_function<unary_single_operation_t>(&var_to_strided_assign_kernel_extra::single);
    e->base.destructor = &var_to_strided_assign_kernel_extra::destruct;

    dtype dst_element_dt;
    const char *dst_element_metadata;
    if (dst_strided_dim_dt.get_type_id() == strided_dim_type_id) {
        const strided_dim_dtype *dst_sad = static_cast<const strided_dim_dtype *>(dst_strided_dim_dt.extended());
        const strided_dim_dtype_metadata *dst_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(dst_metadata);
        e->dst_stride = dst_md->stride;
        e->dst_dim_size = dst_md->size;
        dst_element_dt = dst_sad->get_element_dtype();
        dst_element_metadata = dst_metadata + sizeof(strided_dim_dtype_metadata);
    } else if (dst_strided_dim_dt.get_type_id() == fixed_dim_type_id) {
        const fixed_dim_dtype *dst_fad = static_cast<const fixed_dim_dtype *>(dst_strided_dim_dt.extended());
        e->dst_stride = dst_fad->get_fixed_stride();
        e->dst_dim_size = dst_fad->get_fixed_dim_size();
        dst_element_dt = dst_fad->get_element_dtype();
        dst_element_metadata = dst_metadata;
    } else {
        stringstream ss;
        ss << "make_var_to_strided_dim_assignment_kernel: provided destination dtype " << dst_strided_dim_dt << " is not a strided_dim or fixed_array";
        throw runtime_error(ss.str());
    }

    e->src_md = src_md;
    return ::make_assignment_kernel(out, offset_out + sizeof(var_to_strided_assign_kernel_extra),
                    dst_element_dt, dst_element_metadata,
                    src_vad->get_element_dtype(), src_metadata + sizeof(var_dim_dtype_metadata),
                    errmode, ectx);
}
