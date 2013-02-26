//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>

using namespace std;
using namespace dynd;

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
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t DYND_UNUSED(src_count), const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    size_t undim = dst_dt.get_undim();
    const char *dst_child_metadata;
    const char *src_child_metadata[N];
    dtype dst_child_dt;
    dtype src_child_dt[N];
 
    out->ensure_capacity(offset_out + sizeof(strided_expr_kernel_extra<N>));
    strided_expr_kernel_extra<N> *e = out->get_at<strided_expr_kernel_extra<N> >(offset_out);
    switch (kernreq) {
        case kernel_request_single:
            e->base.set_function<expr_single_operation_t>(&strided_expr_kernel_extra<N>::single);
            break;
        case kernel_request_strided:
            e->base.set_function<expr_strided_operation_t>(&strided_expr_kernel_extra<N>::strided);
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
        dst_child_dt = sdd->get_element_dtype();
    } else {
        const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(dst_dt.extended());
        e->size = fdd->get_fixed_dim_size();
        e->dst_stride = fdd->get_fixed_stride();
        dst_child_metadata = dst_metadata;
        dst_child_dt = fdd->get_element_dtype();
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
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = src_md->stride;
            src_child_metadata[i] = src_metadata[i] + sizeof(strided_dim_dtype_metadata);
            src_child_dt[i] = sdd->get_element_dtype();
        } else {
            const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(src_dt[i].extended());
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride[i] = fdd->get_fixed_stride();
            src_child_metadata[i] = src_metadata[i];
            src_child_dt[i] = fdd->get_element_dtype();
        }
    }
    return elwise_handler->make_expr_kernel(
                    out, offset_out + sizeof(strided_expr_kernel_extra<N>),
                    dst_child_dt, dst_child_metadata,
                    N, src_child_dt, src_child_metadata,
                    kernel_request_strided, ectx);
}

inline static size_t make_elwise_strided_dimension_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
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

static size_t make_elwise_strided_or_var_to_strided_dimension_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    throw runtime_error("TODO: make_elwise_strided_or_var_to_strided_dimension_expr_kernel");
}

static size_t make_elwise_strided_to_var_dimension_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    throw runtime_error("TODO: make_elwise_strided_to_var_dimension_expr_kernel");
}

static size_t make_elwise_strided_or_var_to_var_dimension_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *elwise_handler)
{
    throw runtime_error("TODO: make_elwise_strided_or_var_to_var_dimension_expr_kernel");
}

size_t dynd::make_elwise_dimension_expr_kernel(assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
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
                src_all_strided_or_var = false;
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
            if (src_all_strided) {
                return make_elwise_strided_to_var_dimension_expr_kernel(
                                out, offset_out,
                                dst_dt, dst_metadata,
                                src_count, src_dt, src_metadata,
                                kernreq, ectx,
                                elwise_handler);
            } else if (src_all_strided_or_var) {
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
