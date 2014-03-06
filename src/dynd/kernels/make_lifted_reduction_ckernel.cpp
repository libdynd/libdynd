//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;

namespace {

struct ckernel_reduction_prefix {
    // The function pointer exposed through the ckernel_prefix is
    // for the "first call" of the function on a given destination
    // data address.
    ckernel_prefix ckpbase;
    // This function pointer is for all the calls of the function
    // on a given destination data address after the "first call".
    unary_strided_operation_t followup_call_function;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    template<typename T>
    T get_first_call_function() const {
        return ckpbase.get_function<T>();
    }

    template<typename T>
    void set_first_call_function(T fnptr) {
        ckpbase.set_function<T>(fnptr);
    }

    unary_strided_operation_t get_followup_call_function() const {
        return followup_call_function;
    }

    void set_followup_call_function(unary_strided_operation_t fnptr) {
        followup_call_function = fnptr;
    }

};

/**
 * STRIDED INITIAL REDUCTION DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a reduction dimension, so dst_stride is zero.
 *  - It's an initial dimension, there are additional dimensions
 *    being processed by its child kernels.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child first_call function must be *single*.
 *  - The child followup_call function must be *strided*.
 * 
 */
struct strided_initial_reduction_kernel_extra {
    typedef strided_initial_reduction_kernel_extra extra_type;

    ckernel_reduction_prefix ckpbase;
    // The code assumes that size >= 1
    intptr_t size;
    intptr_t src_stride;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        // The first call at the "dst" address
        unary_single_operation_t opchild_first_call = echild->get_first_call_function<unary_single_operation_t>();
        opchild_first_call(dst, src, &echild->base());
        if (e->size > 1) {
            // All the followup calls at the "dst" address
            unary_strided_operation_t opchild = echild->get_followup_call_function();
            opchild(dst, 0, src + e->src_stride, e->src_stride,
                    e->size - 1, &echild->base());
        }
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        unary_strided_operation_t opchild_followup_call = echild->get_followup_call_function();
        unary_single_operation_t opchild_first_call = echild->get_first_call_function<unary_single_operation_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        if (dst_stride == 0) {
            // With a zero stride, we have one "first", followed by many "followup" calls
            opchild_first_call(dst, src, &echild->base());
            if (inner_size > 1) {
                opchild_followup_call(dst, 0, src + inner_src_stride, inner_src_stride,
                        inner_size - 1, &echild->base());
            }
            src += src_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_followup_call(dst, 0, src, inner_src_stride,
                        inner_size, &echild->base());
                src += src_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop is "first"
            for (size_t i = 0; i != count; ++i) {
                opchild_first_call(dst, src, &echild->base());
                if (inner_size > 1) {
                    opchild_followup_call(dst, 0,
                            src + inner_src_stride, inner_src_stride,
                            inner_size - 1, &echild->base());
                }
                dst += dst_stride;
                src += src_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        unary_strided_operation_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        for (size_t i = 0; i != count; ++i) {
            opchild_followup_call(dst, 0, src, inner_src_stride, inner_size, &echild->base());
            dst += dst_stride;
            src += src_stride;
        }
    }

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base();
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

/**
 * STRIDED INITIAL BROADCAST DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a broadcast dimension, so dst_stride is not zero.
 *  - It's an initial dimension, there are additional dimensions
 *    being processed after this one.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child first_call function must be *strided*.
 *  - The child followup_call function must be *strided*.
 * 
 */
struct strided_initial_broadcast_kernel_extra {
    typedef strided_initial_broadcast_kernel_extra extra_type;

    ckernel_reduction_prefix ckpbase;
    // The code assumes that size >= 1
    intptr_t size;
    intptr_t dst_stride, src_stride;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        unary_strided_operation_t opchild_first_call = echild->get_first_call_function<unary_strided_operation_t>();
        // Because e->dst_stride != 0, all the calls are "first"
        opchild_first_call(dst, e->dst_stride, src, e->src_stride, e->size, &echild->base());
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        unary_strided_operation_t opchild_first_call = echild->get_first_call_function<unary_strided_operation_t>();
        unary_strided_operation_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        if (dst_stride == 0) {
            // With a zero stride, we have one "first", followed by many "followup" calls
            opchild_first_call(dst, inner_dst_stride, src, inner_src_stride,
                        inner_size, &echild->base());
            dst += dst_stride;
            src += src_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_followup_call(dst, inner_dst_stride, src, inner_src_stride,
                        inner_size, &echild->base());
                dst += dst_stride;
                src += src_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop is "first"
            for (size_t i = 0; i != count; ++i) {
                opchild_first_call(dst, inner_dst_stride, src, inner_src_stride,
                        inner_size, &echild->base());
                dst += dst_stride;
                src += src_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild = &(e + 1)->ckpbase;
        unary_strided_operation_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        for (size_t i = 0; i != count; ++i) {
            opchild_followup_call(dst, 0, src, inner_src_stride, inner_size, &echild->base());
            dst += dst_stride;
            src += src_stride;
        }
    }

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base();
        if (echild->destructor) {
            echild->destructor(echild);
        }
    }
};

/**
 * STRIDED INNER REDUCTION DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a reduction dimension, so dst_stride is zero.
 *  - It's an inner dimension, calling the reduction kernel directly.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child destination initialization kernel must be *single*.
 *  - The child reduction kernel must be *strided*.
 * 
 */
struct strided_inner_reduction_kernel_extra {
    typedef strided_inner_reduction_kernel_extra extra_type;

    ckernel_reduction_prefix ckpbase;
    // The code assumes that size >= 1
    intptr_t size;
    intptr_t src_stride;
    size_t dst_init_kernel_offset;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = &(e + 1)->base();
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        // The first call to initialize the "dst" value
        unary_single_operation_t opchild_dst_init = echild_dst_init->get_function<unary_single_operation_t>();
        unary_strided_operation_t opchild_reduce = echild_reduce->get_function<unary_strided_operation_t>();
        opchild_dst_init(dst, src, echild_dst_init);
        if (e->size > 1) {
            // All the followup calls to accumulate at the "dst" address
            opchild_reduce(dst, 0, src + e->src_stride, e->src_stride,
                    e->size - 1, &echild_reduce->base());
        }
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = &(e + 1)->base();
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        unary_single_operation_t opchild_dst_init = echild_dst_init->get_function<unary_single_operation_t>();
        unary_strided_operation_t opchild_reduce = echild_reduce->get_function<unary_strided_operation_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many accumulations
            opchild_dst_init(dst, src, echild_dst_init);
            if (inner_size > 1) {
                opchild_reduce(dst, 0, src + inner_src_stride, inner_src_stride,
                        inner_size - 1, &echild_reduce->base());
            }
            src += src_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, 0, src, inner_src_stride,
                        inner_size, &echild_reduce->base());
                dst += dst_stride;
                src += src_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop has to
            // initialize then reduce
            for (size_t i = 0; i != count; ++i) {
                opchild_dst_init(dst, src, echild_dst_init);
                if (inner_size > 1) {
                    opchild_reduce(dst, 0,
                            src + inner_src_stride, inner_src_stride,
                            inner_size - 1, &echild_reduce->base());
                }
                dst += dst_stride;
                src += src_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = &(e + 1)->base();
        // No initialization, all reduction
        unary_strided_operation_t opchild_reduce = echild_reduce->get_function<unary_strided_operation_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        for (size_t i = 0; i != count; ++i) {
            opchild_reduce(dst, 0, src, inner_src_stride, inner_size, &echild_reduce->base());
            dst += dst_stride;
            src += src_stride;
        }
    }

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        // The reduction kernel
        ckernel_prefix *echild = &(e + 1)->base();
        if (echild->destructor) {
            echild->destructor(echild);
        }
        // The destination initialization kernel
        if (e->dst_init_kernel_offset != 0) {
            echild = reinterpret_cast<ckernel_prefix *>(
                        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    }
};

/**
 * STRIDED INNER BROADCAST DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a broadcast dimension, so dst_stride is not zero.
 *  - It's an inner dimension, calling the reduction kernel directly.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child reduction kernel must be *strided*.
 *  - The child destination initialization kernel must be *strided*.
 * 
 */
struct strided_inner_broadcast_kernel_extra {
    typedef strided_inner_broadcast_kernel_extra extra_type;

    ckernel_reduction_prefix ckpbase;
    // The code assumes that size >= 1
    intptr_t size;
    intptr_t dst_stride, src_stride;
    size_t dst_init_kernel_offset;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *src,
                    ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild = &(e + 1)->base();
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        // All we do is initialize the dst values
        unary_strided_operation_t opchild_dst_init = echild_dst_init->get_function<unary_strided_operation_t>();
        opchild_dst_init(dst, e->dst_stride, src, e->src_stride, e->size, echild_dst_init);
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = &(e + 1)->base();
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        unary_strided_operation_t opchild_dst_init = echild_dst_init->get_function<unary_strided_operation_t>();
        unary_strided_operation_t opchild_reduce = echild_reduce->get_function<unary_strided_operation_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many accumulations
            opchild_dst_init(dst, e->dst_stride, src, e->src_stride, e->size, echild_dst_init);
            dst += dst_stride;
            src += src_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, 0, src, inner_src_stride,
                        inner_size, &echild_reduce->base());
                dst += dst_stride;
                src += src_stride;
            }
        } else {
            // With a non-zero stride, every iteration is an initialization
            for (size_t i = 0; i != count; ++i) {
                opchild_dst_init(dst, e->dst_stride, src, e->src_stride,
                            e->size, echild_dst_init);
                dst += dst_stride;
                src += src_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = &(e + 1)->base();
        // No initialization, all reduction
        unary_strided_operation_t opchild_reduce = echild_reduce->get_function<unary_strided_operation_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        for (size_t i = 0; i != count; ++i) {
            opchild_reduce(dst, 0, src, inner_src_stride, inner_size, &echild_reduce->base());
            dst += dst_stride;
            src += src_stride;
        }
    }

    static void destruct(ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        // The reduction kernel
        ckernel_prefix *echild = &(e + 1)->base();
        if (echild->destructor) {
            echild->destructor(echild);
        }
        // The destination initialization kernel
        if (e->dst_init_kernel_offset != 0) {
            echild = reinterpret_cast<ckernel_prefix *>(
                        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    }
};

} // anonymous namespace

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_reduction_dimension_kernel(
            ckernel_builder *out_ckb, size_t ckb_offset,
            intptr_t src_stride, intptr_t src_size,
            kernel_request_t kernreq)
{
    intptr_t ckb_end = ckb_offset + sizeof(strided_initial_reduction_kernel_extra);
    out_ckb->ensure_capacity(ckb_end);
    strided_initial_reduction_kernel_extra *e = out_ckb->get_at<strided_initial_reduction_kernel_extra>(ckb_offset);
    e->base().destructor = &strided_initial_reduction_kernel_extra::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
        e->ckpbase.set_first_call_function(&strided_initial_reduction_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
        e->ckpbase.set_first_call_function(&strided_initial_reduction_kernel_extra::strided_first);
    } else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->ckpbase.set_followup_call_function(&strided_initial_reduction_kernel_extra::strided_followup);
    // The striding parameters
    e->src_stride = src_stride;
    e->size = src_size;
    return ckb_end;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_broadcast_dimension_kernel(
            ckernel_builder *out_ckb, size_t ckb_offset,
            intptr_t dst_stride, intptr_t src_stride, intptr_t src_size,
            kernel_request_t kernreq)
{
    intptr_t ckb_end = ckb_offset + sizeof(strided_initial_broadcast_kernel_extra);
    out_ckb->ensure_capacity(ckb_end);
    strided_initial_broadcast_kernel_extra *e = out_ckb->get_at<strided_initial_broadcast_kernel_extra>(ckb_offset);
    e->base().destructor = &strided_initial_broadcast_kernel_extra::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
        e->ckpbase.set_first_call_function(&strided_initial_broadcast_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
        e->ckpbase.set_first_call_function(&strided_initial_broadcast_kernel_extra::strided_first);
    } else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->ckpbase.set_followup_call_function(&strided_initial_broadcast_kernel_extra::strided_followup);
    // The striding parameters
    e->dst_stride = dst_stride;
    e->src_stride = src_stride;
    e->size = src_size;
    return ckb_end;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is
 * the final dimension before the accumulation operation.
 *
 * If dst_initialization is NULL, an assignment kernel is used.
 */
static size_t make_strided_inner_reduction_dimension_kernel(
            const ckernel_deferred *elwise_reduction,
            const ckernel_deferred *dst_initialization,
            ckernel_builder *out_ckb, size_t ckb_offset,
            intptr_t src_stride, intptr_t src_size,
            const ndt::type& dst_tp, const char *dst_meta,
            const ndt::type& src_tp, const char *src_meta,
            kernel_request_t kernreq)
{
    intptr_t ckb_end = ckb_offset + sizeof(strided_inner_reduction_kernel_extra);
    out_ckb->ensure_capacity(ckb_end);
    strided_inner_reduction_kernel_extra *e = out_ckb->get_at<strided_inner_reduction_kernel_extra>(ckb_offset);
    e->base().destructor = &strided_inner_reduction_kernel_extra::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
        e->ckpbase.set_first_call_function(&strided_inner_reduction_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
        e->ckpbase.set_first_call_function(&strided_inner_reduction_kernel_extra::strided_first);
    } else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->ckpbase.set_followup_call_function(&strided_inner_reduction_kernel_extra::strided_followup);
    // The striding parameters
    e->src_stride = src_stride;
    e->size = src_size;
    // Validate that the provided deferred_ckernels are unary operations,
    // and have the correct types
    if (elwise_reduction->ckernel_funcproto != unary_operation_funcproto) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "funcproto must be unary, not " << elwise_reduction->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
    if (elwise_reduction->data_dynd_types[0] != dst_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "dst type is " << elwise_reduction->data_dynd_types[0];
        ss << ", expected " << dst_tp;
        throw type_error(ss.str());
    }
    if (elwise_reduction->data_dynd_types[1] != src_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "src type is " << elwise_reduction->data_dynd_types[0];
        ss << ", expected " << src_tp;
        throw type_error(ss.str());
    }
    if (dst_initialization != NULL) {
        if (dst_initialization->ckernel_funcproto != unary_operation_funcproto) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "funcproto must be unary, not " << dst_initialization->ckernel_funcproto;
            throw runtime_error(ss.str());
        }
        if (dst_initialization->data_dynd_types[0] != dst_tp) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "dst type is " << dst_initialization->data_dynd_types[0];
            ss << ", expected " << dst_tp;
            throw type_error(ss.str());
        }
        if (dst_initialization->data_dynd_types[1] != src_tp) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "src type is " << dst_initialization->data_dynd_types[0];
            ss << ", expected " << src_tp;
            throw type_error(ss.str());
        }
    }
    const char *child_ckernel_meta[2] = {dst_meta, src_meta};
    ckb_end = elwise_reduction->instantiate_func(elwise_reduction->data_ptr,
                    out_ckb, ckb_end, child_ckernel_meta, kernel_request_strided);
    e->dst_init_kernel_offset = ckb_end - ckb_offset;
    if (dst_initialization != NULL) {
        ckb_end = dst_initialization->instantiate_func(dst_initialization->data_ptr,
                        out_ckb, ckb_end, child_ckernel_meta, kernel_request_single);
    } else {
        ckb_end = make_assignment_kernel(out_ckb, ckb_end,
                        dst_tp, dst_meta, src_tp, src_meta, kernel_request_single,
                        assign_error_default, &eval::default_eval_context);
    }

    return ckb_end;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_inner_broadcast_dimension_kernel(
            const ckernel_deferred *elwise_reduction,
            const ckernel_deferred *dst_initialization,
            ckernel_builder *out_ckb, size_t ckb_offset,
            intptr_t dst_stride, intptr_t src_stride, intptr_t src_size,
            const ndt::type& dst_tp, const char *dst_meta,
            const ndt::type& src_tp, const char *src_meta,
            kernel_request_t kernreq)
{
    intptr_t ckb_end = ckb_offset + sizeof(strided_inner_broadcast_kernel_extra);
    out_ckb->ensure_capacity(ckb_end);
    strided_inner_broadcast_kernel_extra *e = out_ckb->get_at<strided_inner_broadcast_kernel_extra>(ckb_offset);
    e->base().destructor = &strided_inner_broadcast_kernel_extra::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
        e->ckpbase.set_first_call_function(&strided_inner_broadcast_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
        e->ckpbase.set_first_call_function(&strided_inner_broadcast_kernel_extra::strided_first);
    } else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->ckpbase.set_followup_call_function(&strided_inner_broadcast_kernel_extra::strided_followup);
    // The striding parameters
    e->dst_stride = dst_stride;
    e->src_stride = src_stride;
    e->size = src_size;
    // Validate that the provided deferred_ckernels are unary operations,
    // and have the correct types
    if (elwise_reduction->ckernel_funcproto != unary_operation_funcproto) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "funcproto must be unary, not " << elwise_reduction->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
    if (elwise_reduction->data_dynd_types[0] != dst_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "dst type is " << elwise_reduction->data_dynd_types[0];
        ss << ", expected " << dst_tp;
        throw type_error(ss.str());
    }
    if (elwise_reduction->data_dynd_types[1] != src_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
        ss << "src type is " << elwise_reduction->data_dynd_types[0];
        ss << ", expected " << src_tp;
        throw type_error(ss.str());
    }
    if (dst_initialization != NULL) {
        if (dst_initialization->ckernel_funcproto != unary_operation_funcproto) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "funcproto must be unary, not " << dst_initialization->ckernel_funcproto;
            throw runtime_error(ss.str());
        }
        if (dst_initialization->data_dynd_types[0] != dst_tp) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "dst type is " << dst_initialization->data_dynd_types[0];
            ss << ", expected " << dst_tp;
            throw type_error(ss.str());
        }
        if (dst_initialization->data_dynd_types[1] != src_tp) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
            ss << "src type is " << dst_initialization->data_dynd_types[0];
            ss << ", expected " << src_tp;
            throw type_error(ss.str());
        }
    }
    const char *child_ckernel_meta[2] = {dst_meta, src_meta};
    ckb_end = elwise_reduction->instantiate_func(elwise_reduction->data_ptr,
                    out_ckb, ckb_end, child_ckernel_meta, kernel_request_strided);
    e->dst_init_kernel_offset = ckb_end - ckb_offset;
    if (dst_initialization != NULL) {
        ckb_end = dst_initialization->instantiate_func(dst_initialization->data_ptr,
                        out_ckb, ckb_end, child_ckernel_meta, kernel_request_single);
    } else {
        ckb_end = make_assignment_kernel(out_ckb, ckb_end,
                        dst_tp, dst_meta, src_tp, src_meta, kernel_request_single,
                        assign_error_default, &eval::default_eval_context);
    }

    return ckb_end;
}

size_t dynd::make_lifted_reduction_ckernel(
                const ckernel_deferred *elwise_reduction,
                const ckernel_deferred *dst_initialization,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const ndt::type *lifted_types,
                const char *const* dynd_metadata,
                intptr_t reduction_ndim,
                const bool *reduction_dimflags,
                bool associative,
                bool commutative,
                const nd::array& reduction_identity,
                dynd::kernel_request_t kernreq)
{
    if (!reduction_identity.is_empty()) {
        throw runtime_error("TODO: implement reduction_identity");
    }

    const ndt::type& dst_el_tp = elwise_reduction->data_dynd_types[0];
    const ndt::type& src_el_tp = elwise_reduction->data_dynd_types[1];
    ndt::type dst_tp = lifted_types[0], src_tp = lifted_types[1];

    // Count the number of dimensions being reduced
    intptr_t reducedim_count = 0;
    for (intptr_t i = 0; i < reduction_ndim; ++i) {
        reducedim_count += reduction_dimflags[i];
    }
    if (reducedim_count == 0) {
        throw runtime_error("make_lifted_reduction_ckernel: no dimensions were flagged for reduction");
    }

    // This is the number of dimensions being processed by the reduction
    if (reduction_ndim != src_tp.get_ndim() - src_el_tp.get_ndim()) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: wrong number of reduction dimensions, ";
        ss << "requested " << reduction_ndim << ", but types have ";
        ss << (src_tp.get_ndim() - src_el_tp.get_ndim());
        throw runtime_error(ss.str());
    }
    // Determine whether reduced dimensions are being kept or not
    bool keep_dims;
    if (reduction_ndim == dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
        keep_dims = true;
    } else if (reduction_ndim - reducedim_count == dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
        keep_dims = false;
    } else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: The number of dimensions flagged for reduction, ";
        ss << reducedim_count << ", is not consistent with the destination type";
        throw runtime_error(ss.str());
    }

    const char *dst_meta = dynd_metadata[0];
    const char *src_meta = dynd_metadata[1];
    for (intptr_t i = 0; i < reduction_ndim; ++i) {
        intptr_t dst_stride, dst_size, src_stride, src_size;
        // Get the striding parameters for the source dimension
        switch (src_tp.get_type_id()) {
            case fixed_dim_type_id: {
                const fixed_dim_type *fdt = static_cast<const fixed_dim_type *>(src_tp.extended());
                src_stride = fdt->get_fixed_stride();
                src_size = fdt->get_fixed_dim_size();
                src_tp = fdt->get_element_type();
                break;
            }
            case strided_dim_type_id: {
                const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(src_meta);
                src_stride = md->stride;
                src_size = md->size;
                break;
            }
            default: {
                stringstream ss;
                ss << "make_lifted_reduction_ckernel: type " << src_tp << " not supported as source";
                throw type_error(ss.str());
            }
        }
        if (reduction_dimflags[i]) {
            // This dimension is being reduced
            if (keep_dims) {
                // If the dimensions are being kept, the output should be a
                // a strided dimension of size one
                switch (dst_tp.get_type_id) {
                    case fixed_dim_type_id: {
                        const fixed_dim_type *fdt = static_cast<const fixed_dim_type *>(dst_tp.extended());
                        if (fdt->get_fixed_dim_size() != 1 || fdt->get_fixed_stride() != 0) {
                            stringstream ss;
                            ss << "make_lifted_reduction_ckernel: destination of a reduction dimension ";
                            ss << "must have size 1, not " << dst_tp;
                            throw type_error(ss.str());
                        }
                        dst_tp = fdt->get_element_type();
                        break;
                    }
                    case strided_dim_type_id: {
                        const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(dst_meta);
                        const strided_dim_type *sdt = static_cast<const strided_dim_type *>(dst_tp.extended());
                        if (md->size != 1 || md->stride != 0) {
                            stringstream ss;
                            ss << "make_lifted_reduction_ckernel: destination of a reduction dimension ";
                            ss << "must have size 1, not size" << md->size << "/stride " << md->stride;
                            ss << " in type " << dst_tp;
                            throw type_error(ss.str());
                        }
                        dst_tp = sdt->get_element_type();
                        break;
                    }
                    default: {
                        stringstream ss;
                        ss << "make_lifted_reduction_ckernel: type " << dst_tp;
                        ss << " not supported the destination of a dimension being reduced";
                        throw type_error(ss.str());
                    }
                }
            }
            dst_stride = 0;
            dst_size = 1;
        } else {
            switch (dst_tp.get_type_id()) {
                case fixed_dim_type_id: {
                    const fixed_dim_type *fdt = static_cast<const fixed_dim_type *>(dst_tp.extended());
                    dst_stride = fdt->get_fixed_stride();
                    dst_size = fdt->get_fixed_dim_size();
                    dst_tp = fdt->get_element_type();
                    break;
                }
                case strided_dim_type_id: {
                    const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(dst_meta);
                    const strided_dim_type *sdt = static_cast<const strided_dim_type *>(dst_tp.extended());
                    dst_stride = md->stride;
                    dst_size = md->size;
                    dst_tp = sdt->get_element_type();
                    break;
                }
                default: {
                    stringstream ss;
                    ss << "make_lifted_reduction_ckernel: type " << dst_tp << " not supported as destination";
                    throw type_error(ss.str());
                }
            }
        }
    }
}
 
