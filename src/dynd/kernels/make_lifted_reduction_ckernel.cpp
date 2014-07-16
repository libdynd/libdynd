//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>

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
    expr_strided_t followup_call_function;

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

    expr_strided_t get_followup_call_function() const {
        return followup_call_function;
    }

    void set_followup_call_function(expr_strided_t fnptr) {
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

    static void single_first(char *dst, const char *const *src,
                             ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        // The first call at the "dst" address
        expr_single_t opchild_first_call = echild->get_first_call_function<expr_single_t>();
        opchild_first_call(dst, src, &echild->base());
        if (e->size > 1) {
            // All the followup calls at the "dst" address
            expr_strided_t opchild = echild->get_followup_call_function();
            const char *src_second = src[0] + e->src_stride;
            opchild(dst, 0, &src_second, &e->src_stride, e->size - 1,
                    &echild->base());
        }
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                              const char *const *src,
                              const intptr_t *src_stride, size_t count,
                              ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        expr_strided_t opchild_followup_call = echild->get_followup_call_function();
        expr_single_t opchild_first_call = echild->get_first_call_function<expr_single_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we have one "first", followed by many "followup" calls
            opchild_first_call(dst, &src0, &echild->base());
            if (inner_size > 1) {
                const char *inner_src_second = src0 + inner_src_stride;
                opchild_followup_call(dst, 0, &inner_src_second,
                                      &inner_src_stride, inner_size - 1,
                                      &echild->base());
            }
            src0 += src0_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_followup_call(dst, 0, &src0, &inner_src_stride,
                                      inner_size, &echild->base());
                src0 += src0_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop is "first"
            for (size_t i = 0; i != count; ++i) {
                opchild_first_call(dst, &src0, &echild->base());
                if (inner_size > 1) {
                    const char *inner_src_second = src0 + inner_src_stride;
                    opchild_followup_call(dst, 0, &inner_src_second,
                                          &inner_src_stride, inner_size - 1,
                                          &echild->base());
                }
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                                 const char *const *src,
                                 const intptr_t *src_stride, size_t count,
                                 ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        expr_strided_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            opchild_followup_call(dst, 0, &src0, &inner_src_stride, inner_size, &echild->base());
            dst += dst_stride;
            src0 += src0_stride;
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        self->destroy_child_ckernel(sizeof(extra_type));
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

    static void single_first(char *dst, const char *const *src,
                             ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        expr_strided_t opchild_first_call =
            echild->get_first_call_function<expr_strided_t>();
        opchild_first_call(dst, e->dst_stride, src, &e->src_stride, e->size,
                           &echild->base());
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                              const char *const *src,
                              const intptr_t *src_stride, size_t count,
                              ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        expr_strided_t opchild_first_call = echild->get_first_call_function<expr_strided_t>();
        expr_strided_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we have one "first", followed by many "followup" calls
            opchild_first_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                        inner_size, &echild->base());
            dst += dst_stride;
            src0 += src0_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_followup_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                        inner_size, &echild->base());
                dst += dst_stride;
                src0 += src0_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop is "first"
            for (size_t i = 0; i != count; ++i) {
                opchild_first_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                        inner_size, &echild->base());
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                                 const char *const *src,
                                 const intptr_t *src_stride, size_t count,
                                 ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_reduction_prefix *echild =
            reinterpret_cast<ckernel_reduction_prefix *>(
                extra->get_child_ckernel(sizeof(extra_type)));
        expr_strided_t opchild_followup_call = echild->get_followup_call_function();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            opchild_followup_call(dst, inner_dst_stride, &src0,
                                  &inner_src_stride, inner_size,
                                  &echild->base());
            dst += dst_stride;
            src0 += src0_stride;
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        self->destroy_child_ckernel(sizeof(extra_type));
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
    // For the case with a reduction identity
    const char *ident_data;
    memory_block_data *ident_ref;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *const *src,
                             ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        // The first call to initialize the "dst" value
        expr_single_t opchild_dst_init =
            echild_dst_init->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        opchild_dst_init(dst, src, echild_dst_init);
        if (e->size > 1) {
            // All the followup calls to accumulate at the "dst" address
            const char *child_src = src[0] + e->src_stride;
            opchild_reduce(dst, 0, &child_src, &e->src_stride,
                           e->size - 1, &echild_reduce->base());
        }
    }

    static void single_first_with_ident(char *dst, const char *const *src,
                                        ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        // The first call to initialize the "dst" value
        expr_single_t opchild_ident =
            echild_ident->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        opchild_ident(dst, &e->ident_data, echild_ident);
        // All the followup calls to accumulate at the "dst" address
        opchild_reduce(dst, 0, src, &e->src_stride, e->size,
                       &echild_reduce->base());
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                              const char *const *src,
                              const intptr_t *src_stride, size_t count,
                              ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_single_t opchild_dst_init =
            echild_dst_init->get_function<expr_single_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many accumulations
            opchild_dst_init(dst, &src0, echild_dst_init);
            if (inner_size > 1) {
                const char *inner_child_src = src0 + inner_src_stride;
                opchild_reduce(dst, 0, &inner_child_src, &inner_src_stride,
                               inner_size - 1, &echild_reduce->base());
            }
            src0 += src0_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, 0, &src0, &inner_src_stride,
                        inner_size, &echild_reduce->base());
                dst += dst_stride;
                src0 += src0_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop has to
            // initialize then reduce
            for (size_t i = 0; i != count; ++i) {
                opchild_dst_init(dst, &src0, echild_dst_init);
                if (inner_size > 1) {
                    const char *inner_child_src = src0 + inner_src_stride;
                    opchild_reduce(dst, 0, &inner_child_src, &inner_src_stride,
                                   inner_size - 1, &echild_reduce->base());
                }
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_first_with_ident(char *dst, intptr_t dst_stride,
                                         const char *const *src,
                                         const intptr_t *src_stride,
                                         size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_single_t opchild_ident = echild_ident->get_function<expr_single_t>();
        expr_strided_t opchild_reduce = echild_reduce->get_function<expr_strided_t>();
        const char *ident_data = e->ident_data;
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many accumulations
            opchild_ident(dst, &ident_data, echild_ident);
            for (intptr_t i = 0; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                               &echild_reduce->base());
                dst += dst_stride;
                src0 += src0_stride;
            }
        } else {
            // With a non-zero stride, each iteration of the outer loop has to
            // initialize then reduce
            for (size_t i = 0; i != count; ++i) {
                opchild_ident(dst, &ident_data, echild_ident);
                opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                               &echild_reduce->base());
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                                 const char *const *src,
                                 const intptr_t *src_stride, size_t count,
                                 ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        // No initialization, all reduction
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                           &echild_reduce->base());
            dst += dst_stride;
            src0 += src0_stride;
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        extra_type *e = reinterpret_cast<extra_type *>(self);
        if (e->ident_ref != NULL) {
            memory_block_decref(e->ident_ref);
        }
        // The reduction kernel
        self->destroy_child_ckernel(sizeof(extra_type));
        // The destination initialization kernel
        self->destroy_child_ckernel(e->dst_init_kernel_offset);
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
    // For the case with a reduction identity
    const char *ident_data;
    memory_block_data *ident_ref;

    inline ckernel_prefix& base() {
        return ckpbase.base();
    }

    static void single_first(char *dst, const char *const *src,
                             ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        expr_strided_t opchild_dst_init = echild_dst_init->get_function<expr_strided_t>();
        // All we do is initialize the dst values
        opchild_dst_init(dst, e->dst_stride, src, &e->src_stride, e->size,
                         echild_dst_init);
    }

    static void single_first_with_ident(char *dst, const char *const *src,
                                        ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild_ident =
            echild_ident->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce =
            echild_reduce->get_function<expr_strided_t>();
        // First initialize the dst values (TODO: Do we want to do initialize/reduce in
        // blocks instead of just one then the other?)
        intptr_t zero_stride = 0;
        opchild_ident(dst, e->dst_stride, &e->ident_data, &zero_stride, e->size,
                      echild_ident);
        // Then do the accumulation
        opchild_reduce(dst, e->dst_stride, src, &e->src_stride, e->size,
                       echild_reduce);
    }

    static void strided_first(char *dst, intptr_t dst_stride,
                              const char *const *src,
                              const intptr_t *src_stride, size_t count,
                              ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild_dst_init = echild_dst_init->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce = echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many accumulations
            opchild_dst_init(dst, inner_dst_stride, &src0, &inner_src_stride,
                             inner_size, echild_dst_init);
            dst += dst_stride;
            src0 += src0_stride;
            for (intptr_t i = 1; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                        inner_size, &echild_reduce->base());
                src0 += src0_stride;
            }
        } else {
            // With a non-zero stride, every iteration is an initialization
            for (size_t i = 0; i != count; ++i) {
                opchild_dst_init(dst, inner_dst_stride, &src0, &inner_src_stride,
                            e->size, echild_dst_init);
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_first_with_ident(char *dst, intptr_t dst_stride,
                                         const char *const *src,
                                         const intptr_t *src_stride,
                                         size_t count, ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
                            reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        expr_strided_t opchild_ident = echild_ident->get_function<expr_strided_t>();
        expr_strided_t opchild_reduce = echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        if (dst_stride == 0) {
            // With a zero stride, we initialize "dst" once, then do many
            // accumulations
            intptr_t zero_stride = 0;
            opchild_ident(dst, inner_dst_stride, &e->ident_data, &zero_stride,
                          e->size, echild_ident);
            for (intptr_t i = 0; i < (intptr_t)count; ++i) {
                opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                               inner_size, &echild_reduce->base());
                src0 += src0_stride;
            }
        } else {
            intptr_t zero_stride = 0;
            // With a non-zero stride, every iteration is an initialization
            for (size_t i = 0; i != count; ++i) {
                opchild_ident(dst, inner_dst_stride, &e->ident_data,
                              &zero_stride, inner_size, echild_ident);
                opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                               inner_size, echild_reduce);
                dst += dst_stride;
                src0 += src0_stride;
            }
        }
    }

    static void strided_followup(char *dst, intptr_t dst_stride,
                                 const char *const *src,
                                 const intptr_t *src_stride, size_t count,
                                 ckernel_prefix *extra)
    {
        extra_type *e = reinterpret_cast<extra_type *>(extra);
        ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(extra_type));
        // No initialization, all reduction
        expr_strided_t opchild_reduce = echild_reduce->get_function<expr_strided_t>();
        intptr_t inner_size = e->size;
        intptr_t inner_dst_stride = e->dst_stride;
        intptr_t inner_src_stride = e->src_stride;
        const char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
            opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                           inner_size, &echild_reduce->base());
            dst += dst_stride;
            src0 += src0_stride;
        }
    }

    static void destruct(ckernel_prefix *self)
    {
        extra_type *e = reinterpret_cast<extra_type *>(self);
        if (e->ident_ref != NULL) {
            memory_block_decref(e->ident_ref);
        }
        // The reduction kernel
        self->destroy_child_ckernel(sizeof(extra_type));
        // The destination initialization kernel
        self->destroy_child_ckernel(e->dst_init_kernel_offset);
    }
};

} // anonymous namespace

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_reduction_dimension_kernel(
            ckernel_builder *ckb, intptr_t ckb_offset,
            intptr_t src_stride, intptr_t src_size,
            kernel_request_t kernreq)
{
    strided_initial_reduction_kernel_extra *e = ckb->alloc_ck<strided_initial_reduction_kernel_extra>(ckb_offset);
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
    return ckb_offset;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is not
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_initial_broadcast_dimension_kernel(
            ckernel_builder *ckb, intptr_t ckb_offset,
            intptr_t dst_stride, intptr_t src_stride, intptr_t src_size,
            kernel_request_t kernreq)
{
    strided_initial_broadcast_kernel_extra *e = ckb->alloc_ck<strided_initial_broadcast_kernel_extra>(ckb_offset);
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
    return ckb_offset;
}

static void check_dst_initialization(const arrfunc_type_data *dst_initialization,
                                     const ndt::type &dst_tp,
                                     const ndt::type &src_tp)
{
    if (dst_initialization->get_return_type() != dst_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
        ss << "dst type is " << dst_initialization->get_return_type();
        ss << ", expected " << dst_tp;
        throw type_error(ss.str());
    }
    if (dst_initialization->get_param_type(0) != src_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
        ss << "src type is " << dst_initialization->get_return_type();
        ss << ", expected " << src_tp;
        throw type_error(ss.str());
    }
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being reduced, and is
 * the final dimension before the accumulation operation.
 *
 * If dst_initialization is NULL, an assignment kernel is used.
 */
static size_t make_strided_inner_reduction_dimension_kernel(
    const arrfunc_type_data *elwise_reduction,
    const arrfunc_type_data *dst_initialization, ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t src_stride, intptr_t src_size,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
    const char *src_arrmeta, bool right_associative,
    const nd::array &reduction_identity, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  strided_inner_reduction_kernel_extra *e =
      ckb->alloc_ck<strided_inner_reduction_kernel_extra>(ckb_offset);
  e->base().destructor = &strided_inner_reduction_kernel_extra::destruct;
  // Cannot have both a dst_initialization kernel and a reduction identity
  if (dst_initialization != NULL && !reduction_identity.is_null()) {
    throw invalid_argument(
        "make_lifted_reduction_ckernel: cannot specify"
        " both a dst_initialization kernel and a reduction_identity");
  }
  if (reduction_identity.is_null()) {
    // Get the function pointer for the first_call, for the case with
    // no reduction identity
    if (kernreq == kernel_request_single) {
      e->ckpbase.set_first_call_function(
          &strided_inner_reduction_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
      e->ckpbase.set_first_call_function(
          &strided_inner_reduction_kernel_extra::strided_first);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
  } else {
    // Get the function pointer for the first_call, for the case with
    // a reduction identity
    if (kernreq == kernel_request_single) {
      e->ckpbase.set_first_call_function(
          &strided_inner_reduction_kernel_extra::single_first_with_ident);
    } else if (kernreq == kernel_request_strided) {
      e->ckpbase.set_first_call_function(
          &strided_inner_reduction_kernel_extra::strided_first_with_ident);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    if (reduction_identity.get_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: reduction identity type ";
      ss << reduction_identity.get_type() << " does not match dst type ";
      ss << dst_tp;
      throw runtime_error(ss.str());
    }
    e->ident_data = reduction_identity.get_readonly_originptr();
    e->ident_ref = reduction_identity.get_memblock().release();
  }
  // The function pointer for followup accumulation calls
  e->ckpbase.set_followup_call_function(
      &strided_inner_reduction_kernel_extra::strided_followup);
  // The striding parameters
  e->src_stride = src_stride;
  e->size = src_size;
  // Validate that the provided arrfuncs are unary operations,
  // and have the correct types
  if (elwise_reduction->get_param_count() != 1 &&
      elwise_reduction->get_param_count() != 2) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "funcproto must be unary or a binary expr with all equal types";
    throw runtime_error(ss.str());
  }
  if (elwise_reduction->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "dst type is " << elwise_reduction->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (elwise_reduction->get_param_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "src type is " << elwise_reduction->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization != NULL) {
    check_dst_initialization(dst_initialization, dst_tp, src_tp);
  }
  if (elwise_reduction->get_param_count() == 2) {
    ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
        ckb, ckb_offset, right_associative, kernel_request_strided);
    ndt::type src_tp_doubled[2] = {src_tp, src_tp};
    const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp_doubled,
        src_arrmeta_doubled, kernel_request_strided, ectx);
  } else {
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction, ckb, ckb_offset, dst_tp, dst_arrmeta, &src_tp,
        &src_arrmeta, kernel_request_strided, ectx);
  }
  // Make sure there's capacity for the next ckernel
  ckb->ensure_capacity(ckb_offset);
  // Need to retrieve 'e' again because it may have moved
  e = ckb->get_at<strided_inner_reduction_kernel_extra>(root_ckb_offset);
  e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
  if (dst_initialization != NULL) {
    ckb_offset = dst_initialization->instantiate(
        dst_initialization, ckb, ckb_offset, dst_tp, dst_arrmeta, &src_tp,
        &src_arrmeta, kernel_request_single, ectx);
  } else if (reduction_identity.is_null()) {
    ckb_offset =
        make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                               src_arrmeta, kernel_request_single, ectx);
  } else {
    ckb_offset = make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, reduction_identity.get_type(),
        reduction_identity.get_arrmeta(), kernel_request_single, ectx);
  }

  return ckb_offset;
}

/**
 * Adds a ckernel layer for processing one dimension of the reduction.
 * This is for a strided dimension which is being broadcast, and is
 * the final dimension before the accumulation operation.
 */
static size_t make_strided_inner_broadcast_dimension_kernel(
    const arrfunc_type_data *elwise_reduction,
    const arrfunc_type_data *dst_initialization, ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t dst_stride, intptr_t src_stride,
    intptr_t src_size, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, bool right_associative,
    const nd::array &reduction_identity, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  intptr_t root_ckb_offset = ckb_offset;
  strided_inner_broadcast_kernel_extra *e =
      ckb->alloc_ck<strided_inner_broadcast_kernel_extra>(ckb_offset);
  e->base().destructor = &strided_inner_broadcast_kernel_extra::destruct;
  // Cannot have both a dst_initialization kernel and a reduction identity
  if (dst_initialization != NULL && !reduction_identity.is_null()) {
    throw invalid_argument(
        "make_lifted_reduction_ckernel: cannot specify"
        " both a dst_initialization kernel and a reduction_identity");
  }
  if (reduction_identity.is_null()) {
    // Get the function pointer for the first_call, for the case with
    // no reduction identity
    if (kernreq == kernel_request_single) {
      e->ckpbase.set_first_call_function(
          &strided_inner_broadcast_kernel_extra::single_first);
    } else if (kernreq == kernel_request_strided) {
      e->ckpbase.set_first_call_function(
          &strided_inner_broadcast_kernel_extra::strided_first);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
  } else {
    // Get the function pointer for the first_call, for the case with
    // a reduction identity
    if (kernreq == kernel_request_single) {
      e->ckpbase.set_first_call_function(
          &strided_inner_broadcast_kernel_extra::single_first_with_ident);
    } else if (kernreq == kernel_request_strided) {
      e->ckpbase.set_first_call_function(
          &strided_inner_broadcast_kernel_extra::strided_first_with_ident);
    } else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    if (reduction_identity.get_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: reduction identity type ";
      ss << reduction_identity.get_type() << " does not match dst type ";
      ss << dst_tp;
      throw runtime_error(ss.str());
    }
    e->ident_data = reduction_identity.get_readonly_originptr();
    e->ident_ref = reduction_identity.get_memblock().release();
  }
  // The function pointer for followup accumulation calls
  e->ckpbase.set_followup_call_function(
      &strided_inner_broadcast_kernel_extra::strided_followup);
  // The striding parameters
  e->dst_stride = dst_stride;
  e->src_stride = src_stride;
  e->size = src_size;
  // Validate that the provided arrfuncs are unary operations,
  // and have the correct types
  if (elwise_reduction->get_param_count() != 1 &&
      elwise_reduction->get_param_count() != 2) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "funcproto must be unary or a binary expr with all equal types";
    throw runtime_error(ss.str());
  }
  if (elwise_reduction->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "dst type is " << elwise_reduction->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (elwise_reduction->get_param_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
    ss << "src type is " << elwise_reduction->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization != NULL) {
    check_dst_initialization(dst_initialization, dst_tp, src_tp);
  }
  if (elwise_reduction->get_param_count() == 2) {
    ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
        ckb, ckb_offset, right_associative, kernel_request_strided);
    ndt::type src_tp_doubled[2] = {src_tp, src_tp};
    const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp_doubled,
        src_arrmeta_doubled, kernel_request_strided, ectx);
  } else {
    ckb_offset = elwise_reduction->instantiate(
        elwise_reduction, ckb, ckb_offset, dst_tp, dst_arrmeta, &src_tp,
        &src_arrmeta, kernel_request_strided, ectx);
  }
  // Make sure there's capacity for the next ckernel
  ckb->ensure_capacity(ckb_offset);
  // Need to retrieve 'e' again because it may have moved
  e = ckb->get_at<strided_inner_broadcast_kernel_extra>(root_ckb_offset);
  e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
  if (dst_initialization != NULL) {
    ckb_offset = dst_initialization->instantiate(
        dst_initialization, ckb, ckb_offset, dst_tp, dst_arrmeta, &src_tp,
        &src_arrmeta, kernel_request_strided, ectx);
  } else if (reduction_identity.is_null()) {
    ckb_offset =
        make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp,
                               src_arrmeta, kernel_request_strided, ectx);
  } else {
    ckb_offset = make_assignment_kernel(
        ckb, ckb_offset, dst_tp, dst_arrmeta, reduction_identity.get_type(),
        reduction_identity.get_arrmeta(), kernel_request_strided, ectx);
  }

  return ckb_offset;
}

size_t dynd::make_lifted_reduction_ckernel(
    const arrfunc_type_data *elwise_reduction,
    const arrfunc_type_data *dst_initialization, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, intptr_t reduction_ndim,
    const bool *reduction_dimflags, bool associative, bool commutative,
    bool right_associative, const nd::array &reduction_identity,
    dynd::kernel_request_t kernreq, const eval::eval_context *ectx)
{
    // Count the number of dimensions being reduced
    intptr_t reducedim_count = 0;
    for (intptr_t i = 0; i < reduction_ndim; ++i) {
        reducedim_count += reduction_dimflags[i];
    }
    if (reducedim_count == 0) {
        if (reduction_ndim == 0) {
            // If there are no dimensions to reduce, it's
            // just a dst_initialization operation, so create
            // that ckernel directly
            if (dst_initialization != NULL) {
                return dst_initialization->instantiate(
                    dst_initialization, ckb, ckb_offset, dst_tp,
                    dst_arrmeta, &src_tp, &src_arrmeta, kernreq, ectx);
            } else if (reduction_identity.is_null()) {
                return make_assignment_kernel(ckb, ckb_offset, dst_tp,
                                              dst_arrmeta, src_tp, src_arrmeta,
                                              kernreq, ectx);
            } else {
                // Create the kernel which copies the identity and then
                // does one reduction
                return make_strided_inner_reduction_dimension_kernel(
                    elwise_reduction, dst_initialization, ckb, ckb_offset,
                    0, 1, dst_tp, dst_arrmeta, src_tp, src_arrmeta, right_associative,
                    reduction_identity, kernreq, ectx);
            }
        }
        throw runtime_error("make_lifted_reduction_ckernel: no dimensions were flagged for reduction");
    }

    if (!(reducedim_count == 1 || (associative && commutative))) {
        throw runtime_error("make_lifted_reduction_ckernel: for reducing along multiple dimensions,"
                            " the reduction function must be both associative and commutative");
    }
    if (right_associative) {
        throw runtime_error("make_lifted_reduction_ckernel: right_associative is not yet supported");
    }

    ndt::type dst_el_tp = elwise_reduction->get_return_type();
    ndt::type src_el_tp = elwise_reduction->get_param_type(0);

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

    ndt::type dst_i_tp = dst_tp, src_i_tp = src_tp;

    for (intptr_t i = 0; i < reduction_ndim; ++i) {
        intptr_t dst_stride, dst_size, src_stride, src_size;
        // Get the striding parameters for the source dimension
        if (!src_i_tp.get_as_strided(src_arrmeta, &src_size, &src_stride,
                                     &src_i_tp, &src_arrmeta)) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: type " << src_i_tp << " not supported as source";
            throw type_error(ss.str());
        }
        if (reduction_dimflags[i]) {
            // This dimension is being reduced
            if (src_size == 0 && reduction_identity.is_null()) {
                // If the size of the src is 0, a reduction identity is required to get a value
                stringstream ss;
                ss << "cannot reduce a zero-sized dimension (axis ";
                ss << i << " of " << src_i_tp << ") because the operation";
                ss << " has no identity";
                throw invalid_argument(ss.str());
            }
            if (keep_dims) {
                // If the dimensions are being kept, the output should be a
                // a strided dimension of size one
              if (dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                          &dst_i_tp, &dst_arrmeta)) {
                    if (dst_size != 1 || dst_stride != 0) {
                        stringstream ss;
                        ss << "make_lifted_reduction_ckernel: destination of a reduction dimension ";
                        ss << "must have size 1, not size" << dst_size << "/stride " << dst_stride;
                        ss << " in type " << dst_i_tp;
                        throw type_error(ss.str());
                    }
                } else {
                    stringstream ss;
                    ss << "make_lifted_reduction_ckernel: type " << dst_i_tp;
                    ss << " not supported the destination of a dimension being reduced";
                    throw type_error(ss.str());
                }
            }
            if (i < reduction_ndim - 1) {
                // An initial dimension being reduced
                ckb_offset = make_strided_initial_reduction_dimension_kernel(
                                        ckb, ckb_offset,
                                        src_stride, src_size,
                                        kernreq);
                // The next request should be single, as that's the kind of
                // ckernel the 'first_call' should be in this case
                kernreq = kernel_request_single;
            } else {
                // The innermost dimension being reduced
                return make_strided_inner_reduction_dimension_kernel(
                    elwise_reduction, dst_initialization, ckb, ckb_offset,
                    src_stride, src_size, dst_i_tp, dst_arrmeta, src_i_tp, src_arrmeta,
                    right_associative, reduction_identity, kernreq, ectx);
            }
        } else {
            // This dimension is being broadcast, not reduced
          if (!dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                       &dst_i_tp, &dst_arrmeta)) {
                stringstream ss;
                ss << "make_lifted_reduction_ckernel: type " << dst_i_tp << " not supported as destination";
                throw type_error(ss.str());
            }
            if (dst_size != src_size) {
                stringstream ss;
                ss << "make_lifted_reduction_ckernel: the dst dimension size " << dst_size;
                ss << " must equal the src dimension size " << src_size << " for broadcast dimensions";
                throw runtime_error(ss.str());
            }
            if (i < reduction_ndim - 1) {
                // An initial dimension being broadcast
                ckb_offset = make_strided_initial_broadcast_dimension_kernel(
                                        ckb, ckb_offset,
                                        dst_stride, src_stride, src_size,
                                        kernreq);
                // The next request should be strided, as that's the kind of
                // ckernel the 'first_call' should be in this case
                kernreq = kernel_request_strided;
            } else {
                // The innermost dimension being broadcast
                return make_strided_inner_broadcast_dimension_kernel(
                    elwise_reduction, dst_initialization, ckb, ckb_offset,
                    dst_stride, src_stride, src_size, dst_i_tp, dst_arrmeta, src_i_tp,
                    src_arrmeta, right_associative, reduction_identity, kernreq,
                    ectx);
            }
        }
    }

    throw runtime_error("make_lifted_reduction_ckernel: internal error, "
                        "should have returned in the loop");
}
 
