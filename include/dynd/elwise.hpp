//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_HPP_
#define _DYND__ELWISE_HPP_

#include <utility>

#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

using namespace std;

namespace dynd { namespace nd {

namespace detail {

template<class FuncProto>
struct elwise_ckernel_instantiator;

#define ELWISE_CKERNEL_INSTANTIATOR_FUNC_RET_RES(NSRC)

#define ELWISE_CKERNEL_INSTANTIATOR_FUNC_REF_RES(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_ckernel_instantiator<void (*)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC)))> { \
        typedef elwise_ckernel_instantiator extra_type; \
        typedef void (*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))); \
\
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), DYND_PP_OUTER_1(DYND_PP_META_SCOPE, \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE, (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC)), (type)), \
            DYND_PP_META_NAME_RANGE(U, NSRC)); \
\
        ckernel_prefix base; \
        func_type func; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            e->func(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
                    DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST, \
                    DYND_PP_MAP_1(DYND_PP_META_AS_CONST_PTR, \
                    DYND_PP_META_NAME_RANGE(U, NSRC)), DYND_PP_META_AT_RANGE(src, NSRC)))); \
        } \
\
        static void strided(char *dst, intptr_t dst_stride, \
                            const char * const *src, const intptr_t *src_stride, \
                            size_t count, ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            func_type func = e->func; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), DYND_PP_REPEAT_1(const char *, NSRC), \
                DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), DYND_PP_REPEAT_1(intptr_t, NSRC), \
                DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                func(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
                    DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST, \
                    DYND_PP_MAP_1(DYND_PP_META_AS_CONST_PTR, \
                    DYND_PP_META_NAME_RANGE(U, NSRC)), DYND_PP_META_NAME_RANGE(src, NSRC)))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_EQ, (;), \
                    DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC)); \
            } \
        } \
\
        static intptr_t instantiate(void *self_data_ptr, \
                    dynd::ckernel_builder *out_ckb, intptr_t ckb_offset, \
                    const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype, \
                    const eval::eval_context *DYND_UNUSED(ectx)) \
        { \
            extra_type *e = out_ckb->get_at<extra_type>(ckb_offset); \
            if (kerntype == kernel_request_single) { \
                e->base.set_function(&extra_type::single); \
            } else if (kerntype == kernel_request_strided) { \
                e->base.set_function(&extra_type::strided); \
            } else { \
                throw runtime_error("unsupported kernel request in elwise"); \
            } \
            e->func = reinterpret_cast<func_type>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    }; \

#define ELWISE_CKERNEL_INSTANTIATOR(NSRC) \
    ELWISE_CKERNEL_INSTANTIATOR_FUNC_RET_RES(NSRC) \
    ELWISE_CKERNEL_INSTANTIATOR_FUNC_REF_RES(NSRC)

DYND_PP_JOIN_MAP(ELWISE_CKERNEL_INSTANTIATOR, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef ELWISE_CKERNEL_INSTANTIATOR
#undef ELWISE_CKERNEL_INSTANTIATOR_FUNC_REF_RET
#undef ELWISE_CKERNEL_INSTANTIATOR_FUNC_REF_RES

// Remove this
template<typename R, typename T0, typename T1>
struct elwise_ckernel_instantiator<R (*)(T0, T1)> {
    typedef elwise_ckernel_instantiator extra_type;

    ckernel_prefix base;
    R (*func)(T0, T1);

    typedef typename remove_reference<T0>::type U0;
    typedef typename remove_reference<T0>::type U1;

    static void single(char *dst, char * const *src,
                       ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        *reinterpret_cast<R *>(dst) = e->func(
                            *reinterpret_cast<U0 *>(src[0]),
                            *reinterpret_cast<U1 *>(src[1]));
    }

    static void strided(char *dst, intptr_t dst_stride,
                        char * const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        R (*func)(T0, T1);
        func = e->func;
        char *src0 = src[0], *src1 = src[1];
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];
        for (size_t i = 0; i < count; ++i) {
            *reinterpret_cast<R *>(dst) = func(
                                *reinterpret_cast<U0 *>(src0),
                                *reinterpret_cast<U1 *>(src1));
            dst += dst_stride;
            src0 += src0_stride;
            src1 += src1_stride;
        }
    }

    static intptr_t instantiate(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype,
                const eval::eval_context *DYND_UNUSED(ectx))
    {
        extra_type *e = out_ckb->get_at<extra_type>(ckb_offset);
        if (kerntype == kernel_request_single) {
            e->base.set_function(&extra_type::single);
        } else if (kerntype == kernel_request_strided) {
            e->base.set_function(&extra_type::strided);
        } else {
            throw runtime_error("unsupported kernel request in elwise");
        }
        e->func = reinterpret_cast<R (*)(T0, T1)>(self_data_ptr);
        // No need for a destructor function in this ckernel

        return ckb_offset + sizeof(extra_type);
    }
};

// Remove this
template<typename T, typename R, typename A0, typename A1>
struct elwise_ckernel_instantiator<R (T::*)(A0, A1)> {
    typedef elwise_ckernel_instantiator extra_type;

    ckernel_prefix base;
    T *obj;
    R (T::*func)(A0, A1);

    static void single(char *dst, const char * const *src,
                       ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        *reinterpret_cast<R *>(dst) = ((e->obj)->*(e->func))(
                            *reinterpret_cast<const A0 *>(src[0]),
                            *reinterpret_cast<const A1 *>(src[1]));
    }

    static void strided(char *dst, intptr_t dst_stride,
                        const char * const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        T *obj = e->obj;
        R (T::*func)(A0, A1) = e->func;
        const char *src0 = src[0], *src1 = src[1];
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];
        for (size_t i = 0; i < count; ++i) {
            *reinterpret_cast<R *>(dst) = (obj->*func)(
                                *reinterpret_cast<const A0 *>(src0),
                                *reinterpret_cast<const A1 *>(src1));
            dst += dst_stride;
            src0 += src0_stride;
            src1 += src1_stride;
        }
    }

    static intptr_t instantiate(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype,
                const eval::eval_context *DYND_UNUSED(ectx))
    {
        extra_type *e = out_ckb->get_at<extra_type>(ckb_offset);
        if (kerntype == kernel_request_single) {
            e->base.set_function(&extra_type::single);
        } else if (kerntype == kernel_request_strided) {
            e->base.set_function(&extra_type::strided);
        } else {
            throw runtime_error("unsupported kernel request in elwise");
        }
        std::pair<T *, R (T::*)(A0, A1)> *pr = reinterpret_cast<std::pair<T *, R (T::*)(A0, A1)> *>(self_data_ptr);
        e->obj = pr->first;
        e->func = pr->second;
        // No need for a destructor function in this ckernel

        return ckb_offset + sizeof(extra_type);
    }
};

}; // namespace detail

#define ELWISE_BROADCAST(NSRC) \
    template<DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    void elwise_broadcast(DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), \
                (const nd::array &), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        intptr_t& out_ndim, dynd::dimvector& out_shape) { \
\
        DYND_PP_IF_ELSE(DYND_PP_EQ(NSRC, 1)) \
        ( \
            out_ndim = a0.get_ndim() - detail::ndim_from_array<A0>::value; \
            a0.get_shape(out_shape.get()); \
        ) \
        ( \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ASGN, (;), \
                DYND_PP_OUTER_1(DYND_PP_META_DECL, (intptr_t), \
                    DYND_PP_META_NAME_RANGE(ndim, NSRC)), \
                DYND_PP_ELWISE_1(DYND_PP_META_SUB, \
                    DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                        DYND_PP_META_NAME_RANGE(a, NSRC), (get_ndim)), \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_SCOPE, \
                        (detail::ndim_from_array), DYND_PP_META_NAME_RANGE(A, NSRC), (value)))); \
\
            out_ndim = ndim0; \
            DYND_PP_JOIN_OUTER_1(DYND_PP_META_ASGN, (;), (out_ndim), \
                DYND_PP_OUTER_1(DYND_PP_META_CALL, (max), \
                    DYND_PP_META_NAME_RANGE(ndim, 1, NSRC), (out_ndim))); \
\
            out_shape.init(out_ndim); \
            for (intptr_t j = 0; j != out_ndim; ++j) { \
                out_shape[j] = 1; \
            } \
\
            dimvector tmp_shape(out_ndim); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_IF, (), \
                DYND_PP_OUTER_1(DYND_PP_META_GT, \
                    DYND_PP_META_NAME_RANGE(ndim, NSRC), (0)), \
                DYND_PP_ELWISE_1(DYND_PP_META_SEMI, \
                    DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                        DYND_PP_META_NAME_RANGE(a, NSRC), (get_shape), (tmp_shape.get())), \
                            DYND_PP_OUTER_1(DYND_PP_META_FLAT_CALL, \
                                (incremental_broadcast), DYND_PP_OUTER_1(DYND_PP_APPEND, (tmp_shape.get()), DYND_PP_OUTER_1(DYND_PP_APPEND, \
                                DYND_PP_META_NAME_RANGE(ndim, NSRC), ((out_ndim, out_shape.get()))))))) \
        ) \
    }

DYND_PP_JOIN_MAP(ELWISE_BROADCAST, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef ELWISE_BROADCAST

#define ELWISE_FUNC_RET_RES(NSRC)

#define ELWISE_FUNC_REF_RES(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(void (*func)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))), \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_STATIC_ASSERT, (;), \
            DYND_PP_ELWISE_1(DYND_PP_META_OR, \
                DYND_PP_MAP_1(DYND_PP_META_NOT, \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_SCOPE, \
                        (is_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (value))), \
                DYND_PP_ELWISE_1(DYND_PP_META_PARENTHESIZED_AND, \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_SCOPE, \
                        (is_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (value)), \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_SCOPE, (is_const), \
                        DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_SCOPE, \
                            (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                        (value)))), \
            ("all reference arguments must be const")); \
\
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[DYND_PP_INC(NSRC)] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE, (ndt::fixed_dim_from_array), \
            DYND_PP_META_NAME_RANGE(D, NSRC)), DYND_PP_REPEAT(make, NSRC))}; \
\
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ASGN, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_DECL, \
                (const nd::array), DYND_PP_META_NAME_RANGE(acast, NSRC)), \
            DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                DYND_PP_ELWISE_1(DYND_PP_META_DOT, \
                    DYND_PP_META_NAME_RANGE(a, NSRC), \
                        DYND_PP_OUTER_1(DYND_PP_META_CALL, \
                            (ucast), DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                                DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), (eval))); \
\
        intptr_t res_strided_ndim; \
        dimvector res_strided_shape; \
        elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))> \
            (DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(acast, NSRC)), res_strided_ndim, res_strided_shape); \
        nd::array result = nd::make_strided_array(data_dynd_types[0], res_strided_ndim, res_strided_shape.get()); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = DYND_PP_INC(NSRC); \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(func); \
        ckd.instantiate_func = &detail::elwise_ckernel_instantiator<void (*)(R &, \
            DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC)))>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[DYND_PP_INC(NSRC)] = {result.get_type(), \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[DYND_PP_INC(NSRC)] = {result.get_ndo_meta(), \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(result.get_readwrite_originptr(), src, ckprefix); \
\
        return result; \
    }

#define ELWISE_METH_RET_RES(NSRC)

#define ELWISE_METH_REF_RES(NSRC)

#define ELWISE_OBJ_RET_RES(NSRC)

#define ELWISE_OBJ_REF_RES(NSRC)

#define ELWISE(NSRC) \
    ELWISE_FUNC_RET_RES(NSRC) \
    ELWISE_FUNC_REF_RES(NSRC) \
    ELWISE_METH_RET_RES(NSRC) \
    ELWISE_METH_REF_RES(NSRC) \
    ELWISE_OBJ_RET_RES(NSRC) \
    ELWISE_OBJ_REF_RES(NSRC)

DYND_PP_JOIN_MAP(ELWISE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef ELWISE
#undef ELWISE_FUNC_RET_RES
//#undef ELWISE_FUNC_REF_RES
#undef ELWISE_METH_RET_RES
#undef ELWISE_METH_REF_RES


template<typename R, typename T0, typename T1>
inline nd::array elwise(R (*func)(T0, T1), const nd::array& a, const nd::array& b,
    const eval::eval_context *ectx = &eval::default_eval_context)
{
    typedef typename remove_reference<T0>::type U0;
    typedef typename remove_reference<T0>::type U1;

    const nd::array ac = a.ucast<U0>().eval();
    const nd::array bc = b.ucast<U1>().eval();

    // No casting for now
/*
    if (a.get_dtype() != ndt::make_type<U0>()) {
        stringstream ss;
        ss << "initial prototype of elwise doesn't implicitly cast ";
        ss << a.get_dtype() << " to " << ndt::make_type<U0>();
        throw type_error(ss.str());
    }
    if (b.get_dtype() != ndt::make_type<U1>()) {
        stringstream ss;
        ss << "initial prototype of elwise doesn't implicitly cast ";
        ss << b.get_dtype() << " to " << ndt::make_type<U1>();
        throw type_error(ss.str());
    }
*/

    // Create a static ckernel_deferred out of the function
    ckernel_deferred ckd;
    ckd.ckernel_funcproto = expr_operation_funcproto;
    ckd.data_types_size = 3;
    ndt::type data_dynd_types[3] = {ndt::make_type<R>(), ndt::make_type<U0>(), ndt::make_type<U1>()};
    ckd.data_dynd_types = data_dynd_types;
    ckd.data_ptr = reinterpret_cast<void *>(func);
    ckd.instantiate_func = &detail::elwise_ckernel_instantiator<R (*)(T0, T1)>::instantiate;
    ckd.free_func = NULL;

    intptr_t ndim;
    dimvector result_shape;
    elwise_broadcast<T0, T1>(ac, bc, ndim, result_shape);

//void incremental_broadcast_input_shapes(intptr_t ninputs, const nd::array* inputs,
  //              intptr_t &out_undim, dimvector& out_shape, bool flag = false)

    // Get the broadcasted shape
    // TODO: This was hastily grabbed from arithmetic_op.cpp, should be encapsulated much better
//    size_t ndim = max(a.get_ndim(), b.get_ndim());
//    dimvector result_shape(ndim), tmp_shape(ndim);
//    for (size_t j = 0; j != ndim; ++j) {
  //      result_shape[j] = 1;
  //  }
/*    dimvector tmp_shape(ndim);
    size_t ndim_a = a.get_ndim();
    if (ndim_a > 0) {
        a.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_a, tmp_shape.get());
    }
    size_t ndim_b = b.get_ndim();
    if (ndim_b > 0) {
        b.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_b, tmp_shape.get());
    }*/

    // Allocate the output array
    nd::array result = nd::make_strided_array(ndt::make_type<R>(), ndim, result_shape.get());

    // Lift the ckernel_deferred to a ckernel which handles the dimensions
    ckernel_builder ckb;
    ndt::type lifted_types[3] = {result.get_type(), ac.get_type(), bc.get_type()};
    const char *dynd_metadata[3] = {result.get_ndo_meta(), ac.get_ndo_meta(), bc.get_ndo_meta()};
    make_lifted_expr_ckernel(&ckd, &ckb, 0, lifted_types, dynd_metadata, kernel_request_single, ectx);

    // Call the ckernel to do the operation
    ckernel_prefix *ckprefix = ckb.get();
    expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>();
    const char *src[2] = {ac.get_readonly_originptr(), bc.get_readonly_originptr()};
    op(result.get_readwrite_originptr(), src, ckprefix);

    return result;
}


template<typename T, typename R, typename A0, typename A1>
inline nd::array elwise(T& obj, R (T::*func)(A0, A1), const nd::array& a, const nd::array& b,
    const eval::eval_context *ectx = &eval::default_eval_context)
{
    // No casting for now
    if (a.get_dtype() != ndt::make_type<A0>()) {
        stringstream ss;
        ss << "initial prototype of elwise doesn't implicitly cast ";
        ss << a.get_dtype() << " to " << ndt::make_type<A0>();
        throw type_error(ss.str());
    }
    if (b.get_dtype() != ndt::make_type<A1>()) {
        stringstream ss;
        ss << "initial prototype of elwise doesn't implicitly cast ";
        ss << b.get_dtype() << " to " << ndt::make_type<A1>();
        throw type_error(ss.str());
    }

    // Pack the instance and the member function together
    std::pair<T *, R (T::*)(A0, A1)> pr(&obj, func);

    // Create a static ckernel_deferred out of the function
    ckernel_deferred ckd;
    ckd.ckernel_funcproto = expr_operation_funcproto;
    ckd.data_types_size = 3;
    ndt::type data_dynd_types[3] = {ndt::make_type<R>(), ndt::make_type<A0>(), ndt::make_type<A1>()};
    ckd.data_dynd_types = data_dynd_types;
    ckd.data_ptr = reinterpret_cast<void *>(&pr);
    ckd.instantiate_func = &detail::elwise_ckernel_instantiator<R (T::*)(A0, A1)>::instantiate;
    ckd.free_func = NULL;

    // Get the broadcasted shape
    // TODO: This was hastily grabbed from arithmetic_op.cpp, should be encapsulated much better
    size_t ndim = max(a.get_ndim(), b.get_ndim());
    dimvector result_shape(ndim), tmp_shape(ndim);
    for (size_t j = 0; j != ndim; ++j) {
        result_shape[j] = 1;
    }
    size_t ndim_a = a.get_ndim();
    if (ndim_a > 0) {
        a.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_a, tmp_shape.get());
    }
    size_t ndim_b = b.get_ndim();
    if (ndim_b > 0) {
        b.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_b, tmp_shape.get());
    }

    // Allocate the output array
    nd::array result = nd::make_strided_array(ndt::make_type<R>(), ndim, result_shape.get());

    // Lift the ckernel_deferred to a ckernel which handles the dimensions
    ckernel_builder ckb;
    ndt::type lifted_types[3] = {result.get_type(), a.get_type(), b.get_type()};
    const char *dynd_metadata[3] = {result.get_ndo_meta(), a.get_ndo_meta(), b.get_ndo_meta()};
    make_lifted_expr_ckernel(&ckd, &ckb, 0,
                        lifted_types, dynd_metadata, kernel_request_single, ectx);

    // Call the ckernel to do the operation
    ckernel_prefix *ckprefix = ckb.get();
    expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>();
    const char *src[2] = {a.get_readonly_originptr(), b.get_readonly_originptr()};
    op(result.get_readwrite_originptr(), src, ckprefix);

    return result;
}

template<typename T>
inline nd::array elwise(T obj, const nd::array& a, const nd::array& b)
{
    return elwise(obj, &T::operator (), a, b);
}

}} // namespace dynd::nd

#endif // _DYND__ELWISE_HPP_
