//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FOREACH_HPP_
#define _DYND__FOREACH_HPP_

#include <utility>

#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/pp/list.hpp>

#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

#include <dynd/pp/list.hpp>

using namespace std;

#define SRC_TYPE T
#define RES_TYPE R

#define REINTERPRET_CAST_SRC(I) *reinterpret_cast<const DYND_PP_PASTE(SRC_TYPE, I) *>(src[I])
#define DECL_ASGN_SRC(I) const char *DYND_PP_PASTE(src, I) = src[I]


#define RES_TYPE R
#define DECL_RES_TYPE typename RES_TYPE

#define SRC_TYPE_X(I) DYND_PP_PASTE(T, I)
#define DECL_SRC_TYPE(INDEX) typename SRC_TYPE_X(INDEX)

#define SRC_TYPES(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_2(SRC_TYPE_X, DYND_PP_RANGE(STOP)))
#define DECL_SRC_TYPES(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_1(DECL_SRC_TYPE, DYND_PP_RANGE(STOP)))


#define FIXED_DIM(INDEX) DYND_PP_PASTE(N, INDEX)
#define DECL_FIXED_DIM(INDEX) int FIXED_DIM(INDEX)

#define FIXED_DIMS(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_2(FIXED_DIM, DYND_PP_RANGE(STOP)))
#define DECL_FIXED_DIMS(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_2(DECL_FIXED_DIM, DYND_PP_RANGE(STOP)))


#define ARRAY_DIMS(STOP) DYND_PP_ARRAY_DIMS((FIXED_DIMS(STOP)))

#define FIXED_DIMS_AS_ARRAY_DIMS(STOP) DYND_PP_ARRAY_DIMS((FIXED_DIMS(STOP)))

#define SRC_ARRAY(I) DYND_PP_PASTE(a, I)
#define SRC_ARRAYS(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_1(SRC_ARRAY, DYND_PP_RANGE(STOP)))



//#define SRC_ARRAY(I, METH, ) SRC_ARRAY(I).METH()

#define SRC_ARRAY_GET_TYPE(I) SRC_ARRAY(I).get_type()
#define SRC_ARRAY_GET_NDO_META(I) SRC_ARRAY(I).get_ndo_meta()
#define SRC_ARRAY_GET_READONLY_ORIGINPTR(I) SRC_ARRAY(I).get_readonly_originptr()

#define REINTERPRET_CAST_SRC(I) *reinterpret_cast<const DYND_PP_PASTE(SRC_TYPE, I) *>(src[I])
#define REINTERPRET_CAST_SRCS(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_1(REINTERPRET_CAST_SRC, DYND_PP_RANGE(STOP)))

#define SRC(INDEX) DYND_PP_PASTE(src, INDEX)
#define SRCS(STOP) DYND_PP_MAP_1(SRC, DYND_PP_RANGE(STOP))



#define MAP_SRC_TYPES(MAC, SEP, STOP) DYND_PP_JOIN_1(SEP, DYND_PP_MAP_1(MAC, (SRC_TYPES(STOP))))



#define DECL_ASGN_SRCS(STOP) DYND_PP_JOIN_1((;), DYND_PP_MAP_1(DECL_ASGN_SRC, DYND_PP_RANGE(STOP)))

#define DECL_ASGN_SRC_STRIDE(INDEX) intptr_t DYND_PP_PASTE(DYND_PP_PASTE(src, INDEX), _stride) = src_stride[INDEX]
#define DECL_ASGN_SRC_STRIDES(STOP) DYND_PP_JOIN_1((;), DYND_PP_MAP_1(DECL_ASGN_SRC_STRIDE, DYND_PP_RANGE(STOP)))

#define REINTERPRET_CAST_DECLED_SRC(INDEX) *reinterpret_cast<const DYND_PP_PASTE(SRC_TYPE, INDEX) *>(SRC(INDEX))
#define REINTERPRET_CAST_DECLED_SRCS(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_1(REINTERPRET_CAST_DECLED_SRC, DYND_PP_RANGE(STOP)))

#define DECLED_SRC_ADD_SRC_STRIDE(INDEX) SRC(INDEX) += DYND_PP_PASTE(SRC(INDEX), _stride)
#define DECLED_SRCS_ADD_SRC_STRIDE(STOP) DYND_PP_FLATTEN(DYND_PP_MAP_1(DECLED_SRC_ADD_SRC_STRIDE, DYND_PP_RANGE(STOP)))

#define MAKE_TYPE_SRC_TYPE(INDEX) ndt::make_type<SRC_TYPE_X(INDEX)>()
#define MAKE_TYPE_SRC_TYPES(STOP) DYND_PP_JOIN_1((,), DYND_PP_MAP_1(MAKE_TYPE_SRC_TYPE, DYND_PP_RANGE(STOP)))

#define DECL_SRC_ARRAY(INDEX) const nd::array& SRC_ARRAY(INDEX)
#define DECL_SRC_ARRAYS(STOP) DYND_PP_JOIN_1((,), DYND_PP_MAP_1(DECL_SRC_ARRAY, DYND_PP_RANGE(STOP)))

#define INCREMENTAL_BROADCAST_SRC_ARRAY(INDEX) size_t DYND_PP_PASTE(ndim_, SRC_ARRAY(INDEX)) = SRC_ARRAY(INDEX).get_ndim(); \
    if (DYND_PP_PASTE(ndim_, SRC_ARRAY(INDEX)) > 0) { \
        SRC_ARRAY(INDEX).get_shape(tmp_shape.get()); \
        incremental_broadcast(ndim, result_shape.get(), DYND_PP_PASTE(ndim_, SRC_ARRAY(INDEX)), tmp_shape.get()); \
    }
#define INCREMENTAL_BROADCAST_SRC_ARRAYS(STOP) DYND_PP_JOIN_1((), DYND_PP_MAP_1(INCREMENTAL_BROADCAST_SRC_ARRAY, DYND_PP_RANGE(STOP)))

#define SRC_ARRAY_GET_NDIM(INDEX) SRC_ARRAY(INDEX).get_ndim()
#define SRC_ARRAYS_GET_NDIM(STOP) DYND_PP_JOIN_1((,), DYND_PP_MAP_1(SRC_ARRAY_GET_NDIM, DYND_PP_RANGE(STOP)))


#define MAX_NDIM_0(A) A
#define MAX_NDIM_1(A) DYND_PP_REDUCE(max, (A))

//        if (a0.get_dtype() != ndt::make_type<T0>()) { \
//            stringstream ss; \
//            ss << "initial prototype of foreach doesn't implicitly cast "; \
//            ss << a0.get_dtype() << " to " << ndt::make_type<T0>(); \
//            throw type_error(ss.str()); \
//        } \
//        if (a1.get_dtype() != ndt::make_type<T1>()) { \
//            stringstream ss; \
//            ss << "initial prototype of foreach doesn't implicitly cast "; \
//            ss << a1.get_dtype() << " to " << ndt::make_type<T1>(); \
//            throw type_error(ss.str()); \
//        }

#define ELWISE(STRIDED_NDIM, FIXED_NDIM) \
    namespace detail { \
    template<DECL_RES_TYPE, DECL_FIXED_DIMS(FIXED_NDIM), DECL_SRC_TYPES(STRIDED_NDIM)> \
    struct foreach_ckernel_instantiator<void (*)(RES_TYPE (&)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM), SRC_TYPES(STRIDED_NDIM))> { \
        typedef foreach_ckernel_instantiator extra_type; \
        typedef void (*func_type)(RES_TYPE (&)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM), SRC_TYPES(STRIDED_NDIM)); \
\
        ckernel_prefix base; \
        func_type func; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            e->func(*reinterpret_cast<RES_TYPE (*)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM)>(dst), \
                    REINTERPRET_CAST_SRCS(STRIDED_NDIM)); \
        } \
\
        static void strided(char *dst, intptr_t dst_stride, \
                            const char * const *src, const intptr_t *src_stride, \
                            size_t count, ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            func_type func = e->func; \
            DECL_ASGN_SRCS(STRIDED_NDIM); \
            DECL_ASGN_SRC_STRIDES(STRIDED_NDIM); \
            for (size_t i = 0; i < count; ++i) { \
                func(*reinterpret_cast<RES_TYPE (*)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM)>(dst), \
                    REINTERPRET_CAST_DECLED_SRCS(STRIDED_NDIM)); \
                dst += dst_stride; \
                DECLED_SRCS_ADD_SRC_STRIDE(STRIDED_NDIM); \
            } \
        } \
\
        static intptr_t instantiate(void *self_data_ptr, \
                    dynd::ckernel_builder *out_ckb, intptr_t ckb_offset, \
                    const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype) \
        { \
            extra_type *e = out_ckb->get_at<extra_type>(ckb_offset); \
            if (kerntype == kernel_request_single) { \
                e->base.set_function(&extra_type::single); \
            } else if (kerntype == kernel_request_strided) { \
                e->base.set_function(&extra_type::strided); \
            } else { \
                throw runtime_error("unsupported kernel request in foreach"); \
            } \
            e->func = reinterpret_cast<func_type>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    }; \
    } \
\
    template<DECL_RES_TYPE, DECL_FIXED_DIMS(FIXED_NDIM), DECL_SRC_TYPES(STRIDED_NDIM)> \
    inline nd::array foreach(DECL_SRC_ARRAYS(STRIDED_NDIM), \
        void (*func)(RES_TYPE (&)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM), SRC_TYPES(STRIDED_NDIM))) \
    { \
        intptr_t fixed_shape[FIXED_NDIM] = {FIXED_DIMS(FIXED_NDIM)}; \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = DYND_PP_INC(STRIDED_NDIM); \
        ndt::type data_dynd_types[DYND_PP_INC(STRIDED_NDIM)] \
            = {ndt::make_fixed_dim(FIXED_NDIM, fixed_shape, ndt::make_type<RES_TYPE>(), NULL), \
            MAKE_TYPE_SRC_TYPES(STRIDED_NDIM)}; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(func); \
        ckd.instantiate_func = &detail::foreach_ckernel_instantiator<void (*)(RES_TYPE (&)FIXED_DIMS_AS_ARRAY_DIMS(FIXED_NDIM), \
            SRC_TYPES(STRIDED_NDIM))>::instantiate; \
        ckd.free_func = NULL; \
\
        size_t ndim = DYND_PP_PASTE(MAX_NDIM_, DYND_PP_NE(STRIDED_NDIM, 1))(SRC_ARRAYS_GET_NDIM(STRIDED_NDIM)); \
        dimvector result_shape(ndim), tmp_shape(ndim); \
        for (size_t j = 0; j != ndim; ++j) { \
            result_shape[j] = 1; \
        } \
        INCREMENTAL_BROADCAST_SRC_ARRAYS(STRIDED_NDIM) \
\
        nd::array result = nd::make_strided_array(data_dynd_types[0], ndim, result_shape.get()); \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[DYND_PP_INC(STRIDED_NDIM)] = {result.get_type(), \
            DYND_PP_FLATTEN(DYND_PP_MAP_1(SRC_ARRAY_GET_TYPE, DYND_PP_RANGE(STRIDED_NDIM)))}; \
        const char *dynd_metadata[DYND_PP_INC(STRIDED_NDIM)] = {result.get_ndo_meta(), \
            DYND_PP_FLATTEN(DYND_PP_MAP_1(SRC_ARRAY_GET_NDO_META, DYND_PP_RANGE(STRIDED_NDIM)))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[STRIDED_NDIM] = {DYND_PP_FLATTEN(DYND_PP_MAP_1(SRC_ARRAY_GET_READONLY_ORIGINPTR, DYND_PP_RANGE(STRIDED_NDIM)))}; \
        op(result.get_readwrite_originptr(), src, ckprefix); \
\
        return result; \
    }

namespace dynd { namespace nd {

namespace detail {


template<class FuncProto>
struct foreach_ckernel_instantiator;

template<typename R, typename T0, typename T1>
struct foreach_ckernel_instantiator<R (*)(T0, T1)> {
    typedef foreach_ckernel_instantiator extra_type;

    ckernel_prefix base;
    R (*func)(T0, T1);

    static void single(char *dst, const char * const *src,
                       ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        *reinterpret_cast<R *>(dst) = e->func(
                            *reinterpret_cast<const T0 *>(src[0]),
                            *reinterpret_cast<const T1 *>(src[1]));
    }

    static void strided(char *dst, intptr_t dst_stride,
                        const char * const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        R (*func)(T0, T1);
        func = e->func;
        const char *src0 = src[0], *src1 = src[1];
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];
        for (size_t i = 0; i < count; ++i) {
            *reinterpret_cast<R *>(dst) = func(
                                *reinterpret_cast<const T0 *>(src0),
                                *reinterpret_cast<const T1 *>(src1));
            dst += dst_stride;
            src0 += src0_stride;
            src1 += src1_stride;
        }
    }

    static intptr_t instantiate(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype)
    {
        extra_type *e = out_ckb->get_at<extra_type>(ckb_offset);
        if (kerntype == kernel_request_single) {
            e->base.set_function(&extra_type::single);
        } else if (kerntype == kernel_request_strided) {
            e->base.set_function(&extra_type::strided);
        } else {
            throw runtime_error("unsupported kernel request in foreach");
        }
        e->func = reinterpret_cast<R (*)(T0, T1)>(self_data_ptr);
        // No need for a destructor function in this ckernel

        return ckb_offset + sizeof(extra_type);
    }
};

template<typename T, typename R, typename A0, typename A1>
struct foreach_ckernel_instantiator<R (T::*)(A0, A1)> {
    typedef foreach_ckernel_instantiator extra_type;

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
                const char *const* DYND_UNUSED(dynd_metadata), uint32_t kerntype)
    {
        extra_type *e = out_ckb->get_at<extra_type>(ckb_offset);
        if (kerntype == kernel_request_single) {
            e->base.set_function(&extra_type::single);
        } else if (kerntype == kernel_request_strided) {
            e->base.set_function(&extra_type::strided);
        } else {
            throw runtime_error("unsupported kernel request in foreach");
        }
        std::pair<T *, R (T::*)(A0, A1)> *pr = reinterpret_cast<std::pair<T *, R (T::*)(A0, A1)> *>(self_data_ptr);
        e->obj = pr->first;
        e->func = pr->second;
        // No need for a destructor function in this ckernel

        return ckb_offset + sizeof(extra_type);
    }
};

} // namespace detail

DYND_PP_JOIN_OUTER(ELWISE, (), DYND_PP_RANGE(1, 4), DYND_PP_RANGE(1, 4))

template<typename R, typename T0, typename T1>
inline nd::array foreach(const nd::array& a, const nd::array& b, R (*func)(T0, T1))
{
    // No casting for now
    if (a.get_dtype() != ndt::make_type<T0>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
        ss << a.get_dtype() << " to " << ndt::make_type<T0>();
        throw type_error(ss.str());
    }
    if (b.get_dtype() != ndt::make_type<T1>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
        ss << b.get_dtype() << " to " << ndt::make_type<T1>();
        throw type_error(ss.str());
    }

    // Create a static ckernel_deferred out of the function
    ckernel_deferred ckd;
    ckd.ckernel_funcproto = expr_operation_funcproto;
    ckd.data_types_size = 3;
    ndt::type data_dynd_types[3] = {ndt::make_type<R>(), ndt::make_type<T0>(), ndt::make_type<T1>()};
    ckd.data_dynd_types = data_dynd_types;
    ckd.data_ptr = reinterpret_cast<void *>(func);
    ckd.instantiate_func = &detail::foreach_ckernel_instantiator<R (*)(T0, T1)>::instantiate;
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
                        lifted_types, dynd_metadata, kernel_request_single);

    // Call the ckernel to do the operation
    ckernel_prefix *ckprefix = ckb.get();
    expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>();
    const char *src[2] = {a.get_readonly_originptr(), b.get_readonly_originptr()};
    op(result.get_readwrite_originptr(), src, ckprefix);

    return result;
}

//DYND_PP_OUTER(ELWISE, (), DYND_PP_RANGE(1, 4), DYND_PP_RANGE(1, 4))


template<typename T, typename R, typename A0, typename A1>
inline nd::array foreach(const nd::array& a, const nd::array& b, T& obj, R (T::*func)(A0, A1))
{
    // No casting for now
    if (a.get_dtype() != ndt::make_type<A0>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
        ss << a.get_dtype() << " to " << ndt::make_type<A0>();
        throw type_error(ss.str());
    }
    if (b.get_dtype() != ndt::make_type<A1>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
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
    ckd.instantiate_func = &detail::foreach_ckernel_instantiator<R (T::*)(A0, A1)>::instantiate;
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
                        lifted_types, dynd_metadata, kernel_request_single);

    // Call the ckernel to do the operation
    ckernel_prefix *ckprefix = ckb.get();
    expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>();
    const char *src[2] = {a.get_readonly_originptr(), b.get_readonly_originptr()};
    op(result.get_readwrite_originptr(), src, ckprefix);

    return result;
}

template<typename T>
inline nd::array foreach(const nd::array& a, const nd::array& b, T obj)
{
    return foreach(a, b, obj, &T::operator ());
}

}} // namespace dynd::nd

#endif // _DYND__FOREACH_HPP_
