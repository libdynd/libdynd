//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
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
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>



using namespace std;

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


#define ELWISE(NSRC) \
    namespace detail { \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_VAR_RANGE(T, NSRC))> \
    struct foreach_ckernel_instantiator<void (*)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_VAR_RANGE(T, NSRC)))> { \
        typedef foreach_ckernel_instantiator extra_type; \
        typedef void (*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_VAR_RANGE(T, NSRC))); \
\
        ckernel_prefix base; \
        func_type func; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            e->func(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
                    DYND_PP_ELWISE(DYND_PP_META_REINTERPRET_CAST, \
                    DYND_PP_MAP_1(DYND_PP_META_AS_CONST_PTR, DYND_PP_META_VAR_RANGE(T, NSRC)), \
                    DYND_PP_META_VAR_AT_RANGE(src, NSRC)))); \
        } \
\
        static void strided(char *dst, intptr_t dst_stride, \
                            const char * const *src, const intptr_t *src_stride, \
                            size_t count, ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            func_type func = e->func; \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DECL, (;), \
                (const char *), DYND_PP_ELWISE(DYND_PP_META_EQ, DYND_PP_META_VAR_RANGE(src, NSRC), DYND_PP_META_VAR_AT_RANGE(src, NSRC))); \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DECL, (;), \
                (intptr_t), DYND_PP_ELWISE(DYND_PP_META_EQ, DYND_PP_OUTER(DYND_PP_PASTE, DYND_PP_META_VAR_RANGE(src, NSRC), (_stride)), DYND_PP_META_VAR_AT_RANGE(src_stride, NSRC))); \
            for (size_t i = 0; i < count; ++i) { \
                func(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
                    DYND_PP_ELWISE(DYND_PP_META_REINTERPRET_CAST,  DYND_PP_MAP_1(DYND_PP_META_AS_CONST_PTR, DYND_PP_META_VAR_RANGE(T, NSRC)), DYND_PP_META_VAR_RANGE(src, NSRC)))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE(DYND_PP_META_ADD_EQ, (;), DYND_PP_META_VAR_RANGE(src, NSRC), DYND_PP_OUTER(DYND_PP_PASTE, DYND_PP_META_VAR_RANGE(src, NSRC), (_stride))); \
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
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_VAR_RANGE(T, NSRC))> \
    inline nd::array foreach(DYND_PP_JOIN_OUTER(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_VAR_RANGE(a, NSRC)), \
        void (*func)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_VAR_RANGE(T, NSRC)))) \
    { \
        const nd::array *srcs[NSRC] = {DYND_PP_JOIN_MAP_1(DYND_PP_META_ADDRESS, (,), DYND_PP_META_VAR_RANGE(a, NSRC))}; \
\
        const intptr_t res_fixed_ndim = detail::ndim_from_array<R>::value; \
        typedef typename detail::uniform_type_from_array<R>::type res_uniform_type; \
        ndt::type res_data_dynd_type; \
        if (res_fixed_ndim > 0) { \
            intptr_t res_fixed_shape[res_fixed_ndim + 1]; \
            detail::fill_shape<R>::fill(res_fixed_shape); \
            res_data_dynd_type = ndt::make_fixed_dim(res_fixed_ndim, res_fixed_shape, ndt::make_type<res_uniform_type>(), NULL); \
        } else { \
            res_data_dynd_type = ndt::make_type<res_uniform_type>(); \
        } \
\
        intptr_t res_strided_ndim; \
        dimvector res_strided_shape; \
        incremental_broadcast_input_shapes(NSRC, srcs, res_strided_ndim, res_strided_shape); \
        nd::array result = nd::make_strided_array(res_data_dynd_type, res_strided_ndim, res_strided_shape.get()); \
\
        ndt::type data_dynd_types[DYND_PP_INC(NSRC)] = {res_data_dynd_type, DYND_PP_JOIN_MAP_1(DYND_PP_META_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE, (ndt::make_type), DYND_PP_META_VAR_RANGE(T, NSRC)))}; \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = DYND_PP_INC(NSRC); \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(func); \
        ckd.instantiate_func = &detail::foreach_ckernel_instantiator<void (*)(R&, \
            DYND_PP_JOIN_1((,), DYND_PP_META_VAR_RANGE(T, NSRC)))>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[DYND_PP_INC(NSRC)] = {result.get_type(), \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_VAR_RANGE(a, NSRC), (get_type))}; \
        const char *dynd_metadata[DYND_PP_INC(NSRC)] = {result.get_ndo_meta(), \
            DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_VAR_RANGE(a, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER(DYND_PP_META_DOT_CALL, (,), DYND_PP_META_VAR_RANGE(a, NSRC), (get_readonly_originptr))}; \
        op(result.get_readwrite_originptr(), src, ckprefix); \
\
        return result; \
    }

namespace dynd {
void incremental_broadcast_input_shapes(intptr_t ninputs, const nd::array** inputs,
                intptr_t &out_undim, dimvector& out_shape, bool include_scalars = false) {
    out_undim = inputs[0]->get_ndim();
    for (int j = 1; j < ninputs; j++) {
        out_undim = max(inputs[j]->get_ndim(), out_undim);
    }
    out_shape.init(out_undim);
    for (intptr_t j = 0; j != out_undim; ++j) {
        out_shape[j] = 1;
    }

    dimvector tmp_shape(out_undim);
    for (int j = 0; j < ninputs; j++) {
        if (include_scalars || inputs[j]->get_ndim() > 0) {
            inputs[j]->get_shape(tmp_shape.get());
            incremental_broadcast(out_undim, out_shape.get(), inputs[j]->get_ndim(), tmp_shape.get());
        }
    }
}
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

DYND_PP_JOIN_MAP(ELWISE, (), DYND_PP_RANGE(1, 4))


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

    const nd::array *inputs[2] = {&a, &b};

    intptr_t ndim;
    dimvector result_shape;
    incremental_broadcast_input_shapes(2, inputs, ndim, result_shape, false);

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
