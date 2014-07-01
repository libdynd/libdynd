//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_FUNCREFRES_HPP_
#define _DYND__ELWISE_FUNCREFRES_HPP_

#include <dynd/func/elwise_common.hpp>

namespace dynd { namespace nd {

namespace detail {

#define FUNC_REF_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_ckernel_instantiator<void (*)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC)))> { \
        typedef elwise_ckernel_instantiator extra_type; \
\
        typedef void (*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))); \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
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
                    DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR, \
                        DYND_PP_META_NAME_RANGE(D, NSRC)), \
                    DYND_PP_META_AT_RANGE(src, NSRC)))); \
        } \
\
        static void strided(char *dst, intptr_t dst_stride, \
                            const char * const *src, const intptr_t *src_stride, \
                            size_t count, ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            func_type func = e->func; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, NSRC), DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                func(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
                    DYND_PP_ELWISE_1(DYND_PP_META_REINTERPRET_CAST, \
                        DYND_PP_MAP_1(DYND_PP_META_MAKE_CONST_PTR, \
                            DYND_PP_META_NAME_RANGE(D, NSRC)), \
                        DYND_PP_META_NAME_RANGE(src, NSRC)))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC)); \
            } \
        } \
\
        static intptr_t instantiate( \
                    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb, \
                    intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp), \
                    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *DYND_UNUSED(src_tp), \
                    const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, \
                    const eval::eval_context *DYND_UNUSED(ectx)) \
        { \
            extra_type *e = ckb->alloc_ck_leaf<extra_type>(ckb_offset); \
            e->base.template set_expr_function<extra_type>(kernreq); \
            e->func = *af_self->get_data_as<func_type>(); \
\
            return ckb_offset; \
        } \
    };

DYND_PP_JOIN_MAP(FUNC_REF_RES_CKERNEL_INSTANTIATOR, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef FUNC_REF_RES_CKERNEL_INSTANTIATOR

} // namespace detail

#define FUNC_REF_RES(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(void (*func)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))), \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        DYND_STATIC_ASSERT(!is_const<R>::value, "the reference result must not be const"); \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_STATIC_ASSERT, (;), \
            DYND_PP_ELWISE_1(DYND_PP_META_OR, \
                DYND_PP_MAP_1(DYND_PP_META_NOT, \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, \
                        (is_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (value))), \
                DYND_PP_ELWISE_1(DYND_PP_META_PARENTHESIZED_AND, \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, \
                        (is_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (value)), \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (is_const), \
                        DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                            (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                        (value)))), \
            ("all reference arguments must be const")); \
\
        typedef void (*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))); \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type dst_tp = ndt::cfixed_dim_from_array<R>::make(); \
        ndt::type src_tp[NSRC] = {DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::cfixed_dim_from_array), \
            DYND_PP_META_NAME_RANGE(D, NSRC)), DYND_PP_REPEAT(make, NSRC))}; \
\
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ASGN, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_DECL, \
                (const nd::array), DYND_PP_META_NAME_RANGE(acast, NSRC)), \
            DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                DYND_PP_ELWISE_1(DYND_PP_META_DOT, \
                    DYND_PP_META_NAME_RANGE(a, NSRC), \
                        DYND_PP_OUTER_1(DYND_PP_META_CALL, \
                            (ucast), \
                                DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                                    DYND_PP_META_AT_RANGE(src_tp, 0, NSRC), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(dst_tp, res_ndim, res_shape.get()); \
\
        arrfunc_type_data af; \
        af.func_proto = ndt::make_funcproto(src_tp, dst_tp); \
        *af.get_data_as<func_type>() = func; \
        af.instantiate = &detail::elwise_ckernel_instantiator<func_type>::instantiate; \
        af.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_arrmeta[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_arrmeta))}; \
        intptr_t src_ndim[NSRC]; \
        for (int i = 0; i < NSRC; ++i) { \
          src_ndim[i] = lifted_types[i].get_ndim() - src_tp[i].get_ndim(); \
        } \
        make_lifted_expr_ckernel(&af, &ckb, 0, \
                            res.get_type().get_ndim() - dst_tp.get_ndim(), \
                            res.get_type(), res.get_arrmeta(), \
                            src_ndim, lifted_types, dynd_arrmeta, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_t op = ckprefix->get_function<expr_single_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }

DYND_PP_JOIN_MAP(FUNC_REF_RES, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef FUNC_REF_RES

}} // namespace dynd::nd
 
#endif // _DYND__ELWISE_FUNCREFRES_HPP_
