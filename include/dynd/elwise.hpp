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

namespace dynd { namespace nd {

namespace detail {

#define BROADCAST(NSRC) \
    template<DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(D, NSRC))> \
    void elwise_broadcast(DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array &), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        intptr_t& out_ndim, dynd::dimvector& out_shape) { \
\
        DYND_PP_IF_ELSE(DYND_PP_EQ(NSRC, 1)) ( \
            out_ndim = a0.get_ndim() - detail::ndim_from_array<D0>::value; \
            a0.get_shape(out_shape.get()); \
        )( \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ASGN, (;), \
                DYND_PP_OUTER_1(DYND_PP_META_DECL, (intptr_t), \
                    DYND_PP_META_NAME_RANGE(ndim, NSRC)), \
                DYND_PP_ELWISE_1(DYND_PP_META_SUB, \
                    DYND_PP_OUTER_1(DYND_PP_META_DOT_CALL, \
                        DYND_PP_META_NAME_RANGE(a, NSRC), (get_ndim)), \
                    DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, \
                        (detail::ndim_from_array), DYND_PP_META_NAME_RANGE(D, NSRC), (value)))); \
\
            out_ndim = ndim0; \
            DYND_PP_JOIN_OUTER_1(DYND_PP_META_ASGN, (;), (out_ndim), \
                DYND_PP_OUTER_1(DYND_PP_META_CALL, (std::max), \
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

DYND_PP_JOIN_MAP(BROADCAST, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef BROADCAST

template<class FuncProto>
struct elwise_ckernel_instantiator;

#define FUNC_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_ckernel_instantiator<R (*)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC)))> { \
        typedef elwise_ckernel_instantiator extra_type; \
\
        typedef R (*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))); \
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
            *reinterpret_cast<R*>(dst) = e->func(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                *reinterpret_cast<R*>(dst) = func(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            e->func = reinterpret_cast<func_type>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            e->func = reinterpret_cast<func_type>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
#define FUNC_CKERNEL_INSTANTIATORS(NSRC) \
    FUNC_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    FUNC_REF_RES_CKERNEL_INSTANTIATOR(NSRC)

DYND_PP_JOIN_MAP(FUNC_CKERNEL_INSTANTIATORS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef FUNC_CKERNEL_INSTANTIATORS
#undef FUNC_REF_RES_CKERNEL_INSTANTIATOR
#undef FUNC_RET_RES_CKERNEL_INSTANTIATOR

#define METH_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_ckernel_instantiator<R (T::*)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const> { \
        typedef elwise_ckernel_instantiator extra_type; \
\
        typedef R (T::*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ckernel_prefix base; \
        const T *obj; \
        func_type func; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            *reinterpret_cast<R*>(dst) = ((e->obj)->*(e->func))(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
            const T *obj = e->obj; \
            func_type func = e->func; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, NSRC), DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                *reinterpret_cast<R*>(dst) = (obj->*func)(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            std::pair<const T *, func_type> *obj_func = reinterpret_cast<std::pair<const T *, func_type> *>(self_data_ptr); \
            e->obj = obj_func->first; \
            e->func = obj_func->second; \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
#define METH_REF_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_ckernel_instantiator<void (T::*)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const> { \
        typedef elwise_ckernel_instantiator extra_type; \
\
        typedef void (T::*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ckernel_prefix base; \
        const T *obj; \
        func_type func; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            ((e->obj)->*(e->func))(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
            const T *obj = e->obj; \
            func_type func = e->func; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, NSRC), DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                (obj->*func)(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            std::pair<const T *, func_type> *obj_func = reinterpret_cast<std::pair<const T *, func_type> *>(self_data_ptr); \
            e->obj = obj_func->first; \
            e->func = obj_func->second; \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
#define METH_CKERNEL_INSTANTIATORS(NSRC) \
    METH_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    METH_REF_RES_CKERNEL_INSTANTIATOR(NSRC)

DYND_PP_JOIN_MAP(METH_CKERNEL_INSTANTIATORS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef METH_CKERNEL_INSTANTIATORS
#undef METH_REF_RES_CKERNEL_INSTANTIATOR
#undef METH_RET_RES_CKERNEL_INSTANTIATOR

template<class FuncProto>
struct elwise_from_callable_ckernel_instantiator;

#define CALL_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_from_callable_ckernel_instantiator<R (T::*)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const> { \
        typedef elwise_from_callable_ckernel_instantiator extra_type; \
\
        typedef R (T::*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ckernel_prefix base; \
        const T *obj; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            *reinterpret_cast<R*>(dst) = (*(e->obj))(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
            const T *obj = e->obj; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, NSRC), DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                *reinterpret_cast<R*>(dst) = (*obj)(DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            e->obj = reinterpret_cast<const T *>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
#define CALL_REF_RES_CKERNEL_INSTANTIATOR(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    struct elwise_from_callable_ckernel_instantiator<void (T::*)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const> { \
        typedef elwise_from_callable_ckernel_instantiator extra_type; \
\
        typedef void (T::*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ckernel_prefix base; \
        const T *obj; \
\
        static void single(char *dst, const char * const *src, \
                           ckernel_prefix *ckp) \
        { \
            extra_type *e = reinterpret_cast<extra_type *>(ckp); \
            (*(e->obj))(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
            const T *obj = e->obj; \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, NSRC), DYND_PP_META_NAME_RANGE(src, NSRC), DYND_PP_META_AT_RANGE(src, NSRC)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, NSRC), DYND_PP_META_NAME_RANGE(src_stride, NSRC), DYND_PP_META_AT_RANGE(src_stride, NSRC)); \
            for (size_t i = 0; i < count; ++i) { \
                (*obj)(*reinterpret_cast<R*>(dst), DYND_PP_JOIN_MAP_1(DYND_PP_META_DEREFERENCE, (,), \
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
                throw std::runtime_error("unsupported kernel request in elwise"); \
            } \
            e->obj = reinterpret_cast<const T *>(self_data_ptr); \
\
            return ckb_offset + sizeof(extra_type); \
        } \
    };
#define CALL_CKERNEL_INSTANTIATORS(NSRC) \
    CALL_RET_RES_CKERNEL_INSTANTIATOR(NSRC) \
    CALL_REF_RES_CKERNEL_INSTANTIATOR(NSRC)

DYND_PP_JOIN_MAP(CALL_CKERNEL_INSTANTIATORS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef CALL_CKERNEL_INSTANTIATORS
#undef CALL_REF_RES_CKERNEL_INSTANTIATOR
#undef CALL_RET_RES_CKERNEL_INSTANTIATOR

}; // namespace detail

#define FUNC_RET_RES(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(R (*func)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))), \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
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
        typedef R (*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))); \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(func); \
        ckd.instantiate_func = &detail::elwise_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define FUNC_REF_RES(NSRC) \
    template<typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(void (*func)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))), \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
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
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(func); \
        ckd.instantiate_func = &detail::elwise_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define FUNCS(NSRC) \
    FUNC_RET_RES(NSRC) \
    FUNC_REF_RES(NSRC)

DYND_PP_JOIN_MAP(FUNCS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef FUNCS
#undef FUNC_REF_RES
#undef FUNC_RET_RES

#define METH_RET_RES(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(const T& obj, R (T::*func)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const, \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
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
        typedef R (T::*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        std::pair<const T *, func_type> obj_func(&obj, func); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(&obj_func); \
        ckd.instantiate_func = &detail::elwise_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define METH_REF_RES(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise(const T& obj, void (T::*func)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const, \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
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
        typedef void (T::*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        std::pair<const T *, func_type> obj_func(&obj, func); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(&obj_func); \
        ckd.instantiate_func = &detail::elwise_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define METHS(NSRC) \
    METH_RET_RES(NSRC) \
    METH_REF_RES(NSRC)

DYND_PP_JOIN_MAP(METHS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef METHS
#undef METHS_REF_RES
#undef METHS_RET_RES

#define CALL_RET_RES(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise_from_callable(const T& obj, R (T::*)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const, \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
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
        typedef R (T::*func_type)(DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(const_cast<T *>(&obj)); \
        ckd.instantiate_func = &detail::elwise_from_callable_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define CALL_REF_RES(NSRC) \
    template<typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, NSRC))> \
    inline nd::array elwise_from_callable(const T& obj, void (T::*)(R&, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const, \
        DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), \
        const eval::eval_context *ectx = &eval::default_eval_context) \
    { \
        static_assert(!is_const<R>::value, "the reference result must not be const"); \
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
        typedef void (T::*func_type)(R &, DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(A, NSRC))) const; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_OUTER_1(DYND_PP_META_TEMPLATE_INSTANTIATION_SCOPE, (remove_const), \
                DYND_PP_OUTER_1(DYND_PP_META_TYPENAME_TEMPLATE_INSTANTIATION_SCOPE, \
                    (remove_reference), DYND_PP_META_NAME_RANGE(A, NSRC), (type)), \
                (type)), \
            DYND_PP_META_NAME_RANGE(D, NSRC)); \
\
        ndt::type data_dynd_types[NSRC + 1] = {ndt::fixed_dim_from_array<R>::make(), DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE_CALL, (,), \
            DYND_PP_OUTER(DYND_PP_META_TEMPLATE_INSTANTIATION, (ndt::fixed_dim_from_array), \
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
                                    DYND_PP_META_AT_RANGE(data_dynd_types, 1, DYND_PP_INC(NSRC)), (get_dtype)))), \
                (eval))); \
\
        intptr_t res_ndim; \
        dimvector res_shape; \
        detail::elwise_broadcast<DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(D, NSRC))>(DYND_PP_JOIN_1((,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC)), res_ndim, res_shape); \
        nd::array res = nd::make_strided_array(data_dynd_types[0], res_ndim, res_shape.get()); \
\
        ckernel_deferred ckd; \
        ckd.ckernel_funcproto = expr_operation_funcproto; \
        ckd.data_types_size = NSRC + 1; \
        ckd.data_dynd_types = data_dynd_types; \
        ckd.data_ptr = reinterpret_cast<void *>(const_cast<T *>(&obj)); \
        ckd.instantiate_func = &detail::elwise_from_callable_ckernel_instantiator<func_type>::instantiate; \
        ckd.free_func = NULL; \
\
        ckernel_builder ckb; \
        ndt::type lifted_types[NSRC + 1] = {res.get_type(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_type))}; \
        const char *dynd_metadata[NSRC + 1] = {res.get_ndo_meta(), DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_ndo_meta))}; \
        make_lifted_expr_ckernel(&ckd, &ckb, 0, \
                            lifted_types, dynd_metadata, kernel_request_single, ectx); \
\
        ckernel_prefix *ckprefix = ckb.get(); \
        expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>(); \
        const char *src[NSRC] = {DYND_PP_JOIN_OUTER_1(DYND_PP_META_DOT_CALL, (,), \
            DYND_PP_META_NAME_RANGE(acast, NSRC), (get_readonly_originptr))}; \
        op(res.get_readwrite_originptr(), src, ckprefix); \
\
        return res; \
    }
#define CALLS(NSRC) \
    template<typename T> \
    inline nd::array elwise(const T &obj, DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), \
        (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), const eval::eval_context *ectx = &eval::default_eval_context) { \
        return elwise_from_callable(obj, &T::operator (), DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(a, NSRC)), ectx); \
    } \
    CALL_RET_RES(NSRC) \
    CALL_REF_RES(NSRC) \

DYND_PP_JOIN_MAP(CALLS, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef CALLS
#undef CALL_REF_RES
#undef CALL_RET_RES

}} // namespace dynd::nd

#endif // _DYND__ELWISE_HPP_
