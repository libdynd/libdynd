//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_COMMON_HPP_
#define _DYND__ELWISE_COMMON_HPP_

#include <utility>

#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
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

template<class FuncProto>
struct elwise_from_callable_ckernel_instantiator;

} // namespace detail

#define CALL_ELWISE_IMPL(NSRC) \
    template<typename T> \
    inline nd::array elwise(const T &obj, DYND_PP_JOIN_OUTER_1(DYND_PP_META_DECL, (,), \
        (const nd::array&), DYND_PP_META_NAME_RANGE(a, NSRC)), const eval::eval_context *ectx = &eval::default_eval_context) { \
        return elwise_from_callable(obj, &T::operator (), DYND_PP_JOIN_1((,), DYND_PP_META_NAME_RANGE(a, NSRC)), ectx); \
    }

DYND_PP_JOIN_MAP(CALL_ELWISE_IMPL, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ELWISE_MAX)))

#undef CALL_ELWISE_IMPL

}} // namespace dynd::nd
 
#endif // _DYND__ELWISE_COMMON_HPP_
