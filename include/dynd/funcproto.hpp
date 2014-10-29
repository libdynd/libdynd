//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

namespace dynd {

/**
 * A metaprogram that decays a function pointer, or a member function pointer, to
 * a function (proto)type.
 */
template <typename T>
struct funcproto_from;

template <typename R>
struct funcproto_from<R ()> {
    typedef R (type)();
};

#define FUNCPROTO_FROM(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R DYND_PP_META_NAME_RANGE(A, N)> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_FROM

template <typename func_type>
struct funcproto_from<func_type *> {
    typedef typename funcproto_from<func_type>::type type;
};

#define FUNCPROTO_FROM(N) \
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R (T::*) DYND_PP_META_NAME_RANGE(A, N)> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };  \
\
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R (T::*) DYND_PP_META_NAME_RANGE(A, N) const> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_FROM

} // namespace dynd
