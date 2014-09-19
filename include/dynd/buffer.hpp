//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_HPP_
#define _DYND__BUFFER_HPP_

#include <dynd/funcproto.hpp>

namespace dynd {

struct aux_buffer {
};

struct thread_aux_buffer {
};

namespace detail {

template <typename T>
struct is_aux_buffer_type {
    static const bool value = std::is_base_of<aux_buffer, T>::value;
};

template <typename T>
struct assert_aux_buffer_arg_type {
protected:
    typedef typename remove_all_pointers<typename std::decay<T>::type>::type D;
public:
    static const bool value = is_aux_buffer_type<D>::value;
    DYND_STATIC_ASSERT(!value || (value && std::is_pointer<T>::value
        && !std::is_pointer<typename std::remove_pointer<T>::type>::value),
        "aux_buffer, or a subclass of it, must be passed as a pointer");
};

template <typename T>
struct is_thread_aux_buffer_type {
    static const bool value = std::is_base_of<thread_aux_buffer, T>::value;
};

template <typename T>
struct assert_thread_aux_buffer_arg_type {
protected:
    typedef typename remove_all_pointers<typename std::decay<T>::type>::type D;
public:
    static const bool value = is_thread_aux_buffer_type<D>::value;
    DYND_STATIC_ASSERT(!value || (value && std::is_pointer<T>::value
        && !std::is_pointer<typename std::remove_pointer<T>::type>::value),
        "thread_aux_buffer, or a subclass of it, must be passed as a pointer");
};

} // namespace detail

#define AUX_BUFFER_MESSAGE "a subclass of aux_buffer, if present, must be " \
    "only the last argument or only immediately precede a subclass of thread_aux_buffer"
#define THREAD_AUX_BUFFER_MESSAGE "a subclass of thread_aux_buffer, if present, must be " \
    "only the last argument"

template <typename func_type>
struct assert_buffered_args;

template <typename R>
struct assert_buffered_args<R ()> {
    struct is_thread_aux_buffered {
        static const bool value = false;
    };
    struct is_aux_buffered {
        static const bool value = false;
    };
};

template <typename R, typename A0>
struct assert_buffered_args<R (A0)> {
    struct is_thread_aux_buffered {
        static const bool value = detail::assert_thread_aux_buffer_arg_type<A0>::value;
    };
    struct is_aux_buffered {
        static const bool value = detail::assert_aux_buffer_arg_type<A0>::value;
    };
};

template <typename R, typename A0, typename A1>
struct assert_buffered_args<R (A0, A1)> {
    struct is_thread_aux_buffered {
        static const bool value = detail::assert_thread_aux_buffer_arg_type<A1>::value;
    };
    struct is_aux_buffered {
        static const bool value = detail::assert_aux_buffer_arg_type<A1>::value ||
            (is_thread_aux_buffered::value && detail::assert_aux_buffer_arg_type<A0>::value);
    };
    DYND_STATIC_ASSERT(!detail::assert_thread_aux_buffer_arg_type<A0>::value,
        THREAD_AUX_BUFFER_MESSAGE);
    DYND_STATIC_ASSERT(is_thread_aux_buffered::value || !detail::assert_aux_buffer_arg_type<A0>::value,
        AUX_BUFFER_MESSAGE);
};

#define ASSERT_AUX_BUFFER_ARG_TYPE(TYPE) detail::assert_aux_buffer_arg_type<TYPE>
#define ASSERT_THREAD_AUX_BUFFER_ARG_TYPE(TYPE) detail::assert_thread_aux_buffer_arg_type<TYPE>

#define ASSERT_BUFFERED_ARGS(N) _ASSERT_BUFFERED_ARGS(N, DYND_PP_META_NAME_RANGE(A, N))
#define _ASSERT_BUFFERED_ARGS(N, TYPES) __ASSERT_BUFFERED_ARGS(N, TYPES, \
    DYND_PP_MAP_1(ASSERT_AUX_BUFFER_ARG_TYPE, TYPES), DYND_PP_MAP_1(ASSERT_THREAD_AUX_BUFFER_ARG_TYPE, TYPES))
#define __ASSERT_BUFFERED_ARGS(N, TYPES, ASSERT_AUX_BUFFER_ARG_TYPES, ASSERT_THREAD_AUX_BUFFER_ARG_TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES)> \
    struct assert_buffered_args<R (DYND_PP_JOIN_1((,), TYPES))> { \
        struct is_thread_aux_buffered { \
            static const bool value = DYND_PP_LAST(ASSERT_THREAD_AUX_BUFFER_ARG_TYPES)::value; \
        }; \
        struct is_aux_buffered { \
            static const bool value = DYND_PP_LAST(ASSERT_AUX_BUFFER_ARG_TYPES)::value || \
                (is_thread_aux_buffered::value && DYND_PP_LAST(DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_TYPES))::value); \
        }; \
        DYND_STATIC_ASSERT(!DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
            DYND_PP_POP(ASSERT_THREAD_AUX_BUFFER_ARG_TYPES), DYND_PP_REPEAT_1(value, DYND_PP_DEC(N))), \
            THREAD_AUX_BUFFER_MESSAGE); \
        DYND_STATIC_ASSERT(!(is_thread_aux_buffered::value && \
            (DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
                DYND_PP_POP(DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_TYPES)), DYND_PP_REPEAT_1(value, DYND_PP_DEC(DYND_PP_DEC(N)))))), \
            AUX_BUFFER_MESSAGE); \
        DYND_STATIC_ASSERT(!(!is_thread_aux_buffered::value && \
            (DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
                DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_TYPES), DYND_PP_REPEAT_1(value, DYND_PP_DEC(N))))), \
            AUX_BUFFER_MESSAGE); \
    };

DYND_PP_JOIN_MAP(ASSERT_BUFFERED_ARGS, (), DYND_PP_RANGE(3, DYND_PP_INC(DYND_ARG_MAX)))

#undef ASSERT_BUFFERED_ARGS
#undef _ASSERT_BUFFERED_ARGS
#undef __ASSERT_BUFFERED_ARGS

#undef ASSERT_AUX_BUFFER_ARG_TYPE
#undef ASSERT_THREAD_AUX_BUFFER_ARG_TYPE

#undef AUX_BUFFER_MESSAGE
#undef THREAD_AUX_BUFFER_MESSAGE

template <typename func_type>
struct is_aux_buffered {
    static const bool value = assert_buffered_args<typename funcproto_from<func_type>::type>::is_aux_buffered::value;
};

template <typename func_type>
struct is_thread_aux_buffered {
    static const bool value = assert_buffered_args<typename funcproto_from<func_type>::type>::is_thread_aux_buffered::value;
};

} // namespace dynd

#endif // _DYND__BUFFER_HPP_
