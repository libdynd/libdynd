//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_HPP_
#define _DYND__BUFFER_HPP_

#define DYND_PP_POP(A) DYND_PP_DEL(DYND_PP_DEC(DYND_PP_LEN(A)), A)
#define DYND_PP_META_VIRTUAL(NAME) virtual NAME

#include <dynd/config.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

#define DYND_STATIC_ASSERT2(X, Y)

namespace dynd {

template <typename T> class remove_all_pointers{
public:
    typedef T type;
};

template <typename T> class remove_all_pointers<T*>{
public:
    typedef typename remove_all_pointers<T>::type type;
};

template <typename T> class remove_all_pointers<T* const>{
public:
    typedef typename remove_all_pointers<T>::type type;
};

template <typename T> class remove_all_pointers<T* volatile>{
public:
    typedef typename remove_all_pointers<T>::type type;
};

template <typename T> class remove_all_pointers<T* const volatile >{
public:
    typedef typename remove_all_pointers<T>::type type;
};

struct aux_buffer {
};

struct thread_aux_buffer {
};

namespace detail {

template <typename T>
struct is_aux_buffer_type {
    static const bool value = std::tr1::is_base_of<aux_buffer, T>::value;
};

template <typename T>
struct is_aux_buffer_arg_type {
    // DYND_STATIC_ASSERT2 if T is anything else involving aux_buffer
    static const bool value = false;
};

template <typename T>
struct is_aux_buffer_arg_type<T *> {
    static const bool value = is_aux_buffer_type<T>::value;
};

template <typename T>
struct is_aux_buffer_arg_type<const T *> {
    static const bool value = is_aux_buffer_type<T>::value;
};

template <typename T>
struct is_aux_buffer_arg_type<T &> {
    static const bool value = is_aux_buffer_type<T>::value;
};

template <typename T>
struct is_aux_buffer_arg_type<const T &> {
    static const bool value = is_aux_buffer_type<T>::value;
};

template <typename T>
struct is_thread_aux_buffer_type {
    static const bool value = std::tr1::is_base_of<thread_aux_buffer, T>::value;
};

template <typename T>
struct is_thread_aux_buffer_arg_type {
    // DYND_STATIC_ASSERT2 if T is anything else involving aux_buffer
    static const bool value = false;
};

template <typename T>
struct is_thread_aux_buffer_arg_type<T *> {
    static const bool value = is_thread_aux_buffer_type<T>::value;
};

template <typename T>
struct is_thread_aux_buffer_arg_type<const T *> {
    static const bool value = is_thread_aux_buffer_type<T>::value;
};

template <typename T>
struct is_thread_aux_buffer_arg_type<T &> {
    static const bool value = is_thread_aux_buffer_type<T>::value;
};

template <typename T>
struct is_thread_aux_buffer_arg_type<const T &> {
    static const bool value = is_thread_aux_buffer_type<T>::value;
};

//         DYND_STATIC_ASSERT2(!inspect_buffered<void (*) (R &, DYND_PP_FLATTEN(DYND_PP_POP(TYPES)))>::is_thread_local, "error");

} // namespace dynd::detail

template <typename func_type>
struct is_thread_aux_buffer_last;

template <typename func_type>
struct has_thread_aux_buffer; // is there a thread_aux_buffer at all



namespace detail {

template <typename T>
struct assert_aux_buffer_arg_type {
protected:
    typedef typename remove_all_pointers<typename std::decay<T>::type>::type D;
public:
    static const bool value = is_aux_buffer_type<D>::value;
    DYND_STATIC_ASSERT2(!value || (value && std::tr1::is_pointer<T>::value
        && !std::tr1::is_pointer<typename std::tr1::remove_pointer<T>::type>::value),
        "aux_buffer, or a subclass of it, must be passed as a pointer");
};

template <typename T>
struct assert_thread_aux_buffer_arg_type {
protected:
    typedef typename remove_all_pointers<typename std::decay<T>::type>::type D;
public:
    static const bool value = is_thread_aux_buffer_type<D>::value;
    DYND_STATIC_ASSERT2(!value || (value && std::tr1::is_pointer<T>::value
        && !std::tr1::is_pointer<typename std::tr1::remove_pointer<T>::type>::value),
        "thread_aux_buffer, or a subclass of it, must be passed as a pointer");
};

}

#define DYND_AUX_BUFFER_MESSAGE "a subclass of aux_buffer, if present, must be only the last argument or only immediately precede a subclass of thread_aux_buffer"
#define DYND_THREAD_AUX_BUFFER_MESSAGE "a subclass of thread_aux_buffer, if present, must be only the last argument"

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
    DYND_STATIC_ASSERT2(!detail::assert_thread_aux_buffer_arg_type<A0>::value,
        DYND_THREAD_AUX_BUFFER_MESSAGE);
    DYND_STATIC_ASSERT2(is_thread_aux_buffered::value || !detail::assert_aux_buffer_arg_type<A0>::value,
        DYND_AUX_BUFFER_MESSAGE);
};

#define DYND_ASSERT_BUFFERED_ARGS(N) DYND__ASSERT_BUFFERED_ARGS(N, DYND_PP_META_NAME_RANGE(A, N))
#define DYND__ASSERT_BUFFERED_ARGS(N, TYPES) DYND___ASSERT_BUFFERED_ARGS(N, TYPES, \
    DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION, \
        DYND_PP_REPEAT_1(detail::assert_aux_buffer_arg_type, N), TYPES), \
    DYND_PP_ELWISE_1(DYND_PP_META_TEMPLATE_INSTANTIATION, \
        DYND_PP_REPEAT_1(detail::assert_thread_aux_buffer_arg_type, N), TYPES))
#define DYND___ASSERT_BUFFERED_ARGS(N, TYPES, ASSERT_AUX_BUFFER_ARG_BASES, ASSERT_THREAD_AUX_BUFFER_ARG_BASES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES)> \
    struct assert_buffered_args<R (DYND_PP_JOIN_1((,), TYPES))> { \
        struct is_thread_aux_buffered { \
            static const bool value = DYND_PP_LAST(ASSERT_THREAD_AUX_BUFFER_ARG_BASES)::value; \
        }; \
        struct is_aux_buffered { \
            static const bool value = DYND_PP_LAST(ASSERT_AUX_BUFFER_ARG_BASES)::value || \
                (is_thread_aux_buffered::value && DYND_PP_LAST(DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_BASES))::value); \
        }; \
        DYND_STATIC_ASSERT2(!DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
            DYND_PP_POP(ASSERT_THREAD_AUX_BUFFER_ARG_BASES), DYND_PP_REPEAT_1(value, DYND_PP_DEC(N))), \
            DYND_THREAD_AUX_BUFFER_MESSAGE); \
        DYND_STATIC_ASSERT2(!(is_thread_aux_buffered::value && \
            (DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
                DYND_PP_POP(DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_BASES)), DYND_PP_REPEAT_1(value, DYND_PP_DEC(DYND_PP_DEC(N)))))), \
            DYND_AUX_BUFFER_MESSAGE); \
        DYND_STATIC_ASSERT2(!(!is_thread_aux_buffered::value && \
            (DYND_PP_JOIN_ELWISE_1(DYND_PP_META_SCOPE, (||), \
                DYND_PP_POP(ASSERT_AUX_BUFFER_ARG_BASES), DYND_PP_REPEAT_1(value, DYND_PP_DEC(N))))), \
            DYND_AUX_BUFFER_MESSAGE); \
    };

DYND_PP_JOIN_MAP(DYND_ASSERT_BUFFERED_ARGS, (), DYND_PP_RANGE(3, DYND_PP_INC(DYND_ARG_MAX)))

#undef DYND___ASSERT_BUFFERED_ARGS
#undef DYND__ASSERT_BUFFERED_ARGS
#undef DYND_ASSERT_BUFFERED_ARGS

template <typename T>
struct func_like;

template <typename R>
struct func_like<R ()> {
    typedef R (type)();
};

#define DYND_FUNC_LIKE(N) DYND__FUNC_LIKE(N, DYND_PP_META_NAME_RANGE(A, N))
#define DYND__FUNC_LIKE(N, ARG_TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPES)> \
    struct func_like<R ARG_TYPES> { \
        typedef R (type) ARG_TYPES; \
    };

DYND_PP_JOIN_MAP(DYND_FUNC_LIKE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef DYND__FUNC_LIKE
#undef DYND_FUNC_LIKE

template <typename R>
struct func_like<R (*)()> {
    typedef R (type)();
};


#define DYND_FUNC_LIKE(N) DYND__FUNC_LIKE(N, DYND_PP_META_NAME_RANGE(A, N))
#define DYND__FUNC_LIKE(N, ARG_TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPES)> \
    struct func_like<R (*) ARG_TYPES> { \
        typedef R (type) ARG_TYPES; \
    };

DYND_PP_JOIN_MAP(DYND_FUNC_LIKE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef DYND__FUNC_LIKE
#undef DYND_FUNC_LIKE

#define DYND_FUNC_LIKE(N) DYND__FUNC_LIKE(N, DYND_PP_META_NAME_RANGE(A, N))
#define DYND__FUNC_LIKE(N, ARG_TYPES) \
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPES)> \
    struct func_like<R (T::*) ARG_TYPES> { \
        typedef R (type) ARG_TYPES; \
    };  \
\
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), ARG_TYPES)> \
    struct func_like<R (T::*) ARG_TYPES const> { \
        typedef R (type) ARG_TYPES; \
    };

DYND_PP_JOIN_MAP(DYND_FUNC_LIKE, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef DYND__FUNC_LIKE
#undef DYND_FUNC_LIKE

template <typename func_type>
struct is_aux_buffered {
    static const bool value = assert_buffered_args<typename func_like<func_type>::type>::is_aux_buffered::value;
};

template <typename func_type>
struct is_thread_aux_buffered {
    static const bool value = assert_buffered_args<typename func_like<func_type>::type>::is_thread_aux_buffered::value;
};

template <typename func_type, bool aux_buffered, bool thread_aux_buffered>
struct test {
};

template <typename func_type>
struct remove_all_buffers : test<func_type, is_aux_buffered<func_type>::value, is_thread_aux_buffered<func_type>::value> {

};

} // namespace dynd

#endif // _DYND__BUFFER_HPP_
