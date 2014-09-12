//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFER_HPP_
#define _DYND__BUFFER_HPP_

#define DYND_PP_POP(A) DYND_PP_DEL(DYND_PP_DEC(DYND_PP_LEN(A)), A)

#include <tr1/type_traits>

#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

#define DYND_ARITY_MAX 4
//#define DYND_PP_ID(...) __VA_ARGS__

namespace dynd { namespace nd {

struct auxiliary_buffer {
};

struct thread_local_buffer {
};

} // namespace dynd::nd

namespace detail {

template <typename T>
struct is_auxiliary_buffer_type {
    static const bool value = std::tr1::is_base_of<nd::auxiliary_buffer, T>::value;
};

template <typename T>
struct is_auxiliary_buffer_arg_type {
    // static_assert if T is anything else involving auxiliary_buffer
    static const bool value = false;
};

template <typename T>
struct is_auxiliary_buffer_arg_type<T *> {
    static const bool value = is_auxiliary_buffer_type<T>::value;
};

template <typename T>
struct is_auxiliary_buffer_arg_type<const T *> {
    static const bool value = is_auxiliary_buffer_type<T>::value;
};

template <typename T>
struct is_auxiliary_buffer_arg_type<T &> {
    static const bool value = is_auxiliary_buffer_type<T>::value;
};

template <typename T>
struct is_auxiliary_buffer_arg_type<const T &> {
    static const bool value = is_auxiliary_buffer_type<T>::value;
};

template <typename T>
struct is_thread_local_buffer_type {
    static const bool value = std::tr1::is_base_of<nd::thread_local_buffer, T>::value;
};

template <typename T>
struct is_thread_local_buffer_arg_type {
    // static_assert if T is anything else involving auxiliary_buffer
    static const bool value = false;
};

template <typename T>
struct is_thread_local_buffer_arg_type<T *> {
    static const bool value = is_thread_local_buffer_type<T>::value;
};

template <typename T>
struct is_thread_local_buffer_arg_type<const T *> {
    static const bool value = is_thread_local_buffer_type<T>::value;
};

template <typename T>
struct is_thread_local_buffer_arg_type<T &> {
    static const bool value = is_thread_local_buffer_type<T>::value;
};

template <typename T>
struct is_thread_local_buffer_arg_type<const T &> {
    static const bool value = is_thread_local_buffer_type<T>::value;
};

template <typename T>
struct inspect_buffered;

template <typename R>
struct inspect_buffered<R (*)()> {
    static const bool is_auxiliary = false;
    static const bool is_only_auxiliary = false;
    static const bool is_thread_local = false;
    static const bool is_only_thread_local = false;
};

template <typename R, typename A0>
struct inspect_buffered<R (*)(A0)> {
    static const bool is_auxiliary = is_auxiliary_buffer_arg_type<A0>::value;
    static const bool is_only_auxiliary = is_auxiliary;
    static const bool is_thread_local = is_thread_local_buffer_arg_type<A0>::value;
    static const bool is_only_thread_local = is_thread_local;
};

#define DYND_INSPECT_BUFFERED(N) DYND__INSPECT_BUFFERED(DYND_PP_META_NAME_RANGE(A, N))
#define DYND__INSPECT_BUFFERED(TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES)> \
    struct inspect_buffered<R (*) TYPES> { \
        static const bool is_thread_local = is_thread_local_buffer_arg_type<DYND_PP_LAST(TYPES)>::value; \
        static const bool is_only_auxiliary = !is_thread_local \
            && is_auxiliary_buffer_arg_type<DYND_PP_LAST(TYPES)>::value; \
        static const bool is_auxiliary = is_only_auxiliary \
            || inspect_buffered<R (*) DYND_PP_POP(TYPES)>::is_only_auxiliary; \
        static const bool is_only_thread_local = is_thread_local; \
        static_assert(!inspect_buffered<R (*) DYND_PP_POP(TYPES)>::is_thread_local, "error"); \
    };

DYND_PP_JOIN_MAP(DYND_INSPECT_BUFFERED, (), DYND_PP_RANGE(2, DYND_PP_ADD(DYND_ARITY_MAX, 2)))

} // namespace dynd::detail

template <typename T>
struct is_auxiliary_buffered {
    static const bool value = detail::inspect_buffered<T>::is_auxiliary;
};

template <typename T>
struct is_only_auxiliary_buffered {
    static const bool value = detail::inspect_buffered<T>::is_only_auxiliary;
};

template <typename T>
struct is_thread_local_buffered {
    static const bool value = detail::inspect_buffered<T>::is_thread_local;
};

template <typename T>
struct is_only_thread_local_buffered {
    static const bool value = detail::inspect_buffered<T>::is_only_thread_local;
};

} // namespace dynd

/*
namespace dynd {

template <typename F, typename T>
struct count;

template <typename R, typename T>
struct count<R (*)(), T> {
    static const int value = 0;
};

#define DYND_COUNT(N) DYND__COUNT(DYND_PP_META_NAME_RANGE(A, N))
#define DYND__COUNT(TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES), typename T> \
    struct count<R (*) TYPES, T> { \
    private:
        typedef DYND_PP_LAST(TYPES) Q;
    public:
        static const int value = count<R (*) DYND_PP_POP(TYPES), T>::value + std::tr1::is_same<Q0, T>::value; \
    };

DYND_PP_JOIN_MAP(DYND_COUNT, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARITY_MAX)))

#undef DYND__COUNT
#undef DYND_COUNT

} // namespace dynd

namespace dynd { namespace nd {

struct auxiliary_buffer {
};

struct thread_local_buffer {
};

} // namespace dynd::nd

namespace detail {

template <typename T>
struct is_auxiliary_buffer_type {
    enum { value = std::tr1::is_base_of<nd::auxiliary_buffer, T>::value };
};

template <typename T>
struct is_auxiliary_buffer_type_pointer {
    enum { value = false };
};

template <typename T>
struct is_auxiliary_buffer_type_pointer<T *> {
    enum { value = is_auxiliary_buffer_type<T>::value };
};

template <typename T>
struct is_auxiliary_buffer_type_pointer<const T *> {
    enum { value = is_auxiliary_buffer_type<T>::value };
};

template <typename T>
struct is_auxiliary_buffer_type_reference {
    enum { value = false };
};

template <typename T>
struct is_auxiliary_buffer_type_reference<T &> {
    enum { value = is_auxiliary_buffer_type<T>::value };
};

template <typename T>
struct is_auxiliary_buffer_type_reference<const T &> {
    enum { value = is_auxiliary_buffer_type<T>::value };
};

template <typename T>
struct is_thread_local_buffer_type {
    enum { value = std::tr1::is_base_of<nd::thread_local_buffer, T>::value };
};

template <typename T>
struct is_thread_local_buffer_type_pointer {
    enum { value = false };
};

template <typename T>
struct is_thread_local_buffer_type_pointer<T *> {
    enum { value = is_thread_local_buffer_type<T>::value };
};

template <typename T>
struct is_thread_local_buffer_type_pointer<const T *> {
    enum { value = is_thread_local_buffer_type<T>::value };
};

template <typename T>
struct is_thread_local_buffer_type_reference {
    enum { value = false };
};

template <typename T>
struct is_thread_local_buffer_type_reference<T &> {
    enum { value = is_thread_local_buffer_type<T>::value };
};

template <typename T>
struct is_thread_local_buffer_type_reference<const T &> {
    enum { value = is_thread_local_buffer_type<T>::value };
};

} // dynd::detail

template <typename R>
struct helper {
private:
    static const int thread_local_buffer_types = 0;
    static const int thread_local_buffer_type_pointers = 0;
    static const int thread_local_buffer_type_references = 0;
};


template <typename T>
struct is_thread_local_buffered;

template <typename R>
struct is_thread_local_buffered<R (*)()> {
    enum { value = false };
};

#define DYND_IS_THREAD_LOCAL_BUFFERED(N) DYND__IS_THREAD_LOCAL_BUFFERED(DYND_PP_META_NAME_RANGE(A, N))
#define DYND__IS_THREAD_LOCAL_BUFFERED(TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES)> \
    struct is_thread_local_buffered<R (*) TYPES> { \
    private: \
        typedef R (*) TYPES func_type; \
        typedef DYND_PP_LAST(TYPES) Q0; \
    public: \
        enum { value = (detail::is_thread_local_buffer_type_pointer<Q0>::value \
            || detail::is_thread_local_buffer_type_reference<Q0>::value) }; \
    };

DYND_PP_JOIN_MAP(DYND_IS_THREAD_LOCAL_BUFFERED, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARITY_MAX)))

#undef DYND__IS_THEAD_LOCAL_BUFFERED
#undef DYND_IS_THEAD_LOCAL_BUFFERED

template <typename T>
struct is_auxiliary_buffered;

template <typename R>
struct is_auxiliary_buffered<R (*)()> {
    enum { value = false };
};

template <typename R, typename A0>
struct is_auxiliary_buffered<R (*)(A0)> {
    enum { value = detail::is_auxiliary_buffer_type<A0>::value };
};

#define DYND_IS_AUXILIARY_BUFFERED(N) DYND__IS_AUXILIARY_BUFFERED(DYND_PP_META_NAME_RANGE(A, N))
#define DYND__IS_AUXILIARY_BUFFERED(TYPES) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), TYPES)> \
    struct is_auxiliary_buffered<R (*) TYPES> { \
        typedef DYND_PP_LAST(TYPES) Q0; \
        typedef DYND_PP_LAST(DYND_PP_POP(TYPES)) Q1; \
        enum { value = detail::is_auxiliary_buffer_type<Q1>::value || \
            (detail::is_auxiliary_buffer_type<Q0>::value && detail::is_thread_local_buffer_type<Q1>::value) }; \
    };

DYND_PP_JOIN_MAP(DYND_IS_AUXILIARY_BUFFERED, (), DYND_PP_RANGE(2, DYND_PP_INC(DYND_ARITY_MAX)))

#undef DYND__IS_AUXILIARY_BUFFERED
#undef DYND_IS_AUXILIARY_BUFFERED

} // namespace dynd
*/

#endif // _DYND__BUFFER_HPP_
