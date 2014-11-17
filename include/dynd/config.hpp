//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <cstdlib>
#include <stdint.h>
#include <limits>

#include <dynd/cmake_config.hpp>

/** The number of elements to process at once when doing chunking/buffering */
#define DYND_BUFFER_CHUNK_SIZE 128

#ifdef __clang__
// It appears that on OSX, one can have a configuration with
// clang that supports rvalue references but no implementation
// of std::move, and there doesn't seem to be a way to detect
// this. :P
//#if __has_feature(cxx_rvalue_references)
//#  define DYND_RVALUE_REFS
//#endif

#if __has_feature(cxx_generalized_initializers) && \
    __has_include(<initializer_list>)
#  define DYND_INIT_LIST
#endif

#if __has_feature(cxx_constexpr)
#  define DYND_CONSTEXPR constexpr
#else
#  define DYND_CONSTEXPR
#endif

#if __has_feature(cxx_static_assert)
#  define DYND_STATIC_ASSERT(value, message) static_assert(value, message)
#endif

#if __has_feature(cxx_lambdas)
#  define DYND_CXX_LAMBDAS
#endif

#if __has_include(<type_traits>)
#  define DYND_CXX_TYPE_TRAITS
#elif __has_include(<tr1/type_traits>)
#  define DYND_CXX_TR1_TYPE_TRAITS
#endif

#if __has_feature(cxx_variadic_templates)
#  define DYND_CXX_VARIADIC_TEMPLATES
#endif

#include <cmath>

// Ran into some weird issues with
// clang + gcc std library's C11
// polymorphic macros. Our workaround
// is to wrap this in inline functions.
inline bool DYND_ISNAN(float x) {
    return std::isnan(x);
}
inline bool DYND_ISNAN(double x) {
    return std::isnan(x);
}
inline bool DYND_ISNAN(long double x) {
    return std::isnan(x);
}

#elif defined(__GNUC__)


#if __GNUC__ > 4 || \
              (__GNUC__ == 4 && (__GNUC_MINOR__ >= 7))
// Use initializer lists on gcc >= 4.7
#  define DYND_INIT_LIST
// Use constexpr on gcc >= 4.7
#  define DYND_CONSTEXPR constexpr
// Use rvalue references on gcc >= 4.7
#  define DYND_RVALUE_REFS
#  define DYND_ISNAN(x) (std::isnan(x))
// Use static_assert on gcc >= 4.7
#  define DYND_STATIC_ASSERT(value, message) static_assert(value, message)
#  define DYND_CXX_TYPE_TRAITS
#  define DYND_CXX_VARIADIC_TEMPLATES
#else
// Don't use constexpr on gcc < 4.7
#  define DYND_CONSTEXPR
#  define DYND_ISNAN(x) isnan(x)
#  define DYND_CXX_TR1_TYPE_TRAITS
#endif


// Check for __float128 (added in gcc 4.6)
// #if __GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 6))
// #include <iostream>
// #define DYND_HAS_FLOAT128
// typedef __float128 dynd_float128;
// inline std::ostream& operator<<(std::ostream& o, const __float128&)
// {
//     return (o << "<unimplemented float128 printing>");
// }
// #endif

#elif defined(_MSC_VER)

#include <float.h>

// If set, uses the FP status registers.
// On some compilers, there is no proper
// way to tell the compiler that these are
// important, and it reorders instructions
// so as to make them useless. On MSVC, there
// is #pragma fenv_access(on), which works.
# define DYND_USE_FPSTATUS

// MSVC 2010 and later
# define DYND_CXX_TYPE_TRAITS
# define DYND_USE_TR1_ENABLE_IF
# define DYND_RVALUE_REFS
# define DYND_STATIC_ASSERT(value, message) static_assert(value, message)
# define DYND_CXX_LAMBDAS

#if _MSC_VER < 1700
// Older than MSVC 2012
#define DYND_ATOLL(x) (_atoi64(x))
namespace std {
    inline bool isfinite(double x) {
        return _finite(x) != 0;
    }
    inline bool isnan(double x) {
        return _isnan(x) != 0;
    }
    inline bool isinf(double x) {
        return x == std::numeric_limits<double>::infinity() ||
               x == -std::numeric_limits<double>::infinity();
    }
    inline double copysign(double num, double sign) {
        return _copysign(num, sign);
    }
    inline int signbit(double x) {
        union {
            double d;
            uint64_t u;
        } val;
        val.d = x;
        return (val.u & 0x8000000000000000ULL) ? 1 : 0;
    }
}
#endif

#if _MSC_VER == 1700
// MSVC 2012
#define DYND_ATOLL(x) (_atoi64(x))
inline double copysign(double num, double sign) { return _copysign(num, sign); }
inline int signbit(double x)
{
  union {
    double d;
    uint64_t u;
  } val;
  val.d = x;
  return (val.u & 0x8000000000000000ULL) ? 1 : 0;
}
#endif

#if _MSC_VER >= 1700
// MSVC 2012 and later
# define DYND_USE_STD_ATOMIC
#endif

#if _MSC_VER >= 1800
// MSVC 2013 and later
# define DYND_CXX_VARIADIC_TEMPLATES

// MSVC 2013 doesn't appear to support nested initializer lists
// https://stackoverflow.com/questions/23965565/how-to-do-nested-initializer-lists-in-visual-c-2013
//#define DYND_INIT_LIST
#endif

// No DYND_CONSTEXPR yet, define it as nothing
#  define DYND_CONSTEXPR

#include <stdlib.h>
// Some C library calls will abort if given bad format strings, etc
// This RAII class temporarily disables that
class disable_invalid_parameter_handler {
    _invalid_parameter_handler m_saved;

    disable_invalid_parameter_handler(const disable_invalid_parameter_handler&);
    disable_invalid_parameter_handler& operator=(const disable_invalid_parameter_handler&);

    static void nop_parameter_handler(const wchar_t *, const wchar_t *, const wchar_t *, 
                       unsigned int, uintptr_t) {
    }
public:
    disable_invalid_parameter_handler() {
        m_saved = _set_invalid_parameter_handler(&nop_parameter_handler);
    }
    ~disable_invalid_parameter_handler() {
        _set_invalid_parameter_handler(m_saved);
    }
};

# define DYND_ISNAN(x) (_isnan(x) != 0)

#endif // end of compiler vendor checks

// If RValue References are supported
#ifdef DYND_RVALUE_REFS
#  include <utility>
#  define DYND_MOVE(x) (std::move(x))
#else
#  define DYND_MOVE(x) (x)
#endif

#ifndef DYND_ATOLL
#define DYND_ATOLL(x) (atoll(x))
#endif

// If Initializer Lists are supported
#ifdef DYND_INIT_LIST
#include <initializer_list>
#endif

// If being run from the CLING C++ interpreter
#ifdef DYND_CLING
// Don't use the memcpy function (it has inline assembly).

inline void DYND_MEMCPY(char *dst, const char *src, intptr_t count)
{
    char *cdst = (char *)dst;
    const char *csrc = (const char *)src;
    while (count--) {
        *cdst++ = *csrc++;
    }
}
#else
#include <cstring>
#define DYND_MEMCPY(dst, src, count) std::memcpy(dst, src, count)
#endif

// This static_assert fails at compile-time when expected, but with a more general message
// TODO: This doesn't work as a member of a class, need to fix that and reenable
#ifndef DYND_STATIC_ASSERT
#define DYND_STATIC_ASSERT(value, message) // do { enum { dynd_static_assertion = 1 / (int)(value) }; } while (0)
#endif

#if defined(DYND_CXX_TYPE_TRAITS)
#include <type_traits>
#elif defined(DYND_CXX_TR1_TYPE_TRAITS)
#include <tr1/type_traits>
namespace std {

using std::tr1::add_pointer;
using std::tr1::is_array;
using std::tr1::is_base_of;
using std::tr1::is_const;
using std::tr1::is_function;
using std::tr1::is_pointer;
using std::tr1::is_reference;
using std::tr1::remove_const;
using std::tr1::remove_cv;
using std::tr1::remove_extent;
using std::tr1::remove_reference;
using std::tr1::remove_pointer;

template <bool B, typename T, typename F>
struct conditional {
    typedef T type;
};

template <typename T, typename F>
struct conditional<false, T, F> {
    typedef F type;
};

template <typename T>
struct decay {
    typedef typename std::remove_reference<T>::type U;
    typedef typename std::conditional<std::is_array<U>::value,
        typename std::remove_extent<U>::type *,
        typename std::conditional<std::is_function<U>::value,
            typename std::add_pointer<U>::type,
            typename std::remove_cv<U>::type>::type>::type type;
};

} // namespace std
#endif

// These are small templates 'missing' from the standard library
namespace dynd {

template <typename T>
struct is_function_pointer {
    static const bool value = std::is_pointer<T>::value ?
        std::is_function<typename std::remove_pointer<T>::type>::value : false;
};

template <typename T>
struct remove_all_pointers {
    typedef T type;
};

template <typename T>
struct remove_all_pointers<T *> {
    typedef typename remove_all_pointers<typename std::remove_cv<T>::type>::type type;
};

template <typename... T>
struct type_sequence {
    static constexpr size_t size = sizeof...(T);
};

template <typename U, typename... T>
struct prepend {
    typedef type_sequence<U, T...> type;
};

template <typename U, typename... T>
struct prepend<U, type_sequence<T...> > {
    typedef typename prepend<U, T...>::type type;
};

template <typename U, typename... T>
struct append {
    typedef type_sequence<T..., U> type;
};

template <typename U, typename... T>
struct append<U, type_sequence<T...> > {
    typedef typename append<T..., U>::type type;
};

template <int n, typename... T>
struct from;

template <typename... T>
struct from<0, T...> {
    typedef type_sequence<T...> type;
};

template <typename T0, typename... T>
struct from<0, T0, T...> {
    typedef type_sequence<T0, T...> type;
};

template <int n, typename T0, typename... T>
struct from<n, T0, T...> {
    typedef typename from<n - 1, T...>::type type;
};

template <int n, typename... T>
struct from<n, type_sequence<T...> > {
    typedef typename from<n, T...>::type type;
};

template <int n, typename... T>
struct to;

template <typename... T>
struct to<0, T...> {
    typedef type_sequence<> type;
};

template <typename T0, typename... T>
struct to<0, T0, T...> {
    typedef type_sequence<> type;
};

template <int n, typename T0, typename... T>
struct to<n, T0, T...> {
    typedef typename prepend<T0, typename to<n - 1, T...>::type>::type type;
};

template <int n, typename... T>
struct to<n, type_sequence<T...> > {
    typedef typename to<n, T...>::type type;
};

template <int i, typename... T>
struct at;

template <typename T0, typename... T>
struct at<0, T0, T...> {
    typedef T0 type;
};

template <int i, typename T0, typename... T>
struct at<i, T0, T...> {
    typedef typename at<i - 1, T...>::type type;
};

template<typename T, T... I>
struct integer_sequence {
    static_assert(std::is_integral<T>::value, "Integral type" );

    static constexpr T size = sizeof...(I);

    using type = T;

//    template<T N>
  //  using append = integer_sequence<T, I..., N>;
//        using next = typename append<size>::type;

    template <T J>
    struct append {
        typedef integer_sequence<T, I..., J> type;
    };

    typedef typename append<size>::type next;
};

template<typename T, T... I>
constexpr T integer_sequence<T, I...>::size;

template<std::size_t... I>
using index_sequence = integer_sequence<std::size_t, I...>;

namespace detail {

template <typename T, T Nt, std::size_t N>
struct iota {
    static_assert( Nt >= 0, "N cannot be negative" );

    using type = typename iota<T, Nt-1, N-1>::type::next;
};

template <typename T, T Nt>
struct iota<T, Nt, 0ul> {
    using type = integer_sequence<T>;
};

} // namespace detail

template <typename T, T N>
using make_integer_sequence = typename detail::iota<T, N, N>::type;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

/*
template <typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;
*/

} // namespace dynd


#ifdef DYND_USE_TR1_ENABLE_IF
#include <type_traits>
namespace dynd {
    using std::tr1::enable_if;
}
#else
// These are small templates, so we just replicate them here
namespace dynd {
	template<bool B, class T = void>
	struct enable_if {};
 
	template<class T>
	struct enable_if<true, T> { typedef T type; };
}
#endif

/////////////////////////////////////////
// Diagnostic configurations
//

/**
 * This preprocessor symbol enables or disables assertions
 * that pointers are aligned as they are supposed to be. This helps
 * test that alignment is being done correctly on platforms which
 * do not segfault on misaligned data.
 *
 * An exception is thrown if an improper unalignment is detected.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DYND_ALIGNMENT_ASSERTIONS
# define DYND_ALIGNMENT_ASSERTIONS 0
#endif

/**
 * This preprocessor symbol enables or disables tracing
 * of individual builtin type assignment operations.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DYND_ASSIGNMENT_TRACING
# define DYND_ASSIGNMENT_TRACING 0
#endif


/**
 * Preprocessor macro for marking variables unused, and suppressing
 * warnings for them.
 */
#define DYND_UNUSED(x)

namespace dynd {
    // These are defined in git_version.cpp, generated from
    // git_version.cpp.in by the CMake build configuration.
    extern const char dynd_git_sha1[];
    extern const char dynd_version_string[];
} // namespace dynd

// Check endian: define DYND_BIG_ENDIAN if big endian, otherwise assume little
#if defined(__GLIBC__)
#include <endian.h>
#if __BYTE_ORDER == __BIG_ENDIAN
#define DYND_BIG_ENDIAN
#elif __BYTE_ORDER != __LITTLE_ENDIAN
#error Unrecognized endianness from endian.h.
#endif
#endif

namespace dynd {
  /**
   * Function to call for initializing dynd's global state, such
   * as cached ndt::type objects, the arrfunc registry, etc.
   */
  int libdynd_init();
  /**
   * Function to call to free all resources associated with
   * dynd's global state, that were initialized by libdynd_init.
   */
  void libdynd_cleanup();
  /**
    * A function which can be used at runtime to identify whether
    * the build of dynd being linked against was built with CUDA
    * support enabled.
    */
  bool built_with_cuda();
} // namespace dynd

#include <dynd/cuda_config.hpp>
