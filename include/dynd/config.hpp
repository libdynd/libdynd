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

// On OSX build machine, there is no <initializer_list> header.
// TODO: Remove this build workaround once this is resolved
#if !(__has_feature(cxx_generalized_initializers) &&                           \
      __has_include(<initializer_list>))
#  define DYND_DISABLE_INIT_LIST
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

// Hack trying to work around gcc isfinite problems
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#define DYND_CONDITIONAL_UNUSED(NAME) NAME  __attribute__((unused))

#  define DYND_CONSTEXPR constexpr
#  define DYND_RVALUE_REFS
#  define DYND_ISNAN(x) (std::isnan(x))
#  define DYND_STATIC_ASSERT(value, message) static_assert(value, message)
#  define DYND_CXX_TYPE_TRAITS
#  define DYND_CXX_VARIADIC_TEMPLATES

#elif defined(_MSC_VER)

#include <float.h>

// If set, uses the FP status registers.
// On some compilers, there is no proper
// way to tell the compiler that these are
// important, and it reorders instructions
// so as to make them useless. On MSVC, there
// is #pragma fenv_access(on), which works.
//
// Update 2014-11-29: Found on 32-bit Windows, MSVC 2013 does not respect
//                    pragma fenv_access(on), and reorders incorrectly, so
//                    disabling this on 32-bit.
# ifdef _WIN64
#  define DYND_USE_FPSTATUS
# endif

# define DYND_CXX_TYPE_TRAITS
# define DYND_USE_TR1_ENABLE_IF
# define DYND_RVALUE_REFS
# define DYND_STATIC_ASSERT(value, message) static_assert(value, message)
# define DYND_CXX_LAMBDAS
# define DYND_USE_STD_ATOMIC
# define DYND_CXX_VARIADIC_TEMPLATES

#if _MSC_VER == 1800
// MSVC 2013 doesn't support nested initializer lists
// https://stackoverflow.com/questions/23965565/how-to-do-nested-initializer-lists-in-visual-c-2013
#define DYND_NESTED_INIT_LIST_BUG
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
#ifndef DYND_DISABLE_INIT_LIST
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

#include <tuple>
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

template <size_t I, typename T>
struct at;

template <typename... T>
struct type_sequence {
    static const size_t size = sizeof...(T);

    template <typename U>
    struct append {
        typedef type_sequence<T..., U> type;
    };
};

template <size_t I>
struct at<I, type_sequence<>> {
};

template <typename T0, typename... T>
struct at<0, type_sequence<T0, T...>> {
    typedef T0 type;
};

template <size_t I, typename T0, typename... T>
struct at<I, type_sequence<T0, T...>> {
    typedef typename at<I - 1, type_sequence<T...>>::type type;
};

template <size_t I, typename A0, typename... A>
typename std::enable_if<I == 0, A0>::type get(A0 &&a0, A &&...) {
    return a0;
}

template <size_t I, typename A0, typename... A>
typename std::enable_if<I != 0,
                        typename at<I, type_sequence<A0, A...>>::type>::type
get(A0 &&, A &&... a)
{
  return get<I - 1>(std::forward<A>(a)...);
}

template <typename T, T... I>
struct integer_sequence {
    static_assert(std::is_integral<T>::value, "Integral type" );

    static const T size = sizeof...(I);

    typedef T type;

//    template<T N>
  //  using append = integer_sequence<T, I..., N>;
//        using next = typename append<size>::type;

    template <T J>
    struct prepend {
        typedef integer_sequence<T, J, I...> type;
    };

    template <T J>
    struct append {
        typedef integer_sequence<T, I..., J> type;
    };

    typedef typename append<size>::type next;
};

template <size_t... I>
using index_sequence = integer_sequence<size_t, I...>;

template <typename U, typename... T>
struct prepend {
    typedef type_sequence<U, T...> type;
};

template <typename U, typename... T>
struct prepend<U, type_sequence<T...> > {
    typedef typename prepend<U, T...>::type type;
};

template <typename T, typename U>
struct concatenate;

template <typename... T, typename... U>
struct concatenate<type_sequence<T...>, type_sequence<U...>> {
    typedef type_sequence<T..., U...> type;
};

template <typename T, T... I, T... J>
struct concatenate<integer_sequence<T, I...>, integer_sequence<T, J...>> {
    typedef integer_sequence<T, I..., J...> type;
};

template <typename... T>
struct zip;

template <typename T>
struct zip<integer_sequence<T>, integer_sequence<T>> {
    typedef integer_sequence<T> type;
};

#if defined(_MSC_VER) && _MSC_VER == 1800
// This case shouldn't be necessary, but was added to work around bug:
// https://connect.microsoft.com/VisualStudio/feedback/details/1045397/recursive-variadic-template-metaprogram-ice-when-pack-gets-to-empty-size
template <typename T, T I0, T J0>
struct zip<integer_sequence<T, I0>, integer_sequence<T, J0>> {
  typedef integer_sequence<T, I0, J0> type;
};
#endif

template <typename T, T I0, T... I, T J0, T... J>
struct zip<integer_sequence<T, I0, I...>, integer_sequence<T, J0, J...>> {
  typedef typename concatenate<
      integer_sequence<T, I0, J0>,
      typename zip<integer_sequence<T, I...>, integer_sequence<T, J...>>::type>::type type;
};

template <typename T>
struct zip<integer_sequence<T>, integer_sequence<T>,
           integer_sequence<T>, integer_sequence<T>> {
    typedef integer_sequence<T> type;
};

#if defined(_MSC_VER) && _MSC_VER == 1800
// This case shouldn't be necessary, but was added to work around bug:
// https://connect.microsoft.com/VisualStudio/feedback/details/1045397/recursive-variadic-template-metaprogram-ice-when-pack-gets-to-empty-size
template <typename T, T I0, T J0, T K0, T L0>
struct zip<integer_sequence<T, I0>, integer_sequence<T, J0>,
           integer_sequence<T, K0>, integer_sequence<T, L0>> {
  typedef integer_sequence<T, I0, J0, K0, L0> type;
};
#endif

template <typename T, T I0, T... I, T J0, T... J, T K0, T... K, T L0, T... L>
struct zip<integer_sequence<T, I0, I...>, integer_sequence<T, J0, J...>,
           integer_sequence<T, K0, K...>, integer_sequence<T, L0, L...>> {
  typedef typename concatenate<
      integer_sequence<T, I0, J0, K0, L0>,
      typename zip<integer_sequence<T, I...>, integer_sequence<T, J...>,
                   integer_sequence<T, K...>,
                   integer_sequence<T, L...>>::type>::type type;
};

template <size_t n, typename... T>
struct from;

template <typename... T>
struct from<0, T...> {
    typedef type_sequence<T...> type;
};

template <typename T0, typename... T>
struct from<0, T0, T...> {
    typedef type_sequence<T0, T...> type;
};

template <size_t n, typename T0, typename... T>
struct from<n, T0, T...> {
    typedef typename from<n - 1, T...>::type type;
};

template <typename... T>
struct from<0, type_sequence<T...>> {
    typedef type_sequence<T...> type;
};

template <size_t n, typename... T>
struct from<n, type_sequence<T...> > {
    typedef typename from<n, T...>::type type;
};

template <size_t n, typename... T>
struct to;

template <typename... T>
struct to<0, T...> {
    typedef type_sequence<> type;
};

template <typename T0, typename... T>
struct to<0, T0, T...> {
    typedef type_sequence<> type;
};

template <size_t n, typename T0, typename... T>
struct to<n, T0, T...> {
    typedef typename prepend<T0, typename to<n - 1, T...>::type>::type type;
};

template <typename... T>
struct to<0, type_sequence<T...>> {
    typedef type_sequence<> type;
};

template <size_t n, typename... T>
struct to<n, type_sequence<T...>> {
    typedef typename to<n, T...>::type type;
};

template <typename T, T J0, T... J>
struct at<0, integer_sequence<T, J0, J...>> {
    static const T value = J0;
};

template <size_t I, typename T, T J0, T... J>
struct at<I, integer_sequence<T, J0, J...>> {
    static const T value = at<I - 1, integer_sequence<T, J...>>::value;
};

template <typename T, typename U>
struct as_ {
  typedef U type;
};

template <typename I, typename T>
struct take;

template <size_t... I, typename T>
struct take<index_sequence<I...>, T> {
    typedef type_sequence<typename at<I, T>::type...> type;
};

template <template <typename...> class C, typename T>
struct instantiate;

template <template <typename...> class C, typename... T>
struct instantiate<C, type_sequence<T...>> {
  typedef C<T...> type;  
};

namespace detail {

template <typename T, T Start, T Stop, T Step, bool Empty = Start >= Stop>
struct make_integer_sequence;

template <typename T, T Start, T Stop, T Step>
struct make_integer_sequence<T, Start, Stop, Step, false> {
  typedef typename make_integer_sequence<
      T, Start + Step, Stop, Step>::type::template prepend<Start>::type type;
};

template <typename T, T Start, T Stop, T Step>
struct make_integer_sequence<T, Start, Stop, Step, true> {
    typedef integer_sequence<T> type;
};

} // namespace detail

template <typename T, T Start, T Stop, T Step = 1>
struct make_integer_sequence {
    typedef typename detail::make_integer_sequence<T, Start, Stop, Step>::type type;
};

template <size_t Start, size_t Stop, size_t Step = 1>
using make_index_sequence = make_integer_sequence<size_t, Start, Stop, Step>;

template <typename I>
struct index_proxy;

template <size_t... I>
struct index_proxy<index_sequence<I...>> {
  enum { size = index_sequence<I...>::size };

  template <typename R, typename... A>
  static R apply(R (*func)(A...), A &&... a)
  {
    return (*func)(get<I>(std::forward<A>(a)...)...);
  }

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
  template <typename R, typename... A>
  static R make(A &&... a)
  {
    return R(get<I>(std::forward<A>(a)...)...);
  }
#else
  // Workaround for MSVC 2013 compiler bug reported here:
  // https://connect.microsoft.com/VisualStudio/feedback/details/1045260/unpacking-std-forward-a-a-fails-when-nested-with-another-unpacking
  template <typename R>
  static R make()
  {
    return R(get<I>()...);
  }
  template <typename R, typename A0>
  static R make(A0 &&a0)
  {
    return R(get<I>(std::forward<A0>(a0))...);
  }
  template <typename R, typename A0, typename A1>
  static R make(A0 &&a0, A1 &&a1)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1))...);
  }
  template <typename R, typename A0, typename A1, typename A2>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5))...);
  }
  template <typename R, typename A0, typename A1, typename A2, typename A3,
            typename A4, typename A5, typename A6>
  static R make(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6)
  {
    return R(get<I>(std::forward<A0>(a0), std::forward<A1>(a1),
                    std::forward<A2>(a2), std::forward<A3>(a3),
                    std::forward<A4>(a4), std::forward<A5>(a5),
                    std::forward<A6>(a6))...);
  }
#endif
};

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

#ifndef DYND_CONDITIONAL_UNUSED
#define DYND_CONDITIONAL_UNUSED(NAME) NAME
#endif

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
