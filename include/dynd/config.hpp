//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/cmake_config.hpp>

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>

#include <dynd/git_version.hpp>
#include <dynd/visibility.hpp>

/** The number of elements to process at once when doing chunking/buffering */
#define DYND_BUFFER_CHUNK_SIZE 128

#ifdef __clang__

#if __has_feature(cxx_constexpr)
#define DYND_CONSTEXPR constexpr
#else
#define DYND_CONSTEXPR
#endif

// Workaround for a clang issue
#define DYND_ISSPACE std::isspace
#define DYND_TOLOWER std::tolower

#define DYND_USED(NAME) NAME __attribute__((used))
#define DYND_EMIT_LLVM(NAME) __attribute__((annotate(#NAME), annotate("emit_llvm"))) NAME

#define DYND_ALLOW_UNSIGNED_UNARY_MINUS
#define DYND_END_ALLOW_UNSIGNED_UNARY_MINUS

#elif defined(__GNUC__)

// Hack trying to work around gcc isfinite problems
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#define DYND_IGNORE_UNUSED(NAME) NAME __attribute__((unused))
#define DYND_USED(NAME) NAME
#define DYND_EMIT_LLVM(NAME) NAME

// Ignore erroneous maybe-uninitizlized
// warnings on a given line or code block.
#define DYND_IGNORE_MAYBE_UNINITIALIZED _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define DYND_END_IGNORE_MAYBE_UNINITIALIZED _Pragma("GCC diagnostic pop")

#define DYND_CONSTEXPR constexpr

#define DYND_ISSPACE isspace
#define DYND_TOLOWER tolower

#define DYND_ALLOW_UNSIGNED_UNARY_MINUS
#define DYND_END_ALLOW_UNSIGNED_UNARY_MINUS

#elif defined(_MSC_VER)

#define DYND_ISSPACE isspace
#define DYND_TOLOWER tolower
#define DYND_USED(NAME) NAME
#define DYND_EMIT_LLVM(NAME) NAME

#define DYND_ALLOW_UNSIGNED_UNARY_MINUS __pragma(warning(push)) __pragma(warning(disable : 4146))

#define DYND_END_ALLOW_UNSIGNED_UNARY_MINUS __pragma(warning(pop))

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
#ifdef _WIN64
#define DYND_USE_FPSTATUS
#endif

// No DYND_CONSTEXPR yet, define it as nothing
#define DYND_CONSTEXPR

// Some C library calls will abort if given bad format strings, etc
// This RAII class temporarily disables that
class disable_invalid_parameter_handler {
  _invalid_parameter_handler m_saved;

  disable_invalid_parameter_handler(const disable_invalid_parameter_handler &);
  disable_invalid_parameter_handler &operator=(const disable_invalid_parameter_handler &);

  static void nop_parameter_handler(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t) {}

public:
  disable_invalid_parameter_handler() { m_saved = _set_invalid_parameter_handler(&nop_parameter_handler); }
  ~disable_invalid_parameter_handler() { _set_invalid_parameter_handler(m_saved); }
};

#endif // end of compiler vendor checks

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

#include <dynd/type_sequence.hpp>

// These are small templates 'missing' from the standard library
namespace dynd {

template <typename T, typename U, typename V>
struct is_common_type_of : std::conditional<std::is_same<T, typename std::common_type<U, V>::type>::value,
                                            std::true_type, std::false_type>::type {
};

template <bool Value, template <typename...> class T, template <typename...> class U, typename... As>
struct conditional_make;

template <template <typename...> class T, template <typename...> class U, typename... As>
struct conditional_make<true, T, U, As...> {
  typedef T<As...> type;
};

template <template <typename...> class T, template <typename...> class U, typename... As>
struct conditional_make<false, T, U, As...> {
  typedef U<As...> type;
};

template <typename T>
struct is_function_pointer {
  static const bool value =
      std::is_pointer<T>::value ? std::is_function<typename std::remove_pointer<T>::type>::value : false;
};

template <typename T>
struct is_vector : std::false_type {
};

template <typename T>
struct is_vector<std::vector<T>> : std::true_type {
};

template <typename T>
long intrusive_ptr_use_count(T *ptr);

template <typename T>
void intrusive_ptr_retain(T *ptr);

template <typename T>
void intrusive_ptr_release(T *ptr);

/**
 * A smart pointer, very similar to boost::intrusive_ptr.
 */
template <typename T>
class intrusive_ptr {
protected:
  T *m_ptr;

public:
  /** Default constructor */
  intrusive_ptr() : m_ptr(0) {}

  /** Constructor from a raw pointer */
  explicit intrusive_ptr(T *ptr, bool add_ref = true) : m_ptr(ptr)
  {
    if (m_ptr != 0 && add_ref) {
      intrusive_ptr_retain(m_ptr);
    }
  }

  /** Copy constructor */
  intrusive_ptr(const intrusive_ptr &other) : m_ptr(other.m_ptr)
  {
    if (m_ptr != 0) {
      intrusive_ptr_retain(m_ptr);
    }
  }

  /** Move constructor */
  intrusive_ptr(intrusive_ptr &&other) : m_ptr(other.m_ptr) { other.m_ptr = 0; }

  /** Destructor */
  ~intrusive_ptr()
  {
    if (m_ptr != 0) {
      intrusive_ptr_release(m_ptr);
    }
  }

  intptr_t use_count() const { return intrusive_ptr_use_count(m_ptr); }

  explicit operator bool() const { return m_ptr != NULL; }

  T *operator->() const { return m_ptr; }

  /** Assignment */
  intrusive_ptr &operator=(const intrusive_ptr &rhs)
  {
    if (m_ptr != 0) {
      intrusive_ptr_release(m_ptr);
    }
    if (rhs.m_ptr != 0) {
      m_ptr = rhs.m_ptr;
      intrusive_ptr_retain(m_ptr);
    }
    else {
      m_ptr = 0;
    }
    return *this;
  }

  /** Move assignment */
  intrusive_ptr &operator=(intrusive_ptr &&rhs)
  {
    if (m_ptr != 0) {
      intrusive_ptr_release(m_ptr);
    }
    m_ptr = rhs.m_ptr;
    rhs.m_ptr = 0;
    return *this;
  }

  /** Assignment from raw memory_block pointer */
  intrusive_ptr &operator=(T *rhs)
  {
    if (m_ptr != nullptr) {
      intrusive_ptr_release(m_ptr);
    }

    m_ptr = rhs;
    if (m_ptr != nullptr) {
      intrusive_ptr_retain(m_ptr);
    }

    return *this;
  }

  /** Returns true if there is only one reference to this memory block */
  bool unique() const { return intrusive_ptr_use_count(m_ptr) <= 1; }

  /** Gets the raw memory_block_data pointer */
  T *get() const { return m_ptr; }

  /** Gives away ownership of the reference count */
  T *release()
  {
    T *result = m_ptr;
    m_ptr = 0;
    return result;
  }

  void swap(intrusive_ptr &rhs)
  {
    T *tmp = m_ptr;
    m_ptr = rhs.m_ptr;
    rhs.m_ptr = tmp;
  }
};

template <typename T>
bool operator==(const intrusive_ptr<T> &lhs, const intrusive_ptr<T> &rhs)
{
  return lhs.get() == rhs.get();
}

template <typename T>
bool operator!=(const intrusive_ptr<T> &lhs, const intrusive_ptr<T> &rhs)
{
  return lhs.get() == rhs.get();
}

template <template <typename...> class T, typename U>
struct is_instance {
  static const bool value = false;
};

template <template <typename...> class T, typename... A>
struct is_instance<T, T<A...>> {
  static const bool value = true;
};

template <typename T, typename U>
T alias_cast(U value)
{
  union {
    U tmp;
    T res;
  };

  tmp = value;
  return res;
}

/**
 * Matches string literal arguments.
 */
template <typename T>
struct is_char_string_param {
  static const bool value = false;
};
template <>
struct is_char_string_param<const char *> {
  static const bool value = true;
};
template <>
struct is_char_string_param<char *> {
  static const bool value = true;
};
template <int N>
struct is_char_string_param<const char (&)[N]> {
  static const bool value = true;
};
template <int N>
struct is_char_string_param<const char(&&)[N]> {
  static const bool value = true;
};

/** Returns true if all the packed parameters are char strings */
template <typename... T>
struct all_char_string_params {
  static const bool value = false;
};
template <>
struct all_char_string_params<> {
  static const bool value = true;
};
template <typename T0>
struct all_char_string_params<T0> {
  static const bool value = is_char_string_param<T0>::value;
};
template <typename... T, typename T0>
struct all_char_string_params<T0, T...> {
  static const bool value = is_char_string_param<T0>::value && all_char_string_params<T...>::value;
};

template <typename T>
struct remove_all_pointers {
  typedef T type;
};

template <typename T>
struct remove_reference_then_cv {
  typedef typename std::remove_cv<typename std::remove_reference<T>::type>::type type;
};

template <typename T>
struct remove_all_pointers<T *> {
  typedef typename remove_all_pointers<typename std::remove_cv<T>::type>::type type;
};

namespace detail {

  template <typename func_type, typename... B>
  struct funcproto {
    typedef typename funcproto<decltype(&func_type::operator()), B...>::type type;
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R(A...), B...> {
    typedef R(type)(A..., B...);
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R (*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...) const, B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

} // namespace dynd::detail

template <typename func_type, typename... B>
struct funcproto_of {
  typedef typename detail::funcproto<func_type, B...>::type type;
};

template <typename func_type, typename... B>
struct funcproto_of<func_type *, B...> {
  typedef typename funcproto_of<func_type, B...>::type type;
};

template <typename func_type>
struct return_of {
  typedef typename return_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct return_of<R(A...)> {
  typedef R type;
};

template <typename func_type>
struct args_of {
  typedef typename args_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct args_of<R(A...)> {
  typedef type_sequence<A...> type;
};

template <typename func_type>
struct arity_of {
  static const size_t value = arity_of<typename funcproto_of<func_type>::type>::value;
};

template <typename R, typename... A>
struct arity_of<R(A...)> {
  static const size_t value = sizeof...(A);
};

template <typename func_type, int I>
struct arg_at {
  typedef typename at<typename args_of<func_type>::type, I>::type type;
};

} // namespace dynd

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
#define DYND_ALIGNMENT_ASSERTIONS 0
#endif

/**
 * This preprocessor symbol enables or disables tracing
 * of individual builtin type assignment operations.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DYND_ASSIGNMENT_TRACING
#define DYND_ASSIGNMENT_TRACING 0
#endif

/**
 * Preprocessor macro for marking variables unused, and suppressing
 * warnings for them.
 */
#define DYND_UNUSED(x)

#ifndef DYND_IGNORE_UNUSED
#define DYND_IGNORE_UNUSED(NAME) NAME
#endif

#ifndef DYND_IGNORE_MAYBE_UNINITIALIZED
#define DYND_IGNORE_MAYBE_UNINITIALIZED
#define DYND_END_IGNORE_MAYBE_UNINITIALIZED
#endif

// Check endian: define DYND_BIG_ENDIAN if big endian, otherwise assume little
#if defined(__GLIBC__)
#include <endian.h>
#if __BYTE_ORDER == __BIG_ENDIAN
#define DYND_BIG_ENDIAN
#elif __BYTE_ORDER != __LITTLE_ENDIAN
#error Unrecognized endianness from endian.h.
#endif
#endif

#define DYND_HAS(NAME)                                                                                                 \
  template <typename...>                                                                                               \
  class has_##NAME;                                                                                                    \
                                                                                                                       \
  template <typename T>                                                                                                \
  class has_##NAME<T> {                                                                                                \
    template <typename U,                                                                                              \
              typename = typename std::enable_if<!std::is_member_pointer<decltype(&U::NAME)>::value>::type>            \
    static std::true_type test(int);                                                                                   \
                                                                                                                       \
    template <typename>                                                                                                \
    static std::false_type test(...);                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    static const bool value = decltype(test<T>(0))::value;                                                             \
  };                                                                                                                   \
                                                                                                                       \
  template <typename T, typename StaticMemberType>                                                                     \
  class has_##NAME<T, StaticMemberType> {                                                                              \
    template <typename U,                                                                                              \
              typename = typename std::enable_if<!std::is_member_pointer<decltype(&U::NAME)>::value &&                 \
                                                 std::is_same<decltype(U::NAME), StaticMemberType>::value>::type>      \
    static std::true_type test(int);                                                                                   \
                                                                                                                       \
    template <typename>                                                                                                \
    static std::false_type test(...);                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    static const bool value = decltype(test<T>(0))::value;                                                             \
  }

#define DYND_HAS_MEMBER(NAME)                                                                                          \
  template <typename...>                                                                                               \
  class has_member_##NAME;                                                                                             \
                                                                                                                       \
  template <typename T>                                                                                                \
  class has_member_##NAME<T> {                                                                                         \
    template <typename U, typename = typename std::enable_if<std::is_member_pointer<decltype(&U::NAME)>::value>::type> \
    static std::true_type test(int);                                                                                   \
                                                                                                                       \
    template <typename>                                                                                                \
    static std::false_type test(...);                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    static const bool value = decltype(test<T>(0))::value;                                                             \
  };                                                                                                                   \
                                                                                                                       \
  template <typename T, typename MemberType>                                                                           \
  class has_member_##NAME<T, MemberType> {                                                                             \
    template <typename U,                                                                                              \
              typename = typename std::enable_if<std::is_member_pointer<decltype(&U::NAME)>::value &&                  \
                                                 std::is_same<decltype(&U::NAME), MemberType T::*>::value>::type>      \
    static std::true_type test(int);                                                                                   \
                                                                                                                       \
    template <typename>                                                                                                \
    static std::false_type test(...);                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    static const bool value = decltype(test<T>(0))::value;                                                             \
  }

#define DYND_GET(NAME, TYPE, DEFAULT_VALUE)                                                                            \
  template <typename T, bool ReturnDefaultValue>                                                                       \
  typename std::enable_if<ReturnDefaultValue, TYPE>::type get_##NAME()                                                 \
  {                                                                                                                    \
    return DEFAULT_VALUE;                                                                                              \
  }                                                                                                                    \
                                                                                                                       \
  template <typename T, bool ReturnDefaultValue>                                                                       \
  typename std::enable_if<!ReturnDefaultValue, TYPE>::type get_##NAME()                                                \
  {                                                                                                                    \
    return T::NAME;                                                                                                    \
  }                                                                                                                    \
                                                                                                                       \
  template <typename T>                                                                                                \
  TYPE get_##NAME()                                                                                                    \
  {                                                                                                                    \
    return get_##NAME<T, !has_##NAME<T>::value>();                                                                     \
  }

namespace dynd {

/**
 * Function to call for initializing dynd's global state, such
 * as cached ndt::type objects, the arrfunc registry, etc.
 */
inline int libdynd_init() { return 0; }

/**
 * Function to call to free all resources associated with
 * dynd's global state, that were initialized by libdynd_init.
 */
inline void libdynd_cleanup() {}

/**
  * A function which can be used at runtime to identify whether
  * the build of dynd being linked against was built with CUDA
  * support enabled.
  */
bool built_with_cuda();

} // namespace dynd

namespace dynd {

class bool1;
typedef std::int8_t int8;
typedef std::int16_t int16;
typedef std::int32_t int32;
typedef std::int64_t int64;
#ifndef DYND_HAS_INT128
class int128;
#endif
typedef std::uint8_t uint8;
typedef std::uint16_t uint16;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;
#ifndef DYND_HAS_UINT128
class uint128;
#endif
class float16;
typedef float float32;
typedef double float64;
#ifndef DYND_HAS_FLOAT128
class float128;
#endif

template <typename T>
struct is_boolean {
  static const bool value = false;
};

template <>
struct is_boolean<bool> {
  static const bool value = true;
};

template <>
struct is_boolean<bool1> {
  static const bool value = true;
};

template <typename T>
struct is_integral : std::is_integral<T> {
};

template <typename T>
struct is_floating_point : std::is_floating_point<T> {
};

template <typename T>
struct is_complex : std::false_type {
};

template <typename T>
struct is_arithmetic : std::integral_constant<bool, is_integral<T>::value || is_floating_point<T>::value> {
};

template <typename T>
struct is_numeric : std::integral_constant<bool, is_arithmetic<T>::value || is_complex<T>::value> {
};

template <typename T, typename U>
struct is_mixed_arithmetic : std::integral_constant<bool, is_arithmetic<T>::value && is_arithmetic<U>::value> {
};

template <typename T>
struct is_mixed_arithmetic<T, T> : std::false_type {
};

template <typename... Ts>
using true_t = std::true_type;

template <typename... Ts>
using false_t = std::false_type;

template <typename T>
using not_t = std::integral_constant<bool, !T::value>;

// Checks whether T is not the common type of T and U
template <typename T, typename U>
struct is_lcast_arithmetic : not_t<typename conditional_make<is_arithmetic<T>::value && is_arithmetic<U>::value,
                                                             is_common_type_of, true_t, T, T, U>::type> {
};

// Checks whether U is not the common type of T and U
template <typename T, typename U>
struct is_rcast_arithmetic : not_t<typename conditional_make<is_arithmetic<T>::value && is_arithmetic<U>::value,
                                                             is_common_type_of, true_t, U, T, U>::type> {
};

template <typename T>
struct is_signed {
  static const bool value = std::is_signed<T>::value || std::is_same<T, int128>::value;
};

template <typename T>
struct is_unsigned {
  static const bool value = std::is_unsigned<T>::value || std::is_same<T, uint128>::value;
};

template <typename T>
T strto(const char *begin, char **end);

template <>
inline float strto(const char *begin, char **end)
{
  return std::strtof(begin, end);
}

template <>
inline double strto(const char *begin, char **end)
{
  return std::strtod(begin, end);
}

template <typename T>
T floor(T value)
{
  return std::floor(value);
}

/**
 * An enumeration for the error checks during assignment.
 */
enum assign_error_mode {
  /** No error checking during assignment */
  assign_error_nocheck,
  /** Check overflow, but allow precision loss. Checks loss of imaginary
     component  */
  assign_error_overflow,
  /** Overflow and loss of fractional part (for float -> int) checking */
  assign_error_fractional,
  /** Overflow and floating point precision loss checking */
  assign_error_inexact,
  /** Use the mode specified in the eval_context */
  assign_error_default
};

struct overflow_check_t {
};

inline std::ostream &operator<<(std::ostream &o, assign_error_mode errmode)
{
  switch (errmode) {
  case assign_error_nocheck:
    o << "nocheck";
    break;
  case assign_error_overflow:
    o << "overflow";
    break;
  case assign_error_fractional:
    o << "fractional";
    break;
  case assign_error_inexact:
    o << "inexact";
    break;
  case assign_error_default:
    o << "default";
    break;
  default:
    o << "invalid error mode(" << (int)errmode << ")";
    break;
  }

  return o;
}

namespace detail {
  // Use these declarations before includeing bool1, int128, uint128, etc. so they are usable there.
  // Helper to use for determining if a type is in a given list of unique types.
  template <typename T, typename... Types>
  struct TypeSetCheckInternal : std::is_same<T, Types>... {
  };

  // Determine if a type is in a given list of unique types.
  template <typename T, typename... Types>
  struct TypeSetCheck : std::is_base_of<std::true_type, TypeSetCheckInternal<T, Types...>>::type {
  };

  // Enable a given template only for a given list of unique types.
  template <typename T, typename... Types>
  struct enable_for : std::enable_if<TypeSetCheck<T, Types...>::value, int> {
  };
} // namespace dynd::detail

} // namespace dynd

// Unfortunately, include order matters here (for now), so separate by
// extra lines to avoid having clang-format reorder everything.
#include <dynd/bool1.hpp>

#include <dynd/int128.hpp>

#include <dynd/uint128.hpp>

#include <dynd/float16.hpp>

#include <dynd/float128.hpp>

#include <dynd/complex.hpp>

namespace dynd {

template <typename T, typename U>
struct operator_if_only_lcast_arithmetic
    : std::enable_if<!std::is_same<T, U>::value && !(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value) &&
                         is_lcast_arithmetic<T, U>::value && !is_rcast_arithmetic<T, U>::value,
                     U> {
};

template <typename T, typename U>
struct operator_if_only_rcast_arithmetic
    : std::enable_if<!std::is_same<T, U>::value && !(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value) &&
                         !is_lcast_arithmetic<T, U>::value && is_rcast_arithmetic<T, U>::value,
                     T> {
};

template <typename... T>
struct make_void {
  typedef void type;
};

template <typename T, typename U>
struct operator_if_lrcast_arithmetic
    : std::enable_if<!std::is_same<T, U>::value && !(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value) &&
                         is_lcast_arithmetic<T, U>::value && is_rcast_arithmetic<T, U>::value,
                     typename conditional_make<is_arithmetic<T>::value && is_arithmetic<U>::value, std::common_type,
                                               make_void, T, U>::type::type> {
};

template <typename T, typename U>
typename operator_if_only_rcast_arithmetic<T, U>::type operator+(T lhs, U rhs)
{
  return lhs + static_cast<T>(rhs);
}

template <typename T, typename U>
typename operator_if_only_lcast_arithmetic<T, U>::type operator+(T lhs, U rhs)
{
  return static_cast<U>(lhs) + rhs;
}

template <typename T, typename U>
typename operator_if_lrcast_arithmetic<T, U>::type operator+(T lhs, U rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) +
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T, typename U>
typename operator_if_only_rcast_arithmetic<T, U>::type operator/(T lhs, U rhs)
{
  return lhs / static_cast<T>(rhs);
}

template <typename T, typename U>
typename operator_if_only_lcast_arithmetic<T, U>::type operator/(T lhs, U rhs)
{
  return static_cast<U>(lhs) / rhs;
}

template <typename T, typename U>
typename operator_if_lrcast_arithmetic<T, U>::type operator/(T lhs, U rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) /
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T, typename U>

typename std::enable_if<is_mixed_arithmetic<T, U>::value, complex<typename std::common_type<T, U>::type>>::type
operator/(complex<T> lhs, U rhs)
{
  return static_cast<complex<typename std::common_type<T, U>::type>>(lhs) /
         static_cast<typename std::common_type<T, U>::type>(rhs);
}

template <typename T, typename U>

typename std::enable_if<is_mixed_arithmetic<T, U>::value, complex<typename std::common_type<T, U>::type>>::type
operator/(T lhs, complex<U> rhs)
{
  return static_cast<typename std::common_type<T, U>::type>(lhs) /
         static_cast<complex<typename std::common_type<T, U>::type>>(rhs);
}

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && is_integral<U>::value, T &>::type operator/=(T &lhs, U rhs)
{
  return lhs /= static_cast<T>(rhs);
}

} // namespace dynd

namespace dynd {
namespace detail {

  template <typename T, int N>
  class array_wrapper {
    T m_data[N];

  public:
    array_wrapper() = default;

    array_wrapper(const T *data) { memcpy(m_data, data, sizeof(m_data)); }

    operator T *() { return m_data; }

    operator const T *() const { return m_data; }

    T &operator[](intptr_t i) { return m_data[i]; }

    const T &operator[](intptr_t i) const { return m_data[i]; }
  };

  template <typename T>
  class array_wrapper<T, 0> {
  public:
    array_wrapper() = default;

    array_wrapper(const T *DYND_UNUSED(data)) {}

    operator T *() { return NULL; }

    operator const T *() const { return NULL; }
  };

  template <int N, typename T>
  array_wrapper<T, N> make_array_wrapper(const T *data)
  {
    return array_wrapper<T, N>(data);
  }

  template <typename T>
  class value_wrapper {
    T m_value;

  public:
    value_wrapper(const T &value) : m_value(value) {}

    operator T() const { return m_value; }
  };

  template <typename T>
  value_wrapper<T> make_value_wrapper(const T &value)
  {
    return value_wrapper<T>(value);
  }

} // namespace dynd::detail

using std::cos;
using std::sin;
using std::tan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::exp;
using std::log;
using std::pow;
using std::sqrt;
using std::cbrt;
using std::hypot;
using std::abs;
using std::isfinite;
using std::isinf;
using std::isnan;

namespace ndt {

  class type;

} // namespace dynd::ndt

namespace nd {

  class array;
  class callable;

} // namespace dynd ::nd
} // namespace dynd
