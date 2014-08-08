//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CONFIG_HPP_
#define _DYND__CONFIG_HPP_

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
#else
// Don't use constexpr on gcc < 4.7
#  define DYND_CONSTEXPR
#  define DYND_ISNAN(x) isnan(x)
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
#define DYND_USE_STD_ATOMIC
#endif

#if _MSC_VER >= 1800
// MSVC 2013 and later

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
#ifndef DYND_STATIC_ASSERT
#define DYND_STATIC_ASSERT(value, message) do { enum { dynd_static_assertion = 1 / (int)(value) }; } while (0)
#endif

#ifdef DYND_USE_TR1_ENABLE_IF
#include <type_traits>
namespace dynd {
    using std::tr1::enable_if;
    using std::tr1::is_const;
    using std::tr1::remove_const;
    using std::tr1::is_reference;
    using std::tr1::remove_reference;
}
#else
// These are small templates, so we just replicate them here
namespace dynd {
	template<bool B, class T = void>
	struct enable_if {};
 
	template<class T>
	struct enable_if<true, T> { typedef T type; };

    template<class T>
    struct is_const { enum { value = 0 }; };

    template<class T>
    struct is_const<const T> { enum { value = 1 }; };

    template<class T>
    struct remove_const { typedef T type; };

    template<class T>
    struct remove_const<const T> { typedef T type; };

    template<class T>
    struct is_reference { enum { value = 0 }; };

    template<class T>
    struct is_reference<T&> { enum { value = 1 }; };

#ifdef DYND_RVALUE_REFS
    template<class T>
    struct is_reference<T&&> { enum { value = 1 }; };
#endif

    template<class T>
    struct remove_reference { typedef T type; };

    template<class T>
    struct remove_reference<T&> { typedef T type; };

#ifdef DYND_RVALUE_REFS
    template<class T>
    struct remove_reference<T&&> { typedef T type; };
#endif
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
     * A function which can be used at runtime to identify whether
     * the build of dynd being linked against was built with CUDA
     * support enabled.
     */
    bool built_with_cuda();
} // namespace dynd

#include <dynd/cuda_config.hpp>

#endif // _DYND__CONFIG_HPP_
