//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CONFIG_HPP_
#define _DYND__CONFIG_HPP_

#include <cstdlib>

#ifdef __clang__

//#  define DYND_RVALUE_REFS
//#  define DYND_INIT_LIST

// TODO: versions with constexpr
#  define DYND_CONSTEXPR constexpr

# define DYND_USE_STDINT

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

# define DYND_USE_STDINT

#if __GNUC__ > 4 || \
              (__GNUC__ == 4 && (__GNUC_MINOR__ >= 7))
// Use initializer lists on gcc >= 4.7
#  define DYND_INIT_LIST
// Use constexpr on gcc >= 4.7
#  define DYND_CONSTEXPR constexpr
// Use rvalue references on gcc >= 4.7
#  define DYND_RVALUE_REFS
#  define DYND_ISNAN(x) (std::isnan(x))
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

// If set, uses the FP status registers.
// On some compilers, there is no proper
// way to tell the compiler that these are
// important, and it reorders instructions
// so as to make them useless. On MSVC, there
// is #pragma fenv_access(on), which works.
# define DYND_USE_FPSTATUS

# if _MSC_VER >= 1600
// Use enable_if from std::tr1
#  define DYND_USE_TR1_ENABLE_IF
// Use rvalue refs
#  define DYND_RVALUE_REFS
#  define DYND_USE_STDINT
# else
#  define DYND_USE_BOOST_SHARED_PTR
# endif

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
#  define DYND_MOVE(x) (std::move(x))
#else
#  define DYND_MOVE(x) (x)
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

#ifdef DYND_USE_TR1_ENABLE_IF
#include <type_traits>
namespace dynd {
    using std::tr1::enable_if;
}
#else
// enable_if is a small template, so we just replicate it here
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

#ifdef DYND_USE_STDINT
#include <stdint.h>
#else
typedef signed char      int8_t;
typedef short            int16_t;
typedef int              int32_t;
typedef __int64          int64_t;
typedef ptrdiff_t        intptr_t;
typedef unsigned char    uint8_t;
typedef unsigned short   uint16_t;
typedef unsigned int     uint32_t;
typedef unsigned __int64 uint64_t;
typedef size_t           uintptr_t;
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

#include <dynd/cuda_config.hpp>

#endif // _DYND__CONFIG_HPP_
