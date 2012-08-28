//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__CONFIG_HPP_
#define _DND__CONFIG_HPP_

#ifdef __clang__

//#  define DND_RVALUE_REFS
//#  define DND_INIT_LIST

// NOTE: I hacked the g++ system headers, adding explicit copy constructors and assignment
//       operators to both shared_ptr and __shared_ptr (in shared_ptr_base.h), so that clang
//       would accept them. This is because the LLVM JIT used by CLING complains about inline
//       assembly with the boost version, but not with the g++ version.
#define DND_USE_BOOST_SHARED_PTR
// TODO: versions with constexpr
#  define DND_CONSTEXPR constexpr

# define DND_USE_STDINT

#elif defined(__GNUC__)

# define DND_USE_STDINT

#if __GNUC__ > 4 || \
              (__GNUC__ == 4 && (__GNUC_MINOR__ >= 7))
// Use initializer lists on gcc >= 4.7
#  define DND_INIT_LIST
// Use constexpr on gcc >= 4.7
#  define DND_CONSTEXPR constexpr
// Use rvalue references on gcc >= 4.7
#  define DND_RVALUE_REFS
#else
// Don't use constexpr on gcc < 4.7
#  define DND_CONSTEXPR
// Use boost shared_ptr on gcc < 4.7
#  define DND_USE_BOOST_SHARED_PTR
#endif

#elif defined(_MSC_VER)

# if _MSC_VER >= 1600
// Use enable_if from std::tr1
#  define DND_USE_TR1_ENABLE_IF
// Use rvalue refs
#  define DND_RVALUE_REFS
#  define DND_USE_STDINT
# else
#  define DND_USE_BOOST_SHARED_PTR
# endif

// No DND_CONSTEXPR yet, define it as nothing
#  define DND_CONSTEXPR

#endif

// If RValue References are supported
#ifdef DND_RVALUE_REFS
#  define DND_MOVE(x) (std::move(x))
#else
#  define DND_MOVE(x) (x)
#endif

// Whether to use boost's shared_ptr or the standard library's
#ifdef DND_USE_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
namespace dnd {
    using ::boost::shared_ptr;
    using ::boost::make_shared;
}
#else
#include <memory>
namespace dnd {
    using ::std::shared_ptr;
    using ::std::make_shared;
}
#endif

// If Initializer Lists are supported
#ifdef DND_INIT_LIST
#include <initializer_list>
#endif

// If being run from the CLING C++ interpreter
#ifdef DND_CLING
// 1) Used g++ shared_ptr instead of boost shared_ptr (see above in clang config section).
//    This allowed dtypes to be created in cling!
// 2) Don't use the memcpy function (it has inline assembly).

inline void DND_MEMCPY(char *dst, const char *src, intptr_t count)
{
    char *cdst = (char *)dst;
    const char *csrc = (const char *)src;
    while (count--) {
        *cdst++ = *csrc++;
    }
}
#else
#include <cstring>
#define DND_MEMCPY(a, b, c) std::memcpy(a, b, c)
#endif

#ifdef DND_USE_TR1_ENABLE_IF
#include <type_traits>
namespace dnd {
    using std::tr1::enable_if;
}
#else
#include <boost/utility/enable_if.hpp>
namespace dnd {
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
#ifndef DND_ALIGNMENT_ASSERTIONS
# define DND_ALIGNMENT_ASSERTIONS 0
#endif

/**
 * This preprocessor symbol enables or disables tracing
 * of individual builtin dtype assignment operations.
 *
 * See diagnostics.hpp for the macros which use this.
 */
#ifndef DND_ASSIGNMENT_TRACING
# define DND_ASSIGNMENT_TRACING 0
#endif


/**
 * Preprocessor macro for marking variables unused, and suppressing
 * warnings for them.
 */
#define DND_UNUSED(x)

#ifdef DND_USE_STDINT
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

#endif // _DND__CONFIG_HPP_
