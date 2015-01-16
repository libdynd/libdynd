//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>

#include <dynd/config.hpp>

#define DYND_XSTRINGIFY(s) #s
#define DYND_STRINGIFY(s) DYND_XSTRINGIFY(s)

#if DYND_ALIGNMENT_ASSERTIONS
#include <iostream>
#include <sstream>
#include <dynd/type.hpp>

# define DYND_ASSERT_ALIGNED(ptr, stride, alignment, extra_info) { \
        /*std::cout << "checking alignment for  " << __FILE__ << ": " << __LINE__ << "\n";*/ \
        if ((((uintptr_t)ptr) & (alignment-1)) != 0 || (((uintptr_t)stride) & (alignment-1)) != 0) { \
            std::stringstream ss; \
            ss << "improper unalignment detected, " << __FILE__ << ": " << __LINE__ << "\n"; \
            ss << "pointer " << DYND_STRINGIFY(ptr) << " (" << ((void *)(ptr)) << \
                    ") and stride " << DYND_STRINGIFY(stride) << " (" << ((intptr_t)(stride)) << ")\n"; \
            ss << "expected alignment " << (alignment) << "\n"; \
            ss << extra_info; \
            throw std::runtime_error(ss.str()); \
        } \
    }
#else
# define DYND_ASSERT_ALIGNED(ptr, stride, alignment, extra_info) {}
#endif

#ifdef __APPLE__
# define DYND_TRIGGER_ASSERT(message)
#else
# define DYND_TRIGGER_ASSERT(message) assert(!message)
#endif
#define DYND_TRIGGER_ASSERT_RETURN_ZERO(message) DYND_TRIGGER_ASSERT(message); return 0

#if DYND_ASSIGNMENT_TRACING && !defined(__CUDA_ARCH__)
#include <iostream>
#include <dynd/type.hpp>

# define DYND_TRACE_ASSIGNMENT(dst_value, dst_type, src_value, src_type) { \
        std::cerr << "Assigning value " << src_value << " to value " << dst_value << " from " \
                << dynd::make_type<src_type>() << " to " << dynd::make_type<dst_type>() << std::endl; \
    }
#else
# define DYND_TRACE_ASSIGNMENT(dst_value, dst_type, src_value, src_type) {}
#endif

#ifdef __CUDA_ARCH__
#define DYND_HOST_THROW(EXCEPTION, MESSAGE)
#else
#define DYND_HOST_THROW(EXCEPTION, MESSAGE) throw EXCEPTION(MESSAGE)
#endif

namespace dynd {

#define DYND_ANY_DIAGNOSTICS_ENABLED ((DYND_ALIGNMENT_ASSERTIONS != 0) || (DYND_ASSIGNMENT_TRACING != 0))

/**
 * This function returns true if any diagnostics, which might
 * slow down execution speed, are enabled. For example, in
 * the Python exposure, this is used to print a warning on
 * module import when performance might be hampered by this.
 */
inline bool any_diagnostics_enabled()
{
    // IMPORTANT: All diagnostic macros should be checked here,
    //            and added to the description string below.
    return DYND_ANY_DIAGNOSTICS_ENABLED;
}

/**
 * This function returns a string which lists the enabled
 * diagnostics, including a short description for each.
 */
inline std::string which_diagnostics_enabled()
{
#if DYND_ANY_DIAGNOSTICS_ENABLED
    std::stringstream ss;
#if DYND_ALIGNMENT_ASSERTIONS
    ss << "DYND_ALIGNMENT_ASSERTIONS - checks that data has correct alignment in inner loops\n";
#endif // DYND_ALIGNMENT_ASSERTIONS
#if DYND_ASSIGNMENT_TRACING
    ss << "DYND_ASSIGNMENT_TRACING - prints individual builtin assignment operations\n";
#endif // DYND_ASSIGNMENT_TRACING
    return ss.str();
#else
    return "";
#endif
}

} // namespace dynd
