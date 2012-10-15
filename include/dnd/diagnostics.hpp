//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__DIAGNOSTICS_HPP_
#define _DND__DIAGNOSTICS_HPP_

#include <dnd/config.hpp>

#define DND_XSTRINGIFY(s) #s
#define DND_STRINGIFY(s) DND_XSTRINGIFY(s)

#if DND_ALIGNMENT_ASSERTIONS
#include <iostream>
#include <sstream>
#include <dnd/dtype.hpp>

# define DND_ASSERT_ALIGNED(ptr, stride, alignment, extra_info) { \
        /*std::cout << "checking alignment for  " << __FILE__ << ": " << __LINE__ << "\n";*/ \
        if ((((uintptr_t)ptr) & (alignment-1)) != 0 || (((uintptr_t)stride) & (alignment-1)) != 0) { \
            std::stringstream ss; \
            ss << "improper unalignment detected, " << __FILE__ << ": " << __LINE__ << "\n"; \
            ss << "pointer " << DND_STRINGIFY(ptr) << " (" << ((void *)(ptr)) << \
                    ") and stride " << DND_STRINGIFY(stride) << " (" << ((intptr_t)(stride)) << ")\n"; \
            ss << "expected alignment " << (alignment) << "\n"; \
            ss << extra_info; \
            throw std::runtime_error(ss.str()); \
        } \
    }
#else
# define DND_ASSERT_ALIGNED(ptr, stride, alignment, extra_info) {}
#endif

#if DND_ASSIGNMENT_TRACING
#include <iostream>
#include <dnd/dtype.hpp>

# define DND_TRACE_ASSIGNMENT(dst_value, dst_type, src_value, src_type) { \
        std::cerr << "Assigning value " << src_value << " to value " << dst_value << " from " \
                << dynd::make_dtype<src_type>() << " to " << dynd::make_dtype<dst_type>() << std::endl; \
    }
#else
# define DND_TRACE_ASSIGNMENT(dst_value, dst_type, src_value, src_type) {}
#endif

namespace dynd {

#define DND_ANY_DIAGNOSTICS_ENABLED ((DND_ALIGNMENT_ASSERTIONS != 0) || (DND_ASSIGNMENT_TRACING != 0))

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
    return DND_ANY_DIAGNOSTICS_ENABLED;
}

/**
 * This function returns a string which lists the enabled
 * diagnostics, including a short description for each.
 */
inline std::string which_diagnostics_enabled()
{
#if DND_ANY_DIAGNOSTICS_ENABLED
    std::stringstream ss;
#if DND_ALIGNMENT_ASSERTIONS
    ss << "DND_ALIGNMENT_ASSERTIONS - checks that data has correct alignment in inner loops\n";
#endif // DND_ALIGNMENT_ASSERTIONS
#if DND_ASSIGNMENT_TRACING
    ss << "DND_ASSIGNMENT_TRACING - prints individual builtin assignment operations\n";
#endif // DND_ASSIGNMENT_TRACING
    return ss.str();
#else
    return "";
#endif
}

} // namespace dynd

#endif // _DND__DIAGNOSTICS_HPP_
