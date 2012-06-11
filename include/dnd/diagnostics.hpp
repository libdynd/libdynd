//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__DIAGNOSTICS_HPP_
#define _DND__DIAGNOSTICS_HPP_

#include <dnd/config.hpp>

#define DND_XSTRINGIFY(s) #s
#define DND_STRINGIFY(s) DND_XSTRINGIFY(s)

/**
 * This preprocessor symbol enables or disables assertions
 * that pointers are aligned as they are supposed to be. This helps
 * test that alignment is being done correctly on platforms which
 * do not segfault on misaligned data.
 *
 * An exception is thrown if an improper unalignment is detected.
 */
#define DND_ALIGNMENT_ASSERTIONS 1

#if DND_ALIGNMENT_ASSERTIONS
#include <sstream>
#include <dnd/dtype.hpp>

# define DND_ASSERT_ALIGNED(ptr, stride, alignment, extra_info) { \
        /*std::cout << "checking alignment for  " << __FILE__ << ": " << __LINE__ << "\n";*/ \
        if ((((uintptr_t)ptr) & (alignment-1)) != 0 || (((uintptr_t)stride) & (alignment-1)) != 0) { \
            std::stringstream ss; \
            ss << "improper unalignment detected, " << __FILE__ << ": " << __LINE__ << "\n"; \
            ss << "pointer " << DND_STRINGIFY(ptr) << "(" << ptr << \
                    ") and stride " << DND_STRINGIFY(stride) << "(" << stride << ")\n"; \
            ss << "expected alignment " << alignment << "\n"; \
            ss << extra_info; \
            throw std::runtime_error(ss.str()); \
        } \
    }
#else
# define DND_ASSERT_ALIGNED(ptr, alignment, extra_info) {}
#endif

#endif // _DND__DIAGNOSTICS_HPP_
