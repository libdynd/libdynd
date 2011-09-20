//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _FPSTATUS_HPP_
#define _FPSTATUS_HPP_

#if defined(_MSC_VER)
#include <float.h>

namespace dnd {

inline void clear_fp_status() {
    _clearfp();
}

inline bool is_overflow_fp_status() {
    return (_statusfp() & _EM_OVERFLOW) != 0;
}

inline bool is_inexact_fp_status() {
    return (_statusfp() & _EM_INEXACT) != 0;
}

} // namespace dnd

#else
#include <fenv.h>

namespace dnd {

inline void clear_fp_status() {
    feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
}

inline bool is_overflow_fp_status() {
    return fetestexcept(FE_OVERFLOW) != 0;
}

inline bool is_inexact_fp_status() {
    return fetestexcept(FE_INEXACT) != 0;
}

} // namespace dnd

#endif

#endif // _FPSTATUS_HPP_
