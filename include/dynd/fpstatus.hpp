//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#if defined(_MSC_VER)
#include <float.h>

namespace dynd {

inline void clear_fp_status() {
    _clearfp();
}

inline bool is_overflow_fp_status() {
    return (_statusfp() & _EM_OVERFLOW) != 0;
}

inline bool is_inexact_fp_status() {
    return (_statusfp() & _EM_INEXACT) != 0;
}

} // namespace dynd

#else
#include <fenv.h>

namespace dynd {

inline void clear_fp_status() {
    feclearexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
}

inline bool is_overflow_fp_status() {
    return fetestexcept(FE_OVERFLOW) != 0;
}

inline bool is_inexact_fp_status() {
    return fetestexcept(FE_INEXACT) != 0;
}

} // namespace dynd

#endif
