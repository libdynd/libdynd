//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CLING_ALL_HPP_
#define _DYND__CLING_ALL_HPP_

// When CLING is enabled, we avoid certain glibc operations
// that generate inline assembly, unsupported by the LLVM JIT.
#define DYND_CLING

#include <dynd/ndarray.hpp>
#include <dynd/ndarray_range.hpp>

using namespace std;
using namespace dynd;

#endif // _DYND__CLING_ALL_HPP_
