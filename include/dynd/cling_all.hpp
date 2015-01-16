//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

// When CLING is enabled, we avoid certain glibc operations
// that generate inline assembly, unsupported by the LLVM JIT.
#define DYND_CLING

#include <dynd/ndarray.hpp>
#include <dynd/ndarray_range.hpp>

using namespace std;
using namespace dynd;
