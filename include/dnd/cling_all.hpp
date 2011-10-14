//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__CLING_ALL_HPP_
#define _DND__CLING_ALL_HPP_

// When CLING is enabled, we avoid certain glibc operations
// that generate inline assembly, unsupported by the LLVM JIT.
#define DND_CLING

#include <dnd/ndarray.hpp>
#include <dnd/ndarray_arange.hpp>

using namespace std;
using namespace dnd;

#endif // _DND__CLING_ALL_HPP_
