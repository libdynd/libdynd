//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__LOWLEVEL_API_HPP_
#define _DYND__LOWLEVEL_API_HPP_

#include <dynd/memblock/memory_block.hpp>
#include <dynd/dtypes/base_dtype.hpp>

namespace dynd {

/**
 * This struct contains a bunch of function which provide
 * low level C-level access to the innards of dynd.
 *
 * These functions are static and should not be modified
 * after initialization.
 */
struct lowlevel_api_t {
    uintptr_t version;
    // Reference counting primitives for memory blocks (including ndobjects)
    void (*memory_block_incref)(memory_block_data *mbd);
    void (*memory_block_decref)(memory_block_data *mbd);
    // memory_block_free is *only* exposed for use by code inlining
    // the atomic incref/decref code. If you're not *absolutely sure*
    // you're using it correctly, use the incref/decref instead.
    void (*memory_block_free)(memory_block_data *mbd);
    // Reference counting primitives for dtypes
    void (*base_dtype_incref)(const base_dtype *bd);
    void (*base_dtype_decref)(const base_dtype *bd);
    // Get the base_dtype_members struct from a base dtype
    const base_dtype_members *(*get_base_dtype_members)(const base_dtype *bd);
};

} // namespace dynd

/**
 * Returns a pointer to the static low level API structure.
 */
extern "C" const void *dynd_get_lowlevel_api();

#endif // _DYND__LOWLEVEL_API_HPP_
