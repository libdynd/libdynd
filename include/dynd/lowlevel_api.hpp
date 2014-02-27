//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__LOWLEVEL_API_HPP_
#define _DYND__LOWLEVEL_API_HPP_

#include <dynd/memblock/memory_block.hpp>
#include <dynd/types/base_type.hpp>

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
    // Reference counting primitives for memory blocks (including nd::arrays)
    void (*memory_block_incref)(memory_block_data *mbd);
    void (*memory_block_decref)(memory_block_data *mbd);
    // memory_block_free is *only* exposed for use by code inlining
    // the atomic incref/decref code. If you're not *absolutely sure*
    // you're using it correctly, use the incref/decref instead.
    void (*memory_block_free)(memory_block_data *mbd);
    // Reference counting primitives for dynd types
    void (*base_type_incref)(const base_type *bd);
    void (*base_type_decref)(const base_type *bd);
    // Get the base_type_members struct from a base type
    const base_type_members *(*get_base_type_members)(const base_type *bd);
    // constructor, destructor, member functions of ckernel_builder
    void (*ckernel_builder_construct)(void *ckb);
    void (*ckernel_builder_destruct)(void *ckb);
    void (*ckernel_builder_reset)(void *ckb);
    int (*ckernel_builder_ensure_capacity_leaf)(void *ckb, intptr_t requested_capacity);
    int (*ckernel_builder_ensure_capacity)(void *ckb, intptr_t requested_capacity);
};

} // namespace dynd

/**
 * Returns a pointer to the static low level API structure.
 */
extern "C" const void *dynd_get_lowlevel_api();

#endif // _DYND__LOWLEVEL_API_HPP_
