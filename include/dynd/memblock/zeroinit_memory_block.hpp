//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ZEROINIT_MEMORY_BLOCK_HPP_
#define _DYND__ZEROINIT_MEMORY_BLOCK_HPP_

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * Creates a memory block which can be used to allocate zero-initialized
 * POD output memory for blockref types.
 *
 * The initial capacity can be set if a good estimate is known.
 */
memory_block_ptr make_zeroinit_memory_block(intptr_t initial_capacity_bytes = 2048);

void zeroinit_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent);

} // namespace dynd

#endif // _DYND__ZEROINIT_MEMORY_BLOCK_HPP_
