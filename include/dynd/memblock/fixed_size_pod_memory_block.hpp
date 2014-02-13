//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
#define _DYND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
memory_block_ptr make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer);

void fixed_size_pod_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent);

} // namespace dynd

#endif // _DYND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
