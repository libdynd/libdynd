//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
#define _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_

#include <iostream>
#include <string>

#include <dnd/memblock/memory_block.hpp>

namespace dnd {

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
memory_block_ptr make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer);

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 *
 * The resulting memory block takes over the references in the provided blockref array, all the
 * references are moved out of that array.
 */
memory_block_ptr make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment, char **out_datapointer,
                memory_block_ptr *blockrefs_begin, memory_block_ptr *blockrefs_end);

void fixed_size_pod_memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent);

} // namespace dnd

#endif // _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
