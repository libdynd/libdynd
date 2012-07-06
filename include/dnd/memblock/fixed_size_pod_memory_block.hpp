//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
#define _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_

#include <dnd/memblock/memory_block.hpp>

namespace dnd {

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
memory_block_ptr make_fixed_size_pod_memory_block(intptr_t alignment, intptr_t size, char **out_datapointer);

} // namespace dnd

#endif // _DND__FIXED_SIZE_POD_MEMORY_BLOCK_HPP_
