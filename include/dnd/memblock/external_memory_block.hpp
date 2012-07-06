//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__EXTERNAL_MEMORY_BLOCK_HPP_
#define _DND__EXTERNAL_MEMORY_BLOCK_HPP_

#include <dnd/memblock/memory_block.hpp>

namespace dnd {

typedef void (*external_memory_block_free_t)(void *);

/**
 * Creates a memory block which is a reference to an external object.
 */
memory_block_ptr make_external_memory_block(void *object, external_memory_block_free_t free_fn);

} // namespace dnd

#endif // _DND__EXTERNAL_MEMORY_BLOCK_HPP_
