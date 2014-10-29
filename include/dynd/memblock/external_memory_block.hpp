//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

typedef void (*external_memory_block_free_t)(void *);

/**
 * Creates a memory block which is a reference to an external object.
 */
memory_block_ptr make_external_memory_block(void *object, external_memory_block_free_t free_fn);

void external_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent);

} // namespace dynd
