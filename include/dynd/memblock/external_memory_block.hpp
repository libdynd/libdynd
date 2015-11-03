//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

typedef void (*external_memory_block_free_t)(void *);

/**
 * Creates a memory block which is a reference to an external object.
 */
DYND_API intrusive_ptr<memory_block_data> make_external_memory_block(void *object,
                                                                     external_memory_block_free_t free_fn);

DYND_API void external_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                const std::string &indent);

} // namespace dynd
