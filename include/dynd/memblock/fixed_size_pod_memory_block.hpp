//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

struct fixed_size_pod_memory_block : memory_block_data {
  fixed_size_pod_memory_block(long use_count) : memory_block_data(use_count, fixed_size_pod_memory_block_type) {}
};

/**
 * Creates a memory block of a pre-determined fixed size. A pointer to the
 * memory allocated for data is placed in the output parameter.
 */
DYNDT_API intrusive_ptr<memory_block_data> make_fixed_size_pod_memory_block(intptr_t size_bytes, intptr_t alignment,
                                                                           char **out_datapointer);

DYNDT_API void fixed_size_pod_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                       const std::string &indent);

} // namespace dynd
