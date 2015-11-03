//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>
#include <dynd/type.hpp>

namespace dynd {

/**
 * Creates a memory block which can be used to allocate zero-initialized
 * object type output memory for blockref types.
 *
 * The initial count of elements can be set if a good estimate is known.
 *
 * \param dt  The data type of the objects to allocate.
 * \param arrmeta  The arrmeta corresponding to the data type for the objects to allocate.
 * \param stride  For objects without a fixed size, the size of the memory to allocate
 *                for each element. This would be typically set to the value for
 *                get_default_data_size() corresponding to default-constructed arrmeta.
 * \param initial_count  The number of elements to allocate at the start.
 */
DYND_API intrusive_ptr<memory_block_data> make_objectarray_memory_block(const ndt::type &dt, const char *arrmeta,
                                                                        intptr_t stride, intptr_t initial_count = 64,
                                                                        size_t arrmeta_size = 0);

DYND_API void objectarray_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                   const std::string &indent);

} // namespace dynd
