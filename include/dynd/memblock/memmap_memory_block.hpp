//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream>
#include <string>

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * Creates a memory block of a memory-mapped file.
 *
 * \param filename  The filename of the file to memory map.
 * \param access  A combination of write_access_flag, read_access_flag, immutable_access_flag.
 * \param out_pointer  This is the pointer to the mapped memory.
 * \param out_size  This is the size of the mapped memory. Note that the size may be different
 *                  than requested by begin/end, because this function uses Python semantics to
 *                  clip out of bounds boundaries to the data.
 * \param begin  The position within the file to start the memory map
 *               (default beginning of the file). This value may be
 *               negative, in which case it is interpreted as an offset from the
 *               end of the file.
 * \param end  The position within the file to end the memory map
 *             (default end of the file). This value may be
 *             negative, in which case it is interpreted as an offset from the
 *             end of the file.
 */
DYND_API intrusive_ptr<memory_block_data> make_memmap_memory_block(const std::string &filename, uint32_t access,
                                                                   char **out_pointer, intptr_t *out_size,
                                                                   intptr_t begin = 0,
                                                                   intptr_t end = std::numeric_limits<intptr_t>::max());

DYND_API void memmap_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                              const std::string &indent);

} // namespace dynd
