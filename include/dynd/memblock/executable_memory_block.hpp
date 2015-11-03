//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * Creates a memory block which holds memory which can be used for
 * dynamic code generation. Generally, we want there to be very few of these,
 * and to use a relatively large chunk size. Having this be a separate memory
 * block instead of a global is still useful, because otherwise it would be
 * impossible to temporarily allocate executable memory using this system.
 *
 * @param chunk_size_bytes  When memory has run out, another chunk of this
 *                          size is added for more requests.
 */
DYND_API intrusive_ptr<memory_block_data> make_executable_memory_block(intptr_t chunk_size_bytes = 65536);

#if defined(_WIN32) && defined(_M_X64)
// Windows x64
/**
 * Sets the RUNTIME_FUNCTION structure for the most recently allocated function,
 * for unwinding the stack during exceptions.
 */
DYND_API void set_executable_memory_runtime_function(memory_block_data *self, char *begin, char *end,
                                                     char *unwind_data);
#else
// ov: work in progress
// #error The executable memory block has not been implemented for this platform yet.
#endif

/**
 * Allocates some executable memory for code generation, owned by the memory block.
 */
DYND_API void allocate_executable_memory(memory_block_data *self, intptr_t size_bytes, intptr_t alignment,
                                         char **out_begin, char **out_end);

/**
 * Resizes the most recently allocated executable memory, owned by the memory block.
 * This may move the memory if the new size is larger, so if the code isn't PIC, it may require fix ups.
 */
DYND_API void resize_executable_memory(memory_block_data *self, intptr_t size_bytes, char **inout_begin,
                                       char **inout_end);

DYND_API void executable_memory_block_debug_print(const memory_block_data *memblock, std::ostream &o,
                                                  const std::string &indent);

} // namespace dynd
