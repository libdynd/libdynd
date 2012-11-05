//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__POD_MEMORY_BLOCK_HPP_
#define _DYND__POD_MEMORY_BLOCK_HPP_

#include <iostream>
#include <string>

#include <dynd/dtype.hpp>
#include <dynd/memblock/memory_block.hpp>

namespace dynd {

/**
 * This structure is the start of any ndobject metadata. The
 * metadata after this structure is determined by the m_dtype
 * object.
 */
struct ndobject_preamble {
    extended_dtype *m_dtype;
    void *m_data_pointer;
    memory_block_data *m_data_reference;
};

/**
 * Creates a memory block for holding an ndobject (i.e. a container for ndobject metadata)
 */
memory_block_ptr make_ndobject_memory_block(intptr_t metadata_size);

/**
 * Creates a memory block for holding an ndobject (i.e. a container for ndobject metadata),
 * as well as storage for embedding additional POD storage such as the array data.
 */
memory_block_ptr make_ndobject_memory_block(intptr_t metadata_size, intptr_t extra_size,
                    intptr_t extra_alignment, char **out_extra_ptr);


void ndobject_memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent);

} // namespace dynd

#endif // _DYND__POD_MEMORY_BLOCK_HPP_
