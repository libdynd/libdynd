//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/ndobject_memory_block.hpp>

using namespace std;
using namespace dynd;

namespace dynd { namespace detail {

void free_ndobject_memory_block(memory_block_data *memblock)
{
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(memblock + 1);
    char *metadata = reinterpret_cast<char *>(preamble + 1);
    // If the dtype in the preamble is NULL, it was never initialized
    if (preamble->m_dtype != NULL) {
        // First free the references contained in the metadata
        preamble->m_dtype->metadata_destruct(metadata);
        // Then free the reference to the ndobject data
        if (preamble->m_data_reference != NULL) {
            memory_block_decref(preamble->m_data_reference);
        }
    }
    // Finally free the memory block itself
    free(reinterpret_cast<void *>(memblock));
}

}} // namespace dynd::detail

memory_block_ptr dynd::make_ndobject_memory_block(intptr_t metadata_size)
{
    char *result = (char *)malloc(sizeof(memory_block_data) + sizeof(ndobject_preamble) + metadata_size);
    if (result == 0) {
        throw bad_alloc();
    }
    // Signal that this object is uninitialized by setting its dtype to NULL
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result + sizeof(memory_block_data));
    preamble->m_dtype = NULL;
    return memory_block_ptr(new (result) memory_block_data(1, ndobject_memory_block_type), false);
}

memory_block_ptr make_ndobject_memory_block(intptr_t metadata_size, intptr_t extra_size,
                    intptr_t extra_alignment, char **out_extra_ptr)
{
    intptr_t extra_offset = inc_to_alignment(sizeof(memory_block_data) + sizeof(ndobject_preamble) + metadata_size,
                                        extra_alignment);
    char *result = (char *)malloc(extra_offset + extra_size);
    if (result == 0) {
        throw bad_alloc();
    }
    // Signal that this object is uninitialized by setting its dtype to NULL
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result + sizeof(memory_block_data));
    preamble->m_dtype = NULL;
    // Return a pointer to the extra allocated memory
    *out_extra_ptr = result + extra_offset;
    return memory_block_ptr(new (result) memory_block_data(1, ndobject_memory_block_type), false);
}

void dynd::ndobject_memory_block_debug_dump(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const ndobject_preamble *preamble = reinterpret_cast<const ndobject_preamble *>(memblock + 1);
    const char *metadata = reinterpret_cast<const char *>(preamble + 1);
    if (preamble->m_dtype != NULL) {
        o << indent << " dtype: " << dynd::dtype(preamble->m_dtype) << "\n";
        o << indent << " data pointer: " << (const void *)preamble->m_data_pointer << "\n";
        o << indent << " data memblock:\n";
        memory_block_debug_dump(preamble->m_data_reference, o, indent + " ");
        preamble->m_dtype->metadata_debug_dump(metadata, o, indent + " ");
    } else {
        o << indent << " uninitialized ndobject\n";
    }
}
