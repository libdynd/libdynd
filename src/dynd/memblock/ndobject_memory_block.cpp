//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/ndobject_memory_block.hpp>
#include <dynd/nodes/ndarray_node.hpp>

using namespace std;
using namespace dynd;

namespace dynd { namespace detail {

void free_ndobject_memory_block(memory_block_data *memblock)
{
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(memblock);
    char *metadata = reinterpret_cast<char *>(preamble + 1);
    // If the dtype in the preamble is NULL, it was never initialized
    if (preamble->m_data_pointer != NULL) {
        // First free the references contained in the metadata
        if (!preamble->is_builtin_dtype()) {
            preamble->m_dtype->metadata_destruct(metadata);
            extended_dtype_decref(preamble->m_dtype);
        }
        // Then free the reference to the ndobject data
        if (preamble->m_data_reference != NULL) {
            memory_block_decref(preamble->m_data_reference);
        }
    }
    // Finally free the memory block itself
    free(reinterpret_cast<void *>(memblock));
}

}} // namespace dynd::detail

memory_block_ptr dynd::make_ndobject_memory_block(size_t metadata_size)
{
    char *result = (char *)malloc(sizeof(memory_block_data) + sizeof(ndobject_preamble) + metadata_size);
    if (result == 0) {
        throw bad_alloc();
    }
    // Signal that this object is uninitialized by setting its dtype to NULL
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result + sizeof(memory_block_data));
    preamble->m_dtype = NULL;
    preamble->m_data_pointer = NULL;
    return memory_block_ptr(new (result) memory_block_data(1, ndobject_memory_block_type), false);
}

memory_block_ptr dynd::make_ndobject_memory_block(size_t metadata_size, size_t extra_size,
                    size_t extra_alignment, char **out_extra_ptr)
{
    size_t extra_offset = inc_to_alignment(sizeof(memory_block_data) + sizeof(ndobject_preamble) + metadata_size,
                                        extra_alignment);
    char *result = (char *)malloc(extra_offset + extra_size);
    if (result == 0) {
        throw bad_alloc();
    }
    // Signal that this object is uninitialized by setting its dtype to NULL
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result + sizeof(memory_block_data));
    preamble->m_dtype = NULL;
    preamble->m_data_pointer = NULL;
    // Return a pointer to the extra allocated memory
    *out_extra_ptr = result + extra_offset;
    return memory_block_ptr(new (result) memory_block_data(1, ndobject_memory_block_type), false);
}

memory_block_ptr dynd::make_ndobject_memory_block(const dtype& dt, int ndim, const intptr_t *shape)
{
    size_t metadata_size, element_size;

    if (dt.extended() == NULL) {
        metadata_size = 0;
        element_size = dt.element_size();
    } else {
        metadata_size = dt.extended()->get_metadata_size();
        element_size = dt.extended()->get_default_element_size(ndim, shape);
    }

    char *data = NULL;
    memory_block_ptr result = make_ndobject_memory_block(metadata_size, element_size, dt.alignment(), &data);
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result.get());
    if (dt.extended() == NULL) {
        preamble->m_dtype = reinterpret_cast<extended_dtype *>(dt.type_id());
    } else {
        preamble->m_dtype = dt.extended();
        extended_dtype_incref(preamble->m_dtype);
        dt.extended()->metadata_default_construct(reinterpret_cast<char *>(preamble + 1), ndim, shape);
    }
    preamble->m_data_pointer = data;
    preamble->m_data_reference = NULL;
    preamble->m_flags = read_access_flag|write_access_flag;
    return result;
}

memory_block_ptr dynd::shallow_copy_ndobject_memory_block(const memory_block_ptr& ndo)
{
    // Allocate the new memory block.
    const ndobject_preamble *preamble = reinterpret_cast<const ndobject_preamble *>(ndo.get());
    size_t metadata_size = 0;
    if (!preamble->is_builtin_dtype()) {
        metadata_size = preamble->m_dtype->get_metadata_size();
    }
    memory_block_ptr result = make_ndobject_memory_block(metadata_size);
    ndobject_preamble *result_preamble = reinterpret_cast<ndobject_preamble *>(result.get());

    // Clone the data pointer
    result_preamble->m_data_pointer = preamble->m_data_pointer;
    result_preamble->m_data_reference = preamble->m_data_reference;
    if (result_preamble->m_data_reference == NULL) {
        result_preamble->m_data_reference = ndo.get();
    }
    memory_block_incref(result_preamble->m_data_reference);

    // Copy the flags
    result_preamble->m_flags = preamble->m_flags;

    // Clone the dtype
    result_preamble->m_dtype = preamble->m_dtype;
    if (!preamble->is_builtin_dtype()) {
        extended_dtype_incref(preamble->m_dtype);
        preamble->m_dtype->metadata_copy_construct(reinterpret_cast<char *>(result.get()) + sizeof(ndobject_preamble),
                        reinterpret_cast<const char *>(ndo.get()) + sizeof(ndobject_preamble), ndo.get());
    }

    return result;
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
