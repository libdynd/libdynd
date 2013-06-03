//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/memblock/ndobject_memory_block.hpp>
#include <dynd/ndobject.hpp>

using namespace std;
using namespace dynd;

namespace dynd { namespace detail {

void free_ndobject_memory_block(memory_block_data *memblock)
{
    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(memblock);
    char *metadata = reinterpret_cast<char *>(preamble + 1);

    // Call the data destructor if necessary (i.e. the ndobject owns
    // the data memory, and the dtype has a data destructor)
    if (preamble->m_data_reference == NULL &&
                    !preamble->is_builtin_dtype() &&
                    (preamble->m_dtype->get_flags()&dtype_flag_destructor) != 0) {
        preamble->m_dtype->data_destruct(metadata, preamble->m_data_pointer);
    }

    // Free the references contained in the metadata
    if (!preamble->is_builtin_dtype()) {
        preamble->m_dtype->metadata_destruct(metadata);
        base_dtype_decref(preamble->m_dtype);
    }

    // Free the reference to the ndobject data
    if (preamble->m_data_reference != NULL) {
        memory_block_decref(preamble->m_data_reference);
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
    // Zero out all the metadata to start
    memset(result + sizeof(memory_block_data), 0, sizeof(ndobject_preamble) + metadata_size);
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
    // Zero out all the metadata to start
    memset(result + sizeof(memory_block_data), 0, sizeof(ndobject_preamble) + metadata_size);
    // Return a pointer to the extra allocated memory
    *out_extra_ptr = result + extra_offset;
    return memory_block_ptr(new (result) memory_block_data(1, ndobject_memory_block_type), false);
}

memory_block_ptr dynd::make_ndobject_memory_block(const dtype& dt, size_t ndim, const intptr_t *shape)
{
    size_t metadata_size, data_size;

    if (dt.is_builtin()) {
        metadata_size = 0;
        data_size = dt.get_data_size();
    } else {
        metadata_size = dt.extended()->get_metadata_size();
        data_size = dt.extended()->get_default_data_size(ndim, shape);
    }

    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(metadata_size, data_size, dt.get_data_alignment(), &data_ptr);

    if (dt.get_flags()&dtype_flag_zeroinit) {
        memset(data_ptr, 0, data_size);
    }

    ndobject_preamble *preamble = reinterpret_cast<ndobject_preamble *>(result.get());
    if (dt.is_builtin()) {
        preamble->m_dtype = reinterpret_cast<base_dtype *>(dt.get_type_id());
    } else {
        preamble->m_dtype = dtype(dt).release();
        preamble->m_dtype->metadata_default_construct(reinterpret_cast<char *>(preamble + 1), ndim, shape);
    }
    preamble->m_data_pointer = data_ptr;
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
        base_dtype_incref(preamble->m_dtype);
        preamble->m_dtype->metadata_copy_construct(reinterpret_cast<char *>(result.get()) + sizeof(ndobject_preamble),
                        reinterpret_cast<const char *>(ndo.get()) + sizeof(ndobject_preamble), ndo.get());
    }

    return result;
}

void dynd::ndobject_memory_block_debug_print(const memory_block_data *memblock, std::ostream& o, const std::string& indent)
{
    const ndobject_preamble *preamble = reinterpret_cast<const ndobject_preamble *>(memblock);
    if (preamble->m_dtype != NULL) {
        dtype dt = preamble->is_builtin_dtype() ? dtype(preamble->get_type_id())
                        : dtype(preamble->m_dtype, true);
        o << indent << " dtype: " << dt << "\n";
    } else {
        o << indent << " uninitialized ndobject\n";
    }
}
