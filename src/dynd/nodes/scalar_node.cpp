//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/nodes/scalar_node.hpp>
#include <dynd/nodes/strided_ndarray_node.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

ndarray_node_ptr dynd::scalar_node::as_dtype(const dtype& dt,
                    dynd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else if(m_dtype.element_size() <= 32) {
        // For small amounts of data, make a copy
        return make_scalar_node(
                        make_convert_dtype(dt, m_dtype, errmode),
                        m_originptr, m_access_flags, m_blockref_memblock);
    } else {
        // For larger amounts of data, make a strided node
        // TODO: Make a scalar_node which points at a separate memory block
        return make_strided_ndarray_node(make_convert_dtype(dt, m_dtype, errmode),
                        0, NULL, NULL, m_originptr, read_access_flag | immutable_access_flag, as_ndarray_node_ptr());
    }
}

ndarray_node_ptr dynd::scalar_node::apply_linear_index(
                int DND_UNUSED(ndim), const bool *DND_UNUSED(remove_axis),
                const intptr_t *DND_UNUSED(start_index), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(shape),
                bool DND_UNUSED(allow_in_place))
{
    return as_ndarray_node_ptr();
}

void dynd::scalar_node::debug_dump_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " data: ";
    hexadecimal_print(o, m_originptr, m_dtype.element_size());
    o << "\n";
    if (m_blockref_memblock.get() != NULL) {
        o << indent << " blockref memory block\n";
        memory_block_debug_dump(m_blockref_memblock.get(), o, indent + " ");
    }
}

ndarray_node_ptr dynd::make_scalar_node(const dtype& dt, const char* data, int access_flags)
{
    if (dt.get_memory_management() != pod_memory_management) {
        throw runtime_error("scalar_node doesn't support object dtypes yet");
    }

    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) +
                                        sizeof(scalar_node) + (uintptr_t)(dt.alignment() - 1))
                        & ~((uintptr_t)(dt.alignment() - 1)));
    char *result = reinterpret_cast<char *>(malloc(start + dt.element_size()));
    if (result == NULL) {
        throw bad_alloc();
    }
    memcpy(result + start, data, dt.element_size());
    // Placement new
    new (result + sizeof(memory_block_data))
            scalar_node(dt, result + start, access_flags);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

ndarray_node_ptr dynd::make_scalar_node(const dtype& dt, const char* data, int access_flags,
                const memory_block_ptr& blockref_memblock)
{
    if (blockref_memblock.get() != NULL) {
        if (dt.get_memory_management() != blockref_memory_management) {
            throw runtime_error("scalar node with a blockref memblock needs a blockref dtype");
        }
    } else if (dt.get_memory_management() != pod_memory_management) {
        throw runtime_error("scalar_node doesn't support object dtypes yet");
    }

    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) +
                                        sizeof(scalar_node) + (uintptr_t)(dt.alignment() - 1))
                        & ~((uintptr_t)(dt.alignment() - 1)));
    char *result = reinterpret_cast<char *>(malloc(start + dt.element_size()));
    if (result == NULL) {
        throw bad_alloc();
    }
    memcpy(result + start, data, dt.element_size());
    // Placement new
    new (result + sizeof(memory_block_data))
            scalar_node(dt, result + start, access_flags, blockref_memblock);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}
