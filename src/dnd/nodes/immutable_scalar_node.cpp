//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/immutable_scalar_node.hpp>
#include <dnd/nodes/strided_ndarray_node.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

ndarray_node_ptr dnd::immutable_scalar_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else if(m_dtype.element_size() <= 16) {
        // For small amounts of data, make a copy
        return make_immutable_scalar_node(
                        make_conversion_dtype(dt, m_dtype, errmode),
                        m_originptr);
    } else {
        // For larger amounts of data, make a strided node
        // TODO: Make a scalar_node which isn't necessarily immutable
        return make_strided_ndarray_node(make_conversion_dtype(dt, m_dtype, errmode),
                        0, NULL, NULL, m_originptr, read_access_flag | immutable_access_flag, as_ndarray_node_ptr());
    }
}

ndarray_node_ptr dnd::immutable_scalar_node::apply_linear_index(
                int DND_UNUSED(ndim), const bool *DND_UNUSED(remove_axis),
                const intptr_t *DND_UNUSED(start_index), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(shape),
                bool DND_UNUSED(allow_in_place))
{
    return as_ndarray_node_ptr();
}

void dnd::immutable_scalar_node::debug_dump_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " data: ";
    hexadecimal_print(o, m_originptr, m_dtype.element_size());
    o << "\n";
}

ndarray_node_ptr dnd::make_immutable_scalar_node(const dtype& dt, const char* data)
{
    if (dt.get_memory_management() != pod_memory_management) {
        throw runtime_error("immutable_scalar_node doesn't support object dtypes yet");
    }

    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) +
                                        sizeof(immutable_scalar_node) + (uintptr_t)(dt.alignment() - 1))
                        & ~((uintptr_t)(dt.alignment() - 1)));
    char *result = reinterpret_cast<char *>(malloc(start + dt.element_size()));
    if (result == NULL) {
        throw bad_alloc();
    }
    memcpy(result + start, data, dt.element_size());
    // Placement new
    new (result + sizeof(memory_block_data))
            immutable_scalar_node(dt, result + start);
    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}