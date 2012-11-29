//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/nodes/immutable_scalar_node.hpp>
#include <dynd/nodes/strided_ndarray_node.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/ndarray.hpp>

using namespace std;
using namespace dynd;

ndarray_node_ptr dynd::immutable_scalar_node::as_dtype(const dtype& dt,
                    dynd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else if(m_dtype.element_size() <= 32) {
        // For small amounts of data, make a copy
        return detail::unchecked_make_immutable_scalar_node(
                        make_convert_dtype(dt, m_dtype, errmode),
                        m_originptr);
    } else {
        // For larger amounts of data, make a strided node
        // TODO: Make a scalar_node which points at a separate memory block
        return make_strided_ndarray_node(make_convert_dtype(dt, m_dtype, errmode),
                        0, NULL, NULL, m_originptr, read_access_flag | immutable_access_flag, as_ndarray_node_ptr());
    }
}

ndarray_node_ptr dynd::immutable_scalar_node::apply_linear_index(
                int DYND_UNUSED(ndim), const bool *DYND_UNUSED(remove_axis),
                const intptr_t *DYND_UNUSED(start_index), const intptr_t *DYND_UNUSED(index_strides),
                const intptr_t *DYND_UNUSED(shape),
                bool DYND_UNUSED(allow_in_place))
{
    return as_ndarray_node_ptr();
}

void dynd::immutable_scalar_node::debug_print_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " data: ";
    hexadecimal_print(o, m_originptr, m_dtype.element_size());
    o << "\n";
    o << indent << " value: ";
    try {
        if (m_dtype.get_kind() != expression_kind) {
            m_dtype.print_element(o, m_originptr, NULL); // TODO: ndobject metadata
        } else {
            ndarray a = ndarray(const_cast<immutable_scalar_node *>(this)->as_ndarray_node_ptr()).vals();
            a.get_dtype().print_element(o, a.get_readonly_originptr(), NULL); // TODO: ndobject metadata
        }
        o << "\n";
    } catch(std::exception& e) {
        o << "EXCEPTION: " << e.what() << "\n";
    }
}

ndarray_node_ptr dynd::detail::unchecked_make_immutable_scalar_node(const dtype& dt, const char* data)
{
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
    return ndarray_node_ptr(new (result) memory_block_data(1, deprecated_ndarray_node_memory_block_type), false);
}

ndarray_node_ptr dynd::detail::unchecked_make_immutable_scalar_node(const dtype& dt)
{
    // Calculate the aligned starting point for the data
    intptr_t start = (intptr_t)(((uintptr_t)sizeof(memory_block_data) +
                                        sizeof(immutable_scalar_node) + (uintptr_t)(dt.alignment() - 1))
                        & ~((uintptr_t)(dt.alignment() - 1)));
    char *result = reinterpret_cast<char *>(malloc(start + dt.element_size()));
    if (result == NULL) {
        throw bad_alloc();
    }
    // Placement new
    new (result + sizeof(memory_block_data))
            immutable_scalar_node(dt, result + start);
    return ndarray_node_ptr(new (result) memory_block_data(1, deprecated_ndarray_node_memory_block_type), false);
}
