//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/tuple_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::tuple_dtype::tuple_dtype(const std::vector<dtype>& fields)
    : m_fields(fields), m_offsets(fields.size())
{
    // Calculate the offsets and element size
    size_t offset = 0;
    m_alignment = 1;
    m_memory_management = pod_memory_management;
    for (size_t i = 0, i_end = fields.size(); i != i_end; ++i) {
        size_t field_alignment = fields[i].alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_alignment) {
            m_alignment = field_alignment;
        }
        // Add padding bytes as necessary
        offset = (offset + field_alignment - 1) & (-field_alignment);
        // Save the offset
        m_offsets[i] = offset;
        offset += fields[i].element_size();
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        if (fields[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
    }
    // Pad to get the final element size
    m_element_size = (offset + m_alignment - 1) & (-m_alignment);
}

void dnd::tuple_dtype::print_element(std::ostream& o, const char *data) const
{
    o << "[";
    for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
        m_fields[i].print_element(o, data + m_offsets[i]);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void dnd::tuple_dtype::print_dtype(std::ostream& o) const
{
    o << "tuple<fields=(";
    for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
        o << m_fields[i];
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << ")>";
}

bool dnd::tuple_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.type_id() == tuple_type_id) {
            return src_dt == dst_dt;
        }
    }

    return false;
}

void dnd::tuple_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const
{
    throw runtime_error("tuple_dtype::get_single_compare_kernel is unimplemented"); 
}

void dnd::tuple_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    throw runtime_error("tuple_dtype::get_dtype_assignment_kernel is unimplemented"); 
}

bool dnd::tuple_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != tuple_type_id) {
        return false;
    } else {
        const tuple_dtype *dt = static_cast<const tuple_dtype*>(&rhs);
        return m_element_size == dt->m_element_size &&
                m_alignment == dt->m_alignment &&
                m_memory_management == dt->m_memory_management &&
                m_fields == dt->m_fields &&
                m_offsets == dt->m_offsets;
    }
}
