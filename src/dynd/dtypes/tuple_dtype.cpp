//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/tuple_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

dynd::tuple_dtype::tuple_dtype(const std::vector<dtype>& fields)
    : base_dtype(tuple_type_id, struct_kind, 0, 1, dtype_flag_none, 0, 0),
            m_fields(fields), m_offsets(fields.size()), m_metadata_offsets(fields.size())
{
    // TODO: tuple_dtype should probably not have kind struct_kind?
    // Calculate the offsets and element size
    size_t metadata_offset = 0;
    size_t offset = 0;
    m_members.alignment = 1;
    m_memory_management = pod_memory_management;
    for (size_t i = 0, i_end = fields.size(); i != i_end; ++i) {
        size_t field_alignment = fields[i].get_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_members.alignment) {
            m_members.alignment = (uint8_t)field_alignment;
        }
        // Add padding bytes as necessary
        offset = inc_to_alignment(offset, field_alignment);
        // Save the offset
        m_offsets[i] = offset;
        offset += fields[i].get_data_size();
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        if (fields[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_fields[i].is_builtin() ? 0 : m_fields[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    // Pad to get the final element size
    m_members.data_size = inc_to_alignment(offset, m_members.alignment);
    // This is the standard layout
    m_is_standard_layout = true;
}

tuple_dtype::~tuple_dtype()
{
}

dynd::tuple_dtype::tuple_dtype(const std::vector<dtype>& fields, const std::vector<size_t> offsets,
                    size_t data_size, size_t alignment)
    : base_dtype(tuple_type_id, struct_kind, data_size, alignment, dtype_flag_none, 0, 0),
            m_fields(fields), m_offsets(offsets), m_metadata_offsets(fields.size())
{
    if (!offset_is_aligned(data_size, alignment)) {
        stringstream ss;
        ss << "tuple type cannot be created with size " << data_size;
        ss << " and alignment " << alignment << ", the alignment must divide into the element size";
        throw runtime_error(ss.str());
    }

    size_t metadata_offset = 0;
    m_memory_management = pod_memory_management;
    for (size_t i = 0, i_end = fields.size(); i != i_end; ++i) {
        // Check that the field is within bounds
        if (offsets[i] + fields[i].get_data_size() > data_size) {
            stringstream ss;
            ss << "tuple type cannot be created with field " << i << " of type " << fields[i];
            ss << " at offset " << offsets[i] << ", not fitting within the total element size of " << data_size;
            throw runtime_error(ss.str());
        }
        // Check that the field has proper alignment
        if (((m_members.alignment | offsets[i]) & (fields[i].get_alignment() - 1)) != 0) {
            stringstream ss;
            ss << "tuple type cannot be created with field " << i << " of type " << fields[i];
            ss << " at offset " << offsets[i] << " and tuple alignment " << m_members.alignment;
            ss << " because the field is not properly aligned";
            throw runtime_error(ss.str());
        }
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        //       In particular, object/blockref dtypes should not overlap with each other so
        //       need more code to test for that.
        if (fields[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_fields[i].is_builtin() ? 0 : m_fields[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    // Check whether the layout we were given is standard
    m_is_standard_layout = compute_is_standard_layout();
}

bool dynd::tuple_dtype::compute_is_standard_layout() const
{
    size_t standard_offset = 0, standard_alignment = 1;
    for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
        size_t field_alignment = m_fields[i].get_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > standard_alignment) {
            standard_alignment = field_alignment;
        }
        // Add padding bytes as necessary
        standard_offset = inc_to_alignment(standard_offset, field_alignment);
        if (m_offsets[i] != standard_offset) {
            return false;
        }
        standard_offset += m_fields[i].get_data_size();
    }
    // Pad to get the standard element size
    size_t standard_element_size = inc_to_alignment(standard_offset, standard_alignment);

    return get_data_size() == standard_element_size && get_alignment() == standard_alignment;
}

void dynd::tuple_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    o << "[";
    for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
        m_fields[i].print_data(o, metadata + m_metadata_offsets[i], data + m_offsets[i]);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void dynd::tuple_dtype::print_dtype(std::ostream& o) const
{
    if (is_standard_layout()) {
        o << "tuple<";
        for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
            o << m_fields[i];
            if (i != i_end - 1) {
                o << ", ";
            }
        }
        o << ">";
    } else {
        o << "tuple<fields=(";
        for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
            o << m_fields[i];
            if (i != i_end - 1) {
                o << ", ";
            }
        }
        o << ")";
        o << ", offsets=(";
        for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
            o << m_offsets[i];
            if (i != i_end - 1) {
                o << ", ";
            }
        }
        o << ")";
        o << ", size=" << get_data_size();
        o << ", alignment=" << get_alignment();
        o << ">";
    }
}

void dynd::tuple_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    // Adjust the current shape if necessary
    switch (out_shape[i]) {
        case shape_signal_uninitialized:
            out_shape[i] = m_fields.size();
            break;
        case shape_signal_varying:
            break;
        default:
            if ((size_t)out_shape[i] != m_fields.size()) {
                out_shape[i] = shape_signal_varying;
            }
            break;
    }

    // Process the later shape values
    for (size_t j = 0; j < m_fields.size(); ++j) {
        if (!m_fields[j].is_builtin()) {
            m_fields[j].extended()->get_shape(i+1, out_shape);
        }
    }
}

bool dynd::tuple_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == tuple_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void dynd::tuple_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("tuple_dtype::get_single_compare_kernel is unimplemented"); 
}

bool dynd::tuple_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != tuple_type_id) {
        return false;
    } else {
        const tuple_dtype *dt = static_cast<const tuple_dtype*>(&rhs);
        return get_data_size() == dt->get_data_size() &&
                get_alignment() == dt->get_alignment() &&
                get_memory_management() == dt->get_memory_management() &&
                m_fields == dt->m_fields &&
                m_offsets == dt->m_offsets;
    }
}
