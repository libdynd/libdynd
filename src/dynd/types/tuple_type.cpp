//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

dynd::tuple_type::tuple_type(const std::vector<ndt::type>& field_types)
    : base_type(tuple_type_id, struct_kind, 0, 1, type_flag_none, 0, 0),
            m_fields(field_types), m_offsets(field_types.size()), m_metadata_offsets(field_types.size())
{
    // TODO: tuple_type should probably not have kind struct_kind?
    // Calculate the offsets and element size
    size_t metadata_offset = 0;
    size_t offset = 0;
    m_members.data_alignment = 1;
    for (size_t i = 0, i_end = field_types.size(); i != i_end; ++i) {
        size_t field_alignment = field_types[i].get_data_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_members.data_alignment) {
            m_members.data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        m_members.flags |= (field_types[i].get_flags()&dtype_flags_operand_inherited);
        // Add padding bytes as necessary
        offset = inc_to_alignment(offset, field_alignment);
        // Save the offset
        m_offsets[i] = offset;
        offset += field_types[i].get_data_size();
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_fields[i].is_builtin() ? 0 : m_fields[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    // Pad to get the final element size
    m_members.data_size = inc_to_alignment(offset, m_members.data_alignment);
    // This is the standard layout
    m_is_standard_layout = true;
}

tuple_type::~tuple_type()
{
}

dynd::tuple_type::tuple_type(const std::vector<ndt::type>& field_types, const std::vector<size_t> offsets,
                    size_t data_size, size_t alignment)
    : base_type(tuple_type_id, struct_kind, data_size, alignment, type_flag_none, 0, 0),
            m_fields(field_types), m_offsets(offsets), m_metadata_offsets(field_types.size())
{
    if (!offset_is_aligned(data_size, alignment)) {
        stringstream ss;
        ss << "tuple type cannot be created with size " << data_size;
        ss << " and alignment " << alignment << ", the alignment must divide into the element size";
        throw runtime_error(ss.str());
    }

    size_t metadata_offset = 0;
    for (size_t i = 0, i_end = field_types.size(); i != i_end; ++i) {
        // Check that the field is within bounds
        if (offsets[i] + field_types[i].get_data_size() > data_size) {
            stringstream ss;
            ss << "tuple type cannot be created with field " << i << " of type " << field_types[i];
            ss << " at offset " << offsets[i] << ", not fitting within the total element size of " << data_size;
            throw runtime_error(ss.str());
        }
        // Check that the field has proper alignment
        if (((m_members.data_alignment | offsets[i]) & (field_types[i].get_data_alignment() - 1)) != 0) {
            stringstream ss;
            ss << "tuple type cannot be created with field " << i << " of type " << field_types[i];
            ss << " at offset " << offsets[i] << " and tuple alignment " << m_members.data_alignment;
            ss << " because the field is not properly aligned";
            throw runtime_error(ss.str());
        }
        // Inherit any operand flags from the fields
        m_members.flags |= (field_types[i].get_flags()&dtype_flags_operand_inherited);
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_fields[i].is_builtin() ? 0 : m_fields[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    // Check whether the layout we were given is standard
    m_is_standard_layout = compute_is_standard_layout();
}

bool dynd::tuple_type::compute_is_standard_layout() const
{
    size_t standard_offset = 0, standard_alignment = 1;
    for (size_t i = 0, i_end = m_fields.size(); i != i_end; ++i) {
        size_t field_alignment = m_fields[i].get_data_alignment();
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

    return get_data_size() == standard_element_size && get_data_alignment() == standard_alignment;
}

void dynd::tuple_type::print_data(std::ostream& o, const char *metadata, const char *data) const
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

void dynd::tuple_type::print_type(std::ostream& o) const
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
        o << ", alignment=" << get_data_alignment();
        o << ">";
    }
}

bool dynd::tuple_type::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const
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

bool dynd::tuple_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != tuple_type_id) {
        return false;
    } else {
        const tuple_type *dt = static_cast<const tuple_type*>(&rhs);
        return get_data_size() == dt->get_data_size() &&
                get_data_alignment() == dt->get_data_alignment() &&
                m_fields == dt->m_fields &&
                m_offsets == dt->m_offsets;
    }
}
