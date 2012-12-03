//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

fixedstruct_dtype::fixedstruct_dtype(const std::vector<dtype>& field_types, const std::vector<std::string>& field_names)
    : m_field_types(field_types), m_field_names(field_names),
            m_data_offsets(field_types.size()), m_metadata_offsets(field_types.size())
{
    if (field_types.size() != field_names.size()) {
        throw runtime_error("The field names for a struct dtypes must match the size of the field dtypes");
    }

    // Calculate all the resulting struct data
    size_t metadata_offset = 0, data_offset = 0;
    m_alignment = 1;
    m_memory_management = pod_memory_management;
    for (size_t i = 0, i_end = field_types.size(); i != i_end; ++i) {
        size_t field_alignment = field_types[i].get_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_alignment) {
            m_alignment = field_alignment;
        }
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        if (field_types[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
        // Calculate the data offsets
        data_offset = inc_to_alignment(data_offset, field_types[i].get_alignment());
        m_data_offsets[i] = data_offset;
        size_t field_element_size = field_types[i].get_element_size();
        if (field_element_size == 0) {
            stringstream ss;
            ss << "Cannot create fixedstruct dtype with type " << field_types[i];
            ss << " for field '" << field_names[i] << "', as it does not have a fixed size";
            throw runtime_error(ss.str());
        }
        data_offset += field_element_size;
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].extended() ? m_field_types[i].extended()->get_metadata_size() : 0;
    }
    m_metadata_size = metadata_offset;
}

void fixedstruct_dtype::print_element(std::ostream& o, const char *metadata, const char *data) const
{
    o << "[";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        m_field_types[i].print_element(o, metadata + m_metadata_offsets[i], data + m_data_offsets[i]);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void fixedstruct_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedstruct<";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        o << m_field_types[i] << " " << m_field_names[i];
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << ">";
}

bool fixedstruct_dtype::is_scalar() const
{
    return false;
}

bool fixedstruct_dtype::is_expression() const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_field_types[i].is_expression()) {
            return true;
        }
    }
    return false;
}

dtype fixedstruct_dtype::with_transformed_scalar_types(dtype_transform_fn_t transform_fn, const void *extra) const
{
    std::vector<dtype> field_types(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        field_types[i] = m_field_types[i].with_transformed_scalar_types(transform_fn, extra);
    }

    return dtype(new fixedstruct_dtype(field_types, m_field_names));
}

dtype fixedstruct_dtype::get_canonical_dtype() const
{
    std::vector<dtype> field_types(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        field_types[i] = m_field_types[i].get_canonical_dtype();
    }

    return dtype(new fixedstruct_dtype(field_types, m_field_names));
}

dtype fixedstruct_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt, remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            return m_field_types[start_index].apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_dt);
        } else if (nindices == 1 && start_index == 0 && index_stride == 1 && dimension_size == m_field_types.size()) {
            // This is a do-nothing index, keep the same dtype
            return dtype(this, true);
        } else {
            // Take the subset of the fixed fields in-place
            std::vector<dtype> field_types(dimension_size);
            std::vector<std::string> field_names(dimension_size);

            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                field_types[i] = m_field_types[idx].apply_linear_index(nindices-1, indices+1, current_i+1, root_dt);
                field_names[i] = m_field_names[idx];
            }
            // Return a struct dtype, because the offsets are now not in standard form anymore
            return dtype(new struct_dtype(field_types, field_names));
        }
    }
}

intptr_t fixedstruct_dtype::apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                const dtype& result_dtype, char *out_metadata, int current_i, const dtype& root_dt) const
{
    // A fixedstruct dtype is retained only when the resulting dtype is exactly the same
    if (result_dtype.get_type_id() == fixedstruct_type_id) {
        // Process each element verbatim
        for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
            if (m_field_types[i].extended()) {
                if (m_field_types[i].extended()->apply_linear_index(0, NULL, data + m_data_offsets[i],
                                metadata + m_metadata_offsets[i], m_field_types[i], out_metadata + m_metadata_offsets[i],
                                current_i + 1, root_dt) != 0) {
                    stringstream ss;
                    ss << "Unexpected non-zero offset when applying a NULL index to dtype " << m_field_types[i];
                    throw runtime_error(ss.str());
                }
            }
        }
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt, remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const dtype& dt = m_field_types[start_index];
            if (dt.extended()) {
                intptr_t offset = m_data_offsets[start_index];
                offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1, data + offset,
                                metadata + m_metadata_offsets[start_index], result_dtype,
                                out_metadata, current_i + 1, root_dt);
                return offset;
            }
            return 0;
        } else {
            intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_metadata);
            const struct_dtype *result_e_dt = static_cast<const struct_dtype *>(result_dtype.extended());
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = m_data_offsets[idx];
                const dtype& dt = result_e_dt->get_fields()[i];
                if (dt.extended()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    data + out_offsets[i], metadata + m_metadata_offsets[idx],
                                    dt,out_metadata + result_e_dt->get_metadata_offsets()[i],
                                    current_i + 1, root_dt);
                }
            }
            return 0;
        }
    }
}

intptr_t fixedstruct_dtype::get_dim_size(const char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata)) const
{
    return m_field_types.size();
}

void fixedstruct_dtype::get_shape(int i, intptr_t *out_shape) const
{
    // Adjust the current shape if necessary
    switch (out_shape[i]) {
        case shape_signal_uninitialized:
            out_shape[i] = m_field_types.size();
            break;
        case shape_signal_varying:
            break;
        default:
            if ((size_t)out_shape[i] != m_field_types.size()) {
                out_shape[i] = shape_signal_varying;
            }
            break;
    }

    // Process the later shape values
    for (size_t j = 0; j < m_field_types.size(); ++j) {
        if (m_field_types[j].extended()) {
            m_field_types[j].extended()->get_shape(i+1, out_shape);
        }
    }
}

intptr_t fixedstruct_dtype::get_representative_stride(const char *metadata) const
{
    // Return the first non-zero offset as the representative stride
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_data_offsets[i] != 0) {
            return m_data_offsets[i];
        }
    }
    // Return 0 as the fallback
    return 0;
}

bool fixedstruct_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == fixedstruct_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void fixedstruct_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("fixedstruct_dtype::get_single_compare_kernel is unimplemented");
}

void fixedstruct_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode DYND_UNUSED(errmode),
                unary_specialization_kernel_instance& DYND_UNUSED(out_kernel)) const
{
    stringstream ss;
    ss << "fixedstruct_dtype::get_dtype_assignment_kernel from " << src_dt << " to " << dst_dt << " is unimplemented";
    throw runtime_error(ss.str());
}

bool fixedstruct_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixedstruct_type_id) {
        return false;
    } else {
        const fixedstruct_dtype *dt = static_cast<const fixedstruct_dtype*>(&rhs);
        return m_alignment == dt->m_alignment &&
                m_memory_management == dt->m_memory_management &&
                m_field_types == dt->m_field_types;
    }
}

size_t fixedstruct_dtype::get_metadata_size() const
{
    return m_metadata_size;
}

void fixedstruct_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim > 0) {
        if (shape[0] >= 0 && shape[0] != (intptr_t)m_field_types.size()) {
            stringstream ss;
            ss << "Cannot construct dynd object of dtype " << dtype(this, true);
            ss << " with dimension size " << shape[0] << ", the size must be " << m_field_types.size();
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (field_dt.extended()) {
            try {
                field_dt.extended()->metadata_default_construct(
                            metadata + m_metadata_offsets[i], ndim, shape);
            } catch(...) {
                // Since we're explicitly controlling the memory, need to manually do the cleanup too
                for (size_t j = 0; j < i; ++j) {
                    if (m_field_types[j].extended()) {
                        m_field_types[j].extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
                    }
                }
                throw;
            }
        }
    }
}

void fixedstruct_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy construct all the field's metadata
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (field_dt.extended()) {
            field_dt.extended()->metadata_copy_construct(dst_metadata + m_metadata_offsets[i],
                            src_metadata + m_metadata_offsets[i],
                            embedded_reference);
        }
    }
}

void fixedstruct_dtype::metadata_destruct(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (field_dt.extended()) {
            field_dt.extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
        }
    }
}

void fixedstruct_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
    o << indent << "fixedstruct metadata\n";
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (field_dt.extended() && field_dt.extended()->get_metadata_size() > 0) {
            o << indent << " field " << i << " (name " << m_field_names[i] << ") metadata:\n";
            field_dt.extended()->metadata_debug_print(metadata + m_metadata_offsets[i], o, indent + "  ");
        }
    }
}

void fixedstruct_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    if (!m_field_types.empty()) {
        const dtype *field_types = &m_field_types[0];
        const size_t *metadata_offsets = &m_metadata_offsets[0];
        for (intptr_t i = 0, i_end = m_field_types.size(); i < i_end; ++i) {
            callback(field_types[i], data + m_data_offsets[i], metadata + metadata_offsets[i], callback_data);
        }
    }
}
