//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>

using namespace std;
using namespace dynd;

struct_dtype::struct_dtype(const std::vector<dtype>& fields, const std::vector<std::string>& field_names)
    : extended_dtype(struct_type_id, struct_kind, 0, 1),
            m_field_types(fields), m_field_names(field_names), m_metadata_offsets(fields.size())
{
    if (fields.size() != field_names.size()) {
        throw runtime_error("The field names for a struct dtypes must match the size of the field dtypes");
    }

    // Calculate the needed element alignment
    size_t metadata_offset = fields.size() * sizeof(size_t);
    m_alignment = 1;
    m_memory_management = pod_memory_management;
    for (size_t i = 0, i_end = fields.size(); i != i_end; ++i) {
        size_t field_alignment = fields[i].get_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_alignment) {
            m_alignment = field_alignment;
        }
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        if (fields[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].is_builtin() ? 0 : m_field_types[i].extended()->get_metadata_size();
    }
    m_metadata_size = metadata_offset;

    create_ndobject_properties();
}

struct_dtype::~struct_dtype()
{
}

size_t struct_dtype::get_default_data_size(int ndim, const intptr_t *shape) const
{
    // Default layout is to match the field order - could reorder the elements for more efficient packing
    size_t s = 0;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        s = inc_to_alignment(s, m_field_types[i].get_alignment());
        if (!m_field_types[i].is_builtin()) {
            s += m_field_types[i].extended()->get_default_data_size(ndim, shape);
        } else {
            s += m_field_types[i].get_data_size();
        }
    }
    s = inc_to_alignment(s, m_alignment);
    return s;
}


void struct_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
    o << "[";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        m_field_types[i].print_data(o, metadata + m_metadata_offsets[i], data + offsets[i]);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void struct_dtype::print_dtype(std::ostream& o) const
{
    o << "struct<";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        o << m_field_types[i] << " " << m_field_names[i];
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << ">";
}

bool struct_dtype::is_scalar() const
{
    return false;
}

bool struct_dtype::is_expression() const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_field_types[i].is_expression()) {
            return true;
        }
    }
    return false;
}

void struct_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    std::vector<dtype> tmp_field_types(m_field_types.size());

    bool was_transformed = false;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        transform_fn(m_field_types[i], extra, tmp_field_types[i], was_transformed);
    }
    if (was_transformed) {
        out_transformed_dtype = dtype(new struct_dtype(tmp_field_types, m_field_names));
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}

dtype struct_dtype::get_canonical_dtype() const
{
    std::vector<dtype> fields(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        fields[i] = m_field_types[i].get_canonical_dtype();
    }

    return dtype(new struct_dtype(fields, m_field_names));
}

dtype struct_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt, remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            return m_field_types[start_index].apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_dt);
        } else {
            // Take the subset of the fixed fields in-place
            std::vector<dtype> fields(dimension_size);
            std::vector<std::string> field_names(dimension_size);

            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                fields[i] = m_field_types[idx].apply_linear_index(nindices-1, indices+1, current_i+1, root_dt);
                field_names[i] = m_field_names[idx];
            }

            return dtype(new struct_dtype(fields, field_names));
        }
    }
}

intptr_t struct_dtype::apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                int current_i, const dtype& root_dt) const
{
    const intptr_t *offsets = reinterpret_cast<const intptr_t *>(metadata);
    intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_metadata);
    if (nindices == 0) {
        // Copy the struct offset metadata verbatim
        memcpy(out_metadata, metadata, m_field_types.size() * sizeof(size_t));
        // Then process each element verbatim as well
        for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
            if (!m_field_types[i].is_builtin()) {
                out_offsets[i] += m_field_types[i].extended()->apply_linear_index(0, NULL, data + offsets[i],
                                metadata + m_metadata_offsets[i], m_field_types[i], out_metadata + m_metadata_offsets[i],
                                embedded_reference, current_i + 1, root_dt);
            }
        }
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt, remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const dtype& dt = m_field_types[start_index];
            intptr_t offset = offsets[start_index];
            if (!dt.is_builtin()) {
                offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1, data + offset,
                                metadata + m_metadata_offsets[start_index], result_dtype,
                                out_metadata, embedded_reference, current_i + 1, root_dt);
            }
            return offset;
        } else {
            const struct_dtype *result_e_dt = static_cast<const struct_dtype *>(result_dtype.extended());
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = offsets[idx];
                const dtype& dt = result_e_dt->m_field_types[i];
                if (!dt.is_builtin()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1, data + out_offsets[i],
                                    metadata + m_metadata_offsets[idx], dt, out_metadata + result_e_dt->m_metadata_offsets[i],
                                    embedded_reference, current_i + 1, root_dt);
                }
            }
            return 0;
        }
    }
}

intptr_t struct_dtype::get_dim_size(const char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata)) const
{
    return m_field_types.size();
}

void struct_dtype::get_shape(int i, intptr_t *out_shape) const
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
        if (!m_field_types[j].is_builtin()) {
            m_field_types[j].extended()->get_shape(i+1, out_shape);
        }
    }
}

intptr_t struct_dtype::get_representative_stride(const char *metadata) const
{
    const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
    // Return the first non-zero offset as the representative stride
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (offsets[i] != 0) {
            return offsets[i];
        }
    }
    // Return 0 as the fallback
    return 0;
}

bool struct_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == struct_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void struct_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("struct_dtype::get_single_compare_kernel is unimplemented");
}

void struct_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel) const
{
    if (this == dst_dt.extended()) {
        if (this == src_dt.extended()) {
            get_struct_assignment_kernel(dst_dt, out_kernel);
        } else if (src_dt.get_type_id() == struct_type_id) {
            get_struct_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
        } else if (src_dt.get_type_id() == fixedstruct_type_id) {
            get_fixedstruct_to_struct_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
        } else if (!src_dt.is_builtin()) {
            src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
        } else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
            throw runtime_error(ss.str());
        }
    } else {
        stringstream ss;
        ss << "assignment from " << src_dt << " to " << dst_dt << " is not implemented yet";
        throw runtime_error(ss.str());
    }
}

bool struct_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != struct_type_id) {
        return false;
    } else {
        const struct_dtype *dt = static_cast<const struct_dtype*>(&rhs);
        return m_alignment == dt->m_alignment &&
                m_memory_management == dt->m_memory_management &&
                m_field_types == dt->m_field_types;
    }
}

size_t struct_dtype::get_metadata_size() const
{
    return m_metadata_size;
}

void struct_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
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

    size_t *offsets = reinterpret_cast<size_t *>(metadata);
    size_t offs = 0;
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        offs = inc_to_alignment(offs, field_dt.get_alignment());
        offsets[i] = offs;
        if (!field_dt.is_builtin()) {
            try {
                field_dt.extended()->metadata_default_construct(
                            metadata + m_metadata_offsets[i], ndim, shape);
            } catch(...) {
                // Since we're explicitly controlling the memory, need to manually do the cleanup too
                for (size_t j = 0; j < i; ++j) {
                    if (!m_field_types[j].is_builtin()) {
                        m_field_types[j].extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
                    }
                }
                throw;
            }
            offs += m_field_types[i].extended()->get_default_data_size(ndim, shape);
        } else {
            offs += m_field_types[i].get_data_size();
        }
    }
}

void struct_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy all the field offsets
    memcpy(dst_metadata, src_metadata, m_field_types.size() * sizeof(intptr_t));
    // Copy construct all the field's metadata
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_copy_construct(dst_metadata + m_metadata_offsets[i],
                            src_metadata + m_metadata_offsets[i],
                            embedded_reference);
        }
    }
}

void struct_dtype::metadata_reset_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_reset_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_dtype::metadata_finalize_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_finalize_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_dtype::metadata_destruct(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
    o << indent << "struct metadata\n";
    o << indent << " field offsets: ";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        o << offsets[i];
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "\n";
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin() && field_dt.extended()->get_metadata_size() > 0) {
            o << indent << " field " << i << " (name " << m_field_names[i] << ") metadata:\n";
            field_dt.extended()->metadata_debug_print(metadata + m_metadata_offsets[i], o, indent + "  ");
        }
    }
}

void struct_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    if (!m_field_types.empty()) {
        const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
        const dtype *fields = &m_field_types[0];
        const size_t *metadata_offsets = &m_metadata_offsets[0];
        for (intptr_t i = 0, i_end = m_field_types.size(); i < i_end; ++i) {
            callback(fields[i], data + offsets[i], metadata + metadata_offsets[i], callback_data);
        }
    }
}

static ndobject property_get_field_names(const dtype& dt) {
    const struct_dtype *d = static_cast<const struct_dtype *>(dt.extended());
    // TODO: This property could be an immutable ndobject, which we would just return.
    return ndobject(d->get_field_names());
}

static ndobject property_get_metadata_offsets(const dtype& dt) {
    const struct_dtype *d = static_cast<const struct_dtype *>(dt.extended());
    // TODO: This property could be an immutable ndobject, which we would just return.
    return ndobject(d->get_metadata_offsets());
}

static pair<string, gfunc::callable> dtype_properties[] = {
    pair<string, gfunc::callable>("field_names", gfunc::make_callable(&property_get_field_names, "self")),
    // TODO field_types
    pair<string, gfunc::callable>("metadata_offsets", gfunc::make_callable(&property_get_metadata_offsets, "self"))
};

void struct_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const
{
    *out_properties = dtype_properties;
    *out_count = sizeof(dtype_properties) / sizeof(dtype_properties[0]);
}

dtype struct_dtype::ndobject_parameters_dtype = make_fixedstruct_dtype(dtype(new void_pointer_dtype), "self");

static ndobject_preamble *property_get_ndobject_field(const ndobject_preamble *params, void *extra)
{
    // Get the ndobject 'self' parameter
    ndobject n = ndobject(*(ndobject_preamble **)params->m_data_pointer, true);
    intptr_t i = reinterpret_cast<intptr_t>(extra);
    int ndim = n.get_dtype().get_undim();
    if (ndim == 0) {
        return n.at(i).release();
    } else {
        shortvector<irange> idx(ndim + 1);
        idx[ndim] = irange(i);
        return n.at_array(ndim + 1, idx.get()).release();
    }
}

void struct_dtype::create_ndobject_properties()
{
    m_ndobject_properties.resize(m_field_types.size());
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        // TODO: Transform the name into a valid Python symbol?
        m_ndobject_properties[i].first = m_field_names[i];
        m_ndobject_properties[i].second.set(ndobject_parameters_dtype, &property_get_ndobject_field, (void *)i);
    }
}

void struct_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, int *out_count) const
{
    *out_properties = m_ndobject_properties.empty() ? NULL : &m_ndobject_properties[0];
    *out_count = (int)m_ndobject_properties.size();
}
