//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

fixedstruct_dtype::fixedstruct_dtype(size_t field_count, const dtype *field_types,
                const std::string *field_names)
    : base_struct_dtype(fixedstruct_type_id, 0, 1, field_count, dtype_flag_none, 0),
            m_field_types(field_types, field_types + field_count),
            m_field_names(field_names, field_names + field_count),
           m_data_offsets(field_count), m_metadata_offsets(field_count)
{
    // Calculate all the resulting struct data
    size_t metadata_offset = 0, data_offset = 0;
    m_members.alignment = 1;
    m_memory_management = pod_memory_management;
    for (size_t i = 0; i != field_count; ++i) {
        size_t field_alignment = field_types[i].get_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_members.alignment) {
            m_members.alignment = (uint8_t)field_alignment;
        }
        // Accumulate the correct memory management
        // TODO: Handle object, and object+blockref memory management types as well
        if (field_types[i].get_memory_management() == blockref_memory_management) {
            m_memory_management = blockref_memory_management;
        }
        // If any fields require zero-initialization, flag the struct as requiring it
        m_members.flags |= (field_types[i].get_flags()&dtype_flag_zeroinit);
        // Calculate the data offsets
        data_offset = inc_to_alignment(data_offset, field_types[i].get_alignment());
        m_data_offsets[i] = data_offset;
        size_t field_element_size = field_types[i].get_data_size();
        if (field_element_size == 0) {
            stringstream ss;
            ss << "Cannot create fixedstruct dtype with type " << field_types[i];
            ss << " for field '" << field_names[i] << "', as it does not have a fixed size";
            throw runtime_error(ss.str());
        }
        data_offset += field_element_size;
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].is_builtin() ? 0 : m_field_types[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    m_members.data_size = inc_to_alignment(data_offset, m_members.alignment);

    create_ndobject_properties();
}

fixedstruct_dtype::~fixedstruct_dtype()
{
}

intptr_t fixedstruct_dtype::get_field_index(const std::string& field_name) const
{
    // TODO: Put a map<> or unordered_map<> in the dtype to accelerate this lookup
    vector<string>::const_iterator i = find(m_field_names.begin(), m_field_names.end(), field_name);
    if (i != m_field_names.end()) {
        return i - m_field_names.begin();
    } else {
        return -1;
    }
}

void fixedstruct_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    o << "[";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        m_field_types[i].print_data(o, metadata + m_metadata_offsets[i], data + m_data_offsets[i]);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

static bool is_simple_identifier_name(const string& s)
{
    if (s.empty()) {
        return false;
    } else {
        char c = s[0];
        if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
            return false;
        }
        for (size_t i = 1, i_end = s.size(); i < i_end; ++i) {
            c = s[i];
            if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'z')
                            || ('A' <= c && c <= 'Z') || c == '_')) {
                return false;
            }
        }
        return true;
    }
}

void fixedstruct_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedstruct<";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        o << m_field_types[i] << " ";
        if (is_simple_identifier_name(m_field_names[i])) {
            o << m_field_names[i];
        } else {
            print_escaped_utf8_string(o, m_field_names[i]);
        }
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << ">";
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

bool fixedstruct_dtype::is_unique_data_owner(const char *metadata) const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (!m_field_types[i].is_builtin() &&
                !m_field_types[i].extended()->is_unique_data_owner(metadata + m_metadata_offsets[i])) {
            return false;
        }
    }
    return true;
}

void fixedstruct_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    std::vector<dtype> tmp_field_types(m_field_types.size());

    bool switch_to_struct = false;
    bool was_any_transformed = false;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        bool was_transformed = false;
        transform_fn(m_field_types[i], extra, tmp_field_types[i], was_transformed);
        if (was_transformed) {
            // If the dtype turned into one without fixed size, have to use struct instead of fixedstruct
            if (tmp_field_types[i].get_data_size() == 0) {
                switch_to_struct = true;
            }
            was_any_transformed = true;
        }
    }
    if (was_any_transformed) {
        if (!switch_to_struct) {
            out_transformed_dtype = dtype(new fixedstruct_dtype(
                            tmp_field_types.size(), &tmp_field_types[0], &m_field_names[0]), false);
        } else {
            out_transformed_dtype = dtype(new struct_dtype(tmp_field_types, m_field_names), false);
        }
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}

dtype fixedstruct_dtype::get_canonical_dtype() const
{
    std::vector<dtype> field_types(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        field_types[i] = m_field_types[i].get_canonical_dtype();
    }

    return dtype(new fixedstruct_dtype(m_field_types.size(), &field_types[0], &m_field_names[0]), false);
}

dtype fixedstruct_dtype::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            return m_field_types[start_index].apply_linear_index(nindices - 1, indices + 1,
                            current_i + 1, root_dt, leading_dimension);
        } else if (nindices == 1 && start_index == 0 && index_stride == 1 &&
                        (size_t)dimension_size == m_field_types.size()) {
            // This is a do-nothing index, keep the same dtype
            return dtype(this, true);
        } else {
            // Take the subset of the fixed fields in-place
            std::vector<dtype> field_types(dimension_size);
            std::vector<std::string> field_names(dimension_size);

            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                field_types[i] = m_field_types[idx].apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_dt, false);
                field_names[i] = m_field_names[idx];
            }
            // Return a struct dtype, because the offsets are now not in standard form anymore
            return dtype(new struct_dtype(field_types, field_names), false);
        }
    }
}

intptr_t fixedstruct_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (nindices == 0) {
        // If there are no more indices, copy the metadata verbatim
        metadata_copy_construct(out_metadata, metadata, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const dtype& dt = m_field_types[start_index];
            intptr_t offset = m_data_offsets[start_index];
            if (!dt.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[start_index], result_dtype,
                                    out_metadata, embedded_reference, current_i + 1, root_dt,
                                    true, inout_data, inout_dataref);
                } else {
                    offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[start_index], result_dtype,
                                    out_metadata, embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
            }
            return offset;
        } else if (result_dtype.get_type_id() == fixedstruct_type_id) {
            // This was a no-op, so copy everything verbatim
            for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
                if (!m_field_types[i].is_builtin()) {
                    if (m_field_types[i].extended()->apply_linear_index(0, NULL,
                                    metadata + m_metadata_offsets[i], m_field_types[i], out_metadata + m_metadata_offsets[i],
                                    embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL) != 0) {
                        stringstream ss;
                        ss << "Unexpected non-zero offset when applying a NULL index to dtype " << m_field_types[i];
                        throw runtime_error(ss.str());
                    }
                }
            }
            return 0;
        } else {
            intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_metadata);
            const struct_dtype *result_e_dt = static_cast<const struct_dtype *>(result_dtype.extended());
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = m_data_offsets[idx];
                const dtype& dt = result_e_dt->get_field_types()[i];
                if (!dt.is_builtin()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[idx],
                                    dt, out_metadata + result_e_dt->get_metadata_offsets()[i],
                                    embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
            }
            return 0;
        }
    }
}

dtype fixedstruct_dtype::at_single(intptr_t i0,
                const char **inout_metadata, const char **inout_data) const
{
    // Bounds-checking of the index
    i0 = apply_single_index(i0, m_field_types.size(), NULL);
    if (inout_metadata) {
        // Modify the metadata
        *inout_metadata += m_metadata_offsets[i0];
        // If requested, modify the data
        if (inout_data) {
            *inout_data += m_data_offsets[i0];
        }
    }
    return m_field_types[i0];
}

intptr_t fixedstruct_dtype::get_dim_size(const char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata)) const
{
    return m_field_types.size();
}

void fixedstruct_dtype::get_shape(size_t i, intptr_t *out_shape) const
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

intptr_t fixedstruct_dtype::get_representative_stride(const char *DYND_UNUSED(metadata)) const
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

size_t fixedstruct_dtype::make_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        if (this == src_dt.extended()) {
            return make_struct_identical_assignment_kernel(out, offset_out,
                            dst_dt,
                            dst_metadata, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_dt.get_kind() == struct_kind) {
            return make_struct_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        } else if (!src_dt.is_builtin()) {
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}

size_t fixedstruct_dtype::make_comparison_kernel(
                comparison_kernel *out, size_t offset_out,
                const dtype& src0_dt, const char *src0_metadata,
                const dtype& src1_dt, const char *src1_metadata,
                comparison_type_t comptype,
                const eval::eval_context *ectx) const
{
    if (this == src0_dt.extended()) {
        if (*this == *src1_dt.extended()) {
            return make_struct_comparison_kernel(out, offset_out,
                            src0_dt, src0_metadata, src1_metadata,
                            comptype, ectx);
        } else if (src1_dt.get_kind() == struct_kind) {
            return make_general_struct_comparison_kernel(out, offset_out,
                            src0_dt, src0_metadata,
                            src1_dt, src1_metadata,
                            comptype, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot compare values of types " << src0_dt << " and " << src1_dt;
    throw runtime_error(ss.str());
}

bool fixedstruct_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixedstruct_type_id) {
        return false;
    } else {
        const fixedstruct_dtype *dt = static_cast<const fixedstruct_dtype*>(&rhs);
        return get_alignment() == dt->get_alignment() &&
                get_memory_management() == dt->get_memory_management() &&
                m_field_types == dt->m_field_types;
    }
}

void fixedstruct_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
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
        }
    }
}

void fixedstruct_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
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

void fixedstruct_dtype::metadata_reset_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_reset_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void fixedstruct_dtype::metadata_finalize_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_finalize_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void fixedstruct_dtype::metadata_destruct(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
        }
    }
}

void fixedstruct_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    o << indent << "fixedstruct metadata\n";
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const dtype& field_dt = m_field_types[i];
        if (!field_dt.is_builtin() && field_dt.extended()->get_metadata_size() > 0) {
            o << indent << " field " << i << " (";
            print_escaped_utf8_string(o, m_field_names[i]);
            o << ") metadata:\n";
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

///////// properties on the dtype

static ndobject property_get_field_names(const dtype& dt) {
    const fixedstruct_dtype *d = static_cast<const fixedstruct_dtype *>(dt.extended());
    // TODO: This property should be an immutable ndobject, which we would just return.
    return ndobject(d->get_field_names_vector());
}

static ndobject property_get_data_offsets(const dtype& dt) {
    const fixedstruct_dtype *d = static_cast<const fixedstruct_dtype *>(dt.extended());
    // TODO: This property should be an immutable ndobject, which we would just return.
    return ndobject(d->get_data_offsets_vector());
}

static ndobject property_get_metadata_offsets(const dtype& dt) {
    const fixedstruct_dtype *d = static_cast<const fixedstruct_dtype *>(dt.extended());
    // TODO: This property should be an immutable ndobject, which we would just return.
    return ndobject(d->get_metadata_offsets_vector());
}

static pair<string, gfunc::callable> dtype_properties[] = {
    pair<string, gfunc::callable>("field_names", gfunc::make_callable(&property_get_field_names, "self")),
    // TODO field_types
    pair<string, gfunc::callable>("data_offsets", gfunc::make_callable(&property_get_data_offsets, "self")),
    pair<string, gfunc::callable>("metadata_offsets", gfunc::make_callable(&property_get_metadata_offsets, "self"))
};

void fixedstruct_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = dtype_properties;
    *out_count = sizeof(dtype_properties) / sizeof(dtype_properties[0]);
}

///////// properties on the ndobject

fixedstruct_dtype::fixedstruct_dtype(int, int)
    : base_struct_dtype(fixedstruct_type_id, 0, 1, 1, dtype_flag_none, 0)
{
    // Equivalent to make_fixedstruct_dtype(dtype(new void_pointer_dtype, false), "self");
    // but hardcoded to break the dependency of fixedstruct_dtype::ndobject_parameters_dtype
    m_field_types.push_back(dtype(new void_pointer_dtype, 0));
    m_field_names.push_back("self");
    m_data_offsets.push_back(0);
    m_metadata_offsets.push_back(0);
    m_memory_management = pod_memory_management;
    // TODO: Handle object, and object+blockref memory management types as well
    if (m_field_types[0].get_memory_management() == blockref_memory_management) {
        m_memory_management = blockref_memory_management;
    }
    // If any fields require zero-initialization, flag the struct as requiring it
    m_members.flags |= (m_field_types[0].get_flags()&dtype_flag_zeroinit);
    m_members.alignment = (uint8_t)m_field_types[0].get_alignment();
    m_members.metadata_size = m_field_types[0].get_metadata_size();
    m_members.data_size = m_field_types[0].get_data_size();
    // Leave m_ndobject_properties so there is no reference loop
}

static ndobject_preamble *property_get_ndobject_field(const ndobject_preamble *params, void *extra)
{
    // Get the ndobject 'self' parameter
    ndobject n = ndobject(*(ndobject_preamble **)params->m_data_pointer, true);
    intptr_t i = reinterpret_cast<intptr_t>(extra);
    size_t undim = n.get_undim();
    dtype udt = n.get_udtype();
    if (udt.get_kind() == expression_kind) {
        const string *field_names = static_cast<const fixedstruct_dtype *>(
                        udt.value_dtype().extended())->get_field_names();
        return n.replace_udtype(make_property_dtype(udt, field_names[i], i)).release();
    } else {
        if (undim == 0) {
            return n.at(i).release();
        } else {
            shortvector<irange> idx(undim + 1);
            idx[undim] = irange(i);
            return n.at_array(undim + 1, idx.get()).release();
        }
    }
}

void fixedstruct_dtype::create_ndobject_properties()
{
    dtype ndobject_parameters_dtype(new fixedstruct_dtype(0, 0), false);

    m_ndobject_properties.resize(m_field_types.size());
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        // TODO: Transform the name into a valid Python symbol?
        m_ndobject_properties[i].first = m_field_names[i];
        m_ndobject_properties[i].second.set(ndobject_parameters_dtype, &property_get_ndobject_field, (void *)i);
    }
}

void fixedstruct_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_ndobject_properties.empty() ? NULL : &m_ndobject_properties[0];
    *out_count = (int)m_ndobject_properties.size();
}
