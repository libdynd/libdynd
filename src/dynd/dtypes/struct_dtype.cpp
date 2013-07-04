//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

struct_dtype::struct_dtype(const std::vector<dtype>& field_types, const std::vector<std::string>& field_names)
    : base_struct_dtype(struct_type_id, 0, 1, field_types.size(), dtype_flag_none, 0),
            m_field_types(field_types), m_field_names(field_names), m_metadata_offsets(field_types.size())
{
    if (field_types.size() != field_names.size()) {
        throw runtime_error("The field names for a struct dtypes must match the size of the field dtypes");
    }

    // Calculate the needed element alignment
    size_t metadata_offset = field_types.size() * sizeof(size_t);
    m_members.data_alignment = 1;
    for (size_t i = 0, i_end = field_types.size(); i != i_end; ++i) {
        size_t field_alignment = field_types[i].get_data_alignment();
        // Accumulate the biggest field alignment as the dtype alignment
        if (field_alignment > m_members.data_alignment) {
            m_members.data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        m_members.flags |= (field_types[i].get_flags()&dtype_flags_operand_inherited);
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].is_builtin() ? 0 : m_field_types[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;

    create_array_properties();
}

struct_dtype::~struct_dtype()
{
}

intptr_t struct_dtype::get_field_index(const std::string& field_name) const
{
    // TODO: Put a map<> or unordered_map<> in the dtype to accelerate this lookup
    vector<string>::const_iterator i = find(m_field_names.begin(), m_field_names.end(), field_name);
    if (i != m_field_names.end()) {
        return i - m_field_names.begin();
    } else {
        return -1;
    }
}

size_t struct_dtype::get_default_data_size(size_t ndim, const intptr_t *shape) const
{
    // Default layout is to match the field order - could reorder the elements for more efficient packing
    size_t s = 0;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        s = inc_to_alignment(s, m_field_types[i].get_data_alignment());
        if (!m_field_types[i].is_builtin()) {
            s += m_field_types[i].extended()->get_default_data_size(ndim, shape);
        } else {
            s += m_field_types[i].get_data_size();
        }
    }
    s = inc_to_alignment(s, m_members.data_alignment);
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

bool struct_dtype::is_expression() const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_field_types[i].is_expression()) {
            return true;
        }
    }
    return false;
}

bool struct_dtype::is_unique_data_owner(const char *metadata) const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (!m_field_types[i].is_builtin() &&
                !m_field_types[i].extended()->is_unique_data_owner(metadata + m_metadata_offsets[i])) {
            return false;
        }
    }
    return true;
}

void struct_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    std::vector<dtype> tmp_field_types(m_field_types.size());

    bool was_transformed = false;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        transform_fn(m_field_types[i], extra, tmp_field_types[i], was_transformed);
    }
    if (was_transformed) {
        out_transformed_dtype = dtype(new struct_dtype(tmp_field_types, m_field_names), false);
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

    return dtype(new struct_dtype(fields, m_field_names), false);
}

dtype struct_dtype::apply_linear_index(size_t nindices, const irange *indices,
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

            return dtype(new struct_dtype(field_types, field_names), false);
        }
    }
}

intptr_t struct_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
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
        const intptr_t *offsets = reinterpret_cast<const intptr_t *>(metadata);
        intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_metadata);
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_dt,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const dtype& dt = m_field_types[start_index];
            intptr_t offset = offsets[start_index];
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
        } else {
            const struct_dtype *result_e_dt = static_cast<const struct_dtype *>(result_dtype.extended());
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = offsets[idx];
                const dtype& dt = result_e_dt->m_field_types[i];
                if (!dt.is_builtin()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[idx], dt,
                                    out_metadata + result_e_dt->m_metadata_offsets[i],
                                    embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
            }
            return 0;
        }
    }
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

size_t struct_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
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

size_t struct_dtype::make_comparison_kernel(
                hierarchical_kernel *out, size_t offset_out,
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

    throw not_comparable_error(src0_dt, src1_dt, comptype);
}

bool struct_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != struct_type_id) {
        return false;
    } else {
        const struct_dtype *dt = static_cast<const struct_dtype*>(&rhs);
        return get_data_alignment() == dt->get_data_alignment() &&
                m_field_types == dt->m_field_types &&
                m_field_names == dt->m_field_names;
    }
}

void struct_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
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
        offs = inc_to_alignment(offs, field_dt.get_data_alignment());
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
        if (field_dt.get_metadata_size() > 0) {
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

static nd::array property_get_field_names(const dtype& dt) {
    const struct_dtype *d = static_cast<const struct_dtype *>(dt.extended());
    // TODO: This property could be an immutable nd::array, which we would just return.
    return nd::array(d->get_field_names_vector());
}

static nd::array property_get_field_types(const dtype& dt) {
    const cstruct_dtype *d = static_cast<const cstruct_dtype *>(dt.extended());
    // TODO: This property should be an immutable nd::array, which we would just return.
    return nd::array(d->get_field_types_vector());
}

static nd::array property_get_metadata_offsets(const dtype& dt) {
    const struct_dtype *d = static_cast<const struct_dtype *>(dt.extended());
    // TODO: This property could be an immutable nd::array, which we would just return.
    return nd::array(d->get_metadata_offsets_vector());
}

static pair<string, gfunc::callable> dtype_properties[] = {
    pair<string, gfunc::callable>("field_names", gfunc::make_callable(&property_get_field_names, "self")),
    pair<string, gfunc::callable>("field_types", gfunc::make_callable(&property_get_field_types, "self")),
    pair<string, gfunc::callable>("metadata_offsets", gfunc::make_callable(&property_get_metadata_offsets, "self"))
};

void struct_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = dtype_properties;
    *out_count = sizeof(dtype_properties) / sizeof(dtype_properties[0]);
}

dtype struct_dtype::array_parameters_dtype = make_cstruct_dtype(dtype(new void_pointer_dtype, false), "self");

static array_preamble *property_get_array_field(const array_preamble *params, void *extra)
{
    // Get the nd::array 'self' parameter
    nd::array n = nd::array(*(array_preamble **)params->m_data_pointer, true);
    intptr_t i = reinterpret_cast<intptr_t>(extra);
    size_t undim = n.get_undim();
    dtype udt = n.get_udtype();
    if (udt.get_kind() == expression_kind) {
        const string *field_names = static_cast<const struct_dtype *>(
                        udt.value_dtype().extended())->get_field_names();
        return n.replace_udtype(make_property_dtype(udt, field_names[i], i)).release();
    } else {
        if (undim == 0) {
            return n(i).release();
        } else {
            shortvector<irange> idx(undim + 1);
            idx[undim] = irange(i);
            return n.at_array(undim + 1, idx.get()).release();
        }
    }
}

void struct_dtype::create_array_properties()
{
    m_array_properties.resize(m_field_types.size());
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        // TODO: Transform the name into a valid Python symbol?
        m_array_properties[i].first = m_field_names[i];
        m_array_properties[i].second.set(array_parameters_dtype, &property_get_array_field, (void *)i);
    }
}

void struct_dtype::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
    *out_count = (int)m_array_properties.size();
}
