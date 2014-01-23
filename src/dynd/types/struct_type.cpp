//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

struct_type::struct_type(const std::vector<ndt::type>& field_types, const std::vector<std::string>& field_names)
    : base_struct_type(struct_type_id, 0, 1, field_types.size(), type_flag_none, 0),
            m_field_types(field_types), m_field_names(field_names), m_metadata_offsets(field_types.size())
{
    if (field_types.size() != field_names.size()) {
        throw dynd::type_error("The field names for a dynd struct types must match the size of the field types");
    }

    // Calculate the needed element alignment
    size_t metadata_offset = field_types.size() * sizeof(size_t);
    m_members.data_alignment = 1;
    for (size_t i = 0, i_end = field_types.size(); i != i_end; ++i) {
        size_t field_alignment = field_types[i].get_data_alignment();
        // Accumulate the biggest field alignment as the type alignment
        if (field_alignment > m_members.data_alignment) {
            m_members.data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        m_members.flags |= (field_types[i].get_flags()&type_flags_operand_inherited);
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].is_builtin() ? 0 : m_field_types[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;

    create_array_properties();
}

struct_type::~struct_type()
{
}

intptr_t struct_type::get_field_index(const std::string& field_name) const
{
    // TODO: Put a map<> or unordered_map<> in the type to accelerate this lookup
    vector<string>::const_iterator i = find(m_field_names.begin(), m_field_names.end(), field_name);
    if (i != m_field_names.end()) {
        return i - m_field_names.begin();
    } else {
        return -1;
    }
}

size_t struct_type::get_default_data_size(intptr_t ndim, const intptr_t *shape) const
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


void struct_type::print_data(std::ostream& o, const char *metadata, const char *data) const
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

void struct_type::print_type(std::ostream& o) const
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

bool struct_type::is_expression() const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_field_types[i].is_expression()) {
            return true;
        }
    }
    return false;
}

bool struct_type::is_unique_data_owner(const char *metadata) const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (!m_field_types[i].is_builtin() &&
                !m_field_types[i].extended()->is_unique_data_owner(metadata + m_metadata_offsets[i])) {
            return false;
        }
    }
    return true;
}

void struct_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    std::vector<ndt::type> tmp_field_types(m_field_types.size());

    bool was_transformed = false;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        transform_fn(m_field_types[i], extra, tmp_field_types[i], was_transformed);
    }
    if (was_transformed) {
        out_transformed_tp = ndt::type(new struct_type(tmp_field_types, m_field_names), false);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type struct_type::get_canonical_type() const
{
    std::vector<ndt::type> fields(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        fields[i] = m_field_types[i].get_canonical_type();
    }

    return ndt::type(new struct_type(fields, m_field_names), false);
}

ndt::type struct_type::apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const
{
    if (nindices == 0) {
        return ndt::type(this, true);
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_tp,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            return m_field_types[start_index].apply_linear_index(nindices - 1, indices + 1,
                            current_i + 1, root_tp, leading_dimension);
        } else if (nindices == 1 && start_index == 0 && index_stride == 1 &&
                        (size_t)dimension_size == m_field_types.size()) {
            // This is a do-nothing index, keep the same type
            return ndt::type(this, true);
        } else {
            // Take the subset of the fixed fields in-place
            std::vector<ndt::type> field_types(dimension_size);
            std::vector<std::string> field_names(dimension_size);

            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                field_types[i] = m_field_types[idx].apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_tp, false);
                field_names[i] = m_field_names[idx];
            }

            return ndt::type(new struct_type(field_types, field_names), false);
        }
    }
}

intptr_t struct_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_tp, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_tp,
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
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_tp,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const ndt::type& dt = m_field_types[start_index];
            intptr_t offset = offsets[start_index];
            if (!dt.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[start_index], result_tp,
                                    out_metadata, embedded_reference, current_i + 1, root_tp,
                                    true, inout_data, inout_dataref);
                } else {
                    offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[start_index], result_tp,
                                    out_metadata, embedded_reference, current_i + 1, root_tp,
                                    false, NULL, NULL);
                }
            }
            return offset;
        } else {
            const struct_type *result_e_dt = static_cast<const struct_type *>(result_tp.extended());
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = offsets[idx];
                const ndt::type& dt = result_e_dt->m_field_types[i];
                if (!dt.is_builtin()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[idx], dt,
                                    out_metadata + result_e_dt->m_metadata_offsets[i],
                                    embedded_reference, current_i + 1, root_tp,
                                    false, NULL, NULL);
                }
            }
            return 0;
        }
    }
}

bool struct_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == struct_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

size_t struct_type::make_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
        if (this == src_tp.extended()) {
            return make_struct_identical_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp,
                            dst_metadata, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_tp.get_kind() == struct_kind) {
            return make_struct_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_tp.is_builtin()) {
            return make_broadcast_to_struct_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else {
            return src_tp.extended()->make_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t struct_type::make_comparison_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& src0_dt, const char *src0_metadata,
                const ndt::type& src1_dt, const char *src1_metadata,
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

bool struct_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != struct_type_id) {
        return false;
    } else {
        const struct_type *dt = static_cast<const struct_type*>(&rhs);
        return get_data_alignment() == dt->get_data_alignment() &&
                m_field_types == dt->m_field_types &&
                m_field_names == dt->m_field_names;
    }
}

void struct_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim > 0) {
        if (shape[0] >= 0 && shape[0] != (intptr_t)m_field_types.size()) {
            stringstream ss;
            ss << "Cannot construct dynd object of type " << ndt::type(this, true);
            ss << " with dimension size " << shape[0] << ", the size must be " << m_field_types.size();
            throw dynd::type_error(ss.str());
        }
    }

    size_t *offsets = reinterpret_cast<size_t *>(metadata);
    size_t offs = 0;
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
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

void struct_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy all the field offsets
    memcpy(dst_metadata, src_metadata, m_field_types.size() * sizeof(intptr_t));
    // Copy construct all the field's metadata
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_copy_construct(dst_metadata + m_metadata_offsets[i],
                            src_metadata + m_metadata_offsets[i],
                            embedded_reference);
        }
    }
}

void struct_type::metadata_reset_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (field_dt.get_metadata_size() > 0) {
            field_dt.extended()->metadata_reset_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_type::metadata_finalize_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_finalize_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_type::metadata_destruct(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
        }
    }
}

void struct_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
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
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin() && field_dt.extended()->get_metadata_size() > 0) {
            o << indent << " field " << i << " (name " << m_field_names[i] << ") metadata:\n";
            field_dt.extended()->metadata_debug_print(metadata + m_metadata_offsets[i], o, indent + "  ");
        }
    }
}

void struct_type::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    if (!m_field_types.empty()) {
        const size_t *offsets = reinterpret_cast<const size_t *>(metadata);
        const ndt::type *fields = &m_field_types[0];
        const size_t *metadata_offsets = &m_metadata_offsets[0];
        for (intptr_t i = 0, i_end = m_field_types.size(); i < i_end; ++i) {
            callback(fields[i], data + offsets[i], metadata + metadata_offsets[i], callback_data);
        }
    }
}

static nd::array property_get_field_names(const ndt::type& dt) {
    const struct_type *d = static_cast<const struct_type *>(dt.extended());
    // TODO: This property could be an immutable nd::array, which we would just return.
    return nd::array(d->get_field_names_vector());
}

static nd::array property_get_field_types(const ndt::type& dt) {
    const cstruct_type *d = static_cast<const cstruct_type *>(dt.extended());
    // TODO: This property should be an immutable nd::array, which we would just return.
    return nd::array(d->get_field_types_vector());
}

static nd::array property_get_metadata_offsets(const ndt::type& dt) {
    const struct_type *d = static_cast<const struct_type *>(dt.extended());
    // TODO: This property could be an immutable nd::array, which we would just return.
    return nd::array(d->get_metadata_offsets_vector());
}

static pair<string, gfunc::callable> type_properties[] = {
    pair<string, gfunc::callable>("field_names", gfunc::make_callable(&property_get_field_names, "self")),
    pair<string, gfunc::callable>("field_types", gfunc::make_callable(&property_get_field_types, "self")),
    pair<string, gfunc::callable>("metadata_offsets", gfunc::make_callable(&property_get_metadata_offsets, "self"))
};

void struct_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

ndt::type struct_type::array_parameters_type = ndt::make_cstruct(ndt::type(new void_pointer_type, false), "self");

static array_preamble *property_get_array_field(const array_preamble *params, void *extra)
{
    // Get the nd::array 'self' parameter
    nd::array n = nd::array(*(array_preamble **)params->m_data_pointer, true);
    intptr_t i = reinterpret_cast<intptr_t>(extra);
    intptr_t undim = n.get_ndim();
    ndt::type udt = n.get_dtype();
    if (udt.get_kind() == expression_kind) {
        const string *field_names = static_cast<const struct_type *>(
                        udt.value_type().extended())->get_field_names();
        return n.replace_dtype(ndt::make_property(udt, field_names[i], i)).release();
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

void struct_type::create_array_properties()
{
    m_array_properties.resize(m_field_types.size());
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        // TODO: Transform the name into a valid Python symbol?
        m_array_properties[i].first = m_field_names[i];
        m_array_properties[i].second.set(array_parameters_type, &property_get_array_field, (void *)i);
    }
}

void struct_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
    *out_count = (int)m_array_properties.size();
}
