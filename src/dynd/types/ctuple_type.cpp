//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/ctuple_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/struct_comparison_kernels.hpp>

using namespace std;
using namespace dynd;

ctuple_type::ctuple_type(size_t field_count, const ndt::type *field_types)
    : base_tuple_type(ctuple_type_id, 0, 1, field_count, type_flag_none, 0),
      m_field_types(field_types, field_types + field_count),
      m_data_offsets(field_count), m_metadata_offsets(field_count)
{
    // Calculate all the resulting struct data
    size_t metadata_offset = 0, data_offset = 0;
    m_members.data_alignment = 1;
    for (size_t i = 0; i != field_count; ++i) {
        size_t field_alignment = field_types[i].get_data_alignment();
        // Accumulate the biggest field alignment as the type alignment
        if (field_alignment > m_members.data_alignment) {
            m_members.data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        m_members.flags |= (field_types[i].get_flags()&type_flags_operand_inherited);
        // Calculate the data offsets
        data_offset = inc_to_alignment(data_offset, field_types[i].get_data_alignment());
        m_data_offsets[i] = data_offset;
        size_t field_element_size = field_types[i].get_data_size();
        if (field_element_size == 0) {
            stringstream ss;
            ss << "Cannot create dynd ctuple type with type " << field_types[i];
            ss << " for field at index " << i << ", as it does not have a fixed size";
            throw runtime_error(ss.str());
        }
        data_offset += field_element_size;
        // Calculate the metadata offsets
        m_metadata_offsets[i] = metadata_offset;
        metadata_offset += m_field_types[i].is_builtin() ? 0 : m_field_types[i].extended()->get_metadata_size();
    }
    m_members.metadata_size = metadata_offset;
    m_members.data_size = inc_to_alignment(data_offset, m_members.data_alignment);
}

ctuple_type::~ctuple_type()
{
}

void ctuple_type::print_data(std::ostream& o, const char *metadata, const char *data) const
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

void ctuple_type::print_type(std::ostream& o) const
{
    // Use the tuple datashape syntax prefixed with a "c"
    o << "c(";
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (i != 0) {
            o << ", ";
        }
        o << m_field_types[i];
    }
    o << ")";
}

bool ctuple_type::is_expression() const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (m_field_types[i].is_expression()) {
            return true;
        }
    }
    return false;
}

bool ctuple_type::is_unique_data_owner(const char *metadata) const
{
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        if (!m_field_types[i].is_builtin() &&
                !m_field_types[i].extended()->is_unique_data_owner(metadata + m_metadata_offsets[i])) {
            return false;
        }
    }
    return true;
}

void ctuple_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    std::vector<ndt::type> tmp_field_types(m_field_types.size());

    bool switch_to_tuple = false;
    bool was_any_transformed = false;
    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        bool was_transformed = false;
        transform_fn(m_field_types[i], extra, tmp_field_types[i], was_transformed);
        if (was_transformed) {
            // If the type turned into one without fixed size, have to use tuple instead of ctuple
            if (tmp_field_types[i].get_data_size() == 0) {
                switch_to_tuple = true;
            }
            was_any_transformed = true;
        }
    }
    if (was_any_transformed) {
        if (!switch_to_tuple) {
            out_transformed_tp =
                ndt::make_ctuple(tmp_field_types.size(), &tmp_field_types[0]);
        } else {
            out_transformed_tp =
                ndt::make_tuple(tmp_field_types.size(), &tmp_field_types[0]);
        }
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type ctuple_type::get_canonical_type() const
{
    std::vector<ndt::type> field_types(m_field_types.size());

    for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
        field_types[i] = m_field_types[i].get_canonical_type();
    }

    return ndt::make_tuple(m_field_types.size(), &field_types[0]);
}

ndt::type ctuple_type::apply_linear_index(intptr_t nindices, const irange *indices,
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

            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                field_types[i] = m_field_types[idx].apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_tp, false);
            }
            // Return a tuple type, because the offsets are now not in standard form anymore
            return ndt::make_tuple(field_types.size(),
                                field_types.empty() ? NULL : &field_types[0]);
        }
    }
}

intptr_t ctuple_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
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
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_field_types.size(), current_i, &root_tp,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            const ndt::type& dt = m_field_types[start_index];
            intptr_t offset = m_data_offsets[start_index];
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
        } else if (result_tp.get_type_id() == ctuple_type_id) {
            // This was a no-op, so copy everything verbatim
            for (size_t i = 0, i_end = m_field_types.size(); i != i_end; ++i) {
                if (!m_field_types[i].is_builtin()) {
                    if (m_field_types[i].extended()->apply_linear_index(0, NULL,
                                    metadata + m_metadata_offsets[i], m_field_types[i], out_metadata + m_metadata_offsets[i],
                                    embedded_reference, current_i + 1, root_tp,
                                    false, NULL, NULL) != 0) {
                        stringstream ss;
                        ss << "Unexpected non-zero offset when applying a NULL index to dynd type " << m_field_types[i];
                        throw runtime_error(ss.str());
                    }
                }
            }
            return 0;
        } else {
            intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_metadata);
            const tuple_type *result_etp = result_tp.tcast<tuple_type>();
            for (intptr_t i = 0; i < dimension_size; ++i) {
                intptr_t idx = start_index + i * index_stride;
                out_offsets[i] = m_data_offsets[idx];
                const ndt::type& dt = result_etp->get_field_types()[i];
                if (!dt.is_builtin()) {
                    out_offsets[i] += dt.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + m_metadata_offsets[idx],
                                    dt, out_metadata + result_etp->get_metadata_offsets()[i],
                                    embedded_reference, current_i + 1, root_tp,
                                    false, NULL, NULL);
                }
            }
            return 0;
        }
    }
}

ndt::type ctuple_type::at_single(intptr_t i0,
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

bool ctuple_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        if (src_tp.extended() == this) {
            return true;
        } else if (src_tp.get_type_id() == ctuple_type_id) {
            return *dst_tp.extended() == *src_tp.extended();
        }
    }

    return false;
}

size_t ctuple_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(out_ckb), size_t DYND_UNUSED(ckb_offset),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_metadata),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    /*
    if (this == dst_tp.extended()) {
        if (this == src_tp.extended()) {
            return make_tuple_identical_assignment_kernel(
                out_ckb, ckb_offset, dst_tp, dst_metadata, src_metadata,
                kernreq, errmode, ectx);
        } else if (src_tp.get_kind() == struct_kind) {
            return make_tuple_assignment_kernel(out_ckb, ckb_offset,
                            dst_tp, dst_metadata,
                            src_tp, src_metadata,
                            kernreq, errmode, ectx);
        } else if (src_tp.is_builtin()) {
            return make_broadcast_to_tuple_assignment_kernel(out_ckb, ckb_offset,
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
    */

    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}

size_t ctuple_type::make_comparison_kernel(
                ckernel_builder *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const ndt::type& src0_tp, const char *DYND_UNUSED(src0_metadata),
                const ndt::type& src1_tp, const char *DYND_UNUSED(src1_metadata),
                comparison_type_t comptype,
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    /*
    if (this == src0_tp.extended()) {
        if (*this == *src1_tp.extended()) {
            return make_tuple_comparison_kernel(out, offset_out,
                            src0_tp, src0_metadata, src1_metadata,
                            comptype, ectx);
        } else if (src1_tp.get_kind() == struct_kind) {
            return make_general_tuple_comparison_kernel(out, offset_out,
                            src0_tp, src0_metadata,
                            src1_tp, src1_metadata,
                            comptype, ectx);
        }
    }
    */

    throw not_comparable_error(src0_tp, src1_tp, comptype);
}

bool ctuple_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != ctuple_type_id) {
        return false;
    } else {
        const ctuple_type *dt = static_cast<const ctuple_type*>(&rhs);
        return get_data_alignment() == dt->get_data_alignment() &&
                m_field_types == dt->m_field_types;
    }
}

void ctuple_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim > 0) {
        if (shape[0] >= 0 && shape[0] != (intptr_t)m_field_types.size()) {
            stringstream ss;
            ss << "Cannot construct dynd object of type " << ndt::type(this, true);
            ss << " with dimension size " << shape[0] << ", the size must be " << m_field_types.size();
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
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

void ctuple_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
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

void ctuple_type::metadata_reset_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (field_dt.get_metadata_size() > 0) {
            field_dt.extended()->metadata_reset_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void ctuple_type::metadata_finalize_buffers(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_finalize_buffers(metadata + m_metadata_offsets[i]);
        }
    }
}

void ctuple_type::metadata_destruct(char *metadata) const
{
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin()) {
            field_dt.extended()->metadata_destruct(metadata + m_metadata_offsets[i]);
        }
    }
}

void ctuple_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    o << indent << "ctuple metadata\n";
    for (size_t i = 0; i < m_field_types.size(); ++i) {
        const ndt::type& field_dt = m_field_types[i];
        if (!field_dt.is_builtin() && field_dt.extended()->get_metadata_size() > 0) {
            o << indent << " field " << i << " metadata:\n";
            field_dt.extended()->metadata_debug_print(metadata + m_metadata_offsets[i], o, indent + "  ");
        }
    }
}

void ctuple_type::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    if (!m_field_types.empty()) {
        const ndt::type *field_types = &m_field_types[0];
        const size_t *metadata_offsets = &m_metadata_offsets[0];
        for (intptr_t i = 0, i_end = m_field_types.size(); i < i_end; ++i) {
            callback(field_types[i], data + m_data_offsets[i], metadata + metadata_offsets[i], callback_data);
        }
    }
}

///////// properties on the type

static nd::array property_get_field_types(const ndt::type& dt) {
    const ctuple_type *d = dt.tcast<ctuple_type>();
    // TODO: This property should be an immutable nd::array, which we would just return.
    return nd::array(d->get_field_types_vector());
}

static nd::array property_get_data_offsets(const ndt::type& dt) {
    const ctuple_type *d = dt.tcast<ctuple_type>();
    // TODO: This property should be an immutable nd::array, which we would just return.
    return nd::array(d->get_data_offsets_vector());
}

static nd::array property_get_metadata_offsets(const ndt::type& dt) {
    const ctuple_type *d = dt.tcast<ctuple_type>();
    // TODO: This property should be an immutable nd::array, which we would just return.
    return nd::array(d->get_metadata_offsets_vector());
}

static pair<string, gfunc::callable> type_properties[] = {
    pair<string, gfunc::callable>("field_types", gfunc::make_callable(&property_get_field_types, "self")),
    pair<string, gfunc::callable>("data_offsets", gfunc::make_callable(&property_get_data_offsets, "self")),
    pair<string, gfunc::callable>("metadata_offsets", gfunc::make_callable(&property_get_metadata_offsets, "self"))
};

void ctuple_type::get_dynamic_type_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

