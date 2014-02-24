//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>

using namespace std;
using namespace dynd;

base_memory_type::~base_memory_type()
{
}

void base_memory_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    m_target_tp.print_data(o, metadata + m_target_metadata_offset, data);
}

void base_memory_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_target_tp, extra, tmp_tp, was_transformed);
    if (was_transformed) {
        out_transformed_tp = with_replaced_target_type(tmp_tp);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

ndt::type base_memory_type::get_canonical_type() const
{
    return m_target_tp.get_canonical_type();
}

ndt::type base_memory_type::apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const
{
    return with_replaced_target_type(m_target_tp.apply_linear_index(nindices, indices,
                    current_i, root_tp, leading_dimension));
}

intptr_t base_memory_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_tp, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_tp,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    return m_target_tp.extended()->apply_linear_index(nindices, indices, metadata + m_target_metadata_offset, result_tp,
                    out_metadata, embedded_reference, current_i, root_tp, leading_dimension, inout_data, inout_dataref);
}

ndt::type base_memory_type::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    return with_replaced_target_type(m_target_tp.at_single(i0, inout_metadata ?
                    inout_metadata + m_target_metadata_offset : NULL, inout_data));
}

ndt::type base_memory_type::get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim) const
{
    return with_replaced_target_type(m_target_tp.get_type_at_dimension(inout_metadata ?
                    inout_metadata + m_target_metadata_offset : NULL, i, total_ndim));
}

void base_memory_type::get_shape(intptr_t ndim, intptr_t i,
                intptr_t *out_shape, const char *metadata, const char *data) const
{
    return m_target_tp.extended()->get_shape(ndim, i, out_shape, metadata ?
                    metadata + m_target_metadata_offset : NULL, data);
}

void base_memory_type::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    return m_target_tp.extended()->get_strides(i, out_strides, metadata ?
                    metadata + m_target_metadata_offset : NULL);
}

void base_memory_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata + m_target_metadata_offset, ndim, shape);
    }
}

void base_memory_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_copy_construct(dst_metadata + m_target_metadata_offset,
                        src_metadata + m_target_metadata_offset, embedded_reference);
    }
}

void base_memory_type::metadata_destruct(char *metadata) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(metadata + m_target_metadata_offset);
    }
}

























size_t base_memory_type::get_default_data_size(intptr_t ndim, const intptr_t *shape) const {
    if (m_target_tp.is_builtin()) {
        return m_target_tp.get_data_size();
    } else {
        return m_target_tp.extended()->get_default_data_size(ndim, shape);
    }
}

bool base_memory_type::is_memory() const
{
    return true;
}


bool base_memory_type::is_type_subarray(const ndt::type& subarray_tp) const
{
    return m_target_tp.extended()->is_type_subarray(subarray_tp);
}

bool base_memory_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_kind() != memory_kind) {
        return false;
    } else {
        const base_memory_type *dt = static_cast<const base_memory_type*>(&rhs);
        return m_target_tp == dt->m_target_tp;
    }
}
