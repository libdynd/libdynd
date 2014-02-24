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

bool base_memory_type::is_strided() const
{
    return m_target_tp.extended()->is_strided();
}

void base_memory_type::process_strided(const char *metadata, const char *data,
                ndt::type& out_dt, const char *&out_origin,
                intptr_t& out_stride, intptr_t& out_dim_size) const
{
    return m_target_tp.extended()->process_strided(metadata, data, out_dt, out_origin, out_stride, out_dim_size);
}


bool base_memory_type::is_type_subarray(const ndt::type& subarray_tp) const
{
    return m_target_tp.extended()->is_type_subarray(subarray_tp);
}

ndt::type base_memory_type::get_canonical_type() const
{
    return m_target_tp;
}

intptr_t base_memory_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_type, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_tp,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    return m_target_tp.extended()->apply_linear_index(nindices, indices, metadata, result_type, out_metadata,
        embedded_reference, current_i, root_tp, leading_dimension, inout_data, inout_dataref);
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
