//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_MEMORY_TYPE_HPP_
#define _DYND__BASE_MEMORY_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {

/**
 * Base class for all types of memory_kind.
 */
class base_memory_type : public base_type {
protected:
    ndt::type m_target_tp;
public:
    inline base_memory_type(type_id_t type_id, type_kind_t kind, size_t data_size,
            size_t alignment, flags_type flags, size_t metadata_size, size_t undim,
            const ndt::type& target_tp)
        : base_type(type_id, kind, data_size, alignment, flags, metadata_size, undim),
        m_target_tp(target_tp)
    {
    }

    virtual ~base_memory_type();

    virtual size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    void shift();

    virtual bool is_type_subarray(const ndt::type& subarray_tp) const;

    bool is_memory() const;

    virtual ndt::type get_canonical_type() const;

    virtual bool is_strided() const;

    virtual intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_type, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    virtual bool operator==(const base_type& rhs) const;

    virtual void data_alloc(char **data, size_t size) const = 0;
    virtual void data_zeroinit(char *data, size_t size) const = 0;
};

} // namespace dynd

#endif // _DYND__BASE_MEMORY_TYPE_HPP_
