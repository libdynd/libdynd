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
    size_t m_target_metadata_offset;
public:
    inline base_memory_type(type_id_t type_id, type_kind_t kind, size_t data_size,
            size_t alignment, flags_type flags, size_t metadata_size, size_t undim,
            const ndt::type& target_tp)
        : base_type(type_id, kind, data_size, alignment, flags, metadata_size, undim),
        m_target_tp(target_tp), m_target_metadata_offset(0)
    {
    }

    virtual ~base_memory_type();

    inline const ndt::type& get_target_type() const {
        return m_target_tp;
    }

    virtual void print_data(std::ostream& o, const char *metadata, const char *data) const;

    virtual void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;

    virtual ndt::type get_canonical_type() const;

    virtual ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    virtual intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *metadata,
                    const ndt::type& result_tp, char *out_metadata,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    virtual ndt::type at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;
    virtual ndt::type get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim = 0) const;

    virtual void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                    const char *metadata, const char *data) const;
    virtual void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;

    virtual ndt::type with_replaced_target_type(const ndt::type& target_tp) const = 0;

    virtual void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    virtual void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    virtual void metadata_destruct(char *metadata) const;

    virtual void data_alloc(char **data, size_t size) const = 0;
    virtual void data_zeroinit(char *data, size_t size) const = 0;
    virtual void data_free(char *data) const = 0;









    virtual size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    bool is_type_subarray(const ndt::type& subarray_tp) const;

    bool is_memory() const;

    virtual bool operator==(const base_type& rhs) const;

};

inline const ndt::type& get_target_type(const ndt::type& tp) {
    if (tp.get_kind() == memory_kind) {
        return static_cast<const base_memory_type*>(tp.extended())->get_target_type();
    } else {
        return tp;
    }
}

} // namespace dynd

#endif // _DYND__BASE_MEMORY_TYPE_HPP_
