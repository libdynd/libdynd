//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_TUPLE_TYPE_HPP_
#define _DYND__BASE_TUPLE_TYPE_HPP_

#include <dynd/types/base_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/array.hpp>

namespace dynd {


/**
 * Base class for all tuple types. If a type
 * has kind tuple_kind, it must be a subclass of
 * base_tuple_type.
 */
class base_tuple_type : public base_type {
protected:
    intptr_t m_field_count;
    // m_field_types always has type "strided * type",
    // and m_arrmeta_offsets always has type "strided * uintptr"
    nd::array m_field_types, m_arrmeta_offsets;

    virtual uintptr_t *get_arrmeta_data_offsets(char *DYND_UNUSED(arrmeta)) const {
        return NULL;
    }
public:
    base_tuple_type(type_id_t type_id, const nd::array &field_types,
                    flags_type flags, bool variable_layout);

    virtual ~base_tuple_type();

    /** The number of fields in the struct. This is the size of the other arrays. */
    inline intptr_t get_field_count() const {
        return m_field_count;
    }
    /** The array of the field types */
    inline const nd::array& get_field_types() const {
        return m_field_types;
    }
    inline const ndt::type *get_field_types_raw() const {
        return reinterpret_cast<const ndt::type *>(
            m_field_types.get_readonly_originptr());
    }
    /** The array of the field data offsets */
    virtual const uintptr_t *get_data_offsets(const char *arrmeta) const = 0;
    /** The array of the field arrmeta offsets */
    inline const nd::array &get_arrmeta_offsets() const {
        return m_arrmeta_offsets;
    }
    inline const uintptr_t *get_arrmeta_offsets_raw() const {
        return reinterpret_cast<const uintptr_t *>(
            m_arrmeta_offsets.get_readonly_originptr());
    }

    inline const ndt::type& get_field_type(intptr_t i) const {
        return unchecked_strided_dim_get<ndt::type>(m_field_types, i);
    }
    inline const uintptr_t& get_arrmeta_offset(intptr_t i) const {
        return unchecked_strided_dim_get<uintptr_t>(m_arrmeta_offsets, i);
    }


    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;
    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;

    size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                    const char *arrmeta, const char *data) const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                    const ndt::type& result_tp, char *out_arrmeta,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                 memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data,
                    intptr_t stride, size_t count) const;

    void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                         void *callback_data) const;
};


} // namespace dynd

#endif // _DYND__BASE_TUPLE_TYPE_HPP_
 
