//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BASE_TUPLE_TYPE_HPP_
#define _DYND__BASE_TUPLE_TYPE_HPP_

#include <dynd/types/base_type.hpp>

namespace dynd {


/**
 * Base class for all tuple types. If a type
 * has kind tuple_kind, it must be a subclass of
 * base_tuple_type.
 */
class base_tuple_type : public base_type {
protected:
    size_t m_field_count;
public:
    inline base_tuple_type(type_id_t type_id, size_t data_size, size_t alignment,
                           size_t field_count, flags_type flags,
                           size_t metadata_size)
      : base_type(type_id, tuple_kind, data_size, alignment, flags,
                  metadata_size, 0),
        m_field_count(field_count)
    {}

    inline base_tuple_type(type_id_t type_id, type_kind_t kind,
                           size_t data_size, size_t alignment,
                           size_t field_count, flags_type flags,
                           size_t metadata_size)
        : base_type(type_id, kind, data_size, alignment, flags, metadata_size,
                    0),
          m_field_count(field_count)
    {}

    virtual ~base_tuple_type();

    /** The number of fields in the struct. This is the size of the other arrays. */
    inline size_t get_field_count() const {
        return m_field_count;
    }
    /** The array of the field types */
    virtual const ndt::type *get_field_types() const = 0;
    /** The array of the field data offsets */
    virtual const size_t *get_data_offsets(const char *metadata) const = 0;
    /** The array of the field metadata offsets */
    virtual const size_t *get_metadata_offsets() const = 0;

    size_t get_default_data_size(intptr_t ndim, const intptr_t *shape) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                    const char *metadata, const char *data) const;

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;
};


} // namespace dynd

#endif // _DYND__BASE_TUPLE_TYPE_HPP_
 
